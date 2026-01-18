"""
FastAPI server for semantic LLM caching.
Orchestrates embedding, cache lookup, LLM calls, and metrics.
"""

import logging
import time
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.embeddings import get_embedding
from app.cache import SemanticCache
from app.llm import get_llm_response, LLMError
from app.metrics import metrics
from app.config import LLM_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Validate API key on startup
if not LLM_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

# Rate limiter - uses client IP address
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Semantic LLM Cache")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize cache
cache = SemanticCache()

# Query length limits
MAX_QUERY_LENGTH = 2000
MIN_QUERY_LENGTH = 1

# Rate limit configuration
RATE_LIMIT = "10/minute"


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=MIN_QUERY_LENGTH,
        max_length=MAX_QUERY_LENGTH,
        description="The query to process",
    )


class QueryResponse(BaseModel):
    response: str
    cached: bool
    latency_ms: float
    similarity_score: float | None = None


@app.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT)
async def query(request: Request, query_request: QueryRequest):
    """
    Process query with semantic caching.

    Flow:
    1. Generate embedding for query
    2. Search cache for similar queries
    3. If hit: return cached response
    4. If miss: call LLM, cache result, return response

    Rate limited to 10 requests per minute per IP.
    """
    start_time = time.time()
    logger.info(f"Processing query: {query_request.query[:50]}...")

    try:
        # Step 1: Generate embedding
        query_embedding = get_embedding(query_request.query)

        # Step 2: Search cache
        cached_result = cache.search(query_embedding)

        if cached_result:
            # Cache hit
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_hit(latency_ms)
            logger.info(f"Cache HIT - similarity: {cached_result['score']:.4f}, latency: {latency_ms:.2f}ms")

            return QueryResponse(
                response=cached_result["response"],
                cached=True,
                latency_ms=round(latency_ms, 2),
                similarity_score=round(cached_result["score"], 4),
            )

        # Cache miss - call LLM
        logger.info("Cache MISS - calling LLM")
        llm_response = get_llm_response(query_request.query)

        # Store in cache
        cache.store(query_request.query, llm_response, query_embedding)

        latency_ms = (time.time() - start_time) * 1000
        metrics.record_miss(latency_ms)
        logger.info(f"LLM response cached, latency: {latency_ms:.2f}ms")

        return QueryResponse(
            response=llm_response,
            cached=False,
            latency_ms=round(latency_ms, 2),
            similarity_score=None,
        )

    except LLMError as e:
        logger.error(f"LLM error: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
async def get_metrics():
    """Get current cache metrics."""
    return metrics.get_stats()


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics to zero."""
    metrics.reset()
    logger.info("Metrics reset")
    return {"status": "metrics reset"}


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached entries."""
    cache.clear()
    logger.info("Cache cleared")
    return {"status": "cache cleared"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
