"""
FastAPI server for semantic LLM caching.
Orchestrates embedding, cache lookup, LLM calls, and metrics.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from app.embeddings import get_embedding
from app.cache import SemanticCache
from app.llm import get_llm_response
from app.metrics import metrics

app = FastAPI(title="Semantic LLM Cache")

# Initialize cache
cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    response: str
    cached: bool
    latency_ms: float
    similarity_score: float | None = None


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process query with semantic caching.

    Flow:
    1. Generate embedding for query
    2. Search cache for similar queries
    3. If hit: return cached response
    4. If miss: call LLM, cache result, return response
    """
    start_time = time.time()

    try:
        # Step 1: Generate embedding
        query_embedding = get_embedding(request.query)

        # Step 2: Search cache
        cached_result = cache.search(query_embedding)

        if cached_result:
            # Cache hit
            latency_ms = (time.time() - start_time) * 1000
            metrics.record_hit(latency_ms)

            return QueryResponse(
                response=cached_result["response"],
                cached=True,
                latency_ms=round(latency_ms, 2),
                similarity_score=round(cached_result["score"], 4),
            )

        # Cache miss - call LLM
        llm_response = get_llm_response(request.query)

        # Store in cache
        cache.store(request.query, llm_response, query_embedding)

        latency_ms = (time.time() - start_time) * 1000
        metrics.record_miss(latency_ms)

        return QueryResponse(
            response=llm_response,
            cached=False,
            latency_ms=round(latency_ms, 2),
            similarity_score=None,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """Get current cache metrics."""
    return metrics.get_stats()


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics to zero."""
    metrics.reset()
    return {"status": "metrics reset"}


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached entries."""
    cache.clear()
    return {"status": "cache cleared"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
