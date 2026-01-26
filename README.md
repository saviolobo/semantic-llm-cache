# Semantic LLM Cache

A production-style semantic caching layer for LLM applications that reduces API costs and latency by caching semantically similar queries using vector embeddings and Redis Stack.

## Demo

![Semantic LLM Cache Demo](assets/demo.gif)

## Overview

LLM API calls are expensive (~$0.01-0.10 per request) and slow (500ms-3s latency). This system intercepts queries, checks for semantically similar past queries, and returns cached responses when similarity exceeds a configurable threshold.

**Key Features:**
- Semantic similarity matching using vector embeddings
- Redis Stack with vector search for fast lookup
- Real-time metrics tracking (hit rate, latency, cost savings)
- Clean separation of concerns across modules
- Live Streamlit dashboard for monitoring and testing
- Rate limiting (10 requests/minute per IP)
- Input validation and error handling

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         FastAPI (main.py)               │
│  Orchestration & Request Handling       │
└─────────────────────────────────────────┘
       │
       ├──► embeddings.py ──► Sentence-Transformers
       │                       (all-MiniLM-L6-v2)
       │
       ├──► cache.py ──────► Redis Stack
       │                     (Vector Search)
       │
       ├──► llm.py ────────► Groq API
       │                     (llama-3.1-8b-instant)
       │
       └──► metrics.py ────► In-Memory Stats
```

### Component Responsibilities

| Module | Purpose |
|--------|---------|
| **main.py** | FastAPI server, orchestrates caching flow |
| **cache.py** | Redis interface for vector similarity search |
| **embeddings.py** | Converts text to vectors using Sentence-Transformers |
| **llm.py** | Thin wrapper around Groq API |
| **metrics.py** | Thread-safe in-memory metrics tracking |
| **config.py** | Centralized configuration and constants |
| **dashboard/** | Streamlit metrics visualization |

## Design Decisions

### Why Build Custom Cache (vs RedisVL/LangChain)?
- Demonstrates understanding of vector similarity and system design
- Can explain every architectural choice
- Minimal dependencies: Only Redis primitives, no abstraction layers
- Flexibility: Full control over caching logic and metrics

### Technology Choices

**Redis Stack (Vector Search)**
- Native vector similarity search with COSINE distance
- Fast KNN lookups (<50ms p95)
- Built-in TTL for automatic cache expiration
- RedisInsight UI for debugging

**Sentence-Transformers (all-MiniLM-L6-v2)**
- Runs locally (no API key needed)
- 384-dimensional embeddings
- Good balance of speed and quality
- ~50ms encoding time

**Groq API (llama-3.1-8b-instant)**
- Fast inference (~200-500ms)
- Free tier available
- Good quality for general queries
- Easy to swap for other providers

### Caching Strategy

**Similarity Threshold: 0.85**
- Cosine similarity metric (range: 0-1)
- Higher = stricter matching (fewer false positives)
- Tunable via `config.py`

**TTL: 1 hour**
- Balances freshness vs cache hit rate
- Prevents stale responses for evolving topics
- Configurable per use case

## Setup

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- Groq API key (free at https://console.groq.com)

### Installation

1. **Clone repository**
```bash
git clone <repo-url>
cd semantic-llm-cache
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Start Redis Stack**
```bash
docker-compose up -d
```

4. **Set environment variables**
Create `.env` file:
```bash
GROQ_API_KEY=gsk_your_key_here
```

5. **Run API server**
```bash
uvicorn app.main:app --reload
```

6. **Run dashboard (optional)**
```bash
streamlit run dashboard/metrics_dashboard.py
```

## Usage

### API Endpoints

**POST /query** - Submit query with semantic caching
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

Response:
```json
{
  "response": "Machine learning is...",
  "cached": false,
  "latency_ms": 245.67,
  "similarity_score": null
}
```

**GET /metrics** - Current cache statistics
```bash
curl http://localhost:8000/metrics
```

**POST /cache/clear** - Clear all cached entries
```bash
curl -X POST http://localhost:8000/cache/clear
```

**POST /metrics/reset** - Reset metrics counters
```bash
curl -X POST http://localhost:8000/metrics/reset
```

### Interactive Docs
FastAPI auto-generates interactive API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Dashboard
Access real-time metrics dashboard:
- Streamlit UI: http://localhost:8501

## Metrics

The system tracks:

| Metric | Description |
|--------|-------------|
| **Total Requests** | All queries processed |
| **Cache Hits** | Queries served from cache |
| **Cache Misses** | Queries requiring LLM call |
| **Hit Rate** | (Hits / Total) × 100% |
| **Avg Latency** | Mean response time (ms) |
| **Cost Savings** | Estimated $ saved from avoided API calls |

### Expected Performance

- **Cache hit latency**: 30-80ms (embedding + Redis lookup)
- **Cache miss latency**: 200-800ms (+ LLM API call)
- **Typical hit rate**: 40-70% for repeated/similar queries
- **Cost reduction**: ~50% API cost savings at 50% hit rate

## Configuration

Edit `app/config.py` to adjust:

```python
# Cache behavior
SIMILARITY_THRESHOLD = 0.85  # 0.0-1.0 (higher = stricter)
CACHE_TTL_SECONDS = 3600     # 1 hour

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence-Transformers
VECTOR_DIM = 384

# LLM settings
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 500
```

## Project Structure

```
semantic-llm-cache/
├── app/
│   ├── __init__.py       # Package init
│   ├── main.py           # FastAPI orchestration
│   ├── cache.py          # Redis vector search
│   ├── embeddings.py     # Sentence-Transformers
│   ├── llm.py            # Groq API client
│   ├── metrics.py        # Performance tracking
│   └── config.py         # Configuration
├── dashboard/
│   ├── __init__.py       # Package init
│   └── metrics_dashboard.py  # Streamlit UI
├── tests/
│   ├── test_api.py       # API endpoint tests
│   ├── test_cache.py     # Cache logic tests
│   ├── test_embeddings.py # Embedding tests
│   └── test_metrics.py   # Metrics tests
├── docker-compose.yml    # Redis Stack setup
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
├── PRD.md               # Technical requirements
└── README.md            # This file
```

## Development

### Testing the Cache

#### Via API (curl)

1. Send initial query (cache miss):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain neural networks"}'
# cached: false, ~500ms latency
```

2. Send similar query (cache hit):
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are neural networks?"}'
# cached: true, ~50ms latency, similarity_score: 0.92
```

3. Check metrics:
```bash
curl http://localhost:8000/metrics
```

#### Via Dashboard

1. Start the Streamlit dashboard:
```bash
streamlit run dashboard/metrics_dashboard.py
```

2. Open http://localhost:8501 in your browser

3. Send queries via API or Swagger UI (http://localhost:8000/docs)

4. Watch the dashboard update:
   - Total requests incrementing
   - Cache hit rate increasing as you repeat similar queries
   - Average latency decreasing on cache hits
   - Cost savings accumulating

Click "🔄 Refresh Metrics" to update the stats after sending queries.

## Running Tests

```bash
pytest tests/ -v
```

## Future Enhancements

- [ ] Persistent metrics (Redis/PostgreSQL)
- [ ] Cache invalidation strategies beyond TTL
- [ ] Multi-tenant support with namespace isolation
- [ ] Distributed caching across regions
- [ ] A/B testing different similarity thresholds
- [ ] Integration with other LLM providers (OpenAI, Anthropic)

## License

MIT
