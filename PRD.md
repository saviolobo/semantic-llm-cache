# Semantic LLM Cache - Product Requirements Document

## Problem
LLM API calls are expensive (~$0.01-0.10 per request) and slow (500ms-3s latency). Identical or semantically similar queries result in redundant API calls.

## Solution
Build a semantic caching layer that intercepts LLM queries, checks for similar past queries using vector embeddings, and returns cached responses when similarity exceeds threshold.

## Technical Architecture

### Core Components
1. **FastAPI Server** - Handles incoming queries, orchestrates caching flow
2. **Embedding Layer** - Converts queries to vectors using Sentence-Transformers
3. **Cache Layer** - Redis Stack with vector search for similarity matching
4. **LLM Client** - Handles upstream LLM API calls on cache miss
5. **Metrics System** - Tracks hit rate, latency, cost savings

### Data Flow
```
Query → Embed → Vector Search → Hit? → Return cached response
                                 ↓ Miss
                          Call LLM → Cache response → Return
```

## Key Metrics
- **Cache Hit Rate**: % of queries served from cache
- **Latency Reduction**: Average response time (cache hit vs miss)
- **Cost Savings**: API calls avoided × cost per call
- **Similarity Threshold**: Configurable (default: 0.85)

## Technical Requirements

### Functional
- Accept text queries via REST API
- Return cached response if similarity > threshold
- Call LLM and cache result on miss
- Store embeddings with TTL (configurable, default: 1 hour)
- Expose metrics endpoint

### Non-Functional
- Cache lookup < 50ms (p95)
- Support concurrent requests
- Handle Redis failures gracefully
- Log all cache decisions for debugging

## Out of Scope
- Multi-tenancy / user authentication
- Distributed caching across regions
- Cache invalidation strategies beyond TTL
- Production deployment configuration

## Success Criteria
- Demonstrable cost savings on repeated queries
- Measurable latency improvement on cache hits
- Clean, testable code suitable for portfolio review
- Working metrics dashboard
