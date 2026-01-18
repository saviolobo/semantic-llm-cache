"""
Semantic cache implementation using Redis Vector Search.
Handles index creation, similarity search, and cache storage.
"""

import redis
import numpy as np
from typing import Optional, Dict, Any
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from app.config import (
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB,
    REDIS_INDEX_NAME,
    VECTOR_DIM,
    SIMILARITY_THRESHOLD,
    CACHE_TTL_SECONDS,
)


class SemanticCache:
    """Redis-backed semantic cache using vector similarity search."""

    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False,
        )
        self._ensure_index()

    def _ensure_index(self):
        """Create vector search index if it doesn't exist."""
        try:
            self.client.ft(REDIS_INDEX_NAME).info()
        except redis.exceptions.ResponseError:
            # Index doesn't exist, create it
            schema = (
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": VECTOR_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
                TextField("query"),
                TextField("response"),
            )
            definition = IndexDefinition(
                prefix=[f"{REDIS_INDEX_NAME}:"], index_type=IndexType.HASH
            )
            self.client.ft(REDIS_INDEX_NAME).create_index(
                fields=schema, definition=definition
            )

    def search(self, query_embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Search for similar cached query using vector similarity.

        Args:
            query_embedding: Query vector embedding

        Returns:
            Dict with 'query', 'response', 'score' if cache hit, else None
        """
        # Convert numpy array to bytes for Redis
        embedding_bytes = query_embedding.astype(np.float32).tobytes()

        # KNN search query
        query = (
            Query(f"*=>[KNN 1 @embedding $vec AS score]")
            .return_fields("query", "response", "score")
            .sort_by("score")
            .dialect(2)
        )

        try:
            results = self.client.ft(REDIS_INDEX_NAME).search(
                query, query_params={"vec": embedding_bytes}
            )

            if results.docs:
                doc = results.docs[0]
                score = float(doc.score)

                # Cosine similarity: lower score = more similar
                # Convert to similarity (1 - distance)
                similarity = 1 - score

                if similarity >= SIMILARITY_THRESHOLD:
                    return {
                        "query": doc.query.decode("utf-8"),
                        "response": doc.response.decode("utf-8"),
                        "score": similarity,
                    }
        except redis.exceptions.ResponseError:
            # Index might be empty
            pass

        return None

    def store(self, query: str, response: str, query_embedding: np.ndarray):
        """
        Store query, response, and embedding in cache with TTL.

        Args:
            query: Original query text
            response: LLM response
            query_embedding: Query vector embedding
        """
        # Generate unique key
        key = f"{REDIS_INDEX_NAME}:{hash(query)}"

        # Convert embedding to bytes
        embedding_bytes = query_embedding.astype(np.float32).tobytes()

        # Store as hash with TTL
        self.client.hset(
            key,
            mapping={
                "query": query,
                "response": response,
                "embedding": embedding_bytes,
            },
        )
        self.client.expire(key, CACHE_TTL_SECONDS)

    def clear(self):
        """Clear all cache entries."""
        keys = self.client.keys(f"{REDIS_INDEX_NAME}:*")
        if keys:
            self.client.delete(*keys)
