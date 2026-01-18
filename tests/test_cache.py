"""
Tests for the cache module.
"""

import hashlib
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.config import VECTOR_DIM, SIMILARITY_THRESHOLD


class TestCacheKeyGeneration:
    """Tests for cache key generation."""

    def test_sha256_key_consistency(self):
        """Same query should always produce same key."""
        query = "test query"
        hash1 = hashlib.sha256(query.encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(query.encode()).hexdigest()[:16]
        assert hash1 == hash2

    def test_sha256_key_uniqueness(self):
        """Different queries should produce different keys."""
        query1 = "test query 1"
        query2 = "test query 2"
        hash1 = hashlib.sha256(query1.encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(query2.encode()).hexdigest()[:16]
        assert hash1 != hash2

    def test_sha256_key_length(self):
        """Key should be 16 characters."""
        query = "test query"
        key = hashlib.sha256(query.encode()).hexdigest()[:16]
        assert len(key) == 16


class TestSimilarityThreshold:
    """Tests for similarity threshold logic."""

    def test_threshold_in_valid_range(self):
        """Threshold should be between 0 and 1."""
        assert 0.0 <= SIMILARITY_THRESHOLD <= 1.0

    def test_default_threshold_value(self):
        """Default threshold should be 0.85."""
        assert SIMILARITY_THRESHOLD == 0.85


class TestVectorDimension:
    """Tests for vector dimension configuration."""

    def test_vector_dim_matches_model(self):
        """Vector dimension should match all-MiniLM-L6-v2 output."""
        assert VECTOR_DIM == 384

    def test_embedding_conversion_to_bytes(self):
        """Embedding should convert to bytes correctly."""
        embedding = np.random.rand(VECTOR_DIM).astype(np.float32)
        embedding_bytes = embedding.tobytes()

        # FLOAT32 = 4 bytes per element
        expected_size = VECTOR_DIM * 4
        assert len(embedding_bytes) == expected_size

    def test_embedding_bytes_roundtrip(self):
        """Embedding should survive bytes conversion roundtrip."""
        original = np.random.rand(VECTOR_DIM).astype(np.float32)
        embedding_bytes = original.tobytes()
        restored = np.frombuffer(embedding_bytes, dtype=np.float32)

        assert np.allclose(original, restored)
