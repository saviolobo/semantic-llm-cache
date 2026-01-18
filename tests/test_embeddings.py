"""
Tests for the embeddings module.
"""

import numpy as np
import pytest
from app.embeddings import get_embedding
from app.config import VECTOR_DIM


class TestEmbeddings:
    """Tests for embedding generation."""

    def test_get_embedding_returns_numpy_array(self):
        """Embedding should return a numpy array."""
        result = get_embedding("test query")
        assert isinstance(result, np.ndarray)

    def test_get_embedding_correct_dimension(self):
        """Embedding should have the configured dimension."""
        result = get_embedding("test query")
        assert result.shape == (VECTOR_DIM,)

    def test_get_embedding_different_inputs_different_outputs(self):
        """Different inputs should produce different embeddings."""
        embedding1 = get_embedding("hello world")
        embedding2 = get_embedding("goodbye moon")
        assert not np.allclose(embedding1, embedding2)

    def test_get_embedding_similar_inputs_similar_outputs(self):
        """Similar inputs should produce similar embeddings."""
        embedding1 = get_embedding("What is machine learning?")
        embedding2 = get_embedding("Explain machine learning")

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        assert similarity > 0.8  # Should be highly similar

    def test_get_embedding_empty_string(self):
        """Empty string should still return valid embedding."""
        result = get_embedding("")
        assert isinstance(result, np.ndarray)
        assert result.shape == (VECTOR_DIM,)
