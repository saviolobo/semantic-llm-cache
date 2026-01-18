"""
Embedding generation using Sentence-Transformers.
Converts text queries to vector embeddings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import EMBEDDING_MODEL

# Load model once at module level
_model = SentenceTransformer(EMBEDDING_MODEL)


def get_embedding(text: str) -> np.ndarray:
    """
    Convert text to vector embedding.

    Args:
        text: Input text query

    Returns:
        numpy array of shape (VECTOR_DIM,)
    """
    embedding = _model.encode(text, convert_to_numpy=True)
    return embedding
