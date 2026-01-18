"""
Embedding generation using Sentence-Transformers.
Converts text queries to vector embeddings.
"""

import logging
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# Load model once at module level
logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
_model = SentenceTransformer(EMBEDDING_MODEL)
logger.info("Embedding model loaded successfully")


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
