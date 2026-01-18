"""
Configuration for semantic LLM cache.
All constants and settings in one place.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Redis Configuration
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
REDIS_INDEX_NAME: str = "llm_cache_index"

# Cache Configuration
CACHE_TTL_SECONDS: int = 3600  # 1 hour
SIMILARITY_THRESHOLD: float = 0.85  # Minimum cosine similarity for cache hit

# Embedding Configuration
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Sentence-Transformers model
VECTOR_DIM: int = 384  # Dimension for all-MiniLM-L6-v2

# LLM Configuration (Groq)
LLM_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
LLM_MODEL: str = "llama-3.1-8b-instant"
LLM_TEMPERATURE: float = 0.7
LLM_MAX_TOKENS: int = 500

# Metrics Configuration
METRICS_ENABLED: bool = True
