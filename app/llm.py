"""
LLM client for Groq API.
Thin wrapper for making LLM requests.
"""

import logging
from groq import Groq, GroqError
from app.config import LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

logger = logging.getLogger(__name__)

# Initialize client once at module level
_client = Groq(api_key=LLM_API_KEY)


class LLMError(Exception):
    """Custom exception for LLM-related errors."""
    pass


def get_llm_response(query: str) -> str:
    """
    Get response from LLM for the given query.

    Args:
        query: User query text

    Returns:
        LLM response as string

    Raises:
        LLMError: If the LLM request fails
    """
    try:
        completion = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": query}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        return completion.choices[0].message.content

    except GroqError as e:
        logger.error(f"Groq API error: {e}")
        raise LLMError(f"Failed to get LLM response: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error calling LLM: {e}")
        raise LLMError(f"Unexpected error: {e}") from e
