"""
LLM client for Groq API.
Thin wrapper for making LLM requests.
"""

from groq import Groq
from app.config import LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Initialize client once at module level
_client = Groq(api_key=LLM_API_KEY)


def get_llm_response(query: str) -> str:
    """
    Get response from LLM for the given query.

    Args:
        query: User query text

    Returns:
        LLM response as string
    """
    completion = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": query}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )

    return completion.choices[0].message.content
