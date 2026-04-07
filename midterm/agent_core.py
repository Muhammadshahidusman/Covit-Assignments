"""
AgentLens — Core AI Engine
Uses OpenAI Responses API with built-in web search to discover & rank LLMs.
"""

from __future__ import annotations
import json
import re
from typing import Any

from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError

from config import OPENAI_API_KEY, OPENAI_MODEL, SYSTEM_PROMPT


# ── Client ────────────────────────────────────────────────────────────────────

def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def is_openai_configured() -> bool:
    return bool(OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"))


# ── Search ────────────────────────────────────────────────────────────────────

def search_llms(query: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Call the OpenAI Responses API with web search enabled.

    Returns:
        (recommendations, error_message)
        recommendations — list of parsed LLM dicts (may be empty on failure)
        error_message   — human-readable error string, or None on success
    """
    if not is_openai_configured():
        return [], "OpenAI API key is not configured. Add it to your .env file."

    client = get_openai_client()

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            tools=[{"type": "web_search_preview"}],
            instructions=SYSTEM_PROMPT,
            input=query,
        )
    except APIConnectionError:
        return [], "Could not connect to OpenAI. Check your internet connection."
    except RateLimitError:
        return [], "OpenAI rate limit reached. Please wait a moment and try again."
    except APIStatusError as exc:
        return [], f"OpenAI API error {exc.status_code}: {exc.message}"
    except Exception as exc:
        return [], f"Unexpected error: {exc}"

    # Extract text from the response output
    raw_text = _extract_text(response)
    recommendations, parse_error = _parse_recommendations(raw_text)

    if parse_error:
        return [], f"Response parsing failed: {parse_error}\n\nRaw response:\n{raw_text[:500]}"

    return recommendations, None


# ── Response parsing ──────────────────────────────────────────────────────────

def _extract_text(response: Any) -> str:
    """Pull the assistant's text content from a Responses API response object."""
    text_parts: list[str] = []
    for item in getattr(response, "output", []):
        item_type = getattr(item, "type", "")
        if item_type == "message":
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    text_parts.append(getattr(content, "text", ""))
    return "\n".join(text_parts).strip()


def _parse_recommendations(raw: str) -> tuple[list[dict], str | None]:
    """
    Extract a JSON array from the model's response text.
    Returns (list_of_dicts, error_or_None).
    """
    if not raw:
        return [], "Empty response from API."

    # Strip markdown code fences if present
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find the first '[' … last ']' block
    start = clean.find("[")
    end = clean.rfind("]")
    if start == -1 or end == -1:
        return [], "No JSON array found in response."

    json_str = clean[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return [], str(exc)

    if not isinstance(data, list):
        return [], "Parsed JSON is not a list."

    # Normalize — fill missing keys with sensible defaults
    normalised: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        normalised.append(
            {
                "name": item.get("name", "Unknown"),
                "provider": item.get("provider", "Unknown"),
                "description": item.get("description", "No description available."),
                "parameters": item.get("parameters", "Unknown"),
                "key_features": item.get("key_features", []),
                "tool_calling": item.get("tool_calling", "Unknown"),
                "cost_tier": item.get("cost_tier", "Unknown"),
                "context_window": item.get("context_window", "Unknown"),
                "suitability_score": int(item.get("suitability_score", 5)),
            }
        )

    normalised.sort(key=lambda x: x["suitability_score"], reverse=True)
    return normalised, None
