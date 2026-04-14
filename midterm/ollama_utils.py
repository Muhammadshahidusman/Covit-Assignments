"""
AgentLens — Ollama Local Model Utilities
"""

from __future__ import annotations
import requests
from typing import Any
from config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT


# ── Connectivity ──────────────────────────────────────────────────────────────

def is_ollama_running() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_TIMEOUT)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


# ── Model listing ─────────────────────────────────────────────────────────────

def list_local_models() -> list[dict[str, Any]]:
    """
    Return metadata for every locally installed Ollama model.

    Each dict contains:
        name, family, parameters, quantization, size_gb
    """
    if not is_ollama_running():
        return []

    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        raw_models: list[dict] = r.json().get("models", [])
    except Exception:
        return []

    models: list[dict[str, Any]] = []
    for m in raw_models:
        details = m.get("details", {})
        size_bytes: int = m.get("size", 0)
        models.append(
            {
                "name": m.get("name", "unknown"),
                "family": details.get("family", "unknown"),
                "parameters": details.get("parameter_size", "unknown"),
                "quantization": details.get("quantization_level", "unknown"),
                "size_gb": round(size_bytes / 1e9, 2) if size_bytes else 0.0,
            }
        )
    return models


# ── Suitability scoring ───────────────────────────────────────────────────────

# Models known to support tool/function calling via Ollama
TOOL_CALLING_MODELS: set[str] = {
    "llama3", "llama3.1", "llama3.2", "llama3.3",
    "mistral", "mistral-nemo", "mixtral",
    "qwen2", "qwen2.5", "qwen2.5-coder",
    "phi3", "phi3.5",
    "hermes3", "nous-hermes2",
    "command-r", "command-r-plus",
    "deepseek-r1", "deepseek-coder-v2",
}


def _supports_tool_calling(model_name: str) -> bool:
    name_lower = model_name.lower().split(":")[0]   # strip tag (e.g. :latest)
    return any(known in name_lower for known in TOOL_CALLING_MODELS)


def score_local_model(model: dict[str, Any], query: str) -> int:
    """
    Heuristic suitability score (1-10) for a local model given a workflow query.
    """
    score = 5
    name_lower = model["name"].lower()
    query_lower = query.lower()

    # Tool calling bonus
    if _supports_tool_calling(model["name"]):
        score += 2

    # Size bonus — larger models tend to be more capable
    param_str = str(model.get("parameters", "")).lower()
    if "70b" in param_str or "72b" in param_str:
        score += 2
    elif "13b" in param_str or "14b" in param_str:
        score += 1
    elif "7b" in param_str or "8b" in param_str:
        pass  # baseline

    # Domain-keyword bonus
    domain_keywords = {
        "code": ["coder", "code", "deepseek", "starcoder", "qwen2.5-coder"],
        "marketing": ["llama", "mistral", "qwen", "command-r"],
        "customer": ["llama", "mistral", "hermes"],
        "math": ["qwen", "deepseek", "phi"],
    }
    for domain, keywords in domain_keywords.items():
        if domain in query_lower:
            if any(kw in name_lower for kw in keywords):
                score += 1
                break

    return min(max(score, 1), 10)


def enrich_local_models(models: list[dict], query: str) -> list[dict]:
    """Add tool_calling and suitability_score fields to each local model dict."""
    enriched = []
    for m in models:
        enriched.append(
            {
                **m,
                "tool_calling": _supports_tool_calling(m["name"]),
                "suitability_score": score_local_model(m, query),
            }
        )
    return sorted(enriched, key=lambda x: x["suitability_score"], reverse=True)


# ── Optional: send a test prompt ──────────────────────────────────────────────

def test_model(model_name: str, prompt: str, timeout: int = 60) -> str:
    """
    Send a test prompt to a local Ollama model and return its response text.
    Uses the streaming=False generate endpoint for simplicity.
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The model may still be loading."
    except Exception as exc:
        return f"⚠️ Error: {exc}"
