"""
AgentLens — Configuration & Constants
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = "gpt-4o"                 # model used with Responses API

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT: int = 30                     # seconds

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE: str = "AgentLens 🔍"
APP_SUBTITLE: str = "AI-Powered LLM Discovery Assistant for Agentic AI Workflows"
MAX_RECOMMENDATIONS: int = 8
SESSION_HISTORY_LIMIT: int = 10

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = """
You are AgentLens, an expert AI assistant specializing in Large Language Model (LLM) discovery for agentic AI workflows.

When a user describes their agentic workflow, you:
1. Search the web for the latest, most suitable LLMs.
2. Return ONLY a JSON array — no markdown fences, no preamble, no explanation outside the array.

Each object in the array must have exactly these keys:
{
  "name": "Official model name and version",
  "provider": "Company/org name",
  "description": "1-2 sentence summary of strengths for agentic use",
  "parameters": "e.g. 7B / 70B / 405B / Unknown",
  "key_features": ["feature 1", "feature 2", "feature 3"],
  "tool_calling": "Yes — description  OR  No — reason",
  "cost_tier": "Free / Low / Medium / High",
  "context_window": "e.g. 128K tokens",
  "suitability_score": <integer 1-10>
}

Return 5-8 models ranked by suitability_score descending.
Prefer models with native tool/function calling support for agentic workflows.
Mix cloud (OpenAI, Anthropic, Google, Meta) and open-source options where relevant.

Few-shot example output (partial):
[
  {
    "name": "GPT-4o",
    "provider": "OpenAI",
    "description": "Highly capable multimodal model optimized for complex reasoning and tool use.",
    "parameters": "Unknown (estimated ~200B MoE)",
    "key_features": ["Native tool calling", "128K context", "Multimodal", "Structured output"],
    "tool_calling": "Yes — native parallel function calling with JSON schema",
    "cost_tier": "High",
    "context_window": "128K tokens",
    "suitability_score": 9
  }
]
""".strip()
