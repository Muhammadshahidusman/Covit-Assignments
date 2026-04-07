"""
AgentLens — Quick Connectivity Test
Run with:  python main.py
"""

from config import OPENAI_API_KEY, OLLAMA_BASE_URL
from agent_core import get_openai_client, is_openai_configured
from ollama_utils import is_ollama_running, list_local_models


def test_openai() -> None:
    print("\n── OpenAI ────────────────────────────────")
    if not is_openai_configured():
        print("  ❌  API key missing or invalid in .env")
        return
    try:
        client = get_openai_client()
        # Lightweight call — no web search, just a completion
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Reply with OK"}],
            max_tokens=5,
        )
        print(f"  ✅  Connected — response: {resp.choices[0].message.content!r}")
    except Exception as exc:
        print(f"  ❌  Error: {exc}")


def test_ollama() -> None:
    print("\n── Ollama ────────────────────────────────")
    print(f"  Base URL: {OLLAMA_BASE_URL}")
    if not is_ollama_running():
        print("  ❌  Ollama is not running. Start it with: ollama serve")
        return
    print("  ✅  Ollama is running")
    models = list_local_models()
    if models:
        print(f"  📦  {len(models)} local model(s) installed:")
        for m in models:
            print(f"      • {m['name']}  ({m['parameters']}, {m['size_gb']} GB)")
    else:
        print("  ℹ️   No models found. Pull one with: ollama pull llama3.2")


if __name__ == "__main__":
    print("AgentLens — Connectivity Test")
    test_openai()
    test_ollama()
    print("\n✅  Test complete. Run the app with:  streamlit run app.py\n")
