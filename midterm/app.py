"""
AgentLens — Streamlit Web Application
AI-Powered LLM Discovery Assistant for Agentic AI Workflows
"""

import streamlit as st
import pandas as pd
from typing import Any

from config import APP_TITLE, APP_SUBTITLE, SESSION_HISTORY_LIMIT
from agent_core import search_llms, is_openai_configured
from ollama_utils import (
    is_ollama_running,
    list_local_models,
    enrich_local_models,
    test_model,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgentLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[str] = []
if "last_results" not in st.session_state:
    st.session_state.last_results: list[dict] = []
if "last_query" not in st.session_state:
    st.session_state.last_query: str = ""

# ── Helpers ───────────────────────────────────────────────────────────────────

COST_COLORS = {"Free": "🟢", "Low": "🟡", "Medium": "🟠", "High": "🔴", "Unknown": "⚪"}

EXAMPLE_QUERIES = [
    "I'm building a marketing automation agent — recommend LLMs",
    "Best models for a customer support agentic workflow",
    "Which LLMs support tool calling for coding agents?",
    "Recommend LLMs for a research summarisation agent with long documents",
    "LLMs for a financial analysis agent that needs structured output",
]


def score_badge(score: int) -> str:
    if score >= 8:
        return f"🌟 {score}/10"
    if score >= 6:
        return f"✅ {score}/10"
    return f"🔵 {score}/10"


def render_llm_card(llm: dict[str, Any], idx: int) -> None:
    """Render a single cloud LLM recommendation card."""
    tool_ok = str(llm.get("tool_calling", "")).lower().startswith("yes")
    cost = llm.get("cost_tier", "Unknown")

    with st.container(border=True):
        col_title, col_badges = st.columns([3, 2])
        with col_title:
            st.markdown(f"### {idx}. {llm['name']}")
            st.caption(f"**Provider:** {llm['provider']}")
        with col_badges:
            st.markdown(
                f"{score_badge(llm['suitability_score'])} &nbsp;&nbsp; "
                f"{COST_COLORS.get(cost, '⚪')} {cost} &nbsp;&nbsp; "
                f"{'🔧 Tool Calling' if tool_ok else '⛔ No Tool Calling'}"
            )

        st.write(llm["description"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Parameters", llm["parameters"])
        c2.metric("Context Window", llm["context_window"])
        c3.metric("Tool Calling", "✅ Yes" if tool_ok else "❌ No")

        with st.expander("Key Features & Tool Calling Details"):
            features = llm.get("key_features", [])
            if features:
                for feat in features:
                    st.markdown(f"- {feat}")
            st.markdown(f"**Tool/Function Calling:** {llm['tool_calling']}")


def render_comparison_table(recommendations: list[dict]) -> None:
    """Render a sortable comparison DataFrame."""
    rows = [
        {
            "Name": r["name"],
            "Provider": r["provider"],
            "Parameters": r["parameters"],
            "Context": r["context_window"],
            "Tool Support": "✅" if r["tool_calling"].lower().startswith("yes") else "❌",
            "Cost": r["cost_tier"],
            "Score": r["suitability_score"],
        }
        for r in recommendations
    ]
    df = pd.DataFrame(rows).set_index("Name")
    st.dataframe(df, use_container_width=True)


def render_local_models(query: str) -> None:
    """Render the Ollama local models section."""
    st.markdown("---")
    st.subheader("🖥️ Local Models (Ollama)")

    if not is_ollama_running():
        st.warning(
            "Ollama is not running. Start it with `ollama serve` "
            "to see your local models here."
        )
        return

    models = list_local_models()
    if not models:
        st.info("No local models found. Pull one with `ollama pull llama3.2`.")
        return

    enriched = enrich_local_models(models, query)

    st.caption(f"**{len(enriched)} local model(s) installed**")

    # Cards
    for m in enriched:
        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"**{m['name']}**  ·  `{m['family']}`")
                st.caption(
                    f"Params: **{m['parameters']}** | "
                    f"Quant: `{m['quantization']}` | "
                    f"Size: **{m['size_gb']} GB**"
                )
            with c2:
                st.markdown(
                    f"{'🔧 Tool Calling' if m['tool_calling'] else '⛔ No Tool Calling'} &nbsp; "
                    f"{score_badge(m['suitability_score'])}"
                )

            # Bonus: test prompt
            with st.expander("🧪 Test this model"):
                test_input = st.text_input(
                    "Prompt",
                    placeholder="Type a quick test prompt…",
                    key=f"test_{m['name']}",
                )
                if st.button("Run", key=f"run_{m['name']}"):
                    with st.spinner(f"Running {m['name']}…"):
                        result = test_model(m["name"], test_input)
                    st.text_area("Response", result, height=150, key=f"resp_{m['name']}")

    # Cloud vs Local comparison
    with st.expander("📊 Cloud vs Local — Tradeoff Summary"):
        st.table(
            pd.DataFrame(
                {
                    "Dimension": ["Cost", "Privacy", "Latency", "Capability", "Rate Limits"],
                    "Cloud Models": ["💰 API pricing", "☁️ Data leaves device", "🌐 Network dependent", "🚀 State-of-the-art", "⚠️ Subject to limits"],
                    "Local (Ollama)": ["🆓 Free", "🔒 Fully private", "⚡ Sub-second local", "📉 Smaller models", "✅ Unlimited"],
                }
            ).set_index("Dimension")
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ System Status")

        # OpenAI
        if is_openai_configured():
            st.success("🟢 OpenAI API — Connected")
        else:
            st.error("🔴 OpenAI API — No key found")

        # Ollama
        if is_ollama_running():
            local_count = len(list_local_models())
            st.success(f"🟢 Ollama — Running ({local_count} models)")
        else:
            st.warning("🟡 Ollama — Not running")

        st.divider()

        # Search history
        st.subheader("📜 Search History")
        if st.session_state.history:
            for i, q in enumerate(reversed(st.session_state.history[-SESSION_HISTORY_LIMIT:]), 1):
                if st.button(f"{i}. {q[:40]}…" if len(q) > 40 else f"{i}. {q}", key=f"hist_{i}"):
                    st.session_state.last_query = q
        else:
            st.caption("No searches yet.")

        st.divider()

        # How it works
        with st.expander("ℹ️ How It Works"):
            st.markdown(
                """
1. **Describe** your agentic AI workflow in natural language.
2. **AgentLens** queries OpenAI's Responses API with live web search.
3. The model searches the internet for the **latest LLM data** and returns structured recommendations.
4. **Local models** from Ollama are listed alongside for cost/privacy comparison.
5. A **comparison table** lets you evaluate all options side by side.
                """
            )


# ── Main layout ───────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()

    st.title(APP_TITLE)
    st.markdown(f"*{APP_SUBTITLE}*")
    st.divider()

    # Query input
    query = st.text_area(
        "Describe your agentic AI workflow",
        value=st.session_state.last_query,
        placeholder=EXAMPLE_QUERIES[0],
        height=100,
        help="Be specific about the task — e.g. 'marketing automation agent that creates campaigns and analyzes A/B tests'.",
    )

    # Example query buttons
    with st.expander("💡 Example queries"):
        cols = st.columns(2)
        for i, eq in enumerate(EXAMPLE_QUERIES):
            if cols[i % 2].button(eq, key=f"eq_{i}"):
                st.session_state.last_query = eq
                st.rerun()

    search_clicked = st.button("🔍 Search LLMs", type="primary", use_container_width=True)

    # ── Run search ────────────────────────────────────────────────────────────
    if search_clicked and query.strip():
        st.session_state.last_query = query

        # Save to history
        if query not in st.session_state.history:
            st.session_state.history.append(query)

        with st.spinner("🔎 Searching the web for the best LLMs…"):
            results, error = search_llms(query)

        if error:
            st.error(f"**Search failed:** {error}")
        else:
            st.session_state.last_results = results
            st.success(f"Found **{len(results)}** recommended LLMs for your workflow!")

    elif search_clicked and not query.strip():
        st.warning("Please describe your agentic workflow before searching.")

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.last_results:
        results = st.session_state.last_results
        active_query = st.session_state.last_query

        st.subheader(f"🤖 Recommended LLMs for: _{active_query}_")

        # Cards
        for idx, llm in enumerate(results, 1):
            render_llm_card(llm, idx)

        # Comparison table
        st.divider()
        st.subheader("📊 Side-by-Side Comparison")
        render_comparison_table(results)

    # ── Local models ──────────────────────────────────────────────────────────
    render_local_models(st.session_state.last_query or "")


if __name__ == "__main__":
    main()
