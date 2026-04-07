# AgentLens 🔍
> AI-Powered LLM Discovery Assistant for Agentic AI Workflows

## Setup

```bash
# 1. Install dependencies (using uv)
uv add openai ollama streamlit python-dotenv pandas

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OpenAI API key

# 3. Start Ollama (separate terminal)
ollama serve
ollama pull llama3.2   # optional — pull at least one model

# 4. Run connectivity test
python main.py

# 5. Launch the app
streamlit run app.py
```

## Usage

1. Describe your agentic AI workflow in the text area  
   *e.g. "I'm building a marketing automation agent that creates campaigns and analyses A/B tests"*
2. Click **Search LLMs**
3. AgentLens queries OpenAI's Responses API with live web search and returns 5-8 ranked LLMs
4. Compare cloud recommendations against your locally installed Ollama models

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `agent_core.py` | OpenAI Responses API + parsing logic |
| `ollama_utils.py` | Local Ollama model utilities |
| `config.py` | Settings & constants |
| `main.py` | Connectivity test script |

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (`sk-…`) |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |
