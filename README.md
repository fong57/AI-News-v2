# LiteNews AI - News Writing Automation Agent

A modular LangGraph agent for automated news writing, research, and content generation.

## Features

- **Multi-LLM Support**: Use Perplexity (with web search) or Qwen (DashScope) for reasoning
- **Tavily Search Integration**: Real-time web search for news research
- **LangSmith Tracing**: Built-in observability, debugging, and monitoring
- **Modular Architecture**: Easily swap LLMs, configure search, and customize workflows
- **LangGraph Cloud Ready**: Deploy to LangGraph Cloud with CLI

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Test Integrations

```bash
python scripts/test_integrations.py
```

### 4. Run Locally

```bash
langgraph dev
```

## Project Structure

```
LiteNews_AI_v2/
├── langgraph.json          # LangGraph Cloud deployment config
├── pyproject.toml          # Python package config
├── .env.example            # Example environment variables
├── src/litenews/
│   ├── config/             # User-configurable settings
│   │   ├── settings.py     # Main settings (Pydantic)
│   │   └── llm_config.py   # LLM-specific configurations
│   ├── llms/               # LLM integrations
│   │   ├── base.py         # Base LLM interface
│   │   ├── perplexity.py   # Perplexity integration
│   │   └── qwen.py         # Qwen/DashScope integration
│   ├── tools/              # Tool integrations
│   │   └── search.py       # Tavily search tool
│   ├── state/              # LangGraph state definitions
│   │   └── news_state.py   # News agent state
│   └── graphs/             # LangGraph workflows
│       └── news_graph.py   # Main news writing graph
├── tests/                  # Test suite
└── scripts/                # Utility scripts
```

## Configuration

All configuration is done via environment variables or the settings module:

| Variable | Description | Default |
|----------|-------------|---------|
| `PRIMARY_LLM` | Primary reasoning LLM (`perplexity` or `qwen`) | `perplexity` |
| `PERPLEXITY_MODEL` | Perplexity model name | `sonar-pro` |
| `QWEN_MODEL` | Qwen model name | `qwen-plus` |
| `TAVILY_SEARCH_DEPTH` | Search depth | `advanced` |
| `MAX_TOKENS` | Max tokens for responses | `4096` |
| `TEMPERATURE` | LLM temperature | `0.7` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `false` |
| `LANGCHAIN_API_KEY` | LangSmith API key | - |
| `LANGCHAIN_PROJECT` | LangSmith project name | `litenews-ai` |

## LangSmith Tracing

LangSmith provides observability for your LangGraph workflows. To enable:

1. Get an API key from [smith.langchain.com](https://smith.langchain.com/)
2. Add to your `.env`:
   ```
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_key_here
   ```
3. View traces at [smith.langchain.com](https://smith.langchain.com/)

## Deploy to LangGraph Cloud

```bash
# Test locally first
langgraph dev

# Deploy
langgraph deploy
```

## License

MIT


# Run all workflow tests
pytest tests/workflow/ -v
# Run specific node tests
pytest tests/workflow/nodes/test_research.py -v