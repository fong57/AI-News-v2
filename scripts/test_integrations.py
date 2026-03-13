#!/usr/bin/env python3
"""Test script for LLM and search integrations.

This script tests the connection to all configured APIs.
Run it to verify your API keys are working correctly.

Usage:
    python scripts/test_integrations.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from litenews.config.settings import get_settings, reload_settings
from litenews.config.tracing import test_langsmith_connection, get_tracing_status
from litenews.llms.perplexity import test_perplexity_connection
from litenews.llms.qwen import test_qwen_connection
from litenews.tools.search import test_tavily_connection


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name: str, result: dict):
    """Print a formatted test result."""
    status = result.get("status", "unknown")
    icon = "✓" if status == "success" else "✗"
    
    print(f"\n{icon} {name}")
    print(f"  Status: {status}")
    
    if status == "success":
        if "model" in result:
            print(f"  Model: {result['model']}")
        if "response" in result:
            response = str(result["response"])[:100]
            print(f"  Response: {response}...")
        if "results_count" in result:
            print(f"  Results: {result['results_count']}")
    else:
        print(f"  Error: {result.get('error', 'Unknown error')}")


def test_perplexity():
    """Test Perplexity API connection."""
    settings = get_settings()
    
    if not settings.has_perplexity_key():
        return {"status": "skipped", "error": "PPLX_API_KEY not configured"}
    
    return test_perplexity_connection(
        api_key=settings.pplx_api_key,
        model=settings.perplexity_model,
    )


def test_qwen():
    """Test Qwen/DashScope API connection."""
    settings = get_settings()
    
    if not settings.has_qwen_key():
        return {"status": "skipped", "error": "DASHSCOPE_API_KEY not configured"}
    
    return test_qwen_connection(
        api_key=settings.dashscope_api_key,
        model=settings.qwen_model,
    )


def test_tavily():
    """Test Tavily API connection."""
    settings = get_settings()
    
    if not settings.has_tavily_key():
        return {"status": "skipped", "error": "TAVILY_API_KEY not configured"}
    
    return test_tavily_connection(api_key=settings.tavily_api_key)


def test_langsmith():
    """Test LangSmith API connection."""
    settings = get_settings()
    
    if not settings.has_langsmith_key():
        return {"status": "skipped", "error": "LANGCHAIN_API_KEY not configured"}
    
    return test_langsmith_connection(api_key=settings.langchain_api_key)


async def test_full_workflow():
    """Test a simplified version of the news workflow."""
    from litenews.state.news_state import create_initial_state
    from litenews.graphs.news_graph import research_node
    
    settings = get_settings()
    
    if not settings.has_tavily_key():
        return {"status": "skipped", "error": "TAVILY_API_KEY not configured for workflow test"}
    
    state = create_initial_state("artificial intelligence latest developments")
    
    try:
        result = await research_node(state)
        if result.get("error"):
            return {"status": "error", "error": result["error"]}
        
        search_count = len(result.get("search_results", []))
        return {
            "status": "success",
            "search_results_count": search_count,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    """Run all integration tests."""
    print_header("LiteNews AI - Integration Tests")
    
    reload_settings()
    settings = get_settings()
    
    print("\nConfiguration:")
    print(f"  Primary LLM: {settings.primary_llm}")
    print(f"  Perplexity Model: {settings.perplexity_model}")
    print(f"  Qwen Model: {settings.qwen_model}")
    print(f"  Search Depth: {settings.tavily_search_depth}")
    print(f"  LangSmith Project: {settings.langchain_project}")
    
    print_header("Testing LLM Connections")
    
    print("\n1. Testing Perplexity...")
    result = test_perplexity()
    print_result("Perplexity API", result)
    perplexity_ok = result.get("status") == "success"
    
    print("\n2. Testing Qwen (DashScope)...")
    result = test_qwen()
    print_result("Qwen/DashScope API", result)
    qwen_ok = result.get("status") == "success"
    
    print_header("Testing Search")
    
    print("\n3. Testing Tavily Search...")
    result = test_tavily()
    print_result("Tavily Search API", result)
    tavily_ok = result.get("status") == "success"
    
    print_header("Testing Observability")
    
    print("\n4. Testing LangSmith...")
    result = test_langsmith()
    print_result("LangSmith API", result)
    langsmith_ok = result.get("status") == "success"
    
    tracing_status = get_tracing_status()
    print(f"\n  Tracing enabled: {tracing_status['enabled']}")
    print(f"  Project: {tracing_status['project']}")
    
    print_header("Testing Workflow")
    
    print("\n5. Testing Research Node...")
    result = asyncio.run(test_full_workflow())
    print_result("Research Node", result)
    workflow_ok = result.get("status") == "success"
    
    print_header("Summary")
    
    all_ok = all([perplexity_ok or qwen_ok, tavily_ok])
    
    print(f"\n  Perplexity:  {'✓ Ready' if perplexity_ok else '✗ Not configured/failed'}")
    print(f"  Qwen:        {'✓ Ready' if qwen_ok else '✗ Not configured/failed'}")
    print(f"  Tavily:      {'✓ Ready' if tavily_ok else '✗ Not configured/failed'}")
    print(f"  LangSmith:   {'✓ Ready' if langsmith_ok else '○ Optional (not configured)'}")
    print(f"  Workflow:    {'✓ Ready' if workflow_ok else '✗ Not configured/failed'}")
    
    if all_ok:
        print("\n✓ All core integrations are working!")
        if langsmith_ok:
            print("  LangSmith tracing is enabled for observability.")
        print("  You can now run: langgraph dev")
    elif perplexity_ok or qwen_ok:
        print("\n⚠ At least one LLM is configured.")
        if not tavily_ok:
            print("  Configure TAVILY_API_KEY for search functionality.")
    else:
        print("\n✗ No LLMs configured. Please set API keys in .env file.")
        print("  See .env.example for required variables.")
    
    print("")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
