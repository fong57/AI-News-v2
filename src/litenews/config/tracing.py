"""LangSmith tracing configuration for LiteNews AI.

This module provides utilities for setting up and managing LangSmith tracing.
Tracing enables observability, debugging, and monitoring of LLM workflows.

Usage:
    from litenews.config.tracing import setup_tracing, trace_run
    
    # Setup tracing at application start
    setup_tracing()
    
    # Use context manager for custom traces
    with trace_run("my_operation"):
        # your code here
"""

import os
from contextlib import contextmanager
from typing import Any, Generator

from litenews.config.settings import Settings, get_settings


def setup_tracing(settings: Settings | None = None) -> bool:
    """Setup LangSmith tracing from settings.
    
    This function sets the required environment variables for LangSmith.
    Call this at application startup before any LangChain/LangGraph calls.
    
    Args:
        settings: Optional settings instance. Uses default if not provided.
        
    Returns:
        bool: True if tracing was enabled, False otherwise.
    """
    settings = settings or get_settings()
    
    if not settings.is_tracing_enabled():
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    
    return True


def disable_tracing():
    """Disable LangSmith tracing.
    
    Useful for testing or when you want to temporarily disable tracing.
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


@contextmanager
def trace_run(
    name: str,
    run_type: str = "chain",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Generator[Any, None, None]:
    """Context manager for creating a traced run.
    
    Use this to wrap custom operations that should appear as separate
    runs in LangSmith.
    
    Args:
        name: Name of the run.
        run_type: Type of run (chain, tool, llm, etc.).
        metadata: Optional metadata to attach to the run.
        tags: Optional tags for filtering in LangSmith.
        
    Yields:
        The run context if tracing is enabled, None otherwise.
        
    Example:
        with trace_run("process_news", tags=["news", "processing"]):
            result = process_news_article(article)
    """
    settings = get_settings()
    
    if not settings.is_tracing_enabled():
        yield None
        return
    
    try:
        from langsmith import traceable
        from langsmith.run_helpers import get_current_run_tree
        
        @traceable(name=name, run_type=run_type, metadata=metadata, tags=tags)
        def _traced_block():
            return get_current_run_tree()
        
        run_tree = _traced_block()
        yield run_tree
    except ImportError:
        yield None
    except Exception:
        yield None


def get_tracing_status() -> dict[str, Any]:
    """Get current tracing configuration status.
    
    Returns:
        dict: Tracing configuration details.
    """
    settings = get_settings()
    
    return {
        "enabled": settings.is_tracing_enabled(),
        "tracing_v2": settings.langchain_tracing_v2,
        "has_api_key": settings.has_langsmith_key(),
        "project": settings.langchain_project,
        "endpoint": settings.langchain_endpoint,
    }


def test_langsmith_connection(api_key: str) -> dict[str, Any]:
    """Test LangSmith API connection.
    
    Args:
        api_key: LangSmith API key.
        
    Returns:
        dict: Test result with status and details.
    """
    try:
        from langsmith import Client
        
        client = Client(api_key=api_key)
        
        projects = list(client.list_projects(limit=1))
        
        return {
            "status": "success",
            "message": "Successfully connected to LangSmith",
            "projects_accessible": len(projects) > 0,
        }
    except ImportError:
        return {
            "status": "error",
            "error": "langsmith package not installed",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }
