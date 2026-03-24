"""LangGraph workflow for LiteNews AI.

This module exports the compiled workflow graph and related utilities.
The graph is exported as `graph` for LangGraph Cloud deployment.
"""

from litenews.config.tracing import setup_tracing
from litenews.workflow.graph_builder import create_news_graph

setup_tracing()

graph = create_news_graph()

__all__ = ["graph", "create_news_graph"]
