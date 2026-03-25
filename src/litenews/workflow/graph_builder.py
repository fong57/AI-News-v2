"""Core graph topology for the news writing workflow.

This module defines the LangGraph workflow structure by connecting nodes
with edges and conditional routing.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer

from litenews.state.news_state import NewsState
from litenews.workflow.nodes import (
    analyze_node,
    configure_human_node,
    configure_workflow_node,
    fact_check_node,
    outline_human_node,
    outline_node,
    research_node,
    review_node,
    revise_human_node,
    write_node,
)
from litenews.workflow.nodes.fact_check import route_after_fact_check
from litenews.workflow.nodes.revise import fact_check_remarks_node, revise_node
from litenews.workflow.nodes.revise_human import route_after_revise_human
from litenews.workflow.utils import should_continue  # Import Conditional Logic Utility


def create_news_graph(*, checkpointer: Checkpointer | None = None) -> StateGraph:
    """Create the news writing workflow graph.

    Do not pass a checkpointer for LangGraph API / ``langgraph dev`` — the platform
    supplies persistence. For standalone scripts using ``outline_human`` (interrupt),
    compile with an in-memory saver, for example
    ``create_news_graph(checkpointer=MemorySaver())`` from
    ``langgraph.checkpoint.memory``.

    The workflow follows these steps:
    0. Configure: Resolve target word count (±10% in later prompts) and LLM provider/model
    0b. Configure (human): Confirm or edit settings required for research (interrupt)
    1. Research: Search for news sources on the topic
    2. Analyze: Process and validate sources
    3. Outline: Create article outline
    4. Outline (human): Confirm AI outline or replace with a pasted outline (interrupt)
    5. Write: Generate the article draft
    6. Fact check: Extract claims, search, score factual support
    7. Revise (loop): Soften contradicted/uncertain claims, re-run fact-check up to 5 rounds;
       if still unresolved, append 【事實查核備註】 then continue
    8. Revise (human): Human decides to accept, give feedback for another revise loop,
       or replace draft manually
    9. Review: Polish and finalize the article

    Each step has error handling that will terminate the workflow
    if an error occurs.

    Returns:
        StateGraph: Compiled LangGraph workflow.
    """
    workflow = StateGraph(NewsState)

    workflow.add_node("configure", configure_workflow_node)
    workflow.add_node("configure_human", configure_human_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("outline", outline_node)
    workflow.add_node("outline_human", outline_human_node)
    workflow.add_node("write", write_node)
    workflow.add_node("fact_check", fact_check_node)
    workflow.add_node("revise", revise_node)
    workflow.add_node("revise_human", revise_human_node)
    workflow.add_node("fact_check_remarks", fact_check_remarks_node)
    workflow.add_node("review", review_node)

    workflow.add_edge(START, "configure")
    workflow.add_conditional_edges(
        "configure",
        should_continue,
        {"continue": "configure_human", "error": END},
    )
    workflow.add_conditional_edges(
        "configure_human",
        should_continue,
        {"continue": "research", "error": END},
    )
    # Conditional Edges:
    workflow.add_conditional_edges(
        "research",
        should_continue,
        {"continue": "analyze", "error": END},
    )
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {"continue": "outline", "error": END},
    )
    workflow.add_conditional_edges(
        "outline",
        should_continue,
        {"continue": "outline_human", "error": END},
    )
    workflow.add_conditional_edges(
        "outline_human",
        should_continue,
        {"continue": "write", "error": END},
    )
    workflow.add_conditional_edges(
        "write",
        should_continue,
        {"continue": "fact_check", "error": END},
    )
    workflow.add_conditional_edges(
        "fact_check",
        route_after_fact_check,
        {
            "error": END,
            "revise": "revise",
            "fact_check_remarks": "fact_check_remarks",
            "review": "review",
        },
    )
    workflow.add_conditional_edges(
        "revise",
        should_continue,
        {"continue": "revise_human", "error": END},
    )
    workflow.add_conditional_edges(
        "revise_human",
        route_after_revise_human,
        {"error": END, "revise": "revise", "review": "review"},
    )
    workflow.add_edge("fact_check_remarks", "review")
    workflow.add_edge("review", END)

    return workflow.compile(checkpointer=checkpointer)
