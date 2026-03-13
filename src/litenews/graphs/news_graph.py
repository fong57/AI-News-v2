"""Main LangGraph workflow for news writing.

This module defines the news writing workflow using LangGraph.
The workflow follows these steps:
1. Research: Search for news sources on the topic
2. Analyze: Process and validate sources
3. Outline: Create article outline
4. Write: Generate the article draft
5. Review: Polish and finalize the article

The graph is exported as `graph` for LangGraph Cloud deployment.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START

from litenews.config.settings import get_settings
from litenews.config.tracing import setup_tracing
from litenews.llms.base import get_llm
from litenews.state.news_state import NewsState, NewsSource, NewsArticle
from litenews.tools.search import get_tavily_search_tool

setup_tracing()


RESEARCH_SYSTEM_PROMPT = """You are a news research assistant. Your task is to analyze search results 
and extract key information for writing a news article.

Given the topic and search results, provide:
1. A summary of the key facts and developments
2. Important quotes or statistics
3. Different perspectives or viewpoints
4. Timeline of events if applicable

Be factual and objective. Note the sources for each piece of information."""


OUTLINE_SYSTEM_PROMPT = """You are a news editor creating an article outline. Based on the research notes,
create a structured outline for a news article.

The outline should include:
1. Headline (attention-grabbing but accurate)
2. Lead paragraph (who, what, when, where, why)
3. Main body sections (3-5 sections with key points)
4. Conclusion/future outlook

Keep the outline concise but comprehensive."""


WRITE_SYSTEM_PROMPT = """You are a professional news writer. Write a well-structured news article
based on the provided outline and research.

Guidelines:
- Use an inverted pyramid structure (most important info first)
- Keep sentences clear and concise
- Include relevant quotes and statistics
- Maintain objectivity and balance
- Cite sources appropriately

Write the article in a professional journalistic style."""


REVIEW_SYSTEM_PROMPT = """You are a news editor reviewing an article draft. Your task is to:
1. Check for accuracy and clarity
2. Improve the headline if needed
3. Ensure proper structure and flow
4. Verify all claims are supported by sources
5. Polish the language and fix any issues

Provide the final polished version of the article."""


async def research_node(state: NewsState) -> dict:
    """Research node: Search for news sources on the topic.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with search results.
    """
    settings = get_settings()
    topic = state.get("topic", "")
    
    if not topic:
        return {"error": "No topic provided", "status": "error"}
    
    query = state.get("query") or f"Latest news about {topic}"
    
    try:
        search_tool = get_tavily_search_tool(settings)
        results = await search_tool.ainvoke(query)
        
        search_results = []
        if isinstance(results, dict):
            search_results = results.get("results", [])
        elif isinstance(results, list):
            search_results = results
            
        return {
            "query": query,
            "search_results": search_results,
            "status": "researched",
        }
    except Exception as e:
        return {"error": f"Research failed: {str(e)}", "status": "error"}


async def analyze_node(state: NewsState) -> dict:
    """Analyze node: Process search results and extract key information.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with sources and research notes.
    """
    settings = get_settings()
    search_results = state.get("search_results", [])
    topic = state.get("topic", "")
    
    if not search_results:
        return {"error": "No search results to analyze", "status": "error"}
    
    sources = []
    for result in search_results:
        if isinstance(result, dict):
            sources.append(NewsSource(
                title=result.get("title", ""),
                url=result.get("url", ""),
                snippet=result.get("content", result.get("snippet", "")),
                published_date=result.get("published_date"),
            ))
    
    llm = get_llm(settings=settings)
    
    sources_text = "\n\n".join([
        f"Source: {s.title}\nURL: {s.url}\nContent: {s.snippet}"
        for s in sources
    ])
    
    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=f"Topic: {topic}\n\nSearch Results:\n{sources_text}"),
    ]
    
    try:
        response = await llm.ainvoke(messages)
        return {
            "sources": sources,
            "research_notes": response.content,
            "status": "analyzed",
            "messages": messages + [response],
        }
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}", "status": "error"}


async def outline_node(state: NewsState) -> dict:
    """Outline node: Create article outline based on research.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with article outline.
    """
    settings = get_settings()
    research_notes = state.get("research_notes", "")
    topic = state.get("topic", "")
    
    if not research_notes:
        return {"error": "No research notes for outline", "status": "error"}
    
    llm = get_llm(settings=settings)
    
    messages = [
        SystemMessage(content=OUTLINE_SYSTEM_PROMPT),
        HumanMessage(content=f"Topic: {topic}\n\nResearch Notes:\n{research_notes}"),
    ]
    
    try:
        response = await llm.ainvoke(messages)
        return {
            "outline": response.content,
            "status": "outlined",
            "messages": messages + [response],
        }
    except Exception as e:
        return {"error": f"Outline creation failed: {str(e)}", "status": "error"}


async def write_node(state: NewsState) -> dict:
    """Write node: Generate article draft based on outline.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with article draft.
    """
    settings = get_settings()
    outline = state.get("outline", "")
    research_notes = state.get("research_notes", "")
    topic = state.get("topic", "")
    
    if not outline:
        return {"error": "No outline for writing", "status": "error"}
    
    llm = get_llm(settings=settings)
    
    messages = [
        SystemMessage(content=WRITE_SYSTEM_PROMPT),
        HumanMessage(content=f"""Topic: {topic}

Outline:
{outline}

Research Notes:
{research_notes}

Please write the full article following the outline."""),
    ]
    
    try:
        response = await llm.ainvoke(messages)
        return {
            "draft": response.content,
            "status": "drafted",
            "messages": messages + [response],
        }
    except Exception as e:
        return {"error": f"Writing failed: {str(e)}", "status": "error"}


async def review_node(state: NewsState) -> dict:
    """Review node: Polish and finalize the article.
    
    Args:
        state: Current workflow state.
        
    Returns:
        dict: Updated state with final article.
    """
    settings = get_settings()
    draft = state.get("draft", "")
    sources = state.get("sources", [])
    topic = state.get("topic", "")
    
    if not draft:
        return {"error": "No draft to review", "status": "error"}
    
    llm = get_llm(settings=settings)
    
    messages = [
        SystemMessage(content=REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=f"""Topic: {topic}

Draft Article:
{draft}

Please review and provide the final polished version."""),
    ]
    
    try:
        response = await llm.ainvoke(messages)
        
        lines = response.content.strip().split("\n")
        headline = lines[0].replace("#", "").strip() if lines else topic
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else response.content
        
        summary_end = body.find("\n\n")
        summary = body[:summary_end] if summary_end > 0 else body[:200]
        
        final_article = NewsArticle(
            headline=headline,
            summary=summary,
            body=body,
            sources=sources,
            keywords=[topic],
        )
        
        return {
            "final_article": final_article,
            "status": "completed",
            "messages": messages + [response],
        }
    except Exception as e:
        return {"error": f"Review failed: {str(e)}", "status": "error"}


def should_continue(state: NewsState) -> Literal["continue", "error"]:
    """Determine if the workflow should continue or stop due to error.
    
    Args:
        state: Current workflow state.
        
    Returns:
        str: "continue" to proceed, "error" to stop.
    """
    if state.get("error"):
        return "error"
    return "continue"


def create_news_graph() -> StateGraph:
    """Create the news writing workflow graph.
    
    Returns:
        StateGraph: Compiled LangGraph workflow.
    """
    workflow = StateGraph(NewsState)
    
    workflow.add_node("research", research_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("outline", outline_node)
    workflow.add_node("write", write_node)
    workflow.add_node("review", review_node)
    
    workflow.add_edge(START, "research")
    
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
        {"continue": "write", "error": END},
    )
    workflow.add_conditional_edges(
        "write",
        should_continue,
        {"continue": "review", "error": END},
    )
    workflow.add_edge("review", END)
    
    return workflow.compile()


graph = create_news_graph()
