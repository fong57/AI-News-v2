"""LangGraph state definitions for news writing workflow.

This module defines the state schema used by the LangGraph workflow.
The state is typed using TypedDict for better type safety and documentation.
"""

from typing import Annotated, Any
from operator import add

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class NewsSource(BaseModel):
    """A news source reference."""
    
    title: str = Field(description="Title of the source article")
    url: str = Field(description="URL of the source")
    snippet: str = Field(default="", description="Relevant snippet from the source")
    published_date: str | None = Field(default=None, description="Publication date if available")


class NewsArticle(BaseModel):
    """A generated news article."""
    
    headline: str = Field(description="Article headline")
    summary: str = Field(description="Brief summary/lead paragraph")
    body: str = Field(description="Full article body")
    sources: list[NewsSource] = Field(default_factory=list, description="Sources used")
    keywords: list[str] = Field(default_factory=list, description="Relevant keywords")


class NewsState(TypedDict, total=False):
    """State for the news writing workflow.
    
    This state is passed between nodes in the LangGraph workflow.
    Using TypedDict allows for partial updates to the state.
    
    Attributes:
        topic: The news topic to research and write about.
        query: The refined search query.
        messages: Conversation history with the LLM.
        search_results: Raw search results from Tavily.
        sources: Processed and validated sources.
        research_notes: Notes from the research phase.
        outline: Article outline before writing.
        draft: Initial article draft.
        final_article: The final polished article.
        feedback: Any feedback or revision requests.
        status: Current workflow status.
        error: Error message if something went wrong.
    """
    
    # Input
    topic: str
    query: str
    
    # Conversation
    messages: Annotated[list[BaseMessage], add]
    
    # Research
    search_results: list[dict[str, Any]]
    sources: list[NewsSource]
    research_notes: str
    
    # Writing
    outline: str
    draft: str
    final_article: NewsArticle | None
    
    # Control
    feedback: str
    status: str
    error: str


def create_initial_state(topic: str) -> NewsState:
    """Create initial state for the news writing workflow.
    
    Args:
        topic: The news topic to research and write about.
        
    Returns:
        NewsState: Initial state with default values.
    """
    return NewsState(
        topic=topic,
        query="",
        messages=[],
        search_results=[],
        sources=[],
        research_notes="",
        outline="",
        draft="",
        final_article=None,
        feedback="",
        status="initialized",
        error="",
    )
