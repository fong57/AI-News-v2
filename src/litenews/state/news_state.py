"""LangGraph state definitions for news writing workflow.

This module defines the state schema used by the LangGraph workflow.
The state is typed using TypedDict for better type safety and documentation.
"""

from operator import add
from typing import Annotated, Any, Literal, cast

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from litenews.config.settings import get_settings

LLMProvider = Literal["perplexity", "qwen"]

WorkflowTask = Literal["write", "edit"]

DEFAULT_TARGET_WORD_COUNT = get_settings().default_target_word_count

ArticleType = Literal["懶人包", "多方觀點", "其他"]

ARTICLE_TYPES: tuple[ArticleType, ...] = ("懶人包", "多方觀點", "其他")


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
        article_type: Required article format — one of 懶人包, 多方觀點, 其他.
        target_word_count: Article body word-count target (±10% in prompts); set by configure.
        llm_provider: perplexity or qwen for LLM nodes after configure; set by configure.
        llm_model: Optional model id for that provider; empty uses env default for provider.
        query: The refined search query.
        messages: Conversation history with the LLM.
        search_results: Raw search results from Tavily.
        sources: Processed and validated sources.
        research_notes: Notes from the research phase.
        outline: Article outline before writing.
        draft: Initial article draft.
        fact_check_results: Structured fact-check output (claims, score, optional skipped).
        fact_check_score: Fraction of important claims marked supported (0–1).
        fact_check_revision_round: Count of revise passes after fact-check (max 5).
        fact_check_evidence_cache: Map strip-normalized claim text to evidence_snippets
            lists from Tavily; reused across revise→fact_check loops; cleared on new write.
        tavily_evidence_pool: Normalized search rows from research, write, and first fact-check
            pass; used for pool-only verification after revise (no further Tavily calls).
        last_fact_checked_draft: Draft text that the last fact_check pass used; enables
            incremental extract/verify after revise vs full-article re-check.
        final_article: The final polished article.
        feedback: Any feedback or revision requests.
        status: Current workflow status.
        error: Error message if something went wrong.
        task: ``write`` (default) full pipeline; ``edit`` human draft then fact_check only.
    """
    
    # Input
    topic: str
    article_type: ArticleType
    task: WorkflowTask
    target_word_count: int
    llm_provider: LLMProvider
    llm_model: str
    query: str
    
    # Conversation
    messages: Annotated[list[BaseMessage], add]
    
    # Research
    search_results: list[dict[str, Any]]
    tavily_evidence_pool: list[dict[str, Any]]
    sources: list[NewsSource]
    research_notes: str
    
    # Writing
    outline: str
    draft: str
    fact_check_results: dict[str, Any]
    fact_check_score: float
    fact_check_revision_round: int
    fact_check_evidence_cache: dict[str, Any]
    last_fact_checked_draft: str
    final_article: NewsArticle | None
    
    # Control
    feedback: str
    status: str
    error: str


def validate_article_type(value: str) -> ArticleType:
    """Return a valid article type or raise ValueError."""
    v = (value or "").strip()
    if v not in ARTICLE_TYPES:
        allowed = "、".join(ARTICLE_TYPES)
        raise ValueError(
            f"article_type 必須為以下其一：{allowed}；收到：{value!r}"
        )
    return cast(ArticleType, v)


def create_initial_state(
    topic: str,
    article_type: str,
    *,
    target_word_count: int | None = None,
    llm_provider: LLMProvider | None = None,
    llm_model: str = "",
    task: WorkflowTask | None = None,
) -> NewsState:
    """Create initial state for the news writing workflow.
    
    Args:
        topic: The news topic to research and write about.
        article_type: One of 懶人包, 多方觀點, 其他 (required).
        target_word_count: Optional word-count target; configure node fills default if omitted.
        llm_provider: Optional perplexity/qwen; configure uses settings.primary_llm if omitted.
        llm_model: Optional model name for the provider; empty means use env default.
        task: Optional ``write`` (default) or ``edit`` (human draft then fact-check only).

    Raises:
        ValueError: If article_type is not one of the allowed values.

    Returns:
        NewsState: Initial state with default values.
    """
    at = validate_article_type(article_type)
    wt: WorkflowTask = task if task is not None else "write"
    if wt not in ("write", "edit"):
        raise ValueError("task 必須為 'write' 或 'edit'")

    init: NewsState = NewsState(
        topic=topic,
        article_type=at,
        task=wt,
        query="",
        messages=[],
        search_results=[],
        tavily_evidence_pool=[],
        sources=[],
        research_notes="",
        outline="",
        draft="",
        fact_check_revision_round=0,
        final_article=None,
        feedback="",
        status="initialized",
        error="",
    )
    if target_word_count is not None:
        init["target_word_count"] = target_word_count
    if llm_provider is not None:
        init["llm_provider"] = llm_provider
    if llm_model:
        init["llm_model"] = llm_model
    return init
