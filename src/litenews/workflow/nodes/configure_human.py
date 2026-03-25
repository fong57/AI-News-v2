"""Human-in-the-loop confirmation after configure, before research.

Shows resolved topic, article type, word target, LLM settings, and optional query so a
human can confirm or adjust values required for the research node.

Interrupts require a checkpointer — see ``outline_human`` / ``graph_builder`` docstring.
"""

from typing import Any, cast

from langgraph.types import interrupt

from litenews.config.settings import get_settings
from litenews.state.news_state import LLMProvider, NewsState, validate_article_type
from litenews.workflow.utils import create_error_response

# Exclude "confirm" — reserved for ``{'action': 'confirm', ...}`` merge semantics.
_ACCEPT_TOKENS = frozenset({"accept", "ok", "yes", "y", "confirmed"})

_CONFIGURE_KEYS = (
    "topic",
    "article_type",
    "target_word_count",
    "llm_provider",
    "llm_model",
    "query",
)


def _validate_merged_configure(state: NewsState) -> dict:
    """Return fields + status on success, or standardized error dict."""
    settings = get_settings()

    topic = (state.get("topic") or "").strip()
    if not topic:
        return create_error_response("Configure review: topic is required")

    raw_at = state.get("article_type")
    if raw_at is None or (isinstance(raw_at, str) and not str(raw_at).strip()):
        return create_error_response(
            "Configure review: article_type is required (懶人包、多方觀點、其他)"
        )
    try:
        validate_article_type(str(raw_at))
    except ValueError as e:
        return create_error_response(str(e))

    twc = state.get("target_word_count")
    if twc is None:
        return create_error_response("Configure review: target_word_count is missing")
    try:
        n = int(twc)
    except (TypeError, ValueError):
        return create_error_response("Configure review: target_word_count must be an integer")
    lo, hi = settings.min_target_word_count, settings.max_target_word_count
    if n < lo or n > hi:
        return create_error_response(
            f"Configure review: target_word_count must be between {lo} and {hi}"
        )

    raw_lp = state.get("llm_provider")
    if not isinstance(raw_lp, str) or not raw_lp.strip():
        return create_error_response("Configure review: llm_provider is required")
    s = raw_lp.strip().lower()
    if s not in ("perplexity", "qwen"):
        return create_error_response("Configure review: llm_provider must be 'perplexity' or 'qwen'")
    llm_provider = cast(LLMProvider, s)

    raw_lm = state.get("llm_model")
    llm_model = "" if raw_lm is None else str(raw_lm).strip()

    if llm_provider == "perplexity" and not settings.has_perplexity_key():
        return create_error_response(
            "Configure review: Perplexity API key is missing; set PPLX_API_KEY or choose qwen."
        )
    if llm_provider == "qwen" and not settings.has_qwen_key():
        return create_error_response(
            "Configure review: DashScope API key is missing; set DASHSCOPE_API_KEY or choose perplexity."
        )

    query = state.get("query")
    query_s = "" if query is None else str(query)

    return {
        "topic": topic,
        "article_type": validate_article_type(str(raw_at)),
        "target_word_count": n,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "query": query_s,
        "status": "configure_confirmed",
    }


def _configure_human_resume(resume: Any, state: NewsState) -> dict:
    """Apply interrupt resume: accept as-is, or merge overrides then validate."""
    if resume is None:
        return create_error_response(
            "Configure review: missing resume value. "
            "Use {'action': 'accept'} to continue with the shown settings, or "
            "{'action': 'confirm', 'topic': '...', ...} to adjust fields."
        )

    if isinstance(resume, str):
        s = resume.strip()
        if s.lower() in _ACCEPT_TOKENS:
            return _validate_merged_configure(state)
        return create_error_response(
            "Configure review: expected an accept token or a dict with 'action' and fields."
        )

    if not isinstance(resume, dict):
        return create_error_response(
            f"Configure review: unsupported resume type {type(resume).__name__}"
        )

    action = str(resume.get("action", "")).strip().lower()
    if action in _ACCEPT_TOKENS or action == "accept" or resume.get("confirmed") is True:
        return _validate_merged_configure(state)

    merged: NewsState = cast(NewsState, dict(state))
    if action == "confirm" or any(k in resume for k in _CONFIGURE_KEYS):
        for k in _CONFIGURE_KEYS:
            if k not in resume:
                continue
            val = resume[k]
            if k == "target_word_count" and val is not None:
                merged["target_word_count"] = val
            elif k == "query":
                merged["query"] = "" if val is None else str(val)
            elif k == "topic" and val is not None:
                merged["topic"] = str(val).strip()
            elif val is not None:
                merged[k] = val  # type: ignore[assignment]

    return _validate_merged_configure(merged)


def configure_human_node(state: NewsState) -> dict:
    """Pause so a human can confirm or edit settings before research."""
    pre = _validate_merged_configure(state)
    if pre.get("error"):
        return pre

    payload = {
        "kind": "configure_review",
        "topic": state.get("topic", ""),
        "article_type": state.get("article_type", ""),
        "target_word_count": state.get("target_word_count"),
        "llm_provider": state.get("llm_provider", ""),
        "llm_model": state.get("llm_model", "") or "",
        "query": state.get("query") or "",
        "resume_help": (
            "To continue without changes: resume with {'action': 'accept'} or the string 'accept'. "
            "To adjust: resume with {'action': 'confirm', 'topic': '...', 'article_type': '...', "
            "'target_word_count': <int>, 'llm_provider': 'perplexity'|'qwen', "
            "'llm_model': '...', 'query': '...'} — include only keys you want to change."
        ),
    }
    resume = interrupt(payload)
    return _configure_human_resume(resume, state)
