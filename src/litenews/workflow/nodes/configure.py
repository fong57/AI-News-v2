"""Apply workflow-level user options: target word count and LLM provider/model."""

from typing import cast

from litenews.config.settings import get_settings
from litenews.state.news_state import LLMProvider, NewsState
from litenews.workflow.utils import create_error_response

_DEFAULT_TARGET_WORDS = 800
_MIN_WORDS = 200
_MAX_WORDS = 20000


async def configure_workflow_node(state: NewsState) -> dict:
    """Resolve defaults, validate inputs, and ensure API keys for the chosen LLM."""
    settings = get_settings()

    raw_twc = state.get("target_word_count")
    if raw_twc is None:
        target_word_count = _DEFAULT_TARGET_WORDS
    else:
        try:
            target_word_count = int(raw_twc)
        except (TypeError, ValueError):
            return create_error_response("target_word_count must be an integer")
        if target_word_count < _MIN_WORDS or target_word_count > _MAX_WORDS:
            return create_error_response(
                f"target_word_count must be between {_MIN_WORDS} and {_MAX_WORDS}"
            )

    raw_lp = state.get("llm_provider")
    if raw_lp is None or (isinstance(raw_lp, str) and not str(raw_lp).strip()):
        llm_provider: LLMProvider = settings.primary_llm
    else:
        s = str(raw_lp).strip().lower()
        if s not in ("perplexity", "qwen"):
            return create_error_response("llm_provider must be 'perplexity' or 'qwen'")
        llm_provider = cast(LLMProvider, s)

    raw_lm = state.get("llm_model")
    llm_model = "" if raw_lm is None else str(raw_lm).strip()

    if llm_provider == "perplexity" and not settings.has_perplexity_key():
        return create_error_response(
            "Perplexity API key is missing; set PPLX_API_KEY or choose qwen."
        )
    if llm_provider == "qwen" and not settings.has_qwen_key():
        return create_error_response(
            "DashScope API key is missing; set DASHSCOPE_API_KEY or choose perplexity."
        )

    return {
        "target_word_count": target_word_count,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "status": "configured",
    }
