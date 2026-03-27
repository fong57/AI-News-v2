"""Apply workflow-level user options: target word count and LLM provider/model."""

from typing import cast

from litenews.config.settings import get_settings
from litenews.state.news_state import LLMProvider, NewsState
from litenews.workflow.utils import create_error_response


async def configure_workflow_node(state: NewsState) -> dict:
    """Resolve defaults, validate inputs, and ensure API keys for the chosen LLM."""
    settings = get_settings()

    raw_twc = state.get("target_word_count")
    if raw_twc is None:
        target_word_count = settings.default_target_word_count
    else:
        try:
            target_word_count = int(raw_twc)
        except (TypeError, ValueError):
            return create_error_response("target_word_count must be an integer")
        lo, hi = settings.min_target_word_count, settings.max_target_word_count
        if target_word_count < lo or target_word_count > hi:
            return create_error_response(
                f"target_word_count must be between {lo} and {hi}"
            )

    raw_lp = state.get("llm_provider")
    if raw_lp is None or (isinstance(raw_lp, str) and not str(raw_lp).strip()):
        llm_provider: LLMProvider = settings.primary_llm
    else:
        s = str(raw_lp).strip().lower()
        if s not in ("perplexity", "qwen", "bailian"):
            return create_error_response(
                "llm_provider must be 'perplexity', 'qwen', or 'bailian'"
            )
        llm_provider = cast(LLMProvider, s)

    raw_lm = state.get("llm_model")
    llm_model = "" if raw_lm is None else str(raw_lm).strip()

    if llm_provider == "perplexity" and not settings.has_perplexity_key():
        return create_error_response(
            "Perplexity API key is missing; set PPLX_API_KEY or choose qwen or bailian."
        )
    if llm_provider == "qwen" and not settings.has_qwen_key():
        return create_error_response(
            "DashScope API key is missing; set DASHSCOPE_API_KEY or choose another provider."
        )
    if llm_provider == "bailian" and not settings.has_bailian_key():
        return create_error_response(
            "Bailian API key is missing; set BAILIAN_API_KEY or choose another provider."
        )

    raw_task = state.get("task")
    if raw_task is None or (isinstance(raw_task, str) and not str(raw_task).strip()):
        task_norm = "write"
    else:
        s = str(raw_task).strip().lower()
        if s not in ("write", "edit"):
            return create_error_response("task must be 'write' or 'edit'")
        task_norm = s

    return {
        "target_word_count": target_word_count,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "task": task_norm,
        "status": "configured",
    }
