"""Revise draft after fact-check; append editor remarks when revision cap is reached."""

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from litenews.config.settings import get_settings
from litenews.state.news_state import (
    DEFAULT_TARGET_WORD_COUNT,
    NewsState,
    validate_article_type,
)
from litenews.workflow.prompts import revise_system_prompt
from litenews.workflow.utils import (
    create_error_response,
    invoke_llm_with_messages,
    workflow_llm_options,
)

_MAX_REVISION_ROUNDS = 5


def _claim_importance(claim: dict[str, Any]) -> int:
    try:
        return max(1, min(5, int(float(claim.get("importance", 3)))))
    except (TypeError, ValueError):
        return 3


def has_actionable_fact_check_issues(state: NewsState) -> bool:
    """True if any important claim (importance ≥ 3) is contradicted or uncertain."""
    fr = state.get("fact_check_results") or {}
    if fr.get("skipped"):
        return False
    claims = fr.get("claims")
    if not isinstance(claims, list):
        return False
    for c in claims:
        if not isinstance(c, dict):
            continue
        if _claim_importance(c) < 3:
            continue
        if c.get("status") in ("contradicted", "uncertain"):
            return True
    return False


def route_after_fact_check(
    state: NewsState,
) -> Literal["error", "revise", "fact_check_remarks", "review"]:
    if state.get("error"):
        return "error"
    if not has_actionable_fact_check_issues(state):
        return "review"
    round_n = int(state.get("fact_check_revision_round") or 0)
    if round_n < _MAX_REVISION_ROUNDS:
        return "revise"
    return "fact_check_remarks"


def _build_unresolved_remarks_block(fr: dict[str, Any]) -> str:
    lines: list[str] = [
        "\n\n---\n【事實查核備註】",
        "以下要點經自動事實查核後仍標示為「與檢索結果不符」或「未能核實」（重要性較高之宣稱），編輯部提醒讀者留意辨識：",
    ]
    for c in fr.get("claims") or []:
        if not isinstance(c, dict):
            continue
        if _claim_importance(c) < 3:
            continue
        if c.get("status") not in ("contradicted", "uncertain"):
            continue
        text = str(c.get("text") or "").strip()
        reason = str(c.get("reason") or "").strip()
        status_zh = "與檢索結果不符" if c.get("status") == "contradicted" else "未能核實"
        suffix = f" — {reason}" if reason else ""
        lines.append(f"- （{status_zh}）{text}{suffix}")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


async def revise_node(state: NewsState) -> dict:
    """Rewrite the draft to soften contradicted/uncertain claims per fact_check_results."""
    settings = get_settings()
    draft = state.get("draft", "") or ""
    outline = (state.get("outline") or "").strip()
    raw_at = state.get("article_type")
    try:
        target_word_count = int(
            state.get("target_word_count", DEFAULT_TARGET_WORD_COUNT) or DEFAULT_TARGET_WORD_COUNT
        )
    except (TypeError, ValueError):
        return create_error_response("target_word_count must be an integer")
    fr = state.get("fact_check_results") or {}

    if raw_at is None or (isinstance(raw_at, str) and not raw_at.strip()):
        return create_error_response(
            "article_type 為必填，請選擇其一：懶人包、多方觀點、其他"
        )
    try:
        article_type = validate_article_type(str(raw_at))
    except ValueError as e:
        return create_error_response(str(e))

    if not draft.strip():
        return create_error_response("No draft to revise")

    try:
        fc_json = json.dumps(fr, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        fc_json = "{}"

    outline_block = (
        outline
        if outline
        else "（無可用大綱文本，請盡量維持現稿之標題與章節結構。）"
    )
    user_content = f"""撰寫大綱（修訂後須與下列標題及章節順序一致）：
{outline_block}

Fact-check results (JSON):
{fc_json}

Current article draft:
{draft}

Revise the draft as instructed in the system prompt."""

    messages = [
        SystemMessage(content=revise_system_prompt(article_type, target_word_count)),
        HumanMessage(content=user_content),
    ]

    try:
        response = await invoke_llm_with_messages(
            messages,
            settings,
            **workflow_llm_options(state, settings),
        )
        prev_round = int(state.get("fact_check_revision_round") or 0)
        return {
            "draft": response.content,
            "fact_check_revision_round": prev_round + 1,
            "status": "revised",
            "messages": messages + [response],
        }
    except Exception as e:
        return create_error_response(f"Revision failed: {str(e)}")


async def fact_check_remarks_node(state: NewsState) -> dict:
    """Append an editor remark block for unresolved contradicted/uncertain claims."""
    draft = state.get("draft", "") or ""
    fr = state.get("fact_check_results") or {}
    block = _build_unresolved_remarks_block(fr)
    if not block:
        return {"status": "fact_check_remarks_skipped"}
    return {
        "draft": draft + block,
        "status": "fact_check_remarks_appended",
    }
