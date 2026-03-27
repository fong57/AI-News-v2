"""Revise draft after fact-check; append unresolved-claim editor remarks on every revise output."""

import json
import unicodedata
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from litenews.config.settings import get_settings
from litenews.state.news_state import (
    DEFAULT_TARGET_WORD_COUNT,
    NewsState,
    validate_article_type,
)
from litenews.workflow.fact_check_diff import normalize_claim_key
from litenews.workflow.nodes.fact_check import _claim_importance
from litenews.workflow.prompts import revise_system_prompt
from litenews.workflow.utils import (
    create_error_response,
    invoke_llm_with_messages,
    workflow_llm_options,
)

_REMARKS_BLOCK_START = "\n\n---\n【事實查核備註】"
_REMARKS_BLOCK_START_ALT = "\n---\n【事實查核備註】"


def _llm_content_to_str(content: Any) -> str:
    """Turn AIMessage.content into plain text (handles str or provider-specific block lists)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and block.get("text") is not None:
                    parts.append(str(block.get("text", "")))
                    continue
                if block.get("text") is not None:
                    parts.append(str(block.get("text", "")))
                    continue
                nested = block.get("content")
                if isinstance(nested, str):
                    parts.append(nested)
                elif nested is not None:
                    parts.append(_llm_content_to_str(nested))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "".join(parts)
    if isinstance(content, dict):
        nested = content.get("content") or content.get("text")
        if nested is not None:
            return _llm_content_to_str(nested)
    return str(content)


def _strip_trailing_remarks_block(draft: str) -> str:
    """Return article body without a trailing 【事實查核備註】 section, if present."""
    if not draft:
        return ""
    for marker in (_REMARKS_BLOCK_START, _REMARKS_BLOCK_START_ALT):
        idx = draft.find(marker)
        if idx >= 0:
            return draft[:idx].rstrip()
    return draft


def _claim_still_in_revised_body(claim: dict[str, Any], article_body: str) -> bool:
    ct = normalize_claim_key(unicodedata.normalize("NFKC", str(claim.get("text") or "")))
    if not ct:
        return False
    an = normalize_claim_key(unicodedata.normalize("NFKC", article_body or ""))
    return ct in an


def _eligible_remark_claims(fr: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in fr.get("claims") or []:
        if not isinstance(c, dict):
            continue
        if _claim_importance(c) < 3:
            continue
        if c.get("status") not in ("contradicted", "uncertain"):
            continue
        out.append(c)
    return out


def _remarks_lines_for_claims(claims: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = [
        "\n\n---\n【事實查核備註】",
        "以下要點經自動事實查核後仍標示為「與檢索結果不符」或「未能核實」（重要性較高之宣稱），請編輯部同事留意辨識：",
    ]
    for c in claims:
        text = str(c.get("text") or "").strip()
        reason = str(c.get("reason") or "").strip()
        status_zh = "與檢索結果不符" if c.get("status") == "contradicted" else "未能核實"
        suffix = f" — {reason}" if reason else ""
        lines.append(f"- （{status_zh}）{text}{suffix}")
    return lines


def _build_unresolved_remarks_block(
    fr: dict[str, Any],
    article_body: str | None = None,
    *,
    substring_miss_fallback: bool = True,
) -> str:
    eligible = _eligible_remark_claims(fr)
    if not eligible:
        return ""
    if article_body is None:
        claims_to_show = eligible
    else:
        matched = [c for c in eligible if _claim_still_in_revised_body(c, article_body)]
        if matched:
            claims_to_show = matched
        elif substring_miss_fallback:
            # Revise usually rewrites contradict/uncertain wording, so exact substring often
            # misses; if nothing matches, show all eligible so editors still see the list.
            claims_to_show = eligible
        else:
            claims_to_show = []
    lines = _remarks_lines_for_claims(claims_to_show)
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


async def revise_node(state: NewsState) -> dict:
    """Rewrite the draft per fact_check_results; append 【事實查核備註】 when unresolved claims remain."""
    settings = get_settings()
    draft = state.get("draft", "") or ""
    outline = (state.get("outline") or "").strip()
    feedback = (state.get("feedback") or "").strip()
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
    feedback_block = (
        f"\n\nHuman feedback for this revise pass:\n{feedback}\n"
        if feedback
        else ""
    )
    user_content = f"""撰寫大綱（修訂後須與下列標題及章節順序一致）：
{outline_block}

Fact-check results (JSON):
{fc_json}
{feedback_block}

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
        body = _llm_content_to_str(response.content)
        if not (body or "").strip():
            raw_text = getattr(response, "text", None)
            if raw_text:
                body = str(raw_text)
        remarks = _build_unresolved_remarks_block(
            fr, article_body=body, substring_miss_fallback=True
        )
        draft_out = body + remarks
        prev_round = int(state.get("fact_check_revision_round") or 0)
        return {
            "draft": draft_out,
            "fact_check_revision_round": prev_round + 1,
            "feedback": "",
            "status": "revised",
            "messages": messages + [response],
        }
    except Exception as e:
        return create_error_response(f"Revision failed: {str(e)}")


async def fact_check_remarks_node(state: NewsState) -> dict:
    """Append an editor remark block for unresolved contradicted/uncertain claims.

    Revise already appends this block when applicable; skip if the draft already ends
    with the same block (e.g. revision cap after a revise pass).
    """
    draft = state.get("draft", "") or ""
    stripped = _strip_trailing_remarks_block(draft)
    fr = state.get("fact_check_results") or {}
    block = _build_unresolved_remarks_block(
        fr, article_body=stripped, substring_miss_fallback=False
    )
    if not block:
        if stripped != draft:
            return {"draft": stripped, "status": "fact_check_remarks_skipped"}
        return {"status": "fact_check_remarks_skipped"}
    combined = stripped + block
    if draft == combined:
        return {"status": "fact_check_remarks_skipped"}
    return {
        "draft": combined,
        "status": "fact_check_remarks_appended",
    }
