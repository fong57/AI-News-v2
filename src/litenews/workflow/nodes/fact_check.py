"""Fact-check node: extract factual claims from the draft, verify via search, score support."""

import json
import logging
import re
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from litenews.config.settings import get_settings
from litenews.state.news_state import NewsState, validate_article_type
from litenews.tools.search import get_tavily_search_tool
from litenews.workflow.fact_check_diff import (
    claim_entirely_in_stable_lines,
    compute_incremental_focus,
    normalize_claim_key,
    stable_new_line_indices,
)
from litenews.workflow.nodes.research import _parse_search_response
from litenews.workflow.tavily_pool import (
    filter_blocked_tavily_rows,
    merge_into_pool,
    normalize_raw_tavily_result,
    select_evidence_for_claim,
)
from litenews.workflow.utils import (
    create_error_response,
    invoke_llm_with_messages,
    workflow_llm_options,
)

logger = logging.getLogger(__name__)

_MAX_DRAFT_CHARS = 2000
_MAX_CLAIMS_TO_VERIFY = 15
_MAX_REVISION_ROUNDS = 5

_CONNECTION_RE = re.compile(
    r"connection error|fetch failed|ECONNREFUSED|ETIMEDOUT|ENOTFOUND|network|"
    r"ConnectError|ReadTimeout|Connection reset|Name or service not known|"
    r"getaddrinfo failed|Temporary failure",
    re.IGNORECASE,
)


def _is_connection_error(err: BaseException) -> bool:
    msg = str(err) if err else ""
    cause = getattr(err, "__cause__", None)
    cause_s = str(cause) if cause is not None else ""
    if _CONNECTION_RE.search(msg) or _CONNECTION_RE.search(cause_s):
        return True
    if isinstance(err, (BrokenPipeError, ConnectionResetError, TimeoutError)):
        return True
    if isinstance(err, ConnectionError):
        return True
    if isinstance(err, OSError) and getattr(err, "errno", None) in (
        51,  # ENETUNREACH
        60,  # ETIMEDOUT (mac)
        61,  # ECONNREFUSED (mac)
        101,  # ENETUNREACH (linux)
        110,  # ETIMEDOUT (linux)
        111,  # ECONNREFUSED (linux)
    ):
        return True
    return False


def _message_text(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(c or "")


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
) -> Literal["error", "revise", "fact_check_remarks"]:
    if state.get("error"):
        return "error"
    if has_actionable_fact_check_issues(state):
        round_n = int(state.get("fact_check_revision_round") or 0)
        if round_n < _MAX_REVISION_ROUNDS:
            return "revise"
        return "fact_check_remarks"
    return "revise"


def _parse_json_object(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_evidence_row(r: Any) -> dict[str, Any]:
    return normalize_raw_tavily_result(r)


_EXTRACT_SYSTEM = """ 你是一名查核事實的助理。
從文章中擷取最重要的事實主張。
回傳 JSON 格式，內容如下：
{"claims": [{"id": "c1", "text": "...", "importance": 1-5}]}
僅輸出 JSON。"""

_EXTRACT_SYSTEM_INCREMENTAL = """你是一名查核事實的助理。
下列文字是整篇文章當中「經修訂的片段」（可能含前後文脈絡），請僅從這些片段中擷取可核實的事實主張（不要臆測未出現的內容）。
回傳 JSON 格式，內容如下：
{"claims": [{"id": "c1", "text": "...", "importance": 1-5}]}
僅輸出 JSON。"""

_CHECK_SYSTEM = """你是一名嚴格的查核事實助理。
根據一項事實主張與若干網路搜尋片段，判斷該主張屬於以下何種狀態：
"supported"（有佐證）
"contradicted"（有反證）
"uncertain"（無法確認）
回傳 JSON 格式，內容如下：
{"status": "...", "reason": "..."}
僅輸出 JSON。"""


def _copy_evidence_cache(
    raw: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not raw:
        return out
    for key, val in raw.items():
        if not isinstance(key, str) or not isinstance(val, list):
            continue
        rows: list[dict[str, Any]] = []
        for item in val:
            if isinstance(item, dict):
                rows.append(dict(item))
        out[key] = rows
    return out


async def _llm_extract_claims(
    *,
    system: str,
    user_content: str,
    state: NewsState,
) -> list[dict[str, Any]]:
    settings = get_settings()
    llm_opts = workflow_llm_options(state, settings)
    extract_messages = [
        SystemMessage(content=system),
        HumanMessage(content=user_content),
    ]
    extract_resp = await invoke_llm_with_messages(
        extract_messages, settings, **llm_opts
    )
    extract_text = _message_text(extract_resp)
    parsed = _parse_json_object(extract_text) or {}
    claims_raw = parsed.get("claims")
    return claims_raw if isinstance(claims_raw, list) else []


async def _verify_claim_rows(
    claims: list[dict[str, Any]],
    evidence_cache: dict[str, Any] | None,
    state: NewsState,
    *,
    allow_tavily: bool,
    evidence_pool: list[dict[str, Any]],
    max_pool_items_per_claim: int,
    skip_normalized_texts: frozenset[str] | set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    settings = get_settings()
    llm_opts = workflow_llm_options(state, settings)
    search_tool = (
        get_tavily_search_tool(settings, node="fact_check") if allow_tavily else None
    )
    pool = merge_into_pool(None, evidence_pool)
    cache = _copy_evidence_cache(evidence_cache)
    skip = skip_normalized_texts or set()
    results: list[dict[str, Any]] = []

    for claim in claims:
        if len(results) >= _MAX_CLAIMS_TO_VERIFY:
            break
        if not isinstance(claim, dict):
            continue
        claim_id = str(claim.get("id") or "")
        claim_text = str(claim.get("text") or "").strip()
        if not claim_text:
            continue
        if normalize_claim_key(claim_text) in skip:
            continue
        try:
            importance = int(float(claim.get("importance", 3)))
        except (TypeError, ValueError):
            importance = 3
        importance = max(1, min(5, importance))

        cached = cache.get(claim_text)
        if cached is not None:
            evidence = [dict(e) for e in cached]
        elif allow_tavily and search_tool is not None:
            raw = await search_tool.ainvoke(claim_text)
            batch, api_error = _parse_search_response(raw)
            if api_error:
                search_rows: list[Any] = []
            else:
                search_rows = batch

            search_rows = filter_blocked_tavily_rows(
                search_rows, settings.tavily_exclude_domains
            )

            evidence = [_normalize_evidence_row(r) for r in search_rows]
            cache[claim_text] = [dict(e) for e in evidence]
            pool = merge_into_pool(pool, search_rows)
        else:
            evidence = select_evidence_for_claim(
                claim_text,
                pool,
                max_items=max_pool_items_per_claim,
            )
            cache[claim_text] = [dict(e) for e in evidence]
        snippets_block = "\n\n".join(
            f"{i + 1}. {e['title']}\n{e['snippet']}" for i, e in enumerate(evidence)
        )
        if not snippets_block.strip():
            snippets_block = "None"

        check_user = f"""Claim:
{claim_text}

Web search snippets:
{snippets_block}

Decide the status and explain briefly."""

        check_messages = [
            SystemMessage(content=_CHECK_SYSTEM),
            HumanMessage(content=check_user),
        ]
        check_resp = await invoke_llm_with_messages(
            check_messages, settings, **llm_opts
        )
        check_parsed = _parse_json_object(_message_text(check_resp)) or {}
        status = str(check_parsed.get("status") or "uncertain")
        if status not in ("supported", "contradicted", "uncertain"):
            status = "uncertain"
        reason = str(check_parsed.get("reason") or "")

        results.append(
            {
                "id": claim_id,
                "text": claim_text,
                "importance": importance,
                "status": status,
                "reason": reason,
                "evidence_snippets": evidence,
            }
        )

    return results, cache, pool


def _score_from_claim_rows(results: list[dict[str, Any]]) -> float:
    important = [r for r in results if (r.get("importance") or 3) >= 3]
    supported = [r for r in important if r.get("status") == "supported"]
    return len(supported) / len(important) if important else 1.0


def _carry_forward_stable_claims(
    prior_claims: list[Any],
    new_draft: str,
    stable_lines: frozenset[int],
) -> tuple[list[dict[str, Any]], set[str]]:
    carried: list[dict[str, Any]] = []
    keys: set[str] = set()
    for c in prior_claims:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text") or "").strip()
        if not claim_entirely_in_stable_lines(text, new_draft, stable_lines):
            continue
        row = {k: v for k, v in c.items()}
        carried.append(row)
        keys.add(normalize_claim_key(text))
    return carried, keys


async def _run_fact_check(
    raw_draft: str,
    lang: str,
    evidence_cache: dict[str, Any] | None,
    state: NewsState,
    *,
    allow_tavily: bool,
    evidence_pool: list[dict[str, Any]],
    max_pool_items_per_claim: int,
) -> dict[str, Any]:
    draft_for_extract = (raw_draft or "")[:_MAX_DRAFT_CHARS]
    truncated = len(raw_draft or "") > _MAX_DRAFT_CHARS
    suffix = "\n…" if truncated else ""

    extract_user = f"""Article (in {lang}):
{draft_for_extract}{suffix}

Extract 10-20 factual claims (short sentences) that can be checked."""

    claims = await _llm_extract_claims(
        system=_EXTRACT_SYSTEM,
        user_content=extract_user,
        state=state,
    )
    results, cache, pool = await _verify_claim_rows(
        claims,
        evidence_cache,
        state,
        allow_tavily=allow_tavily,
        evidence_pool=evidence_pool,
        max_pool_items_per_claim=max_pool_items_per_claim,
    )
    fact_check_score = _score_from_claim_rows(results)
    fact_check_results: dict[str, Any] = {
        "claims": results,
        "score": fact_check_score,
    }
    return {
        "fact_check_results": fact_check_results,
        "fact_check_score": fact_check_score,
        "fact_check_evidence_cache": cache,
        "tavily_evidence_pool": pool,
    }


async def _run_fact_check_incremental(
    raw_draft: str,
    lang: str,
    last_checked: str,
    focus_excerpt: str,
    prior_fact_check: dict[str, Any],
    evidence_cache: dict[str, Any] | None,
    state: NewsState,
    *,
    allow_tavily: bool,
    evidence_pool: list[dict[str, Any]],
    max_pool_items_per_claim: int,
) -> dict[str, Any]:
    prior_list = prior_fact_check.get("claims")
    prior_claims = prior_list if isinstance(prior_list, list) else []
    stable = stable_new_line_indices(last_checked, raw_draft)
    carried, carried_keys = _carry_forward_stable_claims(
        prior_claims, raw_draft, stable
    )

    truncated = len(focus_excerpt) >= _MAX_DRAFT_CHARS
    suffix = "\n…" if truncated else ""
    extract_user = f"""Revised excerpt only (in {lang}):
{focus_excerpt}{suffix}

Extract up to 10 factual claims (short sentences) that appear in this excerpt and can be checked."""

    raw_claims = await _llm_extract_claims(
        system=_EXTRACT_SYSTEM_INCREMENTAL,
        user_content=extract_user,
        state=state,
    )
    new_rows, cache, pool = await _verify_claim_rows(
        raw_claims,
        evidence_cache,
        state,
        allow_tavily=allow_tavily,
        evidence_pool=evidence_pool,
        max_pool_items_per_claim=max_pool_items_per_claim,
        skip_normalized_texts=carried_keys,
    )

    merged_keys = set(carried_keys)
    deduped_new: list[dict[str, Any]] = []
    for row in new_rows:
        k = normalize_claim_key(str(row.get("text") or ""))
        if k in merged_keys:
            continue
        merged_keys.add(k)
        deduped_new.append(row)

    merged = carried + deduped_new
    fact_check_score = _score_from_claim_rows(merged)
    fact_check_results: dict[str, Any] = {
        "claims": merged,
        "score": fact_check_score,
    }
    return {
        "fact_check_results": fact_check_results,
        "fact_check_score": fact_check_score,
        "fact_check_evidence_cache": cache,
        "tavily_evidence_pool": pool,
    }


async def fact_check_node(state: NewsState) -> dict:
    """Fact-check the draft; set fact_check_results and fact_check_score.

    On connection/API transport errors, returns score 1.0 and empty skipped results
    so the pipeline continues.
    """
    raw_draft = state.get("draft", "") or ""
    raw_at = state.get("article_type")

    if raw_at is None or (isinstance(raw_at, str) and not raw_at.strip()):
        return create_error_response(
            "article_type 為必填，請選擇其一：懶人包、多方觀點、其他"
        )
    try:
        validate_article_type(str(raw_at))
    except ValueError as e:
        return create_error_response(str(e))

    if not raw_draft.strip():
        return create_error_response("No draft to fact-check")

    settings = get_settings()
    rev_round = int(state.get("fact_check_revision_round") or 0)
    allow_tavily = rev_round == 0
    if allow_tavily and not settings.has_tavily_key():
        return create_error_response(
            "Tavily API key is missing or still set to the placeholder. "
            "Set TAVILY_API_KEY in your environment."
        )

    lang = "繁體中文"

    last_snapshot = state.get("last_fact_checked_draft") or ""
    prior_fr = state.get("fact_check_results") or {}
    prior_claims = prior_fr.get("claims") if isinstance(prior_fr, dict) else None
    use_incremental = (
        rev_round > 0
        and bool(last_snapshot.strip())
        and isinstance(prior_claims, list)
        and not prior_fr.get("skipped")
    )
    focus_ok = False
    focus_excerpt = ""
    if use_incremental:
        focus_ok, focus_excerpt = compute_incremental_focus(
            last_snapshot,
            raw_draft,
            context_lines=4,
            max_chars=_MAX_DRAFT_CHARS,
            max_changed_ratio=0.5,
        )

    pool_in = [dict(r) for r in (state.get("tavily_evidence_pool") or [])]
    max_items = settings.tavily_fact_check_max_results

    try:
        if use_incremental and focus_ok:
            out = await _run_fact_check_incremental(
                raw_draft,
                lang,
                last_snapshot,
                focus_excerpt,
                prior_fr,
                state.get("fact_check_evidence_cache"),
                state,
                allow_tavily=allow_tavily,
                evidence_pool=pool_in,
                max_pool_items_per_claim=max_items,
            )
        else:
            out = await _run_fact_check(
                raw_draft,
                lang,
                state.get("fact_check_evidence_cache"),
                state,
                allow_tavily=allow_tavily,
                evidence_pool=pool_in,
                max_pool_items_per_claim=max_items,
            )
        out["last_fact_checked_draft"] = raw_draft
        return out
    except Exception as err:
        if _is_connection_error(err):
            logger.warning(
                "fact_check skipped (connection error), continuing pipeline: %s",
                err,
            )
            return {
                "fact_check_results": {
                    "claims": [],
                    "score": 1.0,
                    "skipped": True,
                },
                "fact_check_score": 1.0,
            }
        raise
