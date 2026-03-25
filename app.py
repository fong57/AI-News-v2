import json
import os
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse, urlunparse

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from langgraph.errors import GraphInterrupt
from langgraph.pregel.remote import RemoteException, RemoteGraph
from langgraph.types import Command
from openai import OpenAI  # reused for compatible APIs (Perplexity/DashScope)

from litenews.config.settings import get_settings
from litenews.state.news_state import ARTICLE_TYPES, create_initial_state

# --------------------------
# Load Environment Variables
# --------------------------
load_dotenv()

# LangGraph Platform deployment (RemoteGraph / LangGraph Server API — not LangServe /invoke)
LANGGRAPH_GRAPH_ID = os.getenv("LANGGRAPH_GRAPH_ID", "news_writer")
LANGGRAPH_DEFAULT_ARTICLE_TYPE = os.getenv("LANGGRAPH_DEFAULT_ARTICLE_TYPE", "懶人包")


def _resolve_langgraph_api_key() -> str | None:
    """Match langgraph_sdk: LANGGRAPH_API_KEY, then LANGSMITH_API_KEY, then LANGCHAIN_API_KEY."""
    for name in ("LANGGRAPH_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        v = os.getenv(name)
        if v and v.strip():
            return v.strip().strip('"').strip("'")
    return None


def _resolve_langgraph_base_url() -> tuple[str | None, str | None]:
    """Return (base_url, error_message). Base has no trailing slash; host must resolve at runtime."""
    raw = (os.getenv("LANGSMITH_RESOURCE_URL") or "").strip().strip('"').strip("'")
    if not raw:
        return None, (
            "⚠️ AI傳真優：請在環境變數設定 `LANGSMITH_RESOURCE_URL` 為你的部署網址"
            "（結尾不要加路徑）。"
        )
    candidate = raw.rstrip("/")
    if "://" not in candidate:
        candidate = "https://" + candidate.lstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        return None, (
            "⚠️ AI傳真優：`LANGSMITH_RESOURCE_URL` 必須使用 http:// 或 https:// "
            f"（目前為 `{parsed.scheme or '（空）'}`）。"
        )
    if not (parsed.hostname and str(parsed.hostname).strip()):
        return None, (
            "⚠️ AI傳真優：`LANGSMITH_RESOURCE_URL` 必須包含主機名稱"
            "（例如 `https://<deployment-id>.langchain.run`）。"
            "若只有 `https://`、或仍為佔位符未修改，DNS 解析會失敗。"
        )
    netloc, path = parsed.netloc, (parsed.path or "").rstrip("/")
    if path == "/":
        path = ""
    normalized = urlunparse((parsed.scheme, netloc, path, "", "", "")).rstrip("/")
    return normalized, None


def _langgraph_connect_help(original: str) -> str:
    if "Errno 8" in original or "nodename nor servname" in original.lower():
        return (
            "無法解析部署主機名稱（DNS）。"
            "請確認 `LANGSMITH_RESOURCE_URL` 與 LangSmith／部署控制台顯示的網址完全一致"
            "（不是 `.example` 佔位符）、網絡連線正常，且主機確實存在。\n\n"
            f"原始錯誤：{original}"
        )
    return original


# LLM Keys & Config
PPLX_API_KEY = os.getenv("PPLX_API_KEY")
DASHSCOPE_API_BASE = os.getenv("DASHSCOPE_API_BASE")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
PRIMARY_LLM = os.getenv("PRIMARY_LLM", "perplexity").lower()
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-flash")

# --------------------------
# 🔐 USER AUTH (unchanged)
# --------------------------
credentials = {
    "usernames": {
        "admin": {
            "email": "admin@example.com",
            "name": "Admin",
            "password": "admin1234"
        },
        "user1": {
            "email": "user1@example.com",
            "name": "Regular User",
            "password": "$2b$12$EtbD9bAe/QH0l1vK0y5VUO6aS1G7dXH70dG0dG0dG0dG0dG0dG"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials, "langsmith_cookie", "langsmith_key", 3 * 24 * 60 * 60
)

# Login Screen (streamlit-authenticator 0.4+: first arg is location; state lives in session_state)
authenticator.login(location="main")
authentication_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

if authentication_status is False:
    st.error("用戶名稱或密碼錯誤")
if authentication_status is None:
    st.warning("請先登入")
    st.stop()

# Logged-in area
st.success(f"歡迎，{name}！")
authenticator.logout("登出", "sidebar")

# --------------------------
# Helpers (defined before Chat UI so human-in-the-loop panels / chat can call them on every rerun)
# --------------------------
def should_run_langgraph(user_input):
    triggers = ["寫文章"]
    return any(t in user_input.lower() for t in triggers)


def topic_from_workflow_prompt(user_input: str) -> str:
    """Use text after a trigger as the news topic; otherwise use the whole message."""
    p = user_input.strip()
    low = p.lower()
    for t in ("run workflow", "execute agent", "use workflow", "langgraph"):
        idx = low.find(t)
        if idx == -1:
            continue
        rest = p[idx + len(t) :].strip()
        if rest.startswith(":"):
            rest = rest[1:].strip()
        if rest:
            return rest
    return p


def _truncate_preview(text: str, max_len: int = 480) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _format_final_article_markdown(fa: Any) -> str:
    if fa is None:
        return ""
    if hasattr(fa, "model_dump"):
        fa = fa.model_dump()
    if not isinstance(fa, dict):
        return str(fa)
    lines = [
        f"**{fa.get('headline', '')}**",
        "",
        fa.get("summary", "") or "",
        "",
        fa.get("body", "") or "",
    ]
    srcs = fa.get("sources") or []
    if srcs:
        lines.append("\n**參考來源**\n")
        for s in srcs:
            if isinstance(s, dict):
                lines.append(f"- [{s.get('title', '')}]({s.get('url', '')})")
            else:
                lines.append(f"- {s}")
    return "\n".join(lines)


def _format_workflow_progress(state: Any, *, step: int) -> str:
    """Human-readable snapshot for Streamlit while ``stream_mode='values'`` runs."""
    lines: list[str] = [
        "🔄 **AI傳真優** — 即時狀態",
        "",
        f"*第 {step} 次更新*",
        "",
    ]
    if not isinstance(state, dict):
        lines.append("```")
        lines.append(_truncate_preview(str(state), 720))
        lines.append("```")
        return "\n".join(lines)

    if state.get("__interrupt__"):
        lines.append("⏸ **已暫停**（人機協作）")
        intr = state["__interrupt__"]
        if intr:
            first = intr[0]
            val = first.get("value") if isinstance(first, dict) else getattr(first, "value", None)
            if isinstance(val, dict) and val.get("kind"):
                k = val["kind"]
                step_hint = {
                    "configure_review": "已完成 **configure**，尚未研究或尚未貼上草稿",
                    "edit_draft_review": "編輯模式 — 請貼上完整草稿後送交事實查核",
                    "outline_review": "已有大綱，尚未撰寫",
                    "revise_review": "已完成 **revise**，尚未最終審閱",
                }.get(k, "")
                lines.append(f"- 步驟：`{k}`" + (f"（{step_hint}）" if step_hint else ""))
        lines.append("")

    st_ = state.get("status")
    if st_:
        lines.append(f"**狀態：** `{st_}`")

    topic = state.get("topic")
    if topic:
        lines.append(f"**主題：** {_truncate_preview(str(topic), 200)}")
    if state.get("article_type"):
        lines.append(f"**文章類型：** {state['article_type']}")
    twc = state.get("target_word_count")
    if twc is not None:
        lines.append(f"**目標字數：** {twc}")

    q = state.get("query")
    if q:
        lines.append(f"**搜尋查詢：** {_truncate_preview(str(q), 240)}")

    sr = state.get("search_results")
    if isinstance(sr, list) and sr:
        lines.append(f"**搜尋結果：** {len(sr)} 筆")

    srcs = state.get("sources")
    if isinstance(srcs, list) and srcs:
        lines.append(f"**精選來源：** {len(srcs)}")

    notes = state.get("research_notes")
    if notes:
        lines.append("")
        lines.append("**研究筆記**（預覽）")
        lines.append("```")
        lines.append(_truncate_preview(str(notes), 900))
        lines.append("```")

    outline = state.get("outline")
    if outline:
        lines.append("")
        lines.append("**大綱**（預覽）")
        lines.append("```")
        lines.append(_truncate_preview(str(outline), 900))
        lines.append("```")

    draft = state.get("draft")
    if draft:
        lines.append("")
        lines.append("**草稿**（預覽）")
        lines.append("```")
        lines.append(_truncate_preview(str(draft), 900))
        lines.append("```")

    fc = state.get("fact_check_score")
    if fc is not None:
        lines.append(f"**事實核對分數：** {fc}")
    rnd = state.get("fact_check_revision_round")
    if rnd is not None and rnd > 0:
        lines.append(f"**事實核對／修訂輪次：** {rnd}")

    err = state.get("error")
    if err:
        lines.append("")
        lines.append(f"⚠️ **狀態錯誤：** {_truncate_preview(str(err), 400)}")

    fa = state.get("final_article")
    if fa is not None:
        lines.append("")
        lines.append("---")
        lines.append("**定稿文章**")
        lines.append("")
        lines.append(_format_final_article_markdown(fa))

    return "\n".join(lines)


def _first_interrupt_entry(result: dict) -> tuple[Any, str] | None:
    """Return (value, interrupt_id) for the first __interrupt__ item, if any."""
    raw = result.get("__interrupt__")
    if not raw:
        return None
    first = raw[0]
    if isinstance(first, dict):
        return first.get("value"), str(first.get("id") or "")
    v = getattr(first, "value", first)
    i = getattr(first, "id", "")
    return v, str(i or "")


def _format_remote_graph_result(result: object, *, show_interrupt_json: bool = True) -> str:
    if result is None:
        return "部署未回傳任何輸出。"
    if not isinstance(result, dict):
        return str(result)
    if result.get("__interrupt__"):
        if not show_interrupt_json:
            return (
                "**工作流程已暫停**（人機協作）。"
                "請使用輸入框上方的對應面板：**設定審閱**（研究或貼稿前）、"
                "**草稿審閱**（編輯模式）、**大綱審閱**（撰寫前），"
                "或 **修訂審閱**（事實核對修訂後）。"
            )
        parts = []
        for item in result["__interrupt__"]:
            if isinstance(item, dict) and "value" in item:
                parts.append(json.dumps(item["value"], ensure_ascii=False, indent=2))
            else:
                parts.append(repr(item))
        return (
            "工作流程已暫停（人機協作）。"
            "若應用程式未能取得 thread id，請在 LangGraph Studio 或 SDK 繼續執行。\n\n"
            "**中斷載荷：**\n```json\n"
            + "\n---\n".join(parts)
            + "\n```"
        )
    fa = result.get("final_article")
    if fa is not None:
        return _format_final_article_markdown(fa)
    err = result.get("error")
    if err:
        return f"工作流程狀態錯誤：{err}"
    return "```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```"


def _clear_workflow_pending() -> None:
    st.session_state.pop("workflow_pending_thread_id", None)
    st.session_state.pop("workflow_pending_interrupt_id", None)
    st.session_state.pop("workflow_interrupt_context", None)


def _remote_invoke_values(
    remote: RemoteGraph,
    graph_input: Any,
    *,
    config: dict | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[Any, str | None]:
    """Stream with ``stream_mode='values'`` and resolve ``thread_id`` for resume.

    Prefer ``config['configurable']['thread_id']`` when provided (uses
    ``POST /threads/{thread_id}/runs/stream`` and does not depend on response
    headers). Otherwise fall back to ``on_run_created`` / ``Content-Location``.

    If ``on_progress`` is set, it is called with markdown after each state snapshot
    (for progressive UI).
    """
    run_meta: dict[str, str | None] = {}

    def on_run_created(m):
        run_meta["run_id"] = m.get("run_id")
        tid = m.get("thread_id")
        run_meta["thread_id"] = str(tid) if tid else None

    last: Any = None
    n = 0
    for chunk in remote.stream(
        graph_input,
        config=config or {},
        stream_mode="values",
        on_run_created=on_run_created,
    ):
        last = chunk
        n += 1
        if on_progress is not None:
            on_progress(_format_workflow_progress(chunk, step=n))
    cfg_tid = (config or {}).get("configurable", {}).get("thread_id")
    if isinstance(cfg_tid, str) and cfg_tid.strip():
        return last, cfg_tid.strip()
    return last, run_meta.get("thread_id")


def _command_for_resume(resume_payload: Any, interrupt_id: str | None) -> Command:
    if interrupt_id:
        return Command(resume={interrupt_id: resume_payload})
    return Command(resume=resume_payload)


def _normalize_revise_human_payload(
    *,
    decision: str,
    feedback_text: str = "",
    revised_draft_text: str = "",
) -> tuple[dict[str, str] | None, str | None]:
    """Normalize UI input to a revise_human resume payload."""
    d = (decision or "").strip().lower()
    if d == "accept":
        return {"action": "accept"}, None
    if d == "feedback":
        fb = (feedback_text or "").strip()
        if not fb:
            return None, "提交前請先輸入意見。"
        return {"action": "feedback", "feedback": fb}, None
    if d == "replace":
        draft = (revised_draft_text or "").strip()
        if not draft:
            return None, "提交前請先貼上你修訂後的草稿。"
        return {"action": "replace", "draft": draft}, None
    return None, f"不支援的修訂決定：{decision!r}"


def _store_workflow_interrupt(thread_id: str, intr_id: str | None, value: Any) -> None:
    st.session_state.workflow_pending_thread_id = thread_id
    st.session_state.workflow_pending_interrupt_id = intr_id or None
    if isinstance(value, dict):
        st.session_state.workflow_interrupt_context = value
    else:
        st.session_state.workflow_interrupt_context = {"kind": "unknown", "raw": value}


# --------------------------
# ✅ NEW: Perplexity LLM
# --------------------------
def call_perplexity(prompt):
    client = OpenAI(
        api_key=PPLX_API_KEY,
        base_url="https://api.perplexity.ai"
    )
    response = client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --------------------------
# ✅ NEW: Qwen (DashScope) LLM
# --------------------------
def call_qwen(prompt):
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_API_BASE
    )
    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# --------------------------
# Auto-select Primary LLM
# --------------------------
def call_general_llm(prompt):
    if PRIMARY_LLM == "perplexity":
        return call_perplexity(prompt)
    elif PRIMARY_LLM == "qwen":
        return call_qwen(prompt)
    else:
        return "`.env` 中選取的 LLM 無效"


def call_remote_news_graph(
    prompt: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    discarded = ""
    if st.session_state.get("workflow_pending_thread_id"):
        _clear_workflow_pending()
        discarded = "_先前暫停的工作流程已捨棄。_\n\n"

    base, url_err = _resolve_langgraph_base_url()
    api_key = _resolve_langgraph_api_key()
    if url_err:
        return discarded + url_err
    if not api_key:
        return discarded + (
            "⚠️ AI傳真優：請設定 `LANGGRAPH_API_KEY`、`LANGSMITH_API_KEY` 或 "
            "`LANGCHAIN_API_KEY` 其中一項（與你在 LangSmith 使用的金鑰相同）。"
        )

    topic = topic_from_workflow_prompt(prompt)
    if not topic.strip():
        topic = (
            "未指定主題 — 請在觸發詞後描述新聞主題，例如："
            "「run workflow: 歐盟科技監管」。"
        )
    try:
        init = create_initial_state(topic, LANGGRAPH_DEFAULT_ARTICLE_TYPE)
    except ValueError as e:
        return (
            discarded
            + "⚠️ 文章類型無效："
            + f"{e}。請將 `LANGGRAPH_DEFAULT_ARTICLE_TYPE` 設為 懶人包、多方觀點 或 其他。"
        )

    remote = RemoteGraph(LANGGRAPH_GRAPH_ID, url=base, api_key=api_key)
    run_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        result, thread_id = _remote_invoke_values(
            remote, init, config=run_config, on_progress=on_progress
        )
    except GraphInterrupt as gi:
        payloads = [getattr(i, "value", i) for i in gi.args[0]] if gi.args else []
        return (
            discarded
            + "工作流程已暫停（中斷）。\n\n```json\n"
            + json.dumps(payloads, ensure_ascii=False, indent=2)
            + "\n```"
        )
    except RemoteException as e:
        return discarded + f"⚠️ AI傳真優 部署錯誤：{e}"
    except Exception as e:
        raw = f"{type(e).__name__}: {e}"
        return discarded + "⚠️ AI傳真優 錯誤：" + _langgraph_connect_help(raw)

    if isinstance(result, dict) and result.get("__interrupt__"):
        parsed = _first_interrupt_entry(result)
        if parsed:
            value, intr_id = parsed
            kind = value.get("kind") if isinstance(value, dict) else None
            # Order matches graph: configure_human → … → outline_human → … → revise_human
            if kind == "configure_review":
                if thread_id:
                    _store_workflow_interrupt(thread_id, intr_id or None, value)
                    return (
                        discarded
                        + "🔄 **AI傳真優** 已暫停，等待 **設定審閱**（自動設定完成後、研究或貼稿開始前）。\n\n"
                        + "請使用下方的 **設定審閱** 面板確認或修改主題、文章類型、目標字數、"
                        "LLM 設定、選填搜尋查詢，以及 **工作流程模式**（撰寫／編輯）。"
                    )
                return (
                    discarded
                    + "**執行後缺少 thread id**（非預期）。"
                    + "請在 LangGraph Studio 或 SDK 繼續此執行。\n\n"
                    + _format_remote_graph_result(result)
                )
            if kind == "edit_draft_review":
                if thread_id:
                    _store_workflow_interrupt(thread_id, intr_id or None, value)
                    return (
                        discarded
                        + "🔄 **AI傳真優** 已暫停，等待 **草稿審閱**（編輯模式）。\n\n"
                        + "請在下方面板貼上**完整草稿**；繼續後將直接進行 **事實查核**（略過研究、大綱與撰稿節點）。"
                    )
                return (
                    discarded
                    + "**執行後缺少 thread id**（非預期）。"
                    + "請在 LangGraph Studio 或 SDK 繼續此執行。\n\n"
                    + _format_remote_graph_result(result)
                )
            if kind == "outline_review":
                if thread_id:
                    _store_workflow_interrupt(thread_id, intr_id or None, value)
                    return (
                        discarded
                        + "🔄 **AI傳真優** 已暫停，等待 **大綱審閱**。\n\n"
                        + "請使用下方的 **大綱審閱** 面板接受大綱或貼上替換內容。"
                    )
                return (
                    discarded
                    + "**執行後缺少 thread id**（非預期）。"
                    + "請在 LangGraph Studio 或 SDK 繼續此執行。\n\n"
                    + _format_remote_graph_result(result)
                )
            if kind == "revise_review":
                if thread_id:
                    _store_workflow_interrupt(thread_id, intr_id or None, value)
                    return (
                        discarded
                        + "🔄 **AI傳真優** 已暫停，等待 **修訂審閱**（修訂完成後、最終審閱前）。\n\n"
                        + "請使用下方的 **修訂審閱** 面板接受此草稿、"
                        "提供意見要求再修訂一輪，或貼上你自行修訂的草稿。"
                    )
                return (
                    discarded
                    + "**執行後缺少 thread id**（非預期）。"
                    + "請在 LangGraph Studio 或 SDK 繼續此執行。\n\n"
                    + _format_remote_graph_result(result)
                )
        return discarded + _format_remote_graph_result(result)

    return discarded + _format_remote_graph_result(result)


def resume_remote_news_graph(
    resume_payload: Any,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Resume a paused deployment run after any human-in-the-loop node."""
    base, url_err = _resolve_langgraph_base_url()
    api_key = _resolve_langgraph_api_key()
    thread_id = st.session_state.get("workflow_pending_thread_id")
    intr_id = st.session_state.get("workflow_pending_interrupt_id")

    if url_err:
        return url_err
    if not api_key:
        return "⚠️ AI傳真優 尚未設定（網址或 API 金鑰）。"
    if not thread_id:
        return "此工作階段沒有已暫停的工作流程。"

    remote = RemoteGraph(LANGGRAPH_GRAPH_ID, url=base, api_key=api_key)
    cmd = _command_for_resume(resume_payload, intr_id)
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        result, _ = _remote_invoke_values(remote, cmd, config=cfg, on_progress=on_progress)
    except RemoteException as e:
        return f"⚠️ AI傳真優 部署錯誤：{e}"
    except Exception as e:
        raw = f"{type(e).__name__}: {e}"
        return "⚠️ AI傳真優 錯誤：" + _langgraph_connect_help(raw)

    if isinstance(result, dict) and result.get("__interrupt__"):
        parsed = _first_interrupt_entry(result)
        if parsed:
            value, new_intr_id = parsed
            kind = value.get("kind") if isinstance(value, dict) else None
            if kind == "configure_review":
                _store_workflow_interrupt(thread_id, new_intr_id or None, value)
                return (
                    "工作流程再次暫停於設定審閱（研究前）。"
                    "請在下方面板更新你的選擇。"
                )
            if kind == "edit_draft_review":
                _store_workflow_interrupt(thread_id, new_intr_id or None, value)
                return (
                    "工作流程再次暫停於草稿審閱（編輯模式）。"
                    "請在下方面板貼上完整草稿。"
                )
            if kind == "outline_review":
                _store_workflow_interrupt(thread_id, new_intr_id or None, value)
                return (
                    "工作流程再次暫停於大綱審閱。"
                    "請在下方面板更新你的選擇。"
                )
            if kind == "revise_review":
                _store_workflow_interrupt(thread_id, new_intr_id or None, value)
                return (
                    "工作流程再次暫停於修訂審閱（修訂之後）。"
                    "請在下方面板更新你的選擇。"
                )
        _clear_workflow_pending()
        return _format_remote_graph_result(result)

    _clear_workflow_pending()
    return _format_remote_graph_result(result)


# --------------------------
# Chat UI
# --------------------------
st.title("AI傳真優")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "你好！輸入 **寫文章** + **主題**，即可執行從資料搜集、撰稿、事實查核到風格校對的AI自動化新聞工作流程。"
                "我會在重要節點暫停，請你確認下一步行動"
            ),
        }
    ]

# Show messages first so the page reads top-to-bottom; HITL panels render below (near chat input).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Human-in-the-loop panels sit above the chat input. After a LangGraph pause we call
# ``st.rerun()`` so this block runs on the next pass with ``workflow_pending_*`` already set.
if st.session_state.get("workflow_pending_thread_id"):
    ctx = st.session_state.get("workflow_interrupt_context") or {}
    kind = ctx.get("kind")

    if kind == "configure_review":
        wf_settings = get_settings()
        with st.container(border=True):
            st.subheader("設定審閱")
            st.caption(
                "圖節點 **configure_human**：在自動設定完成後、研究或貼稿開始**前**確認設定。"
                "可選 **編輯模式**：確認後會暫停於 **草稿審閱**，貼上完整稿後直接 **事實查核**（略過研究、大綱、撰稿）。"
                "撰寫模式下之後會暫停於大綱確認；若圖會修訂草稿，定稿前另有修訂審閱。"
            )
            default_at = ctx.get("article_type") or ARTICLE_TYPES[0]
            at_idx = (
                ARTICLE_TYPES.index(default_at)
                if default_at in ARTICLE_TYPES
                else 0
            )
            default_lp = (ctx.get("llm_provider") or "perplexity").strip().lower()
            lp_idx = 0 if default_lp == "perplexity" else 1

            default_task = str(ctx.get("task") or "write").strip().lower()
            task_idx = 1 if default_task == "edit" else 0
            cfg_task_choice = st.radio(
                "工作流程模式（請確認後再繼續）",
                options=["write", "edit"],
                index=task_idx,
                format_func=lambda v: {
                    "write": "撰寫：從研究、大綱到撰稿（預設）",
                    "edit": "編輯：我已有完整草稿 — 略過研究／大綱／撰稿，僅事實查核",
                }[v],
                key="wf_cfg_task",
            )

            cfg_topic = st.text_input(
                "主題",
                value=str(ctx.get("topic") or ""),
                key="wf_cfg_topic",
            )
            cfg_article_type = st.selectbox(
                "文章類型",
                ARTICLE_TYPES,
                index=at_idx,
                key="wf_cfg_article_type",
            )
            cfg_twc = st.number_input(
                "目標字數",
                min_value=wf_settings.min_target_word_count,
                max_value=wf_settings.max_target_word_count,
                value=int(ctx.get("target_word_count") or wf_settings.default_target_word_count),
                step=50,
                key="wf_cfg_twc",
            )
            cfg_llm = st.selectbox(
                "LLM 供應商",
                ["perplexity", "qwen"],
                index=lp_idx,
                key="wf_cfg_llm",
            )
            cfg_model = st.text_input(
                "LLM 模型（選填；留空則用供應商預設）",
                value=str(ctx.get("llm_model") or ""),
                key="wf_cfg_model",
            )
            cfg_query = st.text_area(
                "搜尋查詢（選填；留空則由研究階段自動產生查詢）",
                value=str(ctx.get("query") or ""),
                height=100,
                key="wf_cfg_query",
            )

            col_c, col_d2 = st.columns(2)
            with col_c:
                cfg_accept = st.button(
                    "照現有內容接受（不修改）",
                    type="secondary",
                    use_container_width=True,
                )
            with col_d2:
                cfg_discard = st.button("捨棄暫停", use_container_width=True)

            cfg_go = st.button("使用以上設定繼續", type="primary", use_container_width=True)

            if cfg_discard:
                _clear_workflow_pending()
                st.rerun()
            if cfg_accept:
                wf_prog = st.empty()
                with st.spinner("正在繼續工作流程…"):
                    resumed = resume_remote_news_graph(
                        {"action": "accept", "task": cfg_task_choice},
                        on_progress=wf_prog.markdown,
                    )
                wf_prog.markdown("🔄 **AI傳真優**（已繼續）\n\n" + resumed)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "🔄 **AI傳真優**（已繼續）\n\n" + resumed}
                )
                st.rerun()
            if cfg_go:
                wf_prog = st.empty()
                payload = {
                    "action": "confirm",
                    "topic": (cfg_topic or "").strip(),
                    "article_type": cfg_article_type,
                    "target_word_count": int(cfg_twc),
                    "llm_provider": cfg_llm,
                    "llm_model": (cfg_model or "").strip(),
                    "query": (cfg_query or "").strip(),
                    "task": cfg_task_choice,
                }
                with st.spinner("正在繼續工作流程…"):
                    resumed = resume_remote_news_graph(
                        payload,
                        on_progress=wf_prog.markdown,
                    )
                wf_prog.markdown("🔄 **AI傳真優**（已繼續）\n\n" + resumed)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "🔄 **AI傳真優**（已繼續）\n\n" + resumed}
                )
                st.rerun()

    elif kind == "edit_draft_review":
        with st.container(border=True):
            st.subheader("草稿審閱（編輯模式）")
            st.caption(
                "圖節點 **edit_human**：你已選擇略過研究、大綱與自動撰稿。"
                "請貼上**完整文章草稿**；送出後圖會將此稿作為 ``draft`` 直接進入 **fact_check**。"
            )
            topic = ctx.get("topic", "")
            if topic:
                st.markdown(f"**主題：** {topic}")
            at = ctx.get("article_type", "")
            if at:
                st.markdown(f"**文章類型：** {at}")

            draft_in = st.text_area(
                "完整草稿",
                height=320,
                key="wf_edit_full_draft",
                placeholder="在此貼上全文…",
            )
            col_ed, col_disc = st.columns(2)
            with col_ed:
                submit_draft = st.button(
                    "提交草稿並送交事實查核",
                    type="primary",
                    use_container_width=True,
                )
            with col_disc:
                discard_ed = st.button("捨棄暫停", use_container_width=True)

            if discard_ed:
                _clear_workflow_pending()
                st.rerun()
            if submit_draft:
                text = (draft_in or "").strip()
                if not text:
                    st.warning("請貼上完整草稿後再提交。")
                else:
                    wf_prog = st.empty()
                    with st.spinner("正在繼續工作流程…"):
                        resumed = resume_remote_news_graph(
                            {"action": "submit", "draft": text},
                            on_progress=wf_prog.markdown,
                        )
                    wf_prog.markdown("🔄 **AI傳真優**（已繼續）\n\n" + resumed)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "🔄 **AI傳真優**（已繼續）\n\n" + resumed}
                    )
                    st.rerun()

    elif kind == "outline_review":
        with st.container(border=True):
            st.subheader("大綱審閱")
            st.caption(
                "圖節點 **outline_human**：在分析完成後、撰寫**前**確認大綱。"
                "可在下方接受或貼上替換內容；繼續後，圖執行時會顯示 **即時狀態** 更新。"
            )
            topic = ctx.get("topic", "")
            if topic:
                st.markdown(f"**主題：** {topic}")
            outline = ctx.get("outline", "")
            st.markdown("**建議大綱**")
            st.text(outline if outline else "（空白）")

            col_a, col_d = st.columns(2)
            with col_a:
                accept = st.button("接受大綱", type="primary", use_container_width=True)
            with col_d:
                discard = st.button("捨棄暫停", use_container_width=True)

            replacement = st.text_area(
                "改為使用你的大綱",
                height=200,
                key="wf_outline_replacement",
                placeholder="在此貼上完整大綱，然後按下方按鈕。",
            )
            replace_go = st.button("提交替換大綱", use_container_width=True)

            if discard:
                _clear_workflow_pending()
                st.rerun()
            if accept:
                wf_prog = st.empty()
                with st.spinner("正在繼續工作流程…"):
                    resumed = resume_remote_news_graph(
                        {"action": "accept"},
                        on_progress=wf_prog.markdown,
                    )
                wf_prog.markdown("🔄 **AI傳真優**（已繼續）\n\n" + resumed)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "🔄 **AI傳真優**（已繼續）\n\n" + resumed}
                )
                st.rerun()
            if replace_go:
                text = (replacement or "").strip()
                if not text:
                    st.warning("請貼上完整大綱，或使用 **接受大綱**。")
                else:
                    wf_prog = st.empty()
                    with st.spinner("正在繼續工作流程…"):
                        resumed = resume_remote_news_graph(
                            {"action": "replace", "outline": text},
                            on_progress=wf_prog.markdown,
                        )
                    wf_prog.markdown("🔄 **AI傳真優**（已繼續）\n\n" + resumed)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "🔄 **AI傳真優**（已繼續）\n\n" + resumed}
                    )
                    st.rerun()

    elif kind == "revise_review":
        with st.container(border=True):
            st.subheader("修訂審閱")
            st.caption(
                "圖節點 **revise_human**：在 **revise** 節點（事實核對迴圈）之後暫停。"
                "可接受此草稿、提供意見再請 AI **修訂** 一輪，或貼上你自行修訂的草稿，"
                "之後圖會繼續至 **review**（最終潤飾）。"
            )
            topic = ctx.get("topic", "")
            if topic:
                st.markdown(f"**主題：** {topic}")
            draft = ctx.get("draft", "")
            st.markdown("**目前修訂後草稿**")
            st.text(draft if draft else "（空白）")

            decision = st.radio(
                "下一步要怎麼做？",
                ["accept", "feedback", "replace"],
                format_func=lambda v: {
                    "accept": "接受草稿並繼續至審閱",
                    "feedback": "提供意見並請 AI 再修訂一輪",
                    "replace": "貼上我自行修訂的草稿並繼續至審閱",
                }[v],
                key="wf_revise_decision",
            )
            feedback_text = st.text_area(
                "下一輪修訂的意見",
                height=120,
                key="wf_revise_feedback",
                placeholder="具體說明希望 AI 改動哪些內容…",
            )
            revised_draft_text = st.text_area(
                "你修訂後的草稿",
                height=240,
                key="wf_revise_draft",
                placeholder="在此貼上完整修訂稿…",
            )

            col_submit, col_discard = st.columns(2)
            with col_submit:
                submit_revise = st.button(
                    "提交決定",
                    type="primary",
                    use_container_width=True,
                )
            with col_discard:
                discard_revise = st.button("捨棄暫停", use_container_width=True)

            if discard_revise:
                _clear_workflow_pending()
                st.rerun()
            if submit_revise:
                payload, err = _normalize_revise_human_payload(
                    decision=decision,
                    feedback_text=feedback_text,
                    revised_draft_text=revised_draft_text,
                )
                if err or payload is None:
                    st.warning(err or "修訂決定無效。")
                else:
                    wf_prog = st.empty()
                    with st.spinner("正在繼續工作流程…"):
                        resumed = resume_remote_news_graph(
                            payload,
                            on_progress=wf_prog.markdown,
                        )
                    wf_prog.markdown("🔄 **AI傳真優**（已繼續）\n\n" + resumed)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "🔄 **AI傳真優**（已繼續）\n\n" + resumed}
                    )
                    st.rerun()

    else:
        with st.container(border=True):
            st.subheader("工作流程已暫停")
            st.caption(
                "此工作階段的中斷類型未知或缺失。你可捨棄暫停，或在 LangGraph Studio 繼續執行。"
            )
            if st.button("捨棄暫停", key="wf_unknown_discard"):
                _clear_workflow_pending()
                st.rerun()

# --------------------------
# Chat Input
# --------------------------
if prompt := st.chat_input("輸入訊息…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        use_workflow = should_run_langgraph(prompt)
        progress_slot = st.empty()
        with st.spinner("正在執行 AI傳真優…" if use_workflow else "思考中…"):
            if use_workflow:
                resp = call_remote_news_graph(
                    prompt,
                    on_progress=progress_slot.markdown,
                )
            else:
                resp = call_general_llm(prompt)
        if use_workflow:
            progress_slot.markdown("🔄 **AI傳真優 已啟動**\n\n" + resp)
        else:
            progress_slot.markdown(resp)

    st.session_state.messages.append({"role": "assistant", "content": resp})
    if use_workflow and st.session_state.get("workflow_pending_thread_id"):
        st.rerun()
