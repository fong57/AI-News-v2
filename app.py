import json
import os
import uuid
from collections.abc import Callable
from typing import Any

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
LANGSMITH_RESOURCE_URL = os.getenv("LANGSMITH_RESOURCE_URL")
LANGGRAPH_GRAPH_ID = os.getenv("LANGGRAPH_GRAPH_ID", "news_writer")
LANGGRAPH_DEFAULT_ARTICLE_TYPE = os.getenv("LANGGRAPH_DEFAULT_ARTICLE_TYPE", "懶人包")


def _resolve_langgraph_api_key() -> str | None:
    """Match langgraph_sdk: LANGGRAPH_API_KEY, then LANGSMITH_API_KEY, then LANGCHAIN_API_KEY."""
    for name in ("LANGGRAPH_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        v = os.getenv(name)
        if v and v.strip():
            return v.strip().strip('"').strip("'")
    return None

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
    st.error("Username/password incorrect")
if authentication_status is None:
    st.warning("Please log in")
    st.stop()

# Logged-in area
st.success(f"Welcome, {name}!")
authenticator.logout("Logout", "sidebar")

# --------------------------
# Helpers (defined before Chat UI so outline / chat can call them on every rerun)
# --------------------------
def should_run_langgraph(user_input):
    triggers = ["run workflow", "langgraph", "execute agent", "use workflow"]
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
        lines.append("\n**Sources**\n")
        for s in srcs:
            if isinstance(s, dict):
                lines.append(f"- [{s.get('title', '')}]({s.get('url', '')})")
            else:
                lines.append(f"- {s}")
    return "\n".join(lines)


def _format_workflow_progress(state: Any, *, step: int) -> str:
    """Human-readable snapshot for Streamlit while ``stream_mode='values'`` runs."""
    lines: list[str] = [
        "🔄 **LangGraph** — live state",
        "",
        f"*Update {step}*",
        "",
    ]
    if not isinstance(state, dict):
        lines.append("```")
        lines.append(_truncate_preview(str(state), 720))
        lines.append("```")
        return "\n".join(lines)

    if state.get("__interrupt__"):
        lines.append("⏸ **Paused** (human-in-the-loop)")
        intr = state["__interrupt__"]
        if intr:
            first = intr[0]
            val = first.get("value") if isinstance(first, dict) else getattr(first, "value", None)
            if isinstance(val, dict) and val.get("kind"):
                lines.append(f"- Kind: `{val['kind']}`")
        lines.append("")

    st_ = state.get("status")
    if st_:
        lines.append(f"**Status:** `{st_}`")

    topic = state.get("topic")
    if topic:
        lines.append(f"**Topic:** {_truncate_preview(str(topic), 200)}")
    if state.get("article_type"):
        lines.append(f"**Article type:** {state['article_type']}")
    twc = state.get("target_word_count")
    if twc is not None:
        lines.append(f"**Target words:** {twc}")

    q = state.get("query")
    if q:
        lines.append(f"**Search query:** {_truncate_preview(str(q), 240)}")

    sr = state.get("search_results")
    if isinstance(sr, list) and sr:
        lines.append(f"**Search results:** {len(sr)} row(s)")

    srcs = state.get("sources")
    if isinstance(srcs, list) and srcs:
        lines.append(f"**Curated sources:** {len(srcs)}")

    notes = state.get("research_notes")
    if notes:
        lines.append("")
        lines.append("**Research notes** (preview)")
        lines.append("```")
        lines.append(_truncate_preview(str(notes), 900))
        lines.append("```")

    outline = state.get("outline")
    if outline:
        lines.append("")
        lines.append("**Outline** (preview)")
        lines.append("```")
        lines.append(_truncate_preview(str(outline), 900))
        lines.append("```")

    draft = state.get("draft")
    if draft:
        lines.append("")
        lines.append("**Draft** (preview)")
        lines.append("```")
        lines.append(_truncate_preview(str(draft), 900))
        lines.append("```")

    fc = state.get("fact_check_score")
    if fc is not None:
        lines.append(f"**Fact-check score:** {fc}")
    rnd = state.get("fact_check_revision_round")
    if rnd is not None and rnd > 0:
        lines.append(f"**Fact-check / revise round:** {rnd}")

    err = state.get("error")
    if err:
        lines.append("")
        lines.append(f"⚠️ **Error in state:** {_truncate_preview(str(err), 400)}")

    fa = state.get("final_article")
    if fa is not None:
        lines.append("")
        lines.append("---")
        lines.append("**Final article**")
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
        return "No output returned from the deployment."
    if not isinstance(result, dict):
        return str(result)
    if result.get("__interrupt__"):
        if not show_interrupt_json:
            return (
                "**Workflow paused** (human-in-the-loop). "
                "Use the outline review panel below to continue."
            )
        parts = []
        for item in result["__interrupt__"]:
            if isinstance(item, dict) and "value" in item:
                parts.append(json.dumps(item["value"], ensure_ascii=False, indent=2))
            else:
                parts.append(repr(item))
        return (
            "Workflow paused (human-in-the-loop). "
            "If the app did not capture a thread id, resume from LangGraph Studio or the SDK.\n\n"
            "**Interrupt payload:**\n```json\n"
            + "\n---\n".join(parts)
            + "\n```"
        )
    fa = result.get("final_article")
    if fa is not None:
        return _format_final_article_markdown(fa)
    err = result.get("error")
    if err:
        return f"Workflow error in state: {err}"
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
            return None, "Please enter feedback before submitting."
        return {"action": "feedback", "feedback": fb}, None
    if d == "replace":
        draft = (revised_draft_text or "").strip()
        if not draft:
            return None, "Please paste your revised draft before submitting."
        return {"action": "replace", "draft": draft}, None
    return None, f"Unsupported revise decision: {decision!r}"


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
        return "Invalid LLM selected in .env"


def call_remote_news_graph(
    prompt: str,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    discarded = ""
    if st.session_state.get("workflow_pending_thread_id"):
        _clear_workflow_pending()
        discarded = "_Previous paused workflow was discarded._\n\n"

    base = (LANGSMITH_RESOURCE_URL or "").strip().rstrip("/")
    api_key = _resolve_langgraph_api_key()
    if not base:
        return (
            discarded
            + "⚠️ LangGraph: set `LANGSMITH_RESOURCE_URL` to your deployment URL "
            "(no trailing path)."
        )
    if not api_key:
        return discarded + (
            "⚠️ LangGraph: set one of `LANGGRAPH_API_KEY`, `LANGSMITH_API_KEY`, or "
            "`LANGCHAIN_API_KEY` (same key you use in LangSmith)."
        )

    topic = topic_from_workflow_prompt(prompt)
    if not topic.strip():
        topic = (
            "Unspecified topic — describe a news topic after the trigger, e.g. "
            "'run workflow: EU tech regulation'."
        )
    try:
        init = create_initial_state(topic, LANGGRAPH_DEFAULT_ARTICLE_TYPE)
    except ValueError as e:
        return (
            discarded
            + "⚠️ Invalid article type: "
            + f"{e}. Set `LANGGRAPH_DEFAULT_ARTICLE_TYPE` to 懶人包, 多方觀點, or 其他."
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
            + "Workflow paused (interrupt).\n\n```json\n"
            + json.dumps(payloads, ensure_ascii=False, indent=2)
            + "\n```"
        )
    except RemoteException as e:
        return discarded + f"⚠️ LangGraph deployment error: {e}"
    except Exception as e:
        return discarded + f"⚠️ LangGraph Error: {type(e).__name__}: {e}"

    if isinstance(result, dict) and result.get("__interrupt__"):
        parsed = _first_interrupt_entry(result)
        if parsed:
            value, intr_id = parsed
            kind = value.get("kind") if isinstance(value, dict) else None
            if kind == "outline_review":
                if thread_id:
                    _store_workflow_interrupt(thread_id, intr_id or None, value)
                    return (
                        discarded
                        + "🔄 **LangGraph** paused for **outline review**.\n\n"
                        + "Use the **Outline review** panel below to accept the outline or "
                        "paste a replacement."
                    )
                return (
                    discarded
                    + "**Thread id missing** after run (unexpected). "
                    + "Resume this run from LangGraph Studio or the SDK.\n\n"
                    + _format_remote_graph_result(result)
                )
            if kind == "configure_review":
                if thread_id:
                    _store_workflow_interrupt(thread_id, intr_id or None, value)
                    return (
                        discarded
                        + "🔄 **LangGraph** paused for **configure review**.\n\n"
                        + "Use the **Configure review** panel below to confirm or edit "
                        "topic, article type, word target, LLM settings, and optional search "
                        "query before research."
                    )
                return (
                    discarded
                    + "**Thread id missing** after run (unexpected). "
                    + "Resume this run from LangGraph Studio or the SDK.\n\n"
                    + _format_remote_graph_result(result)
                )
            if kind == "revise_review":
                if thread_id:
                    _store_workflow_interrupt(thread_id, intr_id or None, value)
                    return (
                        discarded
                        + "🔄 **LangGraph** paused for **revise review**.\n\n"
                        + "Use the **Revise review** panel below to accept this draft, "
                        "add feedback for another revise pass, or paste your own revised draft."
                    )
                return (
                    discarded
                    + "**Thread id missing** after run (unexpected). "
                    + "Resume this run from LangGraph Studio or the SDK.\n\n"
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
    base = (LANGSMITH_RESOURCE_URL or "").strip().rstrip("/")
    api_key = _resolve_langgraph_api_key()
    thread_id = st.session_state.get("workflow_pending_thread_id")
    intr_id = st.session_state.get("workflow_pending_interrupt_id")

    if not base or not api_key:
        return "⚠️ LangGraph is not configured (URL / API key)."
    if not thread_id:
        return "No paused workflow in this session."

    remote = RemoteGraph(LANGGRAPH_GRAPH_ID, url=base, api_key=api_key)
    cmd = _command_for_resume(resume_payload, intr_id)
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        result, _ = _remote_invoke_values(remote, cmd, config=cfg, on_progress=on_progress)
    except RemoteException as e:
        return f"⚠️ LangGraph deployment error: {e}"
    except Exception as e:
        return f"⚠️ LangGraph Error: {type(e).__name__}: {e}"

    if isinstance(result, dict) and result.get("__interrupt__"):
        parsed = _first_interrupt_entry(result)
        if parsed:
            value, new_intr_id = parsed
            kind = value.get("kind") if isinstance(value, dict) else None
            if kind == "outline_review":
                _store_workflow_interrupt(thread_id, new_intr_id or None, value)
                return (
                    "Workflow paused again at outline review. "
                    "Update your choice in the panel below."
                )
            if kind == "configure_review":
                _store_workflow_interrupt(thread_id, new_intr_id or None, value)
                return (
                    "Workflow paused again at configure review. "
                    "Update your choice in the panel below."
                )
            if kind == "revise_review":
                _store_workflow_interrupt(thread_id, new_intr_id or None, value)
                return (
                    "Workflow paused again at revise review. "
                    "Update your choice in the panel below."
                )
        _clear_workflow_pending()
        return _format_remote_graph_result(result)

    _clear_workflow_pending()
    return _format_remote_graph_result(result)


# --------------------------
# Chat UI
# --------------------------
st.title("🤖 LangGraph + Perplexity/Qwen Chat")

if st.session_state.get("workflow_pending_thread_id"):
    ctx = st.session_state.get("workflow_interrupt_context") or {}
    kind = ctx.get("kind")

    if kind == "outline_review":
        with st.container(border=True):
            st.subheader("Outline review")
            st.caption(
                "The deployment is paused at outline confirmation. Accept the outline below or "
                "paste a full replacement; the workflow will continue and the article will appear "
                "in chat when finished. After you continue, **live state** updates appear while the "
                "graph runs."
            )
            topic = ctx.get("topic", "")
            if topic:
                st.markdown(f"**Topic:** {topic}")
            outline = ctx.get("outline", "")
            st.markdown("**Proposed outline**")
            st.text(outline if outline else "(empty)")

            col_a, col_d = st.columns(2)
            with col_a:
                accept = st.button("Accept outline", type="primary", use_container_width=True)
            with col_d:
                discard = st.button("Discard pause", use_container_width=True)

            replacement = st.text_area(
                "Replace with your own outline",
                height=200,
                key="wf_outline_replacement",
                placeholder="Paste a full outline here, then click the button below.",
            )
            replace_go = st.button("Submit replacement outline", use_container_width=True)

            if discard:
                _clear_workflow_pending()
                st.rerun()
            if accept:
                wf_prog = st.empty()
                with st.spinner("Resuming workflow…"):
                    resumed = resume_remote_news_graph(
                        {"action": "accept"},
                        on_progress=wf_prog.markdown,
                    )
                wf_prog.markdown("🔄 **LangGraph** (resumed)\n\n" + resumed)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "🔄 **LangGraph** (resumed)\n\n" + resumed}
                )
                st.rerun()
            if replace_go:
                text = (replacement or "").strip()
                if not text:
                    st.warning("Paste a full outline or use **Accept outline**.")
                else:
                    wf_prog = st.empty()
                    with st.spinner("Resuming workflow…"):
                        resumed = resume_remote_news_graph(
                            {"action": "replace", "outline": text},
                            on_progress=wf_prog.markdown,
                        )
                    wf_prog.markdown("🔄 **LangGraph** (resumed)\n\n" + resumed)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "🔄 **LangGraph** (resumed)\n\n" + resumed}
                    )
                    st.rerun()

    elif kind == "configure_review":
        wf_settings = get_settings()
        with st.container(border=True):
            st.subheader("Configure review")
            st.caption(
                "Confirm the values that will be used for **research** (topic, article type, word "
                "target, LLM provider/model, optional search query). Edit fields if needed, then "
                "continue."
            )
            default_at = ctx.get("article_type") or ARTICLE_TYPES[0]
            at_idx = (
                ARTICLE_TYPES.index(default_at)
                if default_at in ARTICLE_TYPES
                else 0
            )
            default_lp = (ctx.get("llm_provider") or "perplexity").strip().lower()
            lp_idx = 0 if default_lp == "perplexity" else 1

            cfg_topic = st.text_input(
                "Topic",
                value=str(ctx.get("topic") or ""),
                key="wf_cfg_topic",
            )
            cfg_article_type = st.selectbox(
                "Article type",
                ARTICLE_TYPES,
                index=at_idx,
                key="wf_cfg_article_type",
            )
            cfg_twc = st.number_input(
                "Target word count",
                min_value=wf_settings.min_target_word_count,
                max_value=wf_settings.max_target_word_count,
                value=int(ctx.get("target_word_count") or wf_settings.default_target_word_count),
                step=50,
                key="wf_cfg_twc",
            )
            cfg_llm = st.selectbox(
                "LLM provider",
                ["perplexity", "qwen"],
                index=lp_idx,
                key="wf_cfg_llm",
            )
            cfg_model = st.text_input(
                "LLM model (optional, empty = provider default)",
                value=str(ctx.get("llm_model") or ""),
                key="wf_cfg_model",
            )
            cfg_query = st.text_area(
                "Search query (optional; empty lets research auto-build a query)",
                value=str(ctx.get("query") or ""),
                height=100,
                key="wf_cfg_query",
            )

            col_c, col_d2 = st.columns(2)
            with col_c:
                cfg_accept = st.button(
                    "Accept as shown (no edits)",
                    type="secondary",
                    use_container_width=True,
                )
            with col_d2:
                cfg_discard = st.button("Discard pause", use_container_width=True)

            cfg_go = st.button("Continue with settings above", type="primary", use_container_width=True)

            if cfg_discard:
                _clear_workflow_pending()
                st.rerun()
            if cfg_accept:
                wf_prog = st.empty()
                with st.spinner("Resuming workflow…"):
                    resumed = resume_remote_news_graph(
                        {"action": "accept"},
                        on_progress=wf_prog.markdown,
                    )
                wf_prog.markdown("🔄 **LangGraph** (resumed)\n\n" + resumed)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "🔄 **LangGraph** (resumed)\n\n" + resumed}
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
                }
                with st.spinner("Resuming workflow…"):
                    resumed = resume_remote_news_graph(
                        payload,
                        on_progress=wf_prog.markdown,
                    )
                wf_prog.markdown("🔄 **LangGraph** (resumed)\n\n" + resumed)
                st.session_state.messages.append(
                    {"role": "assistant", "content": "🔄 **LangGraph** (resumed)\n\n" + resumed}
                )
                st.rerun()

    elif kind == "revise_review":
        with st.container(border=True):
            st.subheader("Revise review")
            st.caption(
                "The deployment is paused after AI revision. Choose whether to accept this draft, "
                "send feedback for another revise pass, or replace it with your own revised draft."
            )
            topic = ctx.get("topic", "")
            if topic:
                st.markdown(f"**Topic:** {topic}")
            draft = ctx.get("draft", "")
            st.markdown("**Current revised draft**")
            st.text(draft if draft else "(empty)")

            decision = st.radio(
                "What should happen next?",
                ["accept", "feedback", "replace"],
                format_func=lambda v: {
                    "accept": "Accept draft and continue to review",
                    "feedback": "Add feedback and ask AI to revise again",
                    "replace": "Paste my own revised draft and continue to review",
                }[v],
                key="wf_revise_decision",
            )
            feedback_text = st.text_area(
                "Feedback for next revise pass",
                height=120,
                key="wf_revise_feedback",
                placeholder="Tell the AI exactly what to change...",
            )
            revised_draft_text = st.text_area(
                "Your revised draft",
                height=240,
                key="wf_revise_draft",
                placeholder="Paste full revised draft here...",
            )

            col_submit, col_discard = st.columns(2)
            with col_submit:
                submit_revise = st.button(
                    "Submit decision",
                    type="primary",
                    use_container_width=True,
                )
            with col_discard:
                discard_revise = st.button("Discard pause", use_container_width=True)

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
                    st.warning(err or "Invalid revise decision.")
                else:
                    wf_prog = st.empty()
                    with st.spinner("Resuming workflow…"):
                        resumed = resume_remote_news_graph(
                            payload,
                            on_progress=wf_prog.markdown,
                        )
                    wf_prog.markdown("🔄 **LangGraph** (resumed)\n\n" + resumed)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "🔄 **LangGraph** (resumed)\n\n" + resumed}
                    )
                    st.rerun()

    else:
        with st.container(border=True):
            st.subheader("Workflow paused")
            st.caption(
                "Unknown or missing interrupt kind in this session. You can discard the pause or "
                "resume from LangGraph Studio."
            )
            if st.button("Discard pause", key="wf_unknown_discard"):
                _clear_workflow_pending()
                st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Type 'run workflow' to use LangGraph."}
    ]

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------
# Chat Input
# --------------------------
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        use_workflow = should_run_langgraph(prompt)
        progress_slot = st.empty()
        with st.spinner("Running LangGraph…" if use_workflow else "Thinking…"):
            if use_workflow:
                resp = call_remote_news_graph(
                    prompt,
                    on_progress=progress_slot.markdown,
                )
            else:
                resp = call_general_llm(prompt)
        if use_workflow:
            progress_slot.markdown("🔄 **LangGraph Activated**\n\n" + resp)
        else:
            progress_slot.markdown(resp)

    st.session_state.messages.append({"role": "assistant", "content": resp})
