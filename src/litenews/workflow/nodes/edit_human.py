"""Human-in-the-loop full draft entry for task=edit (skip research, outline, write).

Used when the user chose ``task: edit`` at configure review: after confirm, the graph
routes here instead of research. The pasted draft becomes ``state['draft']`` and flows
straight into ``fact_check``.
"""

from typing import Any

from langgraph.types import interrupt

from litenews.state.news_state import NewsState
from litenews.workflow.utils import create_error_response

_ACCEPT_TOKENS = frozenset({"accept", "ok", "yes", "y", "confirm", "confirmed"})


def _edit_human_update(resume: Any) -> dict:
    """Turn interrupt resume into draft + status or error."""
    if resume is None:
        return create_error_response(
            "Edit draft: missing resume value. "
            "Use {'action': 'submit', 'draft': '<full article>'} or paste the full draft as a string."
        )
    if isinstance(resume, str):
        s = resume.strip()
        if not s:
            return create_error_response("Edit draft: empty draft text")
        if s.lower() in _ACCEPT_TOKENS:
            return create_error_response(
                "Edit draft: accept tokens are not valid here — paste your full draft text."
            )
        return {"draft": s, "status": "edit_draft_submitted"}
    if not isinstance(resume, dict):
        return create_error_response(
            f"Edit draft: unsupported resume type {type(resume).__name__}"
        )
    action = str(resume.get("action", "")).strip().lower()
    if action == "submit" or action == "replace":
        new_draft = resume.get("draft")
        if isinstance(new_draft, str) and new_draft.strip():
            return {"draft": new_draft.strip(), "status": "edit_draft_submitted"}
        return create_error_response(
            "Edit draft: 'submit' / 'replace' requires non-empty 'draft' string"
        )
    new_draft = resume.get("draft")
    if isinstance(new_draft, str) and new_draft.strip():
        return {"draft": new_draft.strip(), "status": "edit_draft_submitted"}
    return create_error_response(
        "Edit draft: expected {'action': 'submit', 'draft': '<full text>'} or {'draft': '...'}"
    )


def edit_human_node(state: NewsState) -> dict:
    """Pause for a complete human-written draft, then continue to fact_check."""
    payload = {
        "kind": "edit_draft_review",
        "topic": state.get("topic", ""),
        "article_type": state.get("article_type", ""),
        "resume_help": (
            "Submit your complete draft as one string. "
            "Resume with {'action': 'submit', 'draft': '<full article>'} or paste only the draft text."
        ),
    }
    resume = interrupt(payload)
    upd = _edit_human_update(resume)
    if upd.get("error") or upd.get("status") == "error":
        return upd

    reset_fc: dict[str, Any] = {
        "fact_check_revision_round": 0,
        "fact_check_results": {},
        "fact_check_evidence_cache": {},
        "fact_check_score": 0.0,
        "last_fact_checked_draft": "",
        "tavily_evidence_pool": [],
    }
    return {**reset_fc, **upd}
