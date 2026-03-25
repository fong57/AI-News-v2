"""Human-in-the-loop review after revise.

Pauses after each AI revision so a human can:
1) accept current draft and continue to review,
2) provide feedback and send back to revise, or
3) paste a fully revised draft and continue to review.
"""

from typing import Any, Literal

from langgraph.types import interrupt

from litenews.state.news_state import NewsState
from litenews.workflow.utils import create_error_response

_ACCEPT_TOKENS = frozenset({"accept", "ok", "yes", "y", "confirm", "confirmed"})


def _revise_human_update(resume: Any) -> dict:
    """Turn interrupt resume payload into a state update or error response."""
    if resume is None:
        return create_error_response(
            "Revise review: missing resume value. "
            "Use {'action': 'accept'} to continue, "
            "{'action': 'feedback', 'feedback': '...'} to request another revise pass, "
            "or {'action': 'replace', 'draft': '...'} to use your own revised draft."
        )

    if isinstance(resume, str):
        s = resume.strip()
        if not s:
            return create_error_response("Revise review: empty input")
        if s.lower() in _ACCEPT_TOKENS:
            return {"feedback": "", "status": "revise_human_accepted"}
        # Plain string is treated as pasted revised draft.
        return {"draft": s, "feedback": "", "status": "revise_human_replaced"}

    if not isinstance(resume, dict):
        return create_error_response(
            f"Revise review: unsupported resume type {type(resume).__name__}"
        )

    action = str(resume.get("action", "")).strip().lower()
    if action == "accept" or resume.get("confirmed") is True:
        return {"feedback": "", "status": "revise_human_accepted"}

    if action == "feedback" or (
        "feedback" in resume and action not in {"accept", "replace"}
    ):
        fb = resume.get("feedback")
        if isinstance(fb, str) and fb.strip():
            return {"feedback": fb.strip(), "status": "revise_human_feedback"}
        return create_error_response(
            "Revise review: 'feedback' action requires non-empty 'feedback' string"
        )

    if action == "replace" or (
        "draft" in resume and action not in {"accept", "feedback"}
    ):
        new_draft = resume.get("draft")
        if isinstance(new_draft, str) and new_draft.strip():
            return {
                "draft": new_draft.strip(),
                "feedback": "",
                "status": "revise_human_replaced",
            }
        return create_error_response(
            "Revise review: 'replace' action requires non-empty 'draft' string"
        )

    return create_error_response(
        "Revise review: expected {'action': 'accept'} or "
        "{'action': 'feedback', 'feedback': '...'} or "
        "{'action': 'replace', 'draft': '...'}"
    )


def revise_human_node(state: NewsState) -> dict:
    """Pause for human decision after revise output."""
    current = (state.get("draft") or "").strip()
    if not current:
        return create_error_response("Revise review: no draft to review")

    payload = {
        "kind": "revise_review",
        "topic": state.get("topic", ""),
        "draft": state.get("draft", ""),
        "resume_help": (
            "To accept this draft and continue: {'action': 'accept'} or 'accept'. "
            "To ask AI to revise again: {'action': 'feedback', 'feedback': '<your notes>'}. "
            "To replace with your own revised draft: {'action': 'replace', 'draft': '<full draft>'} "
            "or paste your full draft as a plain string."
        ),
    }
    resume = interrupt(payload)
    return _revise_human_update(resume)


def route_after_revise_human(state: NewsState) -> Literal["error", "revise", "review"]:
    """Route based on human decision after revise."""
    if state.get("error"):
        return "error"
    if state.get("status") == "revise_human_feedback":
        return "revise"
    return "review"
