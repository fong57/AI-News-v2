"""Human-in-the-loop outline confirmation.

Pauses after AI outline generation so a human can accept it or supply a replacement.
Interrupts require a checkpointer: LangGraph API / ``langgraph dev`` attach one
automatically; for raw ``.invoke`` / ``.stream`` outside the API, pass
``checkpointer=MemorySaver()`` into ``create_news_graph``.
"""

from typing import Any

from langgraph.types import interrupt

from litenews.state.news_state import NewsState
from litenews.workflow.utils import create_error_response

_ACCEPT_TOKENS = frozenset({"accept", "ok", "yes", "y", "confirm", "confirmed"})


def _outline_human_update(resume: Any) -> dict:
    """Turn interrupt resume payload into a state update or error response."""
    if resume is None:
        return create_error_response(
            "Outline review: missing resume value. "
            "Use {'action': 'accept'} to keep the AI outline, or "
            "{'action': 'replace', 'outline': '...'} or paste your outline as a string."
        )
    if isinstance(resume, str):
        s = resume.strip()
        if s.lower() in _ACCEPT_TOKENS:
            return {"status": "outline_confirmed"}
        if not s:
            return create_error_response("Outline review: empty outline text")
        return {"outline": s, "status": "outline_confirmed"}
    if isinstance(resume, dict):
        action = str(resume.get("action", "")).strip().lower()
        if action == "accept" or resume.get("confirmed") is True:
            return {"status": "outline_confirmed"}
        if action == "replace":
            new_outline = resume.get("outline")
            if isinstance(new_outline, str) and new_outline.strip():
                return {"outline": new_outline.strip(), "status": "outline_confirmed"}
            return create_error_response(
                "Outline review: 'replace' requires non-empty 'outline' string"
            )
        new_outline = resume.get("outline")
        if isinstance(new_outline, str) and new_outline.strip():
            return {"outline": new_outline.strip(), "status": "outline_confirmed"}
        return create_error_response(
            "Outline review: expected {'action': 'accept'} or "
            "{'action': 'replace', 'outline': '...'}"
        )
    return create_error_response(
        f"Outline review: unsupported resume type {type(resume).__name__}"
    )


def outline_human_node(state: NewsState) -> dict:
    """Pause for human approval or replacement of the AI-generated outline."""
    current = (state.get("outline") or "").strip()
    if not current:
        return create_error_response("Outline review: no outline to confirm")

    payload = {
        "kind": "outline_review",
        "topic": state.get("topic", ""),
        "outline": state.get("outline", ""),
        "resume_help": (
            "To keep this outline: resume with {'action': 'accept'} or the string 'accept'. "
            "To replace: resume with {'action': 'replace', 'outline': '<your full outline>'} "
            "or paste the full outline as a plain string."
        ),
    }
    resume = interrupt(payload)
    return _outline_human_update(resume)
