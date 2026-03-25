"""Tests for edit_human resume parsing (edit task → draft → fact_check)."""

import pytest

from litenews.workflow.nodes.edit_human import _edit_human_update


class TestEditHumanUpdate:
    def test_submit_string(self):
        out = _edit_human_update("  Full draft text\n")
        assert out == {"draft": "Full draft text", "status": "edit_draft_submitted"}

    def test_submit_dict(self):
        out = _edit_human_update({"action": "submit", "draft": "  x  "})
        assert out == {"draft": "x", "status": "edit_draft_submitted"}

    def test_replace_dict(self):
        out = _edit_human_update({"action": "replace", "draft": "y"})
        assert out == {"draft": "y", "status": "edit_draft_submitted"}

    def test_draft_only_dict(self):
        out = _edit_human_update({"draft": "only"})
        assert out == {"draft": "only", "status": "edit_draft_submitted"}

    def test_errors(self):
        assert _edit_human_update(None)["status"] == "error"
        assert _edit_human_update("")["status"] == "error"
        assert _edit_human_update("accept")["status"] == "error"
        assert _edit_human_update({"action": "submit", "draft": ""})["status"] == "error"
        assert _edit_human_update({})["status"] == "error"
        assert _edit_human_update(123)["status"] == "error"
