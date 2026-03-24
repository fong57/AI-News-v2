"""Tests for outline human-in-the-loop resume parsing."""

import pytest

from litenews.workflow.nodes.outline_human import _outline_human_update


class TestOutlineHumanUpdate:
    @pytest.mark.parametrize(
        "resume",
        [
            {"action": "accept"},
            {"action": "ACCEPT"},
            {"confirmed": True},
        ],
    )
    def test_accept_dict(self, resume):
        out = _outline_human_update(resume)
        assert out == {"status": "outline_confirmed"}
        assert "error" not in out

    @pytest.mark.parametrize("text", ["accept", "Accept", "OK", "yes", "confirm"])
    def test_accept_string(self, text):
        out = _outline_human_update(text)
        assert out == {"status": "outline_confirmed"}

    def test_replace_string(self):
        out = _outline_human_update("  My custom outline\n")
        assert out == {"outline": "My custom outline", "status": "outline_confirmed"}

    def test_replace_dict(self):
        out = _outline_human_update(
            {"action": "replace", "outline": "  Section A\nSection B  "}
        )
        assert out == {
            "outline": "Section A\nSection B",
            "status": "outline_confirmed",
        }

    def test_replace_dict_shortcut_outline_only(self):
        out = _outline_human_update({"outline": "Only key"})
        assert out == {"outline": "Only key", "status": "outline_confirmed"}

    def test_errors(self):
        assert _outline_human_update(None)["status"] == "error"
        assert _outline_human_update("")["status"] == "error"
        assert _outline_human_update({"action": "replace", "outline": ""})[
            "status"
        ] == "error"
        assert _outline_human_update({})["status"] == "error"
        assert _outline_human_update(123)["status"] == "error"
