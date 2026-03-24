"""Unit tests for incremental fact-check diff and carry-forward (no live APIs)."""

from litenews.workflow.fact_check_diff import (
    build_focus_excerpt,
    claim_entirely_in_stable_lines,
    compute_incremental_focus,
    expanded_touched_lines,
    normalize_claim_key,
    split_lines,
    stable_new_line_indices,
    touched_line_indices,
)
from litenews.workflow.nodes.fact_check import (
    _carry_forward_stable_claims,
    _score_from_claim_rows,
)


class TestStableLines:
    def test_one_line_changed_middle(self):
        d0 = "A\nB\nC\nD"
        d1 = "A\nB fixed\nC\nD"
        stable = stable_new_line_indices(d0, d1)
        assert stable == frozenset({0, 2, 3})

    def test_touched_indices(self):
        n = 4
        stable = frozenset({0, 2, 3})
        touched = touched_line_indices(stable, n)
        assert touched == {1}


class TestExpandedTouchedAndExcerpt:
    def test_context_expands(self):
        touched = {2}
        expanded = expanded_touched_lines(touched, n_lines=10, context=2)
        assert expanded == {0, 1, 2, 3, 4}

    def test_focus_excerpt_merges_ranges(self):
        draft = "L0\nL1\nL2\nL3"
        excerpt = build_focus_excerpt(draft, frozenset({0, 3}), max_chars=1000)
        assert "L0" in excerpt and "L3" in excerpt
        assert "---" in excerpt


class TestComputeIncrementalFocus:
    def test_no_last_snapshot(self):
        ok, ex = compute_incremental_focus("", "hello\nworld")
        assert ok is False and ex == ""

    def test_identical_drafts(self):
        d = "a\nb\nc"
        ok, ex = compute_incremental_focus(d, d)
        assert ok is False

    def test_small_change_uses_focus(self):
        d0 = "intro\nstable block\nfooter"
        d1 = "intro\nstable block edited\nfooter"
        ok, ex = compute_incremental_focus(d0, d1, context_lines=1, max_chars=500)
        assert ok is True
        assert "edited" in ex

    def test_massive_rewrite_falls_back(self):
        d0 = "\n".join(f"line{i}" for i in range(20))
        d1 = "\n".join(f"other{i}" for i in range(20))
        ok, _ex = compute_incremental_focus(
            d0, d1, context_lines=0, max_chars=5000, max_changed_ratio=0.5
        )
        assert ok is False


class TestNormalizeClaimKey:
    def test_collapses_whitespace(self):
        assert normalize_claim_key("  a   b  ") == "a b"


class TestClaimEntirelyInStableLines:
    def test_stable_claim_accepted(self):
        d0 = "keep this fact\nchange me"
        d1 = "keep this fact\nchanged line"
        stable = stable_new_line_indices(d0, d1)
        assert claim_entirely_in_stable_lines("keep this fact", d1, stable)

    def test_claim_on_changed_line_rejected(self):
        d0 = "A\nB"
        d1 = "A\nB revised"
        stable = stable_new_line_indices(d0, d1)
        assert not claim_entirely_in_stable_lines("B revised", d1, stable)


class TestCarryForwardAndScore:
    def test_carry_forward_preserves_row(self):
        d0 = "same\nedit"
        d1 = "same\nedited"
        stable = stable_new_line_indices(d0, d1)
        prior = [
            {
                "id": "c1",
                "text": "same",
                "importance": 4,
                "status": "supported",
                "reason": "ok",
                "evidence_snippets": [],
            }
        ]
        carried, keys = _carry_forward_stable_claims(prior, d1, stable)
        assert len(carried) == 1
        assert carried[0]["status"] == "supported"
        assert "same" in keys or normalize_claim_key("same") in keys

    def test_score_all_supported(self):
        rows = [
            {"importance": 4, "status": "supported"},
            {"importance": 2, "status": "uncertain"},
        ]
        assert _score_from_claim_rows(rows) == 1.0

    def test_score_mixed(self):
        rows = [
            {"importance": 4, "status": "supported"},
            {"importance": 4, "status": "contradicted"},
        ]
        assert _score_from_claim_rows(rows) == 0.5


class TestSplitLinesEmpty:
    def test_empty_string(self):
        assert split_lines("") == [""]
