"""Tests for fact-check → revise routing and remarks."""

from litenews.state.news_state import NewsState
from litenews.workflow.nodes.fact_check import (
    has_actionable_fact_check_issues,
    route_after_fact_check,
)


def _state(**kwargs: object) -> NewsState:
    base: NewsState = NewsState(
        topic="t",
        article_type="其他",
        query="",
        messages=[],
        search_results=[],
        sources=[],
        research_notes="",
        outline="",
        draft="x",
        final_article=None,
        feedback="",
        status="",
        error="",
        fact_check_revision_round=0,
    )
    for k, v in kwargs.items():
        base[k] = v  # type: ignore[literal-required]
    return base


class TestHasActionableFactCheckIssues:
    def test_skipped_false(self):
        s = _state(
            fact_check_results={"skipped": True, "claims": []},
        )
        assert has_actionable_fact_check_issues(s) is False

    def test_supported_only_false(self):
        s = _state(
            fact_check_results={
                "claims": [
                    {"importance": 4, "status": "supported", "text": "a"},
                ]
            },
        )
        assert has_actionable_fact_check_issues(s) is False

    def test_low_importance_contradicted_ignored(self):
        s = _state(
            fact_check_results={
                "claims": [
                    {"importance": 2, "status": "contradicted", "text": "a"},
                ]
            },
        )
        assert has_actionable_fact_check_issues(s) is False

    def test_important_uncertain_true(self):
        s = _state(
            fact_check_results={
                "claims": [
                    {"importance": 3, "status": "uncertain", "text": "a"},
                ]
            },
        )
        assert has_actionable_fact_check_issues(s) is True


class TestRouteAfterFactCheck:
    def test_error_stops(self):
        s = _state(error="x", fact_check_revision_round=0)
        assert route_after_fact_check(s) == "error"

    def test_clean_goes_review(self):
        s = _state(
            fact_check_results={"claims": []},
            fact_check_revision_round=0,
        )
        assert route_after_fact_check(s) == "review"

    def test_issues_under_cap_goes_revise(self):
        s = _state(
            fact_check_results={
                "claims": [
                    {"importance": 4, "status": "contradicted", "text": "a"},
                ]
            },
            fact_check_revision_round=0,
        )
        assert route_after_fact_check(s) == "revise"

    def test_issues_at_cap_goes_remarks(self):
        s = _state(
            fact_check_results={
                "claims": [
                    {"importance": 4, "status": "uncertain", "text": "a"},
                ]
            },
            fact_check_revision_round=5,
        )
        assert route_after_fact_check(s) == "fact_check_remarks"

    def test_issues_round_four_still_revise(self):
        s = _state(
            fact_check_results={
                "claims": [
                    {"importance": 4, "status": "uncertain", "text": "a"},
                ]
            },
            fact_check_revision_round=4,
        )
        assert route_after_fact_check(s) == "revise"
