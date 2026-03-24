"""Tests for Tavily evidence pool helpers."""

from litenews.workflow.tavily_pool import (
    merge_into_pool,
    normalize_raw_tavily_result,
    select_evidence_for_claim,
)


class TestNormalizeRawTavilyResult:
    def test_maps_content_and_snippet(self):
        r = {"title": "T", "url": "https://x.com", "content": "body text"}
        n = normalize_raw_tavily_result(r)
        assert n["title"] == "T"
        assert n["url"] == "https://x.com"
        assert n["snippet"] == "body text"[:300]

    def test_non_dict(self):
        assert normalize_raw_tavily_result(None)["url"] == ""


class TestMergeIntoPool:
    def test_dedupes_by_url(self):
        a = [{"title": "1", "url": "https://u", "content": "c"}]
        b = [{"title": "dup", "url": "https://u", "snippet": "x"}]
        out = merge_into_pool(a, b)
        assert len(out) == 1
        assert out[0]["title"] == "1"

    def test_appends_new_urls(self):
        p = merge_into_pool(
            None,
            [{"title": "a", "url": "https://a", "content": "x"}],
        )
        out = merge_into_pool(
            p,
            [{"title": "b", "url": "https://b", "snippet": "y"}],
        )
        assert len(out) == 2

    def test_empty_url_dedupes_by_fingerprint(self):
        r1 = {"title": "same", "url": "", "snippet": "snip"}
        r2 = {"title": "same", "url": "", "snippet": "snip"}
        out = merge_into_pool(None, [r1, r2])
        assert len(out) == 1


class TestSelectEvidenceForClaim:
    def test_prefers_overlapping_snippet(self):
        pool = [
            {"title": "other", "url": "https://a", "snippet": "unrelated text"},
            {"title": "gdp", "url": "https://b", "snippet": "台灣經濟成長數據"},
        ]
        picked = select_evidence_for_claim(
            "台灣經濟成長",
            pool,
            max_items=2,
        )
        assert len(picked) >= 1
        assert any("台灣" in p.get("snippet", "") for p in picked)

    def test_empty_pool(self):
        assert select_evidence_for_claim("claim", []) == []

    def test_fallback_when_no_overlap(self):
        pool = [{"title": "z", "url": "https://z", "snippet": "abc"}]
        picked = select_evidence_for_claim("qqqq", pool, max_items=3)
        assert len(picked) == 1
