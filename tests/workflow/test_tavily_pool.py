"""Tests for Tavily evidence pool helpers."""

from litenews.workflow.tavily_pool import (
    filter_blocked_tavily_rows,
    merge_into_pool,
    normalize_raw_tavily_result,
    select_evidence_for_claim,
)


class TestFilterBlockedTavilyRows:
    def test_empty_blocklist_returns_all(self):
        rows = [{"title": "a", "url": "https://evil.com/x", "content": "c"}]
        assert filter_blocked_tavily_rows(rows, []) == rows
        assert filter_blocked_tavily_rows(rows, None) == rows

    def test_blocks_exact_host(self):
        rows = [
            {"title": "bad", "url": "https://spam.test/p", "content": "x"},
            {"title": "good", "url": "https://ok.test/q", "content": "y"},
        ]
        out = filter_blocked_tavily_rows(rows, ["spam.test"])
        assert len(out) == 1
        assert out[0]["url"] == "https://ok.test/q"

    def test_blocks_subdomain_and_www(self):
        rows = [
            {"title": "w", "url": "https://www.spam.test/a", "content": "1"},
            {"title": "s", "url": "https://sub.spam.test/b", "content": "2"},
            {"title": "g", "url": "https://other.test/c", "content": "3"},
        ]
        out = filter_blocked_tavily_rows(rows, ["spam.test"])
        urls = {r["url"] for r in out}
        assert "https://other.test/c" in urls
        assert len(out) == 1

    def test_keeps_rows_without_url(self):
        rows = [{"title": "x", "url": "", "content": "z"}]
        assert filter_blocked_tavily_rows(rows, ["anything.com"]) == rows

    def test_non_dict_rows_passthrough(self):
        rows = ["not-a-dict", {"url": "https://a.com", "title": "t", "content": "c"}]
        out = filter_blocked_tavily_rows(rows, ["a.com"])
        assert out[0] == "not-a-dict"
        assert len(out) == 1


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
