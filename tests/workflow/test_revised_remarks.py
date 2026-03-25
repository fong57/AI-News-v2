"""Tests for filtered 【事實查核備註】 blocks after revise."""

import pytest

from litenews.workflow.nodes.revise import (
    _build_unresolved_remarks_block,
    _strip_trailing_remarks_block,
    fact_check_remarks_node,
)

_SAMPLE_FR = {
    "claims": [
        {
            "importance": 4,
            "status": "contradicted",
            "text": "香港人口一千萬",
            "reason": "與統計不符",
        },
        {
            "importance": 3,
            "status": "uncertain",
            "text": "明日會下雨",
            "reason": "",
        },
    ]
}


def test_build_remarks_without_article_body_lists_all():
    block = _build_unresolved_remarks_block(_SAMPLE_FR)
    assert "香港人口一千萬" in block
    assert "明日會下雨" in block


def test_build_remarks_filters_by_article_body():
    body_keeps_first = "報導指香港人口一千萬，其餘從略。"
    block = _build_unresolved_remarks_block(_SAMPLE_FR, article_body=body_keeps_first)
    assert "香港人口一千萬" in block
    assert "明日會下雨" not in block

    body_neither = "全文與上述宣稱無關。"
    empty = _build_unresolved_remarks_block(_SAMPLE_FR, article_body=body_neither)
    assert empty == ""


def test_build_remarks_whitespace_normalized_match():
    claim_text = "去年\nGDP\n增長5%"
    fr = {
        "claims": [
            {"importance": 4, "status": "contradicted", "text": claim_text, "reason": "x"},
        ]
    }
    body = "去年 GDP 增長5% 為市場預期。"
    block = _build_unresolved_remarks_block(fr, article_body=body)
    assert "GDP" in block


def test_strip_trailing_remarks_block():
    body = "標題\n\n內文段落。"
    tail = "\n\n---\n【事實查核備註】\n以下要點經自動"
    assert _strip_trailing_remarks_block(body + tail) == body

    alt = "內文\n---\n【事實查核備註】\n行"
    assert _strip_trailing_remarks_block(alt) == "內文"

    assert _strip_trailing_remarks_block("無備註") == "無備註"


@pytest.mark.asyncio
async def test_fact_check_remarks_node_replaces_stale_longer_block():
    """Stale draft may end with an old remark block (e.g. extra bullets); node rebuilds from fr + body."""
    fr_current = {
        "claims": [
            {"importance": 4, "status": "contradicted", "text": "保留句", "reason": "r"},
        ]
    }
    fr_stale_longer = {
        "claims": [
            {"importance": 4, "status": "contradicted", "text": "保留句", "reason": "r"},
            {"importance": 4, "status": "uncertain", "text": "已不在正文", "reason": "x"},
        ]
    }
    article = "本文保留句。"
    stale_block = _build_unresolved_remarks_block(fr_stale_longer, article_body=None)
    draft_with_stale = article + stale_block
    expected_block = _build_unresolved_remarks_block(fr_current, article_body=article)
    assert len(stale_block) > len(expected_block)

    out = await fact_check_remarks_node(
        {"draft": draft_with_stale, "fact_check_results": fr_current}
    )
    assert out["status"] == "fact_check_remarks_appended"
    assert out["draft"] == article + expected_block
    assert "已不在正文" not in out["draft"]
    assert out["draft"].count("【事實查核備註】") == 1


@pytest.mark.asyncio
async def test_fact_check_remarks_node_skip_when_already_combined():
    fr = {
        "claims": [
            {"importance": 4, "status": "contradicted", "text": "X", "reason": ""},
        ]
    }
    article = "前文X後文"
    block = _build_unresolved_remarks_block(fr, article_body=article)
    combined = article + block
    out = await fact_check_remarks_node({"draft": combined, "fact_check_results": fr})
    assert out["status"] == "fact_check_remarks_skipped"


@pytest.mark.asyncio
async def test_fact_check_remarks_node_removes_stale_when_all_resolved():
    fr = {
        "claims": [
            {"importance": 4, "status": "contradicted", "text": "已刪宣稱", "reason": "r"},
        ]
    }
    stale = "全新正文無該句。\n\n---\n【事實查核備註】\n以下要點"
    out = await fact_check_remarks_node({"draft": stale, "fact_check_results": fr})
    assert out["status"] == "fact_check_remarks_skipped"
    assert out["draft"] == "全新正文無該句。"
