"""Shared Tavily result normalization, pool merge, and claim-local retrieval."""

from __future__ import annotations

import hashlib
import re
from typing import Any


def normalize_raw_tavily_result(r: Any) -> dict[str, Any]:
    """Map a Tavily API / tool result dict to fact-check evidence shape."""
    if not isinstance(r, dict):
        return {"title": "", "snippet": "", "url": ""}
    snippet = r.get("content", r.get("snippet", ""))
    return {
        "title": str(r.get("title") or ""),
        "url": str(r.get("url") or "").strip(),
        "snippet": str(snippet)[:300] if snippet else "",
    }


def _no_url_fingerprint(title: str, snippet: str) -> str:
    return hashlib.sha256(f"{title}|{snippet}".encode("utf-8")).hexdigest()


def merge_into_pool(
    pool: list[dict[str, Any]] | None,
    new_rows: list[Any],
) -> list[dict[str, Any]]:
    """Append normalized rows; dedupe by URL, or by title+snippet hash when URL is empty."""
    out: list[dict[str, Any]] = [dict(r) for r in (pool or [])]
    seen_url: set[str] = set()
    seen_no_url: set[str] = set()
    for row in out:
        url = str(row.get("url") or "").strip()
        if url:
            seen_url.add(url)
        else:
            seen_no_url.add(
                _no_url_fingerprint(
                    str(row.get("title") or ""),
                    str(row.get("snippet") or ""),
                )
            )

    for raw in new_rows:
        norm = normalize_raw_tavily_result(raw)
        url = norm["url"]
        if url:
            if url in seen_url:
                continue
            seen_url.add(url)
            out.append(dict(norm))
        else:
            fp = _no_url_fingerprint(norm["title"], norm["snippet"])
            if fp in seen_no_url:
                continue
            seen_no_url.add(fp)
            out.append(dict(norm))
    return out


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _overlap_score(claim: str, title: str, snippet: str) -> int:
    """Cheap relevance: count of 2-character substrings from claim found in title+snippet."""
    c = _compact(claim)
    hay = _compact(title) + _compact(snippet)
    if not c or not hay:
        return 0
    if len(c) == 1:
        return hay.count(c)
    score = 0
    for i in range(len(c) - 1):
        if c[i : i + 2] in hay:
            score += 1
    return score


def select_evidence_for_claim(
    claim_text: str,
    pool: list[dict[str, Any]] | None,
    *,
    max_items: int = 10,
    max_total_chars: int = 4000,
) -> list[dict[str, Any]]:
    """Rank pool rows by bigram overlap with claim; return top rows within char budget."""
    if not pool or not (claim_text or "").strip():
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for row in pool:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "")
        url = str(row.get("url") or "")
        snippet = str(row.get("snippet") or "")
        sc = _overlap_score(claim_text, title, snippet)
        if sc > 0:
            scored.append((sc, dict(row)))

    scored.sort(key=lambda x: -x[0])
    picked: list[dict[str, Any]] = []
    total = 0
    for _, row in scored[: max_items * 3]:
        block = f"{row.get('title', '')}{row.get('snippet', '')}"
        add_len = len(block)
        if picked and total + add_len > max_total_chars:
            continue
        picked.append(row)
        total += add_len
        if len(picked) >= max_items:
            break

    if picked:
        return picked

    for row in pool:
        if isinstance(row, dict):
            picked.append(dict(row))
            if len(picked) >= max_items:
                break
    return picked
