"""Line-based diff helpers for incremental fact-check after revise."""

from difflib import SequenceMatcher


def split_lines(text: str) -> list[str]:
    return text.split("\n")


def line_start_offsets(lines: list[str]) -> list[int]:
    o = 0
    offsets: list[int] = []
    for i, line in enumerate(lines):
        offsets.append(o)
        o += len(line)
        if i < len(lines) - 1:
            o += 1
    return offsets


def stable_new_line_indices(old_text: str, new_text: str) -> frozenset[int]:
    old_lines = split_lines(old_text)
    new_lines = split_lines(new_text)
    sm = SequenceMatcher(None, old_lines, new_lines)
    stable: set[int] = set()
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for j in range(j1, j2):
                stable.add(j)
    return frozenset(stable)


def touched_line_indices(stable: frozenset[int], n_new: int) -> set[int]:
    return {j for j in range(n_new) if j not in stable}


def expanded_touched_lines(touched: set[int], n_lines: int, context: int) -> set[int]:
    out: set[int] = set()
    for t in touched:
        lo = max(0, t - context)
        hi = min(n_lines, t + context + 1)
        for j in range(lo, hi):
            out.add(j)
    return out


def _contiguous_ranges(indices: set[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    sorted_i = sorted(indices)
    ranges: list[tuple[int, int]] = []
    start = sorted_i[0]
    prev = sorted_i[0]
    for x in sorted_i[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))
    return ranges


def _char_range_for_lines(lines: list[str], lo: int, hi: int) -> tuple[int, int]:
    offsets = line_start_offsets(lines)
    start = offsets[lo]
    end = offsets[hi] + len(lines[hi])
    return start, end


def build_focus_excerpt(new_draft: str, focus_lines: frozenset[int], max_chars: int) -> str:
    if not focus_lines:
        return ""
    lines = split_lines(new_draft)
    if not lines:
        return ""
    ranges = _contiguous_ranges(set(focus_lines))
    parts: list[str] = []
    total = 0
    for lo, hi in ranges:
        lo = max(0, min(lo, len(lines) - 1))
        hi = max(0, min(hi, len(lines) - 1))
        if lo > hi:
            continue
        a, b = _char_range_for_lines(lines, lo, hi)
        chunk = new_draft[a:b]
        if total + len(chunk) > max_chars:
            remain = max(0, max_chars - total)
            if remain > 0:
                parts.append(chunk[:remain])
            break
        parts.append(chunk)
        total += len(chunk)
        if total >= max_chars:
            break
    return "\n---\n".join(parts)


def normalize_claim_key(text: str) -> str:
    return " ".join(str(text or "").split())


def lines_spanned_by_char_range(
    start: int, end: int, offsets: list[int], lines: list[str]
) -> frozenset[int]:
    """Line indices whose character span intersects [start, end)."""
    out: set[int] = set()
    for i, off in enumerate(offsets):
        line_end = off + len(lines[i])
        if line_end > start and off < end:
            out.add(i)
    return frozenset(out)


def claim_entirely_in_stable_lines(
    claim_text: str, draft: str, stable_lines: frozenset[int]
) -> bool:
    t = str(claim_text or "").strip()
    if not t:
        return False
    pos = draft.find(t)
    if pos < 0:
        return False
    lines = split_lines(draft)
    if not lines:
        return False
    offsets = line_start_offsets(lines)
    end = pos + len(t)
    spanned = lines_spanned_by_char_range(pos, end, offsets, lines)
    if not spanned:
        return False
    return spanned <= stable_lines


def compute_incremental_focus(
    last_checked: str,
    new_draft: str,
    *,
    context_lines: int = 4,
    max_chars: int = 2000,
    max_changed_ratio: float = 0.5,
) -> tuple[bool, str]:
    """Return (use_focus_excerpt, excerpt). If first is False, run full-article extract.

    excerpt is built from changed lines in new_draft (plus context), capped at max_chars.
    """
    if not (last_checked or "").strip() or last_checked == new_draft:
        return False, ""
    new_lines = split_lines(new_draft)
    n = len(new_lines)
    if n == 0:
        return False, ""
    stable = stable_new_line_indices(last_checked, new_draft)
    touched = touched_line_indices(stable, n)
    if not touched:
        return False, ""
    if (len(touched) / n) > max_changed_ratio:
        return False, ""
    expanded = expanded_touched_lines(touched, n, context_lines)
    excerpt = build_focus_excerpt(new_draft, frozenset(expanded), max_chars)
    if not excerpt.strip():
        return False, ""
    return True, excerpt
