"""Deterministic Markdown-safe chunking for verified answer replay."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

_FENCE_OPEN = re.compile(r"(?m)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)")
_TABLE_SEPARATOR = re.compile(
    r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$"
)
_HTML_OPEN = re.compile(r"<(details|div|pre|table|math)(?:\s|>)", re.IGNORECASE)


def _is_escaped(content: str, index: int) -> bool:
    slashes = 0
    index -= 1
    while index >= 0 and content[index] == "\\":
        slashes += 1
        index -= 1
    return slashes % 2 == 1


def _line_end(content: str, index: int) -> int:
    newline = content.find("\n", index)
    return len(content) if newline < 0 else newline + 1


def _fence_end(content: str, index: int) -> int | None:
    match = _FENCE_OPEN.match(content, index)
    if match is None:
        return None
    marker = match.group(1)
    close = re.compile(rf"(?m)^[ \t]{{0,3}}{re.escape(marker)}[ \t]*(?:\n|$)")
    closing = close.search(content, match.end())
    return len(content) if closing is None else closing.end()


def _table_end(content: str, index: int) -> int | None:
    first_end = _line_end(content, index)
    first = content[index:first_end].rstrip("\n")
    if "|" not in first or first_end >= len(content):
        return None
    second_end = _line_end(content, first_end)
    second = content[first_end:second_end].rstrip("\n")
    if _TABLE_SEPARATOR.fullmatch(second) is None:
        return None
    end = second_end
    while end < len(content):
        candidate_end = _line_end(content, end)
        if "|" not in content[end:candidate_end]:
            break
        end = candidate_end
    return end


def _markdown_link_end(content: str, index: int) -> int | None:
    bracket_start = index + 1 if content.startswith("![", index) else index
    if bracket_start >= len(content) or content[bracket_start] != "[":
        return None
    bracket_end = bracket_start + 1
    while bracket_end < len(content):
        if content[bracket_end] == "]" and not _is_escaped(content, bracket_end):
            break
        bracket_end += 1
    if bracket_end + 1 >= len(content) or content[bracket_end + 1] != "(":
        return None
    depth = 1
    cursor = bracket_end + 2
    while cursor < len(content):
        char = content[cursor]
        if not _is_escaped(content, cursor):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return cursor + 1
        cursor += 1
    return len(content)


def _math_end(content: str, index: int) -> int | None:
    pairs = (("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"))
    for opening, closing in pairs:
        if content.startswith(opening, index):
            end = content.find(closing, index + len(opening))
            return len(content) if end < 0 else end + len(closing)

    if content[index] != "$" or _is_escaped(content, index):
        return None
    if index + 1 >= len(content) or content[index + 1].isspace():
        return None
    cursor = index + 1
    while cursor < len(content) and content[cursor] != "\n":
        if (
            content[cursor] == "$"
            and not _is_escaped(content, cursor)
            and not content[cursor - 1].isspace()
        ):
            return cursor + 1
        cursor += 1
    return None


def _code_span_end(content: str, index: int) -> int | None:
    if content[index] != "`":
        return None
    marker_end = index
    while marker_end < len(content) and content[marker_end] == "`":
        marker_end += 1
    marker = content[index:marker_end]
    closing = content.find(marker, marker_end)
    return len(content) if closing < 0 else closing + len(marker)


def _html_block_end(content: str, index: int) -> int | None:
    match = _HTML_OPEN.match(content, index)
    if match is None:
        return None
    close = re.search(
        rf"</{re.escape(match.group(1))}\s*>",
        content[match.end() :],
        re.IGNORECASE,
    )
    if close is None:
        return len(content)
    return match.end() + close.end()


def _protected_end(content: str, index: int) -> int | None:
    at_line_start = index == 0 or content[index - 1] == "\n"
    if at_line_start:
        end = _fence_end(content, index) or _table_end(content, index)
        if end is not None:
            return end
    if content.startswith("![", index) or content[index] == "[":
        end = _markdown_link_end(content, index)
        if end is not None:
            return end
    if content.startswith(("\\[", "\\(", "$$"), index) or content[index] == "$":
        end = _math_end(content, index)
        if end is not None:
            return end
    if content[index] == "`":
        return _code_span_end(content, index)
    if content[index] == "<":
        return _html_block_end(content, index)
    return None


def iter_markdown_segments(content: str) -> Iterator[tuple[str, bool]]:
    """Yield exact plain/protected segments without changing source text."""
    plain_start = 0
    index = 0
    while index < len(content):
        protected_end = _protected_end(content, index)
        if protected_end is None:
            index += 1
            continue
        if plain_start < index:
            yield content[plain_start:index], False
        yield content[index:protected_end], True
        index = protected_end
        plain_start = index
    if plain_start < len(content):
        yield content[plain_start:], False


def _is_grapheme_extend(char: str) -> bool:
    codepoint = ord(char)
    return (
        unicodedata.category(char).startswith("M")
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0x1F3FB <= codepoint <= 0x1F3FF
        or 0xE0020 <= codepoint <= 0xE007F
        or 0xE0100 <= codepoint <= 0xE01EF
    )


def _is_regional_indicator(char: str) -> bool:
    return 0x1F1E6 <= ord(char) <= 0x1F1FF


def _safe_grapheme_cut(content: str, preferred_cut: int) -> int:
    cut = min(preferred_cut, len(content))
    while 0 < cut < len(content):
        previous = content[cut - 1]
        following = content[cut]
        if (
            _is_grapheme_extend(previous)
            or _is_grapheme_extend(following)
            or previous == "\u200d"
            or following == "\u200d"
        ):
            cut -= 1
            continue
        if _is_regional_indicator(previous) and _is_regional_indicator(following):
            preceding_count = 0
            cursor = cut - 1
            while cursor >= 0 and _is_regional_indicator(content[cursor]):
                preceding_count += 1
                cursor -= 1
            if preceding_count % 2 == 1:
                cut -= 1
                continue
        break
    if cut > 0:
        return cut

    cut = min(preferred_cut, len(content))
    while cut < len(content):
        previous = content[cut - 1]
        following = content[cut]
        if (
            _is_grapheme_extend(previous)
            or _is_grapheme_extend(following)
            or previous == "\u200d"
            or following == "\u200d"
            or (_is_regional_indicator(previous) and _is_regional_indicator(following))
        ):
            cut += 1
            continue
        break
    return cut


def _split_plain(content: str, max_chunk_chars: int) -> Iterator[str]:
    while len(content) > max_chunk_chars:
        cut = content.rfind("\n", 0, max_chunk_chars + 1)
        if cut >= 0:
            cut += 1
        else:
            whitespace = max(
                content.rfind(" ", 0, max_chunk_chars + 1),
                content.rfind("\t", 0, max_chunk_chars + 1),
            )
            cut = (
                whitespace + 1
                if whitespace >= 0
                else _safe_grapheme_cut(content, max_chunk_chars)
            )
        yield content[:cut]
        content = content[cut:]
    if content:
        yield content


def iter_markdown_replay_chunks(content: str, *, max_chunk_chars: int) -> Iterator[str]:
    """Yield exact content without splitting protected Markdown regions."""
    if not content:
        return
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")

    pending = ""
    for segment, protected in iter_markdown_segments(content):
        if protected:
            if pending:
                yield pending
                pending = ""
            yield segment
            continue
        for part in _split_plain(segment, max_chunk_chars):
            if pending and len(pending) + len(part) > max_chunk_chars:
                yield pending
                pending = ""
            if len(part) >= max_chunk_chars:
                if pending:
                    yield pending
                    pending = ""
                yield part
            else:
                pending += part
    if pending:
        yield pending
