from __future__ import annotations

import re
import unicodedata
from typing import Any

from asmr_dub_pipeline.script.countdown import (
    native_korean_count_number,
    sino_korean_count_number,
)

NUMERIC_CADENCE_VARIANT = "native_periods_no_compact"

_SEQUENCE_SEPARATOR_RE = re.compile(r"^[\s,，、.!?。！？・:：;；/\\\-~〜]*$")
_TRAILING_SENTENCE_PUNCTUATION_RE = re.compile(r"^[.!?。！？]+")
_COUNTING_FILLER_RE = re.compile(r"^\s*말이에요[.!?。！？]*\s*$")
_SPLIT_BLOCK_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _build_korean_number_token_map() -> dict[str, int]:
    tokens: dict[str, int] = {}
    for value in range(0, 100):
        tokens[str(value)] = value
        native = native_korean_count_number(value)
        if native:
            tokens[native] = value
        sino = sino_korean_count_number(value)
        if sino:
            tokens[sino] = value
    tokens["공"] = 0
    return tokens


_KOREAN_NUMBER_TOKEN_TO_VALUE = _build_korean_number_token_map()
_KOREAN_NUMBER_TOKEN_TO_NATIVE = {
    token: native_korean_count_number(value) or token
    for token, value in _KOREAN_NUMBER_TOKEN_TO_VALUE.items()
}
_NATIVE_TOKEN_TO_VALUE = {
    token: value
    for token, value in _KOREAN_NUMBER_TOKEN_TO_VALUE.items()
    if native_korean_count_number(value) == token
}
_KOREAN_NUMBER_TOKEN_RE = re.compile(
    r"(?<![0-9A-Za-z가-힣])("
    + "|".join(
        re.escape(token)
        for token in sorted(_KOREAN_NUMBER_TOKEN_TO_VALUE, key=len, reverse=True)
    )
    + r")(?![0-9A-Za-z가-힣])"
)


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").strip()


def _token_value(token: str) -> int | None:
    return _KOREAN_NUMBER_TOKEN_TO_VALUE.get(token)


def _token_native(token: str) -> str:
    return _KOREAN_NUMBER_TOKEN_TO_NATIVE.get(token, token)


def _consume_trailing_sentence_punctuation(text: str, end: int) -> int:
    match = _TRAILING_SENTENCE_PUNCTUATION_RE.match(text[end:])
    if match is None:
        return end
    return end + match.end()


def _replace_numeric_groups(
    text: str,
    groups: list[list[re.Match[str]]],
) -> tuple[str, list[dict[str, Any]]]:
    output_parts: list[str] = []
    runs: list[dict[str, Any]] = []
    last_index = 0
    for group in groups:
        start = group[0].start()
        end = _consume_trailing_sentence_punctuation(text, group[-1].end())
        before = text[start:end]
        values = [_token_value(match.group(1)) for match in group]
        native_tokens = [_token_native(match.group(1)) for match in group]
        after = ". ".join(native_tokens) + "."
        output_parts.append(text[last_index:start])
        output_parts.append(after)
        runs.append(
            {
                "before": before,
                "after": after,
                "values": [value for value in values if value is not None],
                "token_count": len(group),
            }
        )
        last_index = end
    output_parts.append(text[last_index:])
    return "".join(output_parts).strip(), runs


def periodize_korean_numeric_cadence_text(
    text: str,
    *,
    min_values: int = 3,
) -> tuple[str, dict[str, Any] | None]:
    """Rewrite separated Korean numeric cadence runs as native period-spaced tokens."""
    normalized = _normalize_text(text)
    if not normalized:
        return normalized, None
    matches = list(_KOREAN_NUMBER_TOKEN_RE.finditer(normalized))
    if len(matches) < min_values:
        return normalized, None

    groups: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = [matches[0]]
    for match in matches[1:]:
        separator = normalized[current[-1].end() : match.start()]
        if _SEQUENCE_SEPARATOR_RE.fullmatch(separator):
            current.append(match)
        else:
            if len(current) >= min_values:
                groups.append(current)
            current = [match]
    if len(current) >= min_values:
        groups.append(current)
    if not groups:
        return normalized, None

    periodized, runs = _replace_numeric_groups(normalized, groups)
    removed_counting_filler = False
    if len(groups) == 1:
        group = groups[0]
        prefix = normalized[: group[0].start()]
        suffix = normalized[group[-1].end() :]
        if not prefix.strip() and _COUNTING_FILLER_RE.fullmatch(suffix):
            periodized = runs[0]["after"]
            removed_counting_filler = True
            runs[0]["before"] = normalized

    if periodized == normalized:
        return normalized, None
    return periodized, {
        "variant": NUMERIC_CADENCE_VARIANT,
        "compaction_enabled": False,
        "runs": runs,
        "removed_counting_filler": removed_counting_filler,
    }


def _segment_native_compact_block(block: str) -> list[int] | None:
    values: list[int] = []
    index = 0
    native_tokens = sorted(_NATIVE_TOKEN_TO_VALUE, key=len, reverse=True)
    while index < len(block):
        matched = False
        for token in native_tokens:
            if block.startswith(token, index):
                values.append(_NATIVE_TOKEN_TO_VALUE[token])
                index += len(token)
                matched = True
                break
        if not matched:
            return None
    return values if len(values) >= 2 else None


def extract_korean_numeric_values(text: str) -> list[int]:
    """Extract Korean count values while avoiding one-syllable matches inside words."""
    normalized = _normalize_text(text)
    if not normalized:
        return []
    values: list[int] = []
    for block_match in _SPLIT_BLOCK_RE.finditer(normalized):
        block = block_match.group(0)
        direct = _token_value(block)
        if direct is not None:
            values.append(direct)
            continue
        if block.isdigit() and len(block) > 1:
            values.extend(int(char) for char in block)
            continue
        compact = _segment_native_compact_block(block)
        if compact is not None:
            values.extend(compact)
    return values
