from __future__ import annotations

import re
import unicodedata

COUNTDOWN_EVENT_KEY = "countdown_event"
COUNTDOWN_EVENT_NOTE = "deterministic_countdown_event"

_KANA_ONES_PATTERN = r"いち|イチ|に|ニ|さん|サン|よん|ヨン|し|シ|ご|ゴ|ろく|ロク|なな|ナナ|しち|シチ|はち|ハチ|きゅう|キュウ|く|ク"
_KANA_TENS_PATTERN = rf"(?:じゅう|ジュウ)(?:{_KANA_ONES_PATTERN})?"
_TOKEN_RE = re.compile(
    rf"\d{{1,2}}|ゼロ|ぜろ|れい|レイ|零|〇|{_KANA_TENS_PATTERN}|{_KANA_ONES_PATTERN}|十[一二三四五六七八九]?|[二三四五六七八九]十[一二三四五六七八九]?|[一二三四五六七八九]"
)
_ALLOWED_REMAINDER_RE = re.compile(r"^[\s,，、。!！?？・:：;；／/\\\-~〜…]*$")
_JAPANESE_WORD_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fffー々]")
_TRIPLE_DOT_RE = re.compile(r"\.{2,}")
_SINGLE_DOT_BETWEEN_DIGITS_RE = re.compile(r"\d[.．]\d")
_JAPANESE_ONES = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_JAPANESE_KANA_ONES = {
    "いち": 1,
    "イチ": 1,
    "に": 2,
    "ニ": 2,
    "さん": 3,
    "サン": 3,
    "よん": 4,
    "ヨン": 4,
    "し": 4,
    "シ": 4,
    "ご": 5,
    "ゴ": 5,
    "ろく": 6,
    "ロク": 6,
    "なな": 7,
    "ナナ": 7,
    "しち": 7,
    "シチ": 7,
    "はち": 8,
    "ハチ": 8,
    "きゅう": 9,
    "キュウ": 9,
    "く": 9,
    "ク": 9,
}
_JAPANESE_KANA_TEN_PREFIXES = ("じゅう", "ジュウ")
_JAPANESE_ZERO = {"ゼロ", "ぜろ", "れい", "レイ", "零", "〇"}
_NATIVE_KOREAN_ONES = {
    0: "영",
    1: "하나",
    2: "둘",
    3: "셋",
    4: "넷",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
}
_NATIVE_KOREAN_TENS = {
    10: "열",
    20: "스물",
    30: "서른",
    40: "마흔",
    50: "쉰",
    60: "예순",
    70: "일흔",
    80: "여든",
    90: "아흔",
}
_SINO_KOREAN_ONES = {
    0: "영",
    1: "일",
    2: "이",
    3: "삼",
    4: "사",
    5: "오",
    6: "육",
    7: "칠",
    8: "팔",
    9: "구",
}
_SINO_KOREAN_TENS = {
    10: "십",
    20: "이십",
    30: "삼십",
    40: "사십",
    50: "오십",
    60: "육십",
    70: "칠십",
    80: "팔십",
    90: "구십",
}


def source_countdown_values(text: str) -> list[int] | None:
    normalized = _normalize_source_countdown_text(text)
    if not normalized or _SINGLE_DOT_BETWEEN_DIGITS_RE.search(normalized):
        return None
    matches = _source_countdown_matches(normalized)
    if not matches:
        return None
    remainder = _remove_countdown_matches(normalized, matches)
    if not _ALLOWED_REMAINDER_RE.fullmatch(remainder):
        return None
    values: list[int] = []
    for match in matches:
        value = _source_token_value(match.group(0))
        if value is None or not 0 <= value <= 99:
            return None
        values.append(value)
    return values or None


def source_countdown_token_matches(text: str) -> list[tuple[int, str, int, int]]:
    normalized = _normalize_source_countdown_text(text)
    if not normalized or _SINGLE_DOT_BETWEEN_DIGITS_RE.search(normalized):
        return []
    matches: list[tuple[int, str, int, int]] = []
    for match in _source_countdown_matches(normalized):
        raw = match.group(0)
        value = _source_token_value(raw)
        if value is None or not 0 <= value <= 99:
            continue
        matches.append((value, raw, match.start(), match.end()))
    return matches


def is_descending_countdown(values: list[int], *, min_values: int = 3) -> bool:
    return len(values) >= min_values and all(
        left - right == 1 for left, right in zip(values, values[1:], strict=False)
    )


def is_descending_countdown_prefix(values: list[int]) -> bool:
    return len(values) <= 1 or all(
        left - right == 1 for left, right in zip(values, values[1:], strict=False)
    )


def repair_descending_countdown_text(text: str, *, min_values: int = 4) -> str | None:
    values = source_countdown_values(text)
    if values is None or len(values) < min_values or is_descending_countdown(values):
        return None
    repaired_values = _repair_descending_countdown_values(values, min_values=min_values)
    if repaired_values is not None:
        return " ".join(str(value) for value in repaired_values)
    return None


def _repair_descending_countdown_values(values: list[int], *, min_values: int) -> list[int] | None:
    prefix_len = _descending_countdown_prefix_length(values)
    if prefix_len >= min_values:
        tail = values[prefix_len:]
        if tail and all(value == values[prefix_len - 1] for value in tail):
            return values[:prefix_len]

    deltas = [left - right for left, right in zip(values, values[1:], strict=False)]
    if deltas.count(2) == 1 and all(delta in {1, 2} for delta in deltas):
        repaired = list(range(values[0], values[-1] - 1, -1))
        if len(repaired) == len(values) + 1:
            return repaired

    expected = [values[0] - index for index in range(len(values))]
    if expected[-1] < 0:
        return None
    mismatches = [
        index
        for index, (actual, expected_value) in enumerate(zip(values, expected, strict=True))
        if actual != expected_value
    ]
    if len(mismatches) != 1:
        return None
    return expected


def _descending_countdown_prefix_length(values: list[int]) -> int:
    if not values:
        return 0
    length = 1
    for left, right in zip(values, values[1:], strict=False):
        if left - right != 1:
            break
        length += 1
    return length


def native_korean_count_number(value: int) -> str | None:
    if value in _NATIVE_KOREAN_ONES:
        return _NATIVE_KOREAN_ONES[value]
    if value in _NATIVE_KOREAN_TENS:
        return _NATIVE_KOREAN_TENS[value]
    if 10 < value < 100:
        tens, ones = divmod(value, 10)
        tens_text = _NATIVE_KOREAN_TENS.get(tens * 10)
        ones_text = _NATIVE_KOREAN_ONES.get(ones)
        if tens_text and ones_text:
            return tens_text + ones_text
    return None


def sino_korean_count_number(value: int) -> str | None:
    if value in _SINO_KOREAN_ONES:
        return _SINO_KOREAN_ONES[value]
    if value in _SINO_KOREAN_TENS:
        return _SINO_KOREAN_TENS[value]
    if 10 < value < 100:
        tens, ones = divmod(value, 10)
        tens_text = _SINO_KOREAN_TENS.get(tens * 10)
        ones_text = _SINO_KOREAN_ONES.get(ones)
        if tens_text and ones_text:
            return tens_text + ones_text
    return None


def countdown_korean_tokens(values: list[int]) -> list[str] | None:
    tokens = [sino_korean_count_number(value) for value in values]
    if any(token is None for token in tokens):
        return None
    return [str(token) for token in tokens]


def countdown_korean_text(values: list[int], *, separator: str = ", ") -> str | None:
    tokens = countdown_korean_tokens(values)
    if tokens is None:
        return None
    return separator.join(tokens)


def _normalize_source_countdown_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").strip()
    return _TRIPLE_DOT_RE.sub("…", normalized)


def _source_countdown_matches(text: str) -> list[re.Match[str]]:
    return [match for match in _TOKEN_RE.finditer(text) if _is_countdown_token_match(text, match)]


def _remove_countdown_matches(text: str, matches: list[re.Match[str]]) -> str:
    if not matches:
        return text
    pieces: list[str] = []
    cursor = 0
    for match in matches:
        pieces.append(text[cursor : match.start()])
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces)


def _is_countdown_token_match(text: str, match: re.Match[str]) -> bool:
    raw = match.group(0)
    if not _is_kana_countdown_reading(raw):
        return True
    before = text[match.start() - 1] if match.start() > 0 else ""
    after = text[match.end()] if match.end() < len(text) else ""
    return not _is_japanese_word_char(before) and not _is_japanese_word_char(after)


def _is_kana_countdown_reading(token: str) -> bool:
    return (
        token in _JAPANESE_KANA_ONES
        or token in {"ゼロ", "ぜろ", "れい", "レイ"}
        or any(token.startswith(prefix) for prefix in _JAPANESE_KANA_TEN_PREFIXES)
    )


def _is_japanese_word_char(char: str) -> bool:
    return bool(char and _JAPANESE_WORD_CHAR_RE.fullmatch(char))


def _source_token_value(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    if token in _JAPANESE_ZERO:
        return 0
    if token in _JAPANESE_KANA_ONES:
        return _JAPANESE_KANA_ONES[token]
    for prefix in _JAPANESE_KANA_TEN_PREFIXES:
        if token == prefix:
            return 10
        if token.startswith(prefix):
            ones = _JAPANESE_KANA_ONES.get(token[len(prefix) :])
            return 10 + ones if ones is not None else None
    if token == "十":
        return 10
    if token.startswith("十"):
        ones = _JAPANESE_ONES.get(token[1:])
        return 10 + ones if ones is not None else None
    if "十" in token:
        tens_raw, ones_raw = token.split("十", 1)
        tens = _JAPANESE_ONES.get(tens_raw)
        ones = _JAPANESE_ONES.get(ones_raw) if ones_raw else 0
        if tens is None or ones is None:
            return None
        return tens * 10 + ones
    return _JAPANESE_ONES.get(token)
