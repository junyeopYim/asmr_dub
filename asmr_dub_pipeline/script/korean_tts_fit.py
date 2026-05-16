from __future__ import annotations

import re

from asmr_dub_pipeline.schemas import KoreanTranslation, Segment
from asmr_dub_pipeline.script.duration_rewrite import (
    korean_tts_speech_char_count,
    korean_tts_timing_budget,
)

SANITIZED_NOTE = "korean_tts_sanitized"
BUDGET_FIT_NOTE = "korean_tts_budget_fit"
LITERAL_FALLBACK_NOTE = "korean_tts_literal_fallback"

_KANA_RE = re.compile(r"[\u3040-\u30ffー]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d+")
_PRONUNCIATION_SYMBOL_RE = re.compile(r"[^\uac00-\ud7a3\s,.!?…]")
_SPACE_RE = re.compile(r"\s+")

_ACRONYM_READINGS = {
    "ASMR": "에이에스엠알",
    "TTS": "티티에스",
    "OK": "오케이",
}
_DIGIT_READINGS = {
    "0": "영",
    "1": "일",
    "2": "이",
    "3": "삼",
    "4": "사",
    "5": "오",
    "6": "육",
    "7": "칠",
    "8": "팔",
    "9": "구",
    "10": "십",
}
_SINO_DIGIT_READINGS = {
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
_SHORTENINGS = (
    ("반드시", "꼭"),
    ("그렇게", "그리"),
    ("되어요", "돼요"),
    ("거예요", "예요"),
    ("합니다", "해요"),
    ("습니다", "요"),
    ("이에요", "예요"),
    ("풀어요", "풀어"),
    ("밀어요", "밀어"),
)


def _dedupe_notes(notes: list[str]) -> list[str]:
    return list(dict.fromkeys(note for note in notes if note))


def _clean_spacing(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _digit_reading(match: re.Match[str]) -> str:
    value = match.group(0)
    if value in _DIGIT_READINGS:
        return _DIGIT_READINGS[value]
    if not value.startswith("0") and len(value) <= 4:
        number = int(value)
        if 0 < number <= 9999:
            parts: list[str] = []
            for unit, unit_text in ((1000, "천"), (100, "백"), (10, "십")):
                digit, number = divmod(number, unit)
                if digit == 0:
                    continue
                parts.append(("" if digit == 1 else _SINO_DIGIT_READINGS[digit]) + unit_text)
            if number:
                parts.append(_SINO_DIGIT_READINGS[number])
            return "".join(parts)
    return " ".join(_DIGIT_READINGS[digit] for digit in value)


def sanitize_korean_tts_text(text: str) -> tuple[str, list[str]]:
    original = str(text or "")
    sanitized = original
    for source, target in _ACRONYM_READINGS.items():
        sanitized = re.sub(rf"\b{re.escape(source)}\b", target, sanitized, flags=re.IGNORECASE)
    sanitized = _DIGIT_RE.sub(_digit_reading, sanitized)
    sanitized = sanitized.translate(
        str.maketrans(
            {
                "(": " ",
                ")": " ",
                "[": " ",
                "]": " ",
                "{": " ",
                "}": " ",
                "/": " ",
                "\\": " ",
                ":": " ",
                ";": " ",
                "—": " ",
                "–": " ",
                "―": " ",
                "「": " ",
                "」": " ",
                "『": " ",
                "』": " ",
                "“": " ",
                "”": " ",
                "‘": " ",
                "’": " ",
            }
        )
    )
    sanitized = _clean_spacing(sanitized)
    return sanitized, ([SANITIZED_NOTE] if sanitized != original.strip() else [])


def fit_korean_tts_budget(text: str, *, max_speech_chars: int) -> tuple[str, list[str]]:
    max_speech_chars = max(1, int(max_speech_chars))
    original = _clean_spacing(text)
    if korean_tts_speech_char_count(original) <= max_speech_chars:
        return original, []

    candidates = [original]
    shortened = original
    for source, target in _SHORTENINGS:
        shortened = shortened.replace(source, target)
        candidates.append(_clean_spacing(shortened))

    words = _clean_spacing(shortened).split()
    for end in range(len(words) - 1, 0, -1):
        candidates.append(_clean_spacing(" ".join(words[:end])))

    for candidate in candidates:
        if candidate and korean_tts_speech_char_count(candidate) <= max_speech_chars:
            return candidate, [BUDGET_FIT_NOTE]
    return original, []


def _valid_korean_tts_text(text: str) -> bool:
    if not text.strip() or not _HANGUL_RE.search(text):
        return False
    return not (
        _KANA_RE.search(text)
        or _CJK_RE.search(text)
        or _LATIN_RE.search(text)
        or re.search(r"\d", text)
        or _PRONUNCIATION_SYMBOL_RE.search(text)
    )


def salvage_korean_translation(
    segment: Segment,
    translation: KoreanTranslation | None,
) -> tuple[KoreanTranslation, list[str]] | None:
    if translation is None:
        return None
    source_text = segment.source_script.text if segment.source_script else ""
    budget = korean_tts_timing_budget(segment.duration, source_text)
    max_speech_chars = int(budget["max_speech_chars"])
    notes = list(translation.notes)

    natural, sanitize_notes = sanitize_korean_tts_text(translation.ko_natural)
    notes.extend(sanitize_notes)
    fitted, fit_notes = fit_korean_tts_budget(natural, max_speech_chars=max_speech_chars)
    notes.extend(fit_notes)
    natural = fitted

    if korean_tts_speech_char_count(natural) > max_speech_chars:
        literal, literal_sanitize_notes = sanitize_korean_tts_text(translation.ko_literal)
        literal, literal_fit_notes = fit_korean_tts_budget(
            literal,
            max_speech_chars=max_speech_chars,
        )
        if (
            _valid_korean_tts_text(literal)
            and korean_tts_speech_char_count(literal) <= max_speech_chars
        ):
            natural = literal
            notes.extend([*literal_sanitize_notes, *literal_fit_notes, LITERAL_FALLBACK_NOTE])

    if not _valid_korean_tts_text(natural):
        return None
    if korean_tts_speech_char_count(natural) > max_speech_chars:
        return None
    final_notes = _dedupe_notes(notes)
    return (
        translation.model_copy(update={"ko_natural": natural, "notes": final_notes}),
        [note for note in final_notes if note not in translation.notes],
    )
