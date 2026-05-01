from __future__ import annotations

import re

from asmr_dub_pipeline.schemas import JapaneseScript

KOREAN_LANGUAGES = {"ko", "kr", "kor", "korean"}
JAPANESE_LANGUAGES = {"ja", "jp", "jpn", "japanese"}

def _canonical_language(language: str | None) -> str:
    normalized = str(language or "").strip().lower().replace("-", "_")
    if normalized in KOREAN_LANGUAGES:
        return "ko"
    if normalized in JAPANESE_LANGUAGES:
        return "ja"
    return normalized or "ja"


def estimate_tts_duration(text: str, language: str = "ja") -> float:
    japanese_chars = len(text.replace(" ", ""))
    chars_per_second = 5.5 if _canonical_language(language) == "ko" else 7.5
    return max(0.4, japanese_chars / chars_per_second)


def _shorten_text(text: str, target_sec: float, language: str) -> str:
    canonical_language = _canonical_language(language)
    chars_per_second = 5.5 if canonical_language == "ko" else 7.5
    budget = max(4, int(target_sec * chars_per_second))
    shortened = text
    removable_tokens = (
        ("천천히...", "천천히,", "조금만", "잠깐만", "괜찮아요,", "이제,")
        if canonical_language == "ko"
        else ("ゆっくり……", "ゆっくり、", "もう少しだけ", "すこしだけ", "ねえ、")
    )
    for token in removable_tokens:
        shortened = shortened.replace(token, "")
    shortened = re.sub(r"(……){2,}", "……", shortened).strip()
    if len(shortened.replace(" ", "")) <= budget:
        return shortened or text
    chunks = re.split(r"(?<=[。！？、,.!?])", shortened)
    out = ""
    for chunk in chunks:
        if len((out + chunk).replace(" ", "")) > budget:
            break
        out += chunk
    if not out:
        out = shortened[:budget]
    if canonical_language == "ko":
        return out.rstrip(",.!?。！？ ") + "."
    return out.rstrip("、。！？ ") + "。"


def _lengthen_text(text: str, language: str) -> str:
    if _canonical_language(language) == "ko":
        return text.rstrip(".!?。！？ ") + "... 괜찮아요."
    return text.rstrip("。") + "……ね。"


def rewrite_for_duration(
    script: JapaneseScript,
    target_sec: float,
    tolerance: float = 0.15,
) -> JapaneseScript:
    language = script.tts_language
    canonical_language = _canonical_language(language)
    estimated = estimate_tts_duration(script.tts_text, language)
    if target_sec <= 0 or abs(estimated - target_sec) / target_sec <= tolerance:
        return script
    updated = script.model_copy(deep=True)
    updated.rewrite_count += 1
    if estimated > target_sec:
        updated.tts_text = _shorten_text(updated.tts_text, target_sec, language)
        if canonical_language == "ja":
            updated.ja_text = _shorten_text(updated.ja_text, target_sec, language)
        updated.risk_flags.append("duration_rewrite_shortened")
    else:
        updated.tts_text = _lengthen_text(updated.tts_text, language)
        updated.risk_flags.append("duration_rewrite_lengthened")
    updated.expected_tts_duration_sec = estimate_tts_duration(updated.tts_text, language)
    return updated
