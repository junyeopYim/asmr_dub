from __future__ import annotations

import re
import unicodedata

from asmr_dub_pipeline.schemas import JapaneseScript

KOREAN_LANGUAGES = {"ko", "kr", "kor", "korean"}
JAPANESE_LANGUAGES = {"ja", "jp", "jpn", "japanese"}
KOREAN_TTS_CHARS_PER_SECOND = 4.0
KOREAN_TTS_MAX_SPEED_ALLOWANCE = 1.12
JAPANESE_TTS_CHARS_PER_SECOND = 7.5
KOREAN_TOPIC_PARTICLES = ("은", "는", "이", "가", "도", "에서는", "에게는", "한테는")
JAPANESE_SMALL_KANA = set("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")
JAPANESE_SOKUON = {"っ", "ッ"}
JAPANESE_LONG_VOWEL = {"ー"}


def _canonical_language(language: str | None) -> str:
    normalized = str(language or "").strip().lower().replace("-", "_")
    if normalized in KOREAN_LANGUAGES:
        return "ko"
    if normalized in JAPANESE_LANGUAGES:
        return "ja"
    return normalized or "ja"


def _chars_per_second(language: str | None) -> float:
    return (
        KOREAN_TTS_CHARS_PER_SECOND
        if _canonical_language(language) == "ko"
        else JAPANESE_TTS_CHARS_PER_SECOND
    )


def _speech_char_count(text: str) -> int:
    return sum(1 for char in text if unicodedata.category(char)[0] in {"L", "N"})


def korean_tts_speech_char_count(text: str) -> int:
    return _speech_char_count(text)


def _is_kana(char: str) -> bool:
    codepoint = ord(char)
    return 0x3040 <= codepoint <= 0x30FF or 0x31F0 <= codepoint <= 0x31FF


def _is_cjk_ideograph(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
    )


def japanese_pronunciation_count(text: str) -> int:
    count = 0
    for char in text:
        if char in JAPANESE_SMALL_KANA:
            continue
        if char in JAPANESE_SOKUON or char in JAPANESE_LONG_VOWEL:
            count += 1
            continue
        if _is_kana(char):
            count += 1
            continue
        if _is_cjk_ideograph(char):
            count += 2
            continue
        if unicodedata.category(char)[0] in {"L", "N"}:
            count += 1
    return count


def korean_tts_timing_budget(
    target_sec: float,
    source_text: str | None = None,
) -> dict[str, float | int | str | None]:
    duration_target_chars = max(4, int(max(0.0, target_sec) * KOREAN_TTS_CHARS_PER_SECOND))
    source_pronunciation_count = japanese_pronunciation_count(source_text or "")
    if source_pronunciation_count > 0:
        target_chars = source_pronunciation_count
        budget_basis = "source_japanese_pronunciation"
    else:
        target_chars = duration_target_chars
        budget_basis = "duration_estimate"
    max_chars = max(
        target_chars,
        6,
    )
    return {
        "target_duration_sec": round(max(0.0, target_sec), 3),
        "estimated_chars_per_sec": KOREAN_TTS_CHARS_PER_SECOND,
        "source_japanese_pronunciation_count": source_pronunciation_count or None,
        "target_speech_chars": target_chars,
        "max_speech_chars": max_chars,
        "budget_basis": budget_basis,
        "counting_rule": (
            "Korean budget counts Unicode letters and numbers only; Japanese source "
            "pronunciation count approximates kana mora and two units per kanji when no "
            "reading dictionary is available."
        ),
    }


def korean_tts_slot_timing_budget(target_sec: float) -> dict[str, float | int | str | None]:
    target_chars = max(4, int(max(0.0, target_sec) * KOREAN_TTS_CHARS_PER_SECOND))
    max_chars = max(target_chars + 2, int(target_chars * KOREAN_TTS_MAX_SPEED_ALLOWANCE), 6)
    min_chars = max(4, int(target_chars * 0.55))
    return {
        "target_duration_sec": round(max(0.0, target_sec), 3),
        "estimated_chars_per_sec": KOREAN_TTS_CHARS_PER_SECOND,
        "target_speech_chars": target_chars,
        "min_speech_chars": min(min_chars, max_chars),
        "max_speech_chars": max_chars,
        "budget_basis": "duration_slot",
        "counting_rule": (
            "Korean slot timing budget counts Unicode letters and numbers only; it is "
            "based on the available dubbed segment duration rather than source text length."
        ),
    }


def estimate_tts_duration(text: str, language: str = "ja") -> float:
    return max(0.4, _speech_char_count(text) / _chars_per_second(language))


def _fits_speech_budget(text: str, budget: int) -> bool:
    return _speech_char_count(text) <= budget


def _finish_korean_sentence(text: str) -> str:
    body = re.sub(r"\s+", " ", text).strip(" ,.!?。！？、…")
    return f"{body}." if body else ""


def _first_korean_topic(text: str) -> str:
    before_comma = re.split(r"[,、]", text, maxsplit=1)[0].strip()
    words = [word.strip(" ,.!?。！？、…") for word in before_comma.split()]
    words = [word for word in words if word]
    for end in range(1, min(3, len(words)) + 1):
        candidate = " ".join(words[:end])
        if words[end - 1].endswith(KOREAN_TOPIC_PARTICLES) and _speech_char_count(candidate) <= 8:
            return candidate
    return ""


def _korean_word_tail_with_budget(text: str, budget: int, prefix: str = "") -> str:
    clean_prefix = prefix.strip(" ,.!?。！？、…")
    words = [word.strip(" ,.!?。！？、…") for word in text.split()]
    words = [word for word in words if word]
    if clean_prefix and words and words[0] == clean_prefix:
        words = words[1:]
    tail: list[str] = []
    for word in reversed(words):
        trial = [word, *tail]
        candidate_words = ([clean_prefix] if clean_prefix else []) + trial
        if _fits_speech_budget(" ".join(candidate_words), budget):
            tail = trial
        elif tail:
            break
    if tail:
        candidate_words = ([clean_prefix] if clean_prefix else []) + tail
        return " ".join(candidate_words)
    if clean_prefix and _speech_char_count(clean_prefix) <= budget:
        return clean_prefix
    return words[-1] if words else ""


def _shorten_korean_text(text: str, target_sec: float) -> str:
    budget = int(korean_tts_timing_budget(target_sec)["max_speech_chars"])
    shortened = text
    for token in ("천천히...", "천천히,", "조금만", "잠깐만", "괜찮아요,", "이제,"):
        shortened = shortened.replace(token, "")
    shortened = re.sub(r"(…){2,}", "…", shortened).strip()
    if _fits_speech_budget(shortened, budget):
        return shortened or text

    clauses = [clause.strip(" ,.!?。！？、…") for clause in re.split(r"[,、]", shortened)]
    clauses = [clause for clause in clauses if clause]
    if len(clauses) > 1:
        topic = _first_korean_topic(shortened)
        last_clause = clauses[-1]
        candidates = []
        if topic:
            candidates.append(_finish_korean_sentence(f"{topic} {last_clause}"))
        candidates.append(_finish_korean_sentence(last_clause))
        for candidate in candidates:
            if candidate and _fits_speech_budget(candidate, budget):
                return candidate
        for prefix in (topic, ""):
            candidate = _finish_korean_sentence(
                _korean_word_tail_with_budget(last_clause, budget, prefix)
            )
            if candidate and _fits_speech_budget(candidate, budget):
                return candidate

    chunks = re.split(r"(?<=[。！？、,.!?])", shortened)
    out = ""
    for chunk in chunks:
        if not _fits_speech_budget(out + chunk, budget):
            break
        out += chunk
    if out:
        candidate = _finish_korean_sentence(out)
        lower_budget = max(4, int(budget * 0.6))
        if _speech_char_count(candidate) >= lower_budget:
            return candidate
        tail_candidate = _finish_korean_sentence(_korean_word_tail_with_budget(shortened, budget))
        if _speech_char_count(tail_candidate) > _speech_char_count(candidate):
            return tail_candidate
        return candidate

    return _finish_korean_sentence(_korean_word_tail_with_budget(shortened, budget)) or text


def _shorten_text(text: str, target_sec: float, language: str) -> str:
    canonical_language = _canonical_language(language)
    if canonical_language == "ko":
        return _shorten_korean_text(text, target_sec)
    chars_per_second = _chars_per_second(language)
    budget = max(4, int(target_sec * chars_per_second))
    shortened = text
    for token in ("ゆっくり……", "ゆっくり、", "もう少しだけ", "すこしだけ", "ねえ、"):
        shortened = shortened.replace(token, "")
    shortened = re.sub(r"(……){2,}", "……", shortened).strip()
    if _fits_speech_budget(shortened, budget):
        return shortened or text
    chunks = re.split(r"(?<=[。！？、,.!?])", shortened)
    out = ""
    for chunk in chunks:
        if not _fits_speech_budget(out + chunk, budget):
            break
        out += chunk
    if not out:
        out = shortened[:budget]
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
