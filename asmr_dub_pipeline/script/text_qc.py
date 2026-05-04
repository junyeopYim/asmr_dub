from __future__ import annotations

import re
from dataclasses import dataclass, field

from asmr_dub_pipeline.schemas import JapaneseScript

KANA_RE = re.compile(r"[\u3040-\u30ff]")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIGIT_RE = re.compile(r"\d")
PRONUNCIATION_SYMBOL_RE = re.compile(r"[^\uac00-\ud7a3\s,.!?…]")
SUSPICIOUS_FINAL_FRAGMENT_RE = re.compile(
    r"(?:다음\s+[가-힣]{1,2}|말고,\s*[가-힣]{1,4}|[가-힣]+\s+말고\s+[가-힣]{1,4})$"
)
BRACKETED_RE = re.compile(
    r"(\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|（[^）]*）|【[^】]*】|｛[^｝]*｝|〈[^〉]*〉|《[^》]*》)"
)
KOREAN_LONG_CLAUSE_MAX_CHARS = 44
MINOR_SUBJECT_RE = re.compile(
    r"(?:少女|幼女|小さな女の子|女の子|女子校生|女子高生|学校|"
    r"소녀|여자아이|여아|미성년자?|로리|학교|교복)"
)
SEXUALIZED_CONTENT_RE = re.compile(
    r"(?:裸|全裸|下着|股間|胸|乳首|性的|性器|性交|媚薬|"
    r"나체|벌거벗|속옷|가랑이|가슴|유두|성적|성적인|성기|음부|최음)"
)


@dataclass(frozen=True)
class TextQCResult:
    decision: str
    issues: list[str] = field(default_factory=list)
    hangul_ratio: float = 0.0

    @property
    def blocked(self) -> bool:
        return self.decision == "block"

    def as_payload(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "issues": list(self.issues),
            "hangul_ratio": round(self.hangul_ratio, 6),
        }


def has_kana(text: str) -> bool:
    return bool(KANA_RE.search(text))


def has_hangul(text: str) -> bool:
    return bool(HANGUL_RE.search(text))


def has_bracketed_direction(text: str) -> bool:
    return bool(BRACKETED_RE.search(text))


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text))


def has_digit(text: str) -> bool:
    return bool(DIGIT_RE.search(text))


def has_pronunciation_symbol(text: str) -> bool:
    return bool(PRONUNCIATION_SYMBOL_RE.search(text))


def has_long_korean_clause(text: str, max_chars: int = KOREAN_LONG_CLAUSE_MAX_CHARS) -> bool:
    for clause in re.split(r"[,.!?…]+", text):
        speech_len = sum(1 for char in clause if not char.isspace())
        if speech_len > max_chars:
            return True
    return False


def has_suspicious_truncated_sentence(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return False
    if normalized.endswith(","):
        return True
    if SUSPICIOUS_FINAL_FRAGMENT_RE.search(normalized):
        return True
    last_clause = re.split(r"[,!?…]+", normalized)[-1].strip()
    if last_clause in {"푸"}:
        return True
    return bool(re.search(r"(?:그리고|그런데|하지만|또는|혹은|말고|다음)$", last_clause))


def hangul_ratio(text: str) -> float:
    speech_chars = [char for char in text if not char.isspace() and not char.isdigit()]
    if not speech_chars:
        return 0.0
    hangul = sum(1 for char in speech_chars if HANGUL_RE.match(char))
    return hangul / len(speech_chars)


def has_minor_sexualized_content(text: str, source_text: str = "") -> bool:
    combined = "\n".join(part for part in (source_text.strip(), text.strip()) if part)
    if not combined:
        return False
    return bool(MINOR_SUBJECT_RE.search(combined) and SEXUALIZED_CONTENT_RE.search(combined))


def preflight_tts_text(
    script: JapaneseScript,
    *,
    target_language: str,
    source_text: str = "",
    min_hangul_ratio: float = 0.20,
) -> TextQCResult:
    target = target_language.strip().lower().replace("kr", "ko")
    issues: list[str] = []
    text = script.tts_text.strip()
    if target == "ko":
        if script.tts_language.strip().lower().replace("kr", "ko") != "ko":
            issues.append("tts_language_not_ko")
        if not text:
            issues.append("korean_tts_text_empty")
        if has_kana(text):
            issues.append("korean_tts_contains_kana")
        if has_bracketed_direction(text):
            issues.append("korean_tts_contains_stage_direction")
        if has_latin(text):
            issues.append("korean_tts_contains_latin")
        if has_digit(text):
            issues.append("korean_tts_contains_digit")
        if has_pronunciation_symbol(text):
            issues.append("korean_tts_contains_pronunciation_symbol")
        if has_long_korean_clause(text):
            issues.append("korean_tts_long_clause_without_pause")
        if has_suspicious_truncated_sentence(text):
            issues.append("korean_tts_suspicious_truncated_sentence")
        if has_minor_sexualized_content(text, source_text):
            issues.append("tts_safety_minor_sexualized_content")
        ratio = hangul_ratio(text)
        if source_text.strip() and text == source_text.strip():
            issues.append("korean_tts_matches_source_japanese")
        if source_text.strip() and not has_hangul(text):
            issues.append("korean_tts_missing_hangul")
        if text and ratio < min_hangul_ratio:
            issues.append("korean_tts_hangul_ratio_too_low")
        return TextQCResult(
            decision="block" if issues else "pass",
            issues=issues,
            hangul_ratio=ratio,
        )
    return TextQCResult(decision="pass", issues=[], hangul_ratio=hangul_ratio(text))
