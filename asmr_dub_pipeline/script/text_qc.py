from __future__ import annotations

import re
from dataclasses import dataclass, field

from asmr_dub_pipeline.schemas import JapaneseScript

KANA_RE = re.compile(r"[\u3040-\u30ff]")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
BRACKETED_RE = re.compile(r"(\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|（[^）]*）|【[^】]*】)")


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


def hangul_ratio(text: str) -> float:
    speech_chars = [char for char in text if not char.isspace() and not char.isdigit()]
    if not speech_chars:
        return 0.0
    hangul = sum(1 for char in speech_chars if HANGUL_RE.match(char))
    return hangul / len(speech_chars)


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
