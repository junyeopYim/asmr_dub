from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from asmr_dub_pipeline.schemas import JapaneseScript, NonverbalCue, ScriptRetryPolicy


@dataclass(frozen=True)
class NormalizedScriptText:
    text: str
    cues: list[NonverbalCue]
    risk_flags: list[str]


BRACKET_PATTERN = re.compile(
    r"(\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|（[^）]*）|【[^】]*】|｛[^｝]*｝|〈[^〉]*〉|《[^》]*》)"
)
TOKEN_REPLACEMENTS = {
    "ASMR": "エーエスエムアール",
    "OK": "オーケー",
}
KOREAN_TOKEN_REPLACEMENTS = {
    "ASMR": "에이에스엠알",
    "OK": "오케이",
}
NUMBER_UNITS = {
    "1": "いっ",
    "2": "に",
    "3": "さん",
    "4": "よん",
    "5": "ご",
    "6": "ろっ",
    "7": "なな",
    "8": "はっ",
    "9": "きゅう",
    "10": "じゅう",
}
HEARTS = {"♡", "♥", "❤", "💕", "💗"}
BRACKET_CHARS = "()[]{}（）【】｛｝〈〉《》"
PAUSE_SECONDS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(秒|sec(?:onds?)?|s|ms)", re.IGNORECASE)
PUNCT_TRANSLATION = str.maketrans(
    {
        ",": "、",
        "､": "、",
        ".": "。",
        "｡": "。",
        "!": "！",
        "?": "？",
    }
)
KOREAN_PUNCT_TRANSLATION = str.maketrans(
    {
        "､": ",",
        "｡": ".",
        "。": ".",
        "、": ",",
        "！": "!",
        "？": "?",
    }
)
SPACE_RE = re.compile(r"[\s\u3000]+")
JAPANESE_PUNCT_RE = re.compile(r"\s*([、。！？])\s*")
KOREAN_PUNCT_RE = re.compile(r"\s*([,.!?])\s*")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _cue_kind(content: str) -> tuple[str, str]:
    lowered = content.lower()
    if "笑" in content or "くす" in content or "laugh" in lowered:
        return "laugh", "laugh"
    if "耳" in content or "左" in content or "右" in content or "近" in content or "pan" in lowered:
        return "spatial", "ear_close"
    if "息" in content or "吐息" in content or "呼吸" in content or "breath" in lowered:
        return "breath", "breath"
    if "間" in content or "待" in content or "沈黙" in content or "pause" in lowered:
        return "pause", "pause"
    if "小声" in content or "囁" in content or "ささや" in content or "whisper" in lowered:
        return "style", "whisper"
    if "リップ" in content or "口" in content or "mouth" in lowered:
        return "mouth_sound", "mouth_sound"
    return "stage_direction", content.strip(BRACKET_CHARS)


def _pause_hint_sec(content: str) -> float | None:
    match = PAUSE_SECONDS_RE.search(content)
    if match:
        value = float(match.group(1))
        return value / 1000 if match.group(2).lower() == "ms" else value
    lowered = content.lower()
    if "長" in content or "しばらく" in content or "沈黙" in content or "long" in lowered:
        return 0.8
    if "短" in content or "一拍" in content or "short" in lowered:
        return 0.25
    if "間" in content or "待" in content or "pause" in lowered:
        return 0.45
    return None


def extract_bracketed_cues(text: str) -> tuple[str, list[NonverbalCue]]:
    cues: list[NonverbalCue] = []

    def repl(match: re.Match[str]) -> str:
        source = match.group(0)
        content = source.strip(BRACKET_CHARS)
        kind, normalized = _cue_kind(content)
        cues.append(
            NonverbalCue(
                kind=kind,
                source_text=source,
                normalized_text=normalized,
                position=match.start(),
                pause_sec=_pause_hint_sec(content) if kind == "pause" else None,
            )
        )
        return ""

    return BRACKET_PATTERN.sub(repl, text), cues


def _normalize_numbers(text: str) -> tuple[str, list[str]]:
    risk_flags: list[str] = []

    def repl(match: re.Match[str]) -> str:
        number, unit = match.group(1), match.group(2)
        if number == "3" and unit == "分":
            return "さんぷん"
        if number == "10" and unit == "秒":
            return "じゅうびょう"
        if number in NUMBER_UNITS:
            suffix = "ぷん" if unit == "分" else "びょう"
            return NUMBER_UNITS[number] + suffix
        risk_flags.append("unhandled_numeric_token")
        return match.group(0)

    return re.sub(r"(\d+)(分|秒)", repl, text), risk_flags


def _normalize_punctuation_and_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"(?:\.{3,}|…+|⋯+)", "……", text)
    text = text.translate(PUNCT_TRANSLATION)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"、{2,}", "、", text)
    text = re.sub(r"！{2,}", "！", text)
    text = re.sub(r"？{2,}", "？", text)
    text = re.sub(r"(?:……){2,}", "……", text)
    text = JAPANESE_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"([、。！？])([、。！？])+", r"\1", text)
    return SPACE_RE.sub(" ", text).strip()


def _normalize_korean_punctuation_and_space(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"(?:\.{3,}|…+|⋯+)", "...", text)
    text = text.translate(KOREAN_PUNCT_TRANSLATION)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = KOREAN_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"([,.!?])([,.!?])+", r"\1", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_korean_tts_text(text: str) -> NormalizedScriptText:
    without_brackets, cues = extract_bracketed_cues(text)
    risk_flags: list[str] = []
    for token, replacement in KOREAN_TOKEN_REPLACEMENTS.items():
        without_brackets = without_brackets.replace(token, replacement)
    out_chars: list[str] = []
    for idx, char in enumerate(without_brackets):
        if char in HEARTS:
            cues.append(
                NonverbalCue(
                    kind="soft_affect",
                    source_text=char,
                    normalized_text="soft_affect",
                    position=idx,
                )
            )
            continue
        if ord(char) >= 0x1F300:
            continue
        if char in "*_`#>":
            continue
        out_chars.append(char)
    normalized = _normalize_korean_punctuation_and_space("".join(out_chars))
    if normalized != text.strip():
        risk_flags.append("normalized_tts_text")
    if not normalized:
        risk_flags.append("tts_text_empty")
    return NormalizedScriptText(text=normalized, cues=cues, risk_flags=_dedupe(risk_flags))


def normalize_tts_text(text: str, language: str = "ja") -> NormalizedScriptText:
    if language.strip().lower() in {"ko", "kr", "kor", "korean"}:
        return normalize_korean_tts_text(text)
    without_brackets, cues = extract_bracketed_cues(text)
    risk_flags: list[str] = []
    for token, replacement in TOKEN_REPLACEMENTS.items():
        without_brackets = without_brackets.replace(token, replacement)
    without_brackets, number_flags = _normalize_numbers(without_brackets)
    risk_flags.extend(number_flags)
    out_chars: list[str] = []
    for idx, char in enumerate(without_brackets):
        if char in HEARTS:
            cues.append(
                NonverbalCue(
                    kind="soft_affect",
                    source_text=char,
                    normalized_text="soft_affect",
                    position=idx,
                )
            )
            continue
        if ord(char) >= 0x1F300:
            continue
        out_chars.append(char)
    normalized = _normalize_punctuation_and_space("".join(out_chars))
    if normalized != text.strip():
        risk_flags.append("normalized_tts_text")
    if not normalized:
        risk_flags.append("tts_text_empty")
    return NormalizedScriptText(text=normalized, cues=cues, risk_flags=_dedupe(risk_flags))


def _coerce_retry_policy(value: Any) -> ScriptRetryPolicy:
    if isinstance(value, ScriptRetryPolicy):
        return value
    if isinstance(value, dict):
        return ScriptRetryPolicy.model_validate(value)
    return ScriptRetryPolicy()


def normalize_script_payload(payload: dict[str, Any], language: str | None = None) -> JapaneseScript:
    tts_language = str(language or payload.get("tts_language") or "ja")
    normalized_ja = normalize_tts_text(str(payload.get("ja_text") or ""), "ja")
    normalized = normalize_tts_text(str(payload.get("tts_text") or normalized_ja.text or ""), tts_language)
    existing_cues = payload.get("nonverbal_cues") or []
    cues = [
        cue if isinstance(cue, NonverbalCue) else NonverbalCue.model_validate(cue)
        for cue in existing_cues
    ]
    risk_flags = list(payload.get("risk_flags") or [])
    for flag in normalized_ja.risk_flags:
        if flag not in risk_flags:
            risk_flags.append(flag)
    risk_flags.extend(flag for flag in normalized.risk_flags if flag not in risk_flags)
    return JapaneseScript(
        literal_ja=str(payload.get("literal_ja") or ""),
        ja_text=normalized_ja.text or normalized.text,
        tts_text=normalized.text,
        tts_language=tts_language,
        source_language=str(payload.get("source_language") or "ja"),
        target_language=str(payload.get("target_language") or tts_language),
        ref_style=str(payload.get("ref_style") or "whisper_close"),
        emotion=payload.get("emotion") or "gentle",
        pace=payload.get("pace") or "slow",
        volume=payload.get("volume") or "soft",
        nonverbal_cues=[*cues, *normalized_ja.cues, *normalized.cues],
        spatial_style=payload.get("spatial_style") or "center",
        expected_tts_duration_sec=float(payload.get("expected_tts_duration_sec") or 1.0),
        style_tags=list(payload.get("style_tags") or []),
        retry_policy=_coerce_retry_policy(payload.get("retry_policy")),
        rewrite_count=int(payload.get("rewrite_count") or 0),
        risk_flags=_dedupe(risk_flags),
    )
