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


@dataclass(frozen=True)
class JapaneseKanaText:
    text: str
    original_text: str
    changed: bool
    risk_flags: list[str]


class JapaneseKanaNormalizationError(ValueError):
    pass


BRACKET_PATTERN = re.compile(
    r"(\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|（[^）]*）|【[^】]*】|｛[^｝]*｝|〈[^〉]*〉|《[^》]*》)"
)
TOKEN_REPLACEMENTS = {
    "ASMR": "エーエスエムアール",
    "OK": "オーケー",
}
KOREAN_LATIN_TOKEN_REPLACEMENTS = {
    "ai": "에이아이",
    "asmr": "에이에스엠알",
    "gpt": "지피티",
    "gpt-sovits": "지피티 소비츠",
    "ok": "오케이",
    "rvc": "알브이씨",
    "sns": "에스엔에스",
    "sovits": "소비츠",
    "tts": "티티에스",
    "url": "유알엘",
    "usb": "유에스비",
}
KOREAN_LATIN_LETTERS = {
    "A": "에이",
    "B": "비",
    "C": "씨",
    "D": "디",
    "E": "이",
    "F": "에프",
    "G": "지",
    "H": "에이치",
    "I": "아이",
    "J": "제이",
    "K": "케이",
    "L": "엘",
    "M": "엠",
    "N": "엔",
    "O": "오",
    "P": "피",
    "Q": "큐",
    "R": "알",
    "S": "에스",
    "T": "티",
    "U": "유",
    "V": "브이",
    "W": "더블유",
    "X": "엑스",
    "Y": "와이",
    "Z": "제트",
}
KOREAN_LATIN_DIGITS = {
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
}
KOREAN_SYMBOL_REPLACEMENTS = {
    "&": " 그리고 ",
    "@": " 골뱅이 ",
    "%": " 퍼센트 ",
    "+": " 플러스 ",
    "=": " 는 ",
    "/": " 슬래시 ",
    "\\": " 슬래시 ",
    "#": " 샵 ",
    "$": " 달러 ",
    "€": " 유로 ",
    "₩": " 원 ",
    "°": " 도 ",
    "~": " ",
    "-": " ",
    "—": "…",
    "–": "…",
    "―": "…",
    "−": " ",
    "_": " ",
    "|": " ",
    ":": ", ",
    ";": ", ",
    '"': " ",
    "'": " ",
    "“": " ",
    "”": " ",
    "‘": " ",
    "’": " ",
    "「": " ",
    "」": " ",
    "『": " ",
    "』": " ",
    "*": " ",
    "`": " ",
    "<": " ",
    ">": " ",
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
KOREAN_PUNCT_RE = re.compile(r"\s*([,.!?…])\s*")
KOREAN_LEADING_SENTENCE_FRAGMENT_RE = re.compile(r"^[\s、。！？,.!?]+(?=\s*[\uac00-\ud7a3])")
KOREAN_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-_][A-Za-z]+)*")
KOREAN_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
KOREAN_LONG_CLAUSE_MAX_CHARS = 44
JAPANESE_REMAINING_NON_KANA_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff々〆〤A-Za-z0-9]")
KATAKANA_TO_HIRAGANA = str.maketrans(
    {chr(codepoint): chr(codepoint - 0x60) for codepoint in range(0x30A1, 0x30F7)}
)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _pyopenjtalk_kana(text: str) -> str | None:
    try:
        import pyopenjtalk  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return str(pyopenjtalk.g2p(text, kana=True))
    except Exception:
        return None


def normalize_japanese_kana_text(text: str, *, strict: bool = False) -> JapaneseKanaText:
    """Normalize Japanese model-training/reference text to hiragana pronunciation text."""
    original = str(text or "").strip()
    if not original:
        return JapaneseKanaText(
            text="",
            original_text=original,
            changed=False,
            risk_flags=["japanese_kana_text_empty"],
        )
    converted = _pyopenjtalk_kana(original)
    if converted is None:
        converted = original
    normalized = _normalize_punctuation_and_space(converted).translate(KATAKANA_TO_HIRAGANA)
    risk_flags: list[str] = []
    if normalized != original:
        risk_flags.append("normalized_japanese_kana_text")
    if JAPANESE_REMAINING_NON_KANA_RE.search(normalized):
        risk_flags.append("remaining_non_kana_japanese_text")
        if strict:
            raise JapaneseKanaNormalizationError(
                "Japanese kana normalization left remaining non-kana Japanese text: "
                f"{normalized!r}"
            )
    return JapaneseKanaText(
        text=normalized,
        original_text=original,
        changed=normalized != original,
        risk_flags=_dedupe(risk_flags),
    )


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


def _integer_to_korean(value: int) -> str:
    if value == 0:
        return "영"
    if value >= 100_000:
        return "".join(KOREAN_LATIN_DIGITS[digit] for digit in str(value))
    if value >= 10_000:
        quotient, remainder = divmod(value, 10_000)
        prefix = "" if quotient == 1 else _integer_to_korean(quotient)
        return prefix + "만" + (_integer_to_korean(remainder) if remainder else "")
    pieces: list[str] = []
    for unit_value, unit_name in ((1000, "천"), (100, "백"), (10, "십")):
        quotient, value = divmod(value, unit_value)
        if quotient:
            pieces.append(("" if quotient == 1 else KOREAN_LATIN_DIGITS[str(quotient)]) + unit_name)
    if value:
        pieces.append(KOREAN_LATIN_DIGITS[str(value)])
    return "".join(pieces)


def _korean_digits_to_speech(digits: str) -> str:
    if len(digits) > 1 and digits.startswith("0"):
        return "".join(KOREAN_LATIN_DIGITS[digit] for digit in digits)
    return _integer_to_korean(int(digits or "0"))


def _korean_number_to_speech(token: str) -> str:
    whole, dot, fraction = token.partition(".")
    spoken = _korean_digits_to_speech(whole)
    if dot:
        spoken += "점" + "".join(KOREAN_LATIN_DIGITS[digit] for digit in fraction)
    return spoken


def _normalize_korean_numbers(text: str) -> tuple[str, list[str]]:
    risk_flags: list[str] = []

    def repl(match: re.Match[str]) -> str:
        risk_flags.append("normalized_numeric_token")
        return _korean_number_to_speech(match.group(0))

    return KOREAN_NUMBER_RE.sub(repl, text), risk_flags


def _spell_korean_latin_component(component: str) -> str:
    replacement = KOREAN_LATIN_TOKEN_REPLACEMENTS.get(component.lower())
    if replacement:
        return replacement
    return "".join(KOREAN_LATIN_LETTERS[char.upper()] for char in component if char.upper() in KOREAN_LATIN_LETTERS)


def _korean_latin_token_to_speech(token: str) -> str:
    key = token.lower().replace("_", "-")
    replacement = KOREAN_LATIN_TOKEN_REPLACEMENTS.get(key)
    if replacement:
        return replacement
    components = re.split(r"[-_]+", token)
    return " ".join(_spell_korean_latin_component(component) for component in components if component)


def _normalize_korean_latin_tokens(text: str) -> tuple[str, list[str]]:
    risk_flags: list[str] = []

    def repl(match: re.Match[str]) -> str:
        risk_flags.append("normalized_latin_token")
        return _korean_latin_token_to_speech(match.group(0))

    return KOREAN_LATIN_TOKEN_RE.sub(repl, text), risk_flags


def _normalize_korean_symbols(text: str) -> tuple[str, list[str]]:
    risk_flags: list[str] = []
    out: list[str] = []
    for char in text:
        replacement = KOREAN_SYMBOL_REPLACEMENTS.get(char)
        if replacement is None:
            out.append(char)
            continue
        risk_flags.append("normalized_symbol_token")
        out.append(replacement)
    return "".join(out), risk_flags


def _korean_clause_speech_len(text: str) -> int:
    return sum(1 for char in text if not char.isspace() and char not in ".,!?…")


def _split_korean_clause(clause: str, max_chars: int) -> list[str]:
    if _korean_clause_speech_len(clause) <= max_chars:
        return [clause.strip()]
    words = clause.strip().split(" ")
    if len(words) < 2:
        return [clause.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        word_len = _korean_clause_speech_len(word)
        if current and current_len + word_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = [word]
            current_len = word_len
            continue
        current.append(word)
        current_len += word_len
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


def _split_long_korean_clauses(text: str) -> tuple[str, bool]:
    parts: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in ".!?":
            parts.append("".join(current))
            current = []
    if current:
        parts.append("".join(current))

    changed = False
    out: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped:
            continue
        terminal = stripped[-1] if stripped[-1] in ".!?" else ""
        body = stripped[:-1].strip() if terminal else stripped
        chunks = _split_korean_clause(body, KOREAN_LONG_CLAUSE_MAX_CHARS)
        changed = changed or len(chunks) > 1
        for index, chunk in enumerate(chunks):
            if not chunk:
                continue
            ending = terminal if index == len(chunks) - 1 else ","
            out.append(chunk + ending)
    return " ".join(out), changed


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
    text = re.sub(r"(?:\.{3,}|…+|⋯+)", "…", text)
    text = text.translate(KOREAN_PUNCT_TRANSLATION)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = KOREAN_LEADING_SENTENCE_FRAGMENT_RE.sub("", text)
    text = KOREAN_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"([,.!?])([,.!?])+", r"\1", text)
    text = re.sub(r"([,.!?…])(?=[^\s,.!?…])", r"\1 ", text)
    return SPACE_RE.sub(" ", text).strip()


def normalize_korean_tts_text(text: str) -> NormalizedScriptText:
    without_brackets, cues = extract_bracketed_cues(text)
    risk_flags: list[str] = []
    without_brackets, latin_flags = _normalize_korean_latin_tokens(without_brackets)
    risk_flags.extend(latin_flags)
    without_brackets, number_flags = _normalize_korean_numbers(without_brackets)
    risk_flags.extend(number_flags)
    without_brackets, symbol_flags = _normalize_korean_symbols(without_brackets)
    risk_flags.extend(symbol_flags)
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
    normalized, split_long_clause = _split_long_korean_clauses(normalized)
    if split_long_clause:
        risk_flags.append("split_long_korean_clause")
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
