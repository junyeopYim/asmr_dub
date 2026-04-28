from __future__ import annotations

import re

from asmr_dub_pipeline.schemas import JapaneseScript


def estimate_tts_duration(text: str) -> float:
    japanese_chars = len(text.replace(" ", ""))
    return max(0.4, japanese_chars / 7.5)


def _shorten_text(text: str, target_sec: float) -> str:
    budget = max(4, int(target_sec * 7.5))
    shortened = text
    for token in ("ゆっくり……", "ゆっくり、", "もう少しだけ", "すこしだけ", "ねえ、"):
        shortened = shortened.replace(token, "")
    shortened = re.sub(r"(……){2,}", "……", shortened).strip()
    if len(shortened.replace(" ", "")) <= budget:
        return shortened or text
    chunks = re.split(r"(?<=[。！？、])", shortened)
    out = ""
    for chunk in chunks:
        if len((out + chunk).replace(" ", "")) > budget:
            break
        out += chunk
    if not out:
        out = shortened[:budget]
    return out.rstrip("、。！？ ") + "。"


def rewrite_for_duration(
    script: JapaneseScript,
    target_sec: float,
    tolerance: float = 0.15,
) -> JapaneseScript:
    estimated = estimate_tts_duration(script.tts_text)
    if target_sec <= 0 or abs(estimated - target_sec) / target_sec <= tolerance:
        return script
    updated = script.model_copy(deep=True)
    updated.rewrite_count += 1
    if estimated > target_sec:
        updated.tts_text = _shorten_text(updated.tts_text, target_sec)
        updated.ja_text = _shorten_text(updated.ja_text, target_sec)
        updated.risk_flags.append("duration_rewrite_shortened")
    else:
        updated.tts_text = updated.tts_text.rstrip("。") + "……ね。"
        updated.risk_flags.append("duration_rewrite_lengthened")
    updated.expected_tts_duration_sec = estimate_tts_duration(updated.tts_text)
    return updated
