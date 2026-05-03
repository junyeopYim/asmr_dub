from __future__ import annotations

import base64
import json
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx

from asmr_dub_pipeline.schemas import KoreanTranslation, Segment


class GemmaTextTranslationError(RuntimeError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL)
_KANA_RE = re.compile(r"[\u3040-\u30ffー]")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE = re.compile(r"\d")
_NON_SPEECH_SOURCE_RE = re.compile(r"^[\d\s,.;:!?！？、。・ー~〜…]+$")
_NUMERIC_SOURCE_RE = re.compile(r"^\s*\d+(?:[\s,]+\d+)*\s*$")
_UNSAFE_TTS_SYMBOL_RE = re.compile(r"[—–―“”‘’「」『』]")
_ASR_REVIEW_COMPARE_DROP_RE = re.compile(r"[\s,.;:!?！？、。・ー~〜…'\"“”‘’「」『』（）()]+")


def _strip_fence(text: str) -> str:
    return _FENCE_RE.sub(lambda match: match.group(1), text.strip()).strip()


def _normalize_asr_review_compare_text(text: str) -> str:
    return _ASR_REVIEW_COMPARE_DROP_RE.sub("", text).strip().casefold()


def _balanced_arrays(text: str) -> list[str]:
    candidates: list[str] = []
    starts = [idx for idx, char in enumerate(text) if char == "["]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _load_translation_items(text: str) -> list[dict[str, Any]]:
    stripped = _strip_fence(text)
    candidates = [stripped, _remove_trailing_commas(stripped)]
    with suppress(Exception):
        largest = max(_balanced_arrays(stripped), key=len)
        candidates.extend([largest, _remove_trailing_commas(largest)])
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(value, Mapping) and isinstance(value.get("translations"), list):
            value = value["translations"]
        if not isinstance(value, list):
            last_error = GemmaTextTranslationError("Gemma text response must be a JSON array.")
            continue
        if all(isinstance(item, Mapping) for item in value):
            return [dict(item) for item in value]
        last_error = GemmaTextTranslationError("Gemma text response array items must be objects.")
    raise GemmaTextTranslationError(f"Could not parse translation JSON array: {last_error}")


def _coerce_confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower()
        if not text:
            return None
        labels = {
            "high": 0.9,
            "높음": 0.9,
            "medium": 0.6,
            "med": 0.6,
            "moderate": 0.6,
            "보통": 0.6,
            "low": 0.3,
            "낮음": 0.3,
        }
        if text in labels:
            return labels[text]
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match is None:
            return None
        number = float(match.group(0))
        if "%" in text or number > 1.0:
            number /= 100.0
    return max(0.0, min(1.0, number))


def _coerce_notes(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        note = value.strip()
        return [] if not note or note.lower() in {"none", "n/a", "null"} else [note]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(note) for note in value if str(note).strip()]
    return [str(value)]


def parse_translation_response(
    text: str,
    *,
    batch_id: str,
    model: str,
) -> dict[str, KoreanTranslation]:
    parsed: dict[str, KoreanTranslation] = {}
    for item in _load_translation_items(text):
        segment_id = str(item.get("segment_id") or "").strip()
        if not segment_id:
            raise GemmaTextTranslationError("Translation item missing segment_id.")
        natural = str(item.get("ko_natural") or "")
        parsed[segment_id] = KoreanTranslation(
            ko_literal=str(item.get("ko_literal") or natural),
            ko_natural=natural,
            notes=_coerce_notes(item.get("notes")),
            confidence=_coerce_confidence(item.get("confidence")),
            model=str(item.get("model") or model),
            batch_id=str(item.get("batch_id") or batch_id),
        )
    return parsed


def _coerce_review_decision(value: Any) -> str:
    decision = str(value or "").strip().lower().replace("-", "_")
    if decision in {"replace", "keep", "manual_review"}:
        return decision
    if decision in {"manual", "review", "needs_manual_review"}:
        return "manual_review"
    return "manual_review"


def parse_asr_review_response(
    text: str,
    *,
    batch_id: str,
    model: str,
) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    for item in _load_translation_items(text):
        chunk_id = str(item.get("chunk_id") or item.get("segment_id") or "").strip()
        if not chunk_id:
            raise GemmaTextTranslationError("ASR review item missing chunk_id.")
        selected_candidate_id = str(
            item.get("selected_candidate_id") or item.get("selected_id") or item.get("candidate_id") or ""
        ).strip()
        if not selected_candidate_id:
            raise GemmaTextTranslationError(f"ASR review item missing selected_candidate_id: {chunk_id}")
        parsed[chunk_id] = {
            "chunk_id": chunk_id,
            "decision": _coerce_review_decision(item.get("decision")),
            "selected_candidate_id": selected_candidate_id,
            "confidence": _coerce_confidence(item.get("confidence")),
            "heard_text": str(item.get("heard_text") or "").strip(),
            "reason": str(item.get("reason") or "").strip(),
            "risk_terms": _coerce_notes(item.get("risk_terms")),
            "model": str(item.get("model") or model),
            "batch_id": str(item.get("batch_id") or batch_id),
        }
    return parsed


def _parse_literal_response(
    text: str,
    *,
    batch_id: str,
    model: str,
) -> dict[str, KoreanTranslation]:
    parsed: dict[str, KoreanTranslation] = {}
    for item in _load_translation_items(text):
        segment_id = str(item.get("segment_id") or "").strip()
        if not segment_id:
            raise GemmaTextTranslationError("Translation item missing segment_id.")
        literal = str(item.get("ko_literal") or item.get("ko_natural") or "")
        parsed[segment_id] = KoreanTranslation(
            ko_literal=literal,
            ko_natural=literal,
            notes=_coerce_notes(item.get("notes")),
            confidence=_coerce_confidence(item.get("confidence")),
            model=str(item.get("model") or model),
            batch_id=str(item.get("batch_id") or batch_id),
        )
    return parsed


def _segment_prompt_item(segment: Segment) -> dict[str, Any]:
    return {
        "segment_id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "duration": segment.duration,
        "source_language": segment.source_script.language if segment.source_script else "ja",
        "source_text": segment.source_script.text if segment.source_script else "",
    }


def _translation_prompt_payload(
    segments: Sequence[Segment],
    batch_id: str,
    context_segments: Sequence[Segment] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "segments": [_segment_prompt_item(segment) for segment in segments],
        "target_span": {
            "segment_ids": [segment.id for segment in segments],
            "combined_source_text": "\n".join(
                f"{segment.id}: {segment.source_script.text.strip()}"
                for segment in segments
                if segment.source_script and segment.source_script.text.strip()
            ),
        },
    }
    if context_segments:
        target_ids = {segment.id for segment in segments}
        payload["context"] = [
            {
                **_segment_prompt_item(segment),
                "is_target": segment.id in target_ids,
            }
            for segment in context_segments
            if segment.source_script and segment.source_script.text.strip()
        ]
    return payload


def _translation_quality_errors(
    segments: Sequence[Segment],
    translations: Mapping[str, KoreanTranslation],
) -> list[str]:
    return [
        error
        for segment_errors in _translation_quality_error_map(segments, translations).values()
        for error in segment_errors
    ]


def _translation_quality_error_map(
    segments: Sequence[Segment],
    translations: Mapping[str, KoreanTranslation],
    *,
    field: str = "ko_natural",
) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {}
    for segment in segments:
        source = (segment.source_script.text if segment.source_script else "").strip()
        if not source:
            continue
        translation = translations.get(segment.id)
        if translation is None:
            continue
        natural = getattr(translation, field).strip()
        if not natural:
            errors.setdefault(segment.id, []).append(f"{segment.id}: {field} is empty")
            continue
        if _KANA_RE.search(natural):
            errors.setdefault(segment.id, []).append(
                f"{segment.id}: {field} still contains Japanese kana"
            )
        if _CJK_RE.search(natural):
            errors.setdefault(segment.id, []).append(
                f"{segment.id}: {field} still contains untranslated CJK characters"
            )
        is_numeric_source = bool(_NUMERIC_SOURCE_RE.fullmatch(source))
        if field == "ko_natural":
            if _LATIN_RE.search(natural):
                errors.setdefault(segment.id, []).append(
                    f"{segment.id}: {field} contains Latin letters; spell acronyms in Hangul"
                )
            if _DIGIT_RE.search(natural):
                errors.setdefault(segment.id, []).append(
                    f"{segment.id}: {field} contains raw digits; spell numbers in Korean"
                )
            if _UNSAFE_TTS_SYMBOL_RE.search(natural):
                errors.setdefault(segment.id, []).append(
                    f"{segment.id}: {field} contains TTS-unsafe punctuation"
                )
        if (
            (field == "ko_natural" and is_numeric_source)
            or not _NON_SPEECH_SOURCE_RE.fullmatch(source)
        ) and not _HANGUL_RE.search(natural):
            errors.setdefault(segment.id, []).append(
                f"{segment.id}: {field} has no Hangul for Japanese source text"
            )
        if len(source) >= 30 and len(natural) / max(len(source), 1) < 0.25:
            errors.setdefault(segment.id, []).append(
                f"{segment.id}: {field} is too short for the source text"
            )
    return errors


def build_translate_ko_prompt(
    segments: Sequence[Segment],
    batch_id: str,
    context_segments: Sequence[Segment] | None = None,
) -> str:
    payload = _translation_prompt_payload(segments, batch_id, context_segments)
    return (
        "Return exactly one valid JSON array and no markdown. Translate Japanese ASMR source "
        "text into Korean. Keep tone gentle, natural, and conversational in polite spoken "
        "Korean suitable for ASMR TTS. Each array item must contain only "
        "segment_id and ko_natural. ko_natural must be Korean Hangul prose; do not leave "
        "Japanese kana or untranslated Japanese particles in ko_natural. Do not include "
        "explanations, reasoning, analysis, notes, confidence, model, batch_id, or any text "
        "outside the JSON array. Translate only the items in segments; use context only to "
        "resolve names, pronouns, omitted subjects, and tone. Read target_span.combined_source_text "
        "first to understand the discourse across adjacent segments, but keep exactly one output "
        "item per input segment_id; do not merge, split, omit, duplicate, or move content between "
        "segment_ids. Spell numbers and acronyms in Hangul for Korean TTS. For numeric-only "
        "or digit-heavy source segments, infer the intended reading from target_span and context: "
        "natural counting may use 하나, 둘, 셋; digit/code/phone-like reading may use 일, 이, "
        "삼 or 공; quantities, years, ordinals, time, and measurements should use the idiomatic "
        "Korean form for that context. Never leave raw digits. Translate the full source text "
        "without summarizing or dropping clauses.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_literal_translate_prompt(
    segments: Sequence[Segment],
    batch_id: str,
    context_segments: Sequence[Segment] | None = None,
) -> str:
    payload = _translation_prompt_payload(segments, batch_id, context_segments)
    return (
        "Return exactly one valid JSON array and no markdown. First pass: translate the "
        "Japanese ASMR source text into faithful Korean literal meaning. Each array item "
        "must contain only segment_id and ko_literal. ko_literal must be Korean Hangul prose; "
        "do not leave Japanese kana or untranslated Japanese particles. Do not polish into "
        "final TTS copy yet. Translate only the items in segments; use context only to resolve "
        "names, pronouns, omitted subjects, and tone. Read target_span.combined_source_text "
        "first for adjacent-segment context, but keep exactly one output item per input "
        "segment_id; do not merge, split, omit, duplicate, or move content between segment_ids. "
        "Preserve all clauses without summary.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_naturalize_ko_prompt(
    segments: Sequence[Segment],
    batch_id: str,
    literal_translations: Mapping[str, KoreanTranslation],
    context_segments: Sequence[Segment] | None = None,
) -> str:
    payload = _translation_prompt_payload(segments, batch_id, context_segments)
    payload["literal_translations"] = [
        {
            "segment_id": segment.id,
            "source_text": segment.source_script.text if segment.source_script else "",
            "ko_literal": literal_translations[segment.id].ko_literal,
        }
        for segment in segments
        if segment.id in literal_translations
    ]
    return (
        "Return exactly one valid JSON array and no markdown. Second pass: rewrite the "
        "literal Korean translations into natural Korean ASMR TTS dialogue. Each array item "
        "must contain only segment_id and ko_natural. ko_natural must be gentle, polite, "
        "spoken Korean that can be read aloud naturally; split stiff written phrasing into "
        "short breath-friendly wording when useful. Keep the meaning of ko_literal and the "
        "source text; do not add new events, omit clauses, or leave Japanese kana. Spell "
        "numbers and acronyms in Hangul for TTS. For numeric-only or digit-heavy source segments, "
        "infer the intended reading from target_span and context: natural counting may use 하나, "
        "둘, 셋; digit/code/phone-like reading may use 일, 이, 삼 or 공; quantities, years, "
        "ordinals, time, and measurements should use the idiomatic Korean form for that context. "
        "Never leave raw digits. Translate only the items in literal_translations; use "
        "target_span.combined_source_text and context "
        "only for continuity of names, pronouns, register, and tone. Keep exactly one output item "
        "per input segment_id; do not merge, split, omit, duplicate, or move content between "
        "segment_ids.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_repair_prompt(
    bad_response: str,
    error: str,
    batch_id: str,
    segments: Sequence[Segment] | None = None,
    context_segments: Sequence[Segment] | None = None,
    output_field: str = "ko_natural",
) -> str:
    input_text = ""
    if segments is not None:
        input_text = (
            "\nOriginal input:\n"
            f"{json.dumps(_translation_prompt_payload(segments, batch_id, context_segments), ensure_ascii=False)}"
        )
    return (
        "Repair the previous response into exactly one valid JSON array. Each item must contain "
        f"only segment_id and {output_field}. {output_field} must be Korean Hangul prose in polite "
        "spoken conversational style with no Japanese kana, untranslated CJK, raw Latin letters, "
        "raw digits, or long dash/quote punctuation. Spell acronyms and numbers in Hangul. "
        "For numeric-only or digit-heavy source segments, infer the intended reading from the "
        "original input context, choosing natural counting, digit/code reading, quantity, year, "
        "ordinal, time, or measurement wording as appropriate. Translate the full source "
        "text without summarizing. Preserve the input segment_id boundaries exactly; do not merge, "
        "split, omit, duplicate, or move content between segment_ids. "
        f"Batch id: {batch_id}. Validation error: {error}{input_text}\nPrevious response:\n"
        f"{bad_response[:6000]}"
    )


def build_asr_review_prompt(items: Sequence[Mapping[str, Any]], batch_id: str) -> str:
    prompt_items = [
        {key: value for key, value in item.items() if key not in {"audio_clip_path", "audio_clip"}}
        for item in items
    ]
    payload = {
        "batch_id": batch_id,
        "task": "japanese_asr_audio_candidate_review",
        "domain": "user-authorized Japanese ASMR transcription",
        "items": prompt_items,
    }
    return (
        "Return exactly one valid JSON array and no markdown. You are reviewing Japanese ASR "
        "text candidates for a user-authorized ASMR dubbing workflow with the matching audio "
        "attached to this request. Treat the audio as the primary evidence. First, internally "
        "transcribe the spoken Japanese in the audio. Then choose the candidate whose meaning and "
        "wording are closest to that internal transcript. Candidate labels such as original, "
        "repair, replacement, and ASR confidence are only supporting metadata; do not keep the "
        "original merely because its confidence is high. Ignore harmless spacing and punctuation "
        "differences. Do not invent unrelated Japanese text beyond a short heard_text transcript. "
        "For each input item, write heard_text as the Japanese you hear in the audio, then choose "
        "only one candidate_id from candidates. If selected_candidate_id is not 'original', decision must "
        "be 'replace'. Use decision='replace' when a non-original candidate better matches the "
        "audio than the original transcription; use decision='keep' only when selected_candidate_id "
        "is 'original' and original is closest to the audio. Use decision='manual_review' only "
        "when no candidate is close to the audio, and in that case selected_candidate_id must be "
        "'original'. Treat manual_review as a last resort when every candidate is "
        "unusable and the uncertainty changes translation meaning. Each output item must contain "
        "only chunk_id, heard_text, decision, selected_candidate_id, confidence, reason, and risk_terms. "
        "confidence must be 0.0 to 1.0. risk_terms must be a short array of suspicious source "
        "terms, or an empty array. Prefer ASMR-domain readings such as 絶頂, 媚薬, 耳舐め, 暗示, "
        "快感, 10数える, メスイキ, クリトリス, おまんこ, and 出会いアプリ only when the audio "
        "and surrounding context support them.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_asr_review_repair_prompt(
    bad_response: str,
    error: str,
    batch_id: str,
    items: Sequence[Mapping[str, Any]],
) -> str:
    return (
        "Repair the previous ASR review response into exactly one valid JSON array. Each item "
        "must contain only chunk_id, heard_text, decision, selected_candidate_id, confidence, reason, and "
        "risk_terms. heard_text may be an empty string if absent from the previous response. "
        "selected_candidate_id must exactly match one candidate_id from the original "
        "input item. If selected_candidate_id is not 'original', decision must be 'replace'. "
        "If decision is 'keep' or 'manual_review', selected_candidate_id must be 'original'. "
        "If decision is 'replace', selected_candidate_id must not be 'original'. Do not invent "
        "candidate text. "
        f"Batch id: {batch_id}. Validation error: {error}\nOriginal input:\n"
        f"{json.dumps({'items': list(items)}, ensure_ascii=False)}\nPrevious response:\n"
        f"{bad_response[:6000]}"
    )


def _asr_review_consistency_errors(
    reviews: Mapping[str, Mapping[str, Any]],
    candidate_ids_by_chunk: Mapping[str, set[str]],
) -> list[str]:
    invalid: list[str] = []
    for chunk_id, review in reviews.items():
        selected_id = str(review.get("selected_candidate_id") or "")
        decision = str(review.get("decision") or "")
        if selected_id not in candidate_ids_by_chunk.get(chunk_id, set()):
            invalid.append(f"{chunk_id}: invalid selected_candidate_id {selected_id}")
        if decision == "replace" and selected_id == "original":
            invalid.append(f"{chunk_id}: decision replace requires a non-original candidate")
        if decision in {"keep", "manual_review"} and selected_id != "original":
            invalid.append(f"{chunk_id}: decision {decision} requires selected_candidate_id original")
    return invalid


def _align_asr_review_with_heard_text(
    reviews: dict[str, dict[str, Any]],
    items: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    candidate_text_by_chunk: dict[str, dict[str, str]] = {}
    for item in items:
        chunk_id = str(item.get("chunk_id") or "")
        candidate_text_by_chunk[chunk_id] = {
            str(candidate.get("candidate_id")): str(candidate.get("text") or "")
            for candidate in item.get("candidates", [])
            if isinstance(candidate, Mapping)
        }
    aligned: dict[str, dict[str, Any]] = {}
    for chunk_id, review in reviews.items():
        heard_text = str(review.get("heard_text") or "")
        normalized_heard = _normalize_asr_review_compare_text(heard_text)
        if not normalized_heard:
            aligned[chunk_id] = review
            continue
        candidate_texts = candidate_text_by_chunk.get(chunk_id, {})
        original_text = candidate_texts.get("original", "")
        normalized_original = _normalize_asr_review_compare_text(original_text)
        if normalized_heard == normalized_original:
            aligned[chunk_id] = review
            continue
        matched_id = next(
            (
                candidate_id
                for candidate_id, text in candidate_texts.items()
                if candidate_id != "original"
                and _normalize_asr_review_compare_text(text) == normalized_heard
            ),
            None,
        )
        if matched_id is None:
            aligned[chunk_id] = review
            continue
        confidence = review.get("confidence")
        aligned[chunk_id] = {
            **review,
            "decision": "replace",
            "selected_candidate_id": matched_id,
            "confidence": confidence if confidence is not None else 0.9,
            "reason": str(review.get("reason") or "heard_text matched a non-original candidate"),
        }
    return aligned


class LlamaServerTranslationClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_sec: float = 180.0,
        retries: int = 1,
        n_predict: int = 2048,
        model: str = "gemma4",
        two_pass: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.n_predict = n_predict
        self.model = model
        self.two_pass = two_pass
        self.client = client or httpx.Client(timeout=timeout_sec)

    def _complete(self, prompt: str, *, n_predict: int | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": n_predict or self.n_predict,
            "stream": False,
        }
        response = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        if response.status_code >= 400:
            raise GemmaTextTranslationError(
                f"Gemma text server error {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
        if isinstance(data.get("content"), str):
            return data["content"]
        raise GemmaTextTranslationError("Gemma text server response did not include content.")

    def _complete_with_input_audio(
        self,
        prompt: str,
        audio_path: str | Path,
        *,
        audio_format: str = "wav",
        n_predict: int | None = None,
    ) -> str:
        try:
            audio_data = Path(audio_path).read_bytes()
        except OSError as exc:
            raise GemmaTextTranslationError(f"Could not read ASR review audio clip: {audio_path}") from exc
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_data).decode("ascii"),
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": n_predict or self.n_predict,
            "stream": False,
        }
        response = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        if response.status_code >= 400:
            raise GemmaTextTranslationError(
                f"Gemma text server error {response.status_code}: {response.text[:500]}"
            )
        data = response.json()
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
        if isinstance(data.get("content"), str):
            return data["content"]
        raise GemmaTextTranslationError("Gemma text server response did not include content.")

    def _translate_once(
        self,
        *,
        segments: Sequence[Segment],
        batch_id: str,
        prompt: str,
        parser: Any,
        output_field: str,
        context_segments: Sequence[Segment] | None = None,
    ) -> dict[str, KoreanTranslation]:
        raw_response = ""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                raw_response = self._complete(
                    prompt
                    if attempt == 0
                    else build_repair_prompt(
                        raw_response,
                        str(last_error),
                        batch_id,
                        segments,
                        context_segments=context_segments,
                        output_field=output_field,
                    ),
                )
                translations = parser(
                    raw_response,
                    batch_id=batch_id,
                    model=self.model,
                )
                expected_ids = {segment.id for segment in segments}
                translations = {
                    segment_id: translation
                    for segment_id, translation in translations.items()
                    if segment_id in expected_ids
                }
                quality_error_map = _translation_quality_error_map(
                    segments,
                    translations,
                    field=output_field,
                )
                if quality_error_map:
                    quality_errors = [
                        error
                        for segment_errors in quality_error_map.values()
                        for error in segment_errors
                    ]
                    if attempt < self.retries:
                        raise GemmaTextTranslationError("; ".join(quality_errors[:5]))
                    if len(segments) > 1:
                        valid_translations = {
                            segment_id: translation
                            for segment_id, translation in translations.items()
                            if segment_id not in quality_error_map
                        }
                        if valid_translations:
                            return valid_translations
                    raise GemmaTextTranslationError("; ".join(quality_errors[:5]))
                return translations
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break
            except (ValueError, GemmaTextTranslationError) as exc:
                last_error = exc
                if attempt < self.retries:
                    continue
                break
        raise GemmaTextTranslationError(f"Gemma text translation failed: {last_error}")

    def _translate_batch_single_pass(
        self,
        segments: Sequence[Segment],
        batch_id: str,
        context_segments: Sequence[Segment] | None = None,
    ) -> dict[str, KoreanTranslation]:
        return self._translate_once(
            segments=segments,
            batch_id=batch_id,
            prompt=build_translate_ko_prompt(segments, batch_id, context_segments),
            parser=parse_translation_response,
            output_field="ko_natural",
            context_segments=context_segments,
        )

    def _translate_batch_two_pass(
        self,
        segments: Sequence[Segment],
        batch_id: str,
        context_segments: Sequence[Segment] | None = None,
    ) -> dict[str, KoreanTranslation]:
        literal_translations = self._translate_once(
            segments=segments,
            batch_id=f"{batch_id}_literal",
            prompt=build_literal_translate_prompt(segments, batch_id, context_segments),
            parser=_parse_literal_response,
            output_field="ko_literal",
            context_segments=context_segments,
        )
        natural_translations = self._translate_once(
            segments=segments,
            batch_id=f"{batch_id}_natural",
            prompt=build_naturalize_ko_prompt(
                segments,
                batch_id,
                literal_translations,
                context_segments,
            ),
            parser=parse_translation_response,
            output_field="ko_natural",
            context_segments=context_segments,
        )
        return {
            segment_id: natural.model_copy(
                update={
                    "ko_literal": literal_translations[segment_id].ko_literal,
                    "batch_id": batch_id,
                }
            )
            for segment_id, natural in natural_translations.items()
            if segment_id in literal_translations
        }

    def translate_batch(
        self,
        segments: Sequence[Segment],
        batch_id: str,
        context_segments: Sequence[Segment] | None = None,
    ) -> dict[str, KoreanTranslation]:
        if not self.two_pass:
            return self._translate_batch_single_pass(segments, batch_id, context_segments)
        return self._translate_batch_two_pass(segments, batch_id, context_segments)

    def review_asr_candidates_with_audio(
        self,
        items: Sequence[Mapping[str, Any]],
        batch_id: str,
        audio_path: str | Path,
    ) -> dict[str, dict[str, Any]]:
        if len(items) != 1:
            raise GemmaTextTranslationError("Gemma ASR audio review expects exactly one item per request.")
        candidate_ids_by_chunk = {
            str(item.get("chunk_id")): {
                str(candidate.get("candidate_id"))
                for candidate in item.get("candidates", [])
                if isinstance(candidate, Mapping)
            }
            for item in items
        }
        prompt = build_asr_review_prompt(items, batch_id)
        raw_response = ""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if not raw_response:
                    raw_response = self._complete_with_input_audio(prompt, audio_path)
                else:
                    raw_response = self._complete(
                        build_asr_review_repair_prompt(
                            raw_response,
                            str(last_error),
                            batch_id,
                            items,
                        )
                    )
                reviews = parse_asr_review_response(
                    raw_response,
                    batch_id=batch_id,
                    model=self.model,
                )
                reviews = _align_asr_review_with_heard_text(reviews, items)
                invalid = _asr_review_consistency_errors(reviews, candidate_ids_by_chunk)
                if invalid:
                    raise GemmaTextTranslationError("; ".join(invalid[:5]))
                return {
                    chunk_id: review
                    for chunk_id, review in reviews.items()
                    if chunk_id in candidate_ids_by_chunk
                }
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break
            except (ValueError, GemmaTextTranslationError) as exc:
                last_error = exc
                if attempt < self.retries:
                    continue
                break
        raise GemmaTextTranslationError(f"Gemma ASR audio review failed: {last_error}")


class MockTranslationClient:
    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def translate_batch(
        self,
        segments: Sequence[Segment],
        batch_id: str,
        context_segments: Sequence[Segment] | None = None,
    ) -> dict[str, KoreanTranslation]:
        result: dict[str, KoreanTranslation] = {}
        for segment in segments:
            result[segment.id] = KoreanTranslation(
                ko_literal="직역: 원문 의미를 한국어로 옮긴 문장입니다.",
                ko_natural="자연 번역: 부드럽게 속삭여 드릴게요.",
                notes=[],
                confidence=0.99,
                model=self.model,
                batch_id=batch_id,
            )
        return result

    def review_asr_candidates_for_mock(
        self,
        items: Sequence[Mapping[str, Any]],
        batch_id: str,
    ) -> dict[str, dict[str, Any]]:
        reviews: dict[str, dict[str, Any]] = {}
        for item in items:
            chunk_id = str(item.get("chunk_id") or "")
            candidates = [
                candidate for candidate in item.get("candidates", []) if isinstance(candidate, Mapping)
            ]
            selected = next(
                (
                    candidate
                    for candidate in candidates
                    if str(candidate.get("candidate_id")) != "original"
                ),
                candidates[0] if candidates else {},
            )
            selected_id = str(selected.get("candidate_id") or "original")
            reviews[chunk_id] = {
                "chunk_id": chunk_id,
                "decision": "replace" if selected_id != "original" else "keep",
                "selected_candidate_id": selected_id,
                "confidence": 0.99,
                "reason": "mock ASR review selected the first non-original candidate.",
                "risk_terms": [],
                "model": self.model,
                "batch_id": batch_id,
            }
        return reviews

    def review_asr_candidates_with_audio(
        self,
        items: Sequence[Mapping[str, Any]],
        batch_id: str,
        audio_path: str | Path,
    ) -> dict[str, dict[str, Any]]:
        _ = audio_path
        return self.review_asr_candidates_for_mock(items, batch_id)
