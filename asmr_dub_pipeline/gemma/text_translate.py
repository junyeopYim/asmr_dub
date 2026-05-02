from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
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


def _strip_fence(text: str) -> str:
    return _FENCE_RE.sub(lambda match: match.group(1), text.strip()).strip()


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
            if is_numeric_source and "년" in natural:
                errors.setdefault(segment.id, []).append(
                    f"{segment.id}: pure numeric count was mistranslated as a year"
                )
            if is_numeric_source and "번째" in natural:
                errors.setdefault(segment.id, []).append(
                    f"{segment.id}: pure numeric count was mistranslated as an ordinal"
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
        "segment_ids. Spell numbers and acronyms in Hangul for Korean TTS. Treat pure numeric "
        "source segments as spoken counts; do not preserve raw digits or add year/ordinal wording "
        "unless the source explicitly says year or ordinal. Translate the full source text without "
        "summarizing or dropping clauses.\n"
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
        "numbers and acronyms in Hangul for TTS. Pure numeric source segments are counts, "
        "not years or ordinals, unless the source explicitly says so. Translate "
        "only the items in literal_translations; use target_span.combined_source_text and context "
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
        "For pure numeric source segments, write a spoken count and do not add year or ordinal "
        "wording unless the original source explicitly includes it. Translate the full source "
        "text without summarizing. Preserve the input segment_id boundaries exactly; do not merge, "
        "split, omit, duplicate, or move content between segment_ids. "
        f"Batch id: {batch_id}. Validation error: {error}{input_text}\nPrevious response:\n"
        f"{bad_response[:6000]}"
    )


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
