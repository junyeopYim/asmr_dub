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
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_NON_SPEECH_SOURCE_RE = re.compile(r"^[\d\s,.;:!?！？、。・ー~〜…]+$")


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
        parsed[segment_id] = KoreanTranslation(
            ko_literal=str(item.get("ko_literal") or ""),
            ko_natural=str(item.get("ko_natural") or ""),
            notes=_coerce_notes(item.get("notes")),
            confidence=_coerce_confidence(item.get("confidence")),
            model=str(item.get("model") or model),
            batch_id=str(item.get("batch_id") or batch_id),
        )
    return parsed


def _translation_prompt_payload(segments: Sequence[Segment], batch_id: str) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "segments": [
            {
                "segment_id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "duration": segment.duration,
                "source_language": segment.source_script.language if segment.source_script else "ja",
                "source_text": segment.source_script.text if segment.source_script else "",
            }
            for segment in segments
        ],
    }


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
) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {}
    for segment in segments:
        source = (segment.source_script.text if segment.source_script else "").strip()
        if not source:
            continue
        translation = translations.get(segment.id)
        if translation is None:
            continue
        natural = translation.ko_natural.strip()
        if not natural:
            errors.setdefault(segment.id, []).append(f"{segment.id}: ko_natural is empty")
            continue
        if _KANA_RE.search(natural):
            errors.setdefault(segment.id, []).append(
                f"{segment.id}: ko_natural still contains Japanese kana"
            )
        if not _NON_SPEECH_SOURCE_RE.fullmatch(source) and not _HANGUL_RE.search(natural):
            errors.setdefault(segment.id, []).append(
                f"{segment.id}: ko_natural has no Hangul for Japanese source text"
            )
        if len(source) >= 30 and len(natural) / max(len(source), 1) < 0.25:
            errors.setdefault(segment.id, []).append(
                f"{segment.id}: ko_natural is too short for the source text"
            )
    return errors


def build_translate_ko_prompt(segments: Sequence[Segment], batch_id: str) -> str:
    payload = _translation_prompt_payload(segments, batch_id)
    return (
        "Return exactly one valid JSON array and no markdown. Translate Japanese ASMR "
        "source text into Korean. Keep tone gentle and natural. Each array item must "
        "contain segment_id, ko_literal, ko_natural, notes, confidence, model, and batch_id. "
        "confidence must be a JSON number from 0.0 to 1.0, not a label like High. "
        "Use empty notes when no note is needed. ko_natural must be Korean Hangul prose; "
        "do not leave Japanese kana or untranslated Japanese particles in ko_natural. "
        "Do not include explanations, reasoning, analysis, or any text outside the JSON array. "
        "For pure counts or non-speech tokens, preserving digits is allowed. Translate the full "
        "source text without summarizing or dropping clauses.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def build_repair_prompt(
    bad_response: str,
    error: str,
    batch_id: str,
    segments: Sequence[Segment] | None = None,
) -> str:
    input_text = ""
    if segments is not None:
        input_text = (
            "\nOriginal input:\n"
            f"{json.dumps(_translation_prompt_payload(segments, batch_id), ensure_ascii=False)}"
        )
    return (
        "Repair the previous response into exactly one valid JSON array. Each item must contain "
        "segment_id, ko_literal, ko_natural, notes, confidence, model, and batch_id. "
        "confidence must be a JSON number from 0.0 to 1.0, not a label like High. "
        "ko_natural must be Korean Hangul prose with no Japanese kana unless the source is only "
        "numbers or non-speech tokens. Translate the full source text without summarizing. "
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
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.n_predict = n_predict
        self.model = model
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

    def translate_batch(self, segments: Sequence[Segment], batch_id: str) -> dict[str, KoreanTranslation]:
        prompt = build_translate_ko_prompt(segments, batch_id)
        raw_response = ""
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                raw_response = self._complete(
                    prompt
                    if attempt == 0
                    else build_repair_prompt(raw_response, str(last_error), batch_id, segments),
                )
                translations = parse_translation_response(
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
                quality_error_map = _translation_quality_error_map(segments, translations)
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


class MockTranslationClient:
    def __init__(self, model: str = "mock") -> None:
        self.model = model

    def translate_batch(self, segments: Sequence[Segment], batch_id: str) -> dict[str, KoreanTranslation]:
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
