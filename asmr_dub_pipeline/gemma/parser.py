from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from pydantic import ValidationError

from .json_repair import JSONRepairError, loads_json_dict
from .schemas import TASK_REQUIRED_KEYS, TaskName, validate_gemma_task_response


def _candidate_texts(payload: Mapping[str, Any]) -> Iterable[str]:
    for key in ("content", "text", "response", "generated_text", "output_text"):
        value = payload.get(key)
        if isinstance(value, str):
            yield value
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            message = choice.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                yield message["content"]
            if isinstance(choice.get("text"), str):
                yield choice["text"]


def parse_gemma_task_response(task: TaskName, raw: Any) -> dict[str, Any]:
    """Parse and validate a Gemma task response against the task schema."""
    if isinstance(raw, str):
        required_keys = None if task == "audio_style" else TASK_REQUIRED_KEYS[task]
        payload = loads_json_dict(raw, required_keys=required_keys)
        try:
            return validate_gemma_task_response(task, payload)
        except ValidationError as exc:
            raise JSONRepairError(f"Gemma {task} response failed schema validation: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise JSONRepairError("Gemma response must be a JSON object or text containing one.")

    try:
        return validate_gemma_task_response(task, dict(raw))
    except ValidationError as direct_error:
        last_error: Exception = direct_error

    for text in _candidate_texts(raw):
        try:
            payload = loads_json_dict(text, required_keys=TASK_REQUIRED_KEYS[task])
            return validate_gemma_task_response(task, payload)
        except (JSONRepairError, ValidationError) as exc:
            last_error = exc

    raise JSONRepairError(f"Gemma {task} response did not match the JSON contract: {last_error}")
