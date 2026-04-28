from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import Any


class JSONRepairError(ValueError):
    pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    return _FENCE_RE.sub(lambda match: match.group(1), stripped).strip()


def _balanced_objects(text: str) -> list[str]:
    candidates: list[str] = []
    starts = [idx for idx, char in enumerate(text) if char == "{"]
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
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def _extract_largest_balanced_object(text: str) -> str:
    candidates = _balanced_objects(text)
    if not candidates:
        raise JSONRepairError("No JSON object found in response.")
    return max(candidates, key=len)


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def loads_json_dict(text: str, required_keys: set[str] | None = None) -> dict[str, Any]:
    candidates = []
    stripped = _strip_fence(text)
    candidates.append(stripped)
    with suppress(JSONRepairError):
        candidates.append(_extract_largest_balanced_object(stripped))
    candidates.extend(_remove_trailing_commas(candidate) for candidate in list(candidates))
    last_error: Exception | None = None
    required = required_keys or set()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(value, dict):
            last_error = JSONRepairError("Gemma response must be a JSON object.")
            continue
        missing = required - set(value)
        if missing:
            last_error = JSONRepairError(f"Gemma response missing required keys: {sorted(missing)}")
            continue
        return value
    raise JSONRepairError(f"Could not parse JSON object: {last_error}")
