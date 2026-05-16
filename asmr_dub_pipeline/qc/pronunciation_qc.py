from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from asmr_dub_pipeline.script.numeric_cadence import extract_korean_numeric_values

_HANGUL_SYLLABLE_RE = re.compile(r"[가-힣]")
_KOREAN_NUMBER_CHAR_MAP = str.maketrans(
    {
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
)

PronunciationQCGate = Literal["pass", "warn", "fail", "unavailable", "disabled"]


@dataclass(frozen=True)
class PronunciationQCResult:
    expected_text: str
    transcript: str
    coverage: float
    matched_units: int
    expected_units: int
    observed_units: int
    extra_units: int
    gate: PronunciationQCGate
    issues: list[str]
    backend: str | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coverage"] = round(self.coverage, 6)
        return payload


@dataclass(frozen=True)
class NumericSequenceQCResult:
    expected_text: str
    transcript: str
    expected_values: list[int]
    observed_values: list[int]
    ordered_pass: bool
    contiguous_pass: bool
    missing_values: list[int]
    gate: PronunciationQCGate
    issues: list[str]
    backend: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def normalize_korean_pronunciation_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_KOREAN_NUMBER_CHAR_MAP)
    return "".join(_HANGUL_SYLLABLE_RE.findall(normalized))


def _lcs_length(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        diagonal = 0
        for index, right_char in enumerate(right, start=1):
            above = previous[index]
            if left_char == right_char:
                current.append(diagonal + 1)
            else:
                current.append(max(previous[index], current[-1]))
            diagonal = above
        previous = current
    return previous[-1]


def _looks_like_repetition_overrun(expected: str, observed: str) -> bool:
    if not expected or len(observed) <= len(expected):
        return False
    if len(expected) == 1:
        return observed.startswith(expected)
    repeated = (expected * ((len(observed) // len(expected)) + 2))[: len(observed)]
    return observed == repeated or observed.startswith(expected * 2)


def _looks_like_suffix_contamination(expected: str, observed: str) -> bool:
    return len(expected) == 1 and observed.startswith(expected) and len(observed) > len(expected)


def _apply_short_numeric_asr_alias(expected: str, observed: str) -> str:
    if expected == "일" and observed == "예":
        return "일"
    if expected == "육" and observed in {"유", "류"}:
        return "육"
    return observed


def _subsequence_missing_values(expected: list[int], observed: list[int]) -> list[int]:
    observed_index = 0
    missing: list[int] = []
    for value in expected:
        while observed_index < len(observed) and observed[observed_index] != value:
            observed_index += 1
        if observed_index >= len(observed):
            missing.append(value)
        else:
            observed_index += 1
    return missing


def _is_contiguous_subsequence(expected: list[int], observed: list[int]) -> bool:
    if not expected:
        return True
    if len(observed) < len(expected):
        return False
    width = len(expected)
    return any(observed[index : index + width] == expected for index in range(len(observed) - width + 1))


def evaluate_numeric_sequence_text(
    expected_text: str,
    transcript: str,
    *,
    min_values: int = 3,
    require_contiguous: bool = True,
    backend: str | None = None,
) -> NumericSequenceQCResult:
    expected_values = extract_korean_numeric_values(expected_text)
    observed_values = extract_korean_numeric_values(transcript)
    issues: list[str] = []
    if len(expected_values) < min_values:
        return NumericSequenceQCResult(
            expected_text=expected_text,
            transcript=transcript,
            expected_values=expected_values,
            observed_values=observed_values,
            ordered_pass=True,
            contiguous_pass=True,
            missing_values=[],
            gate="unavailable",
            issues=["numeric_sequence_too_short"],
            backend=backend,
        )
    if not observed_values:
        return NumericSequenceQCResult(
            expected_text=expected_text,
            transcript=transcript,
            expected_values=expected_values,
            observed_values=observed_values,
            ordered_pass=False,
            contiguous_pass=False,
            missing_values=list(expected_values),
            gate="fail",
            issues=["empty_numeric_transcript"],
            backend=backend,
        )

    missing_values = _subsequence_missing_values(expected_values, observed_values)
    ordered_pass = not missing_values
    contiguous_pass = _is_contiguous_subsequence(expected_values, observed_values)
    if not ordered_pass:
        issues.append("numeric_sequence_order_mismatch")
        issues.append("numeric_sequence_missing_values")
    if require_contiguous and not contiguous_pass:
        issues.append("numeric_sequence_contiguous_mismatch")
    gate: PronunciationQCGate = "pass" if ordered_pass and (contiguous_pass or not require_contiguous) else "fail"
    return NumericSequenceQCResult(
        expected_text=expected_text,
        transcript=transcript,
        expected_values=expected_values,
        observed_values=observed_values,
        ordered_pass=ordered_pass,
        contiguous_pass=contiguous_pass,
        missing_values=missing_values,
        gate=gate,
        issues=issues,
        backend=backend,
    )


def evaluate_pronunciation_text(
    expected_text: str,
    transcript: str,
    *,
    pass_coverage: float = 0.82,
    warn_coverage: float = 0.62,
    max_observed_unit_ratio: float | None = 1.8,
    max_extra_units: int | None = 1,
    backend: str | None = None,
) -> PronunciationQCResult:
    expected = normalize_korean_pronunciation_text(expected_text)
    observed = normalize_korean_pronunciation_text(transcript)
    observed = _apply_short_numeric_asr_alias(expected, observed)
    observed_units = len(observed)
    issues: list[str] = []
    if not expected:
        return PronunciationQCResult(
            expected_text=expected_text,
            transcript=transcript,
            coverage=1.0,
            matched_units=0,
            expected_units=0,
            observed_units=observed_units,
            extra_units=0,
            gate="unavailable",
            issues=["empty_expected_text"],
            backend=backend,
        )
    if not observed:
        return PronunciationQCResult(
            expected_text=expected_text,
            transcript=transcript,
            coverage=0.0,
            matched_units=0,
            expected_units=len(expected),
            observed_units=0,
            extra_units=0,
            gate="fail",
            issues=["empty_transcript"],
            backend=backend,
        )
    matched = _lcs_length(expected, observed)
    coverage = matched / len(expected)
    extra_units = max(0, observed_units - len(expected))
    observed_ratio = observed_units / len(expected)
    observed_too_long = False
    if max_extra_units is not None and extra_units > int(max_extra_units):
        observed_too_long = True
    if max_observed_unit_ratio is not None and observed_ratio > float(max_observed_unit_ratio):
        observed_too_long = True

    if observed_too_long and coverage >= warn_coverage:
        gate: PronunciationQCGate = "fail"
        issues.append("observed_pronunciation_too_long")
        if _looks_like_repetition_overrun(expected, observed):
            issues.append("repetition_overrun")
        if _looks_like_suffix_contamination(expected, observed):
            issues.append("suffix_contamination")
    elif _looks_like_suffix_contamination(expected, observed) and coverage >= warn_coverage:
        gate = "fail"
        issues.append("suffix_contamination")
    elif coverage >= pass_coverage:
        gate: PronunciationQCGate = "pass"
    elif coverage >= warn_coverage:
        gate = "warn"
        issues.append("low_pronunciation_coverage")
    else:
        gate = "fail"
        issues.append("pronunciation_coverage_below_threshold")
    return PronunciationQCResult(
        expected_text=expected_text,
        transcript=transcript,
        coverage=coverage,
        matched_units=matched,
        expected_units=len(expected),
        observed_units=observed_units,
        extra_units=extra_units,
        gate=gate,
        issues=issues,
        backend=backend,
    )


def evaluate_pronunciation_chunks(
    expected_text: str,
    chunks: Sequence[Any],
    *,
    pass_coverage: float = 0.82,
    warn_coverage: float = 0.62,
    max_observed_unit_ratio: float | None = 1.8,
    max_extra_units: int | None = 1,
    backend: str | None = None,
) -> PronunciationQCResult:
    transcript = " ".join(str(getattr(chunk, "text", "")).strip() for chunk in chunks).strip()
    return evaluate_pronunciation_text(
        expected_text,
        transcript,
        pass_coverage=pass_coverage,
        warn_coverage=warn_coverage,
        max_observed_unit_ratio=max_observed_unit_ratio,
        max_extra_units=max_extra_units,
        backend=backend,
    )
