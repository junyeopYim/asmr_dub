from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from asmr_dub_pipeline.script.countdown import native_korean_count_number


class NumericRenderKind(StrEnum):
    COUNTDOWN_10_TO_1 = "countdown_10_to_1"
    DESCENDING_COUNTDOWN = "descending_countdown"
    NUMERIC_CADENCE = "numeric_cadence"


@dataclass(frozen=True)
class NumericRenderPlan:
    kind: NumericRenderKind
    values: list[int]
    tokens: list[str]
    target_duration_sec: float
    text: str
    text_variant: str
    render_policy: str
    groups: list[list[int]]


def _native_tokens(values: list[int]) -> list[str] | None:
    tokens = [native_korean_count_number(value) for value in values]
    if any(token is None for token in tokens):
        return None
    return [str(token) for token in tokens]


def _is_strict_descending(values: list[int]) -> bool:
    return len(values) >= 3 and all(left - right == 1 for left, right in zip(values, values[1:], strict=False))


def _build_plan(
    *,
    kind: NumericRenderKind,
    values: list[int],
    tokens: list[str],
    target_duration_sec: float,
    text: str,
    text_variant: str,
    render_policy: str,
    groups: list[list[int]],
) -> NumericRenderPlan:
    return NumericRenderPlan(
        kind=kind,
        values=list(values),
        tokens=list(tokens),
        target_duration_sec=target_duration_sec,
        text=text,
        text_variant=text_variant,
        render_policy=render_policy,
        groups=[list(group) for group in groups],
    )


def _period_separated_text(tokens: list[str]) -> str:
    return ". ".join(tokens) + "."


def build_numeric_render_plan(values: list[int], *, target_duration_sec: float) -> NumericRenderPlan | None:
    tokens = _native_tokens(values)
    if tokens is None or len(values) < 3:
        return None
    if values == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        return _build_plan(
            kind=NumericRenderKind.COUNTDOWN_10_TO_1,
            values=values,
            tokens=tokens,
            target_duration_sec=target_duration_sec,
            text=", ".join(tokens) + ".",
            text_variant="native_countdown",
            render_policy="head_single_rest",
            groups=[[10], [9, 8, 7, 6, 5, 4, 3, 2, 1]],
        )
    if _is_strict_descending(values):
        return _build_plan(
            kind=NumericRenderKind.DESCENDING_COUNTDOWN,
            values=values,
            tokens=tokens,
            target_duration_sec=target_duration_sec,
            text=" ".join(tokens) + ".",
            text_variant="native_spaces",
            render_policy="whole_span_guard120_pad350",
            groups=[values],
        )
    return _build_plan(
        kind=NumericRenderKind.NUMERIC_CADENCE,
        values=values,
        tokens=tokens,
        target_duration_sec=target_duration_sec,
        text=_period_separated_text(tokens),
        text_variant="native_periods_no_compact",
        render_policy="whole_span_guard120_pad350",
        groups=[values],
    )
