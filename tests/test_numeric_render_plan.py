from asmr_dub_pipeline.script.numeric_render_plan import (
    NumericRenderKind,
    build_numeric_render_plan,
)


def test_builds_native_whole_span_plan_for_ascending_count_run() -> None:
    plan = build_numeric_render_plan([3, 4, 5, 6, 7, 8, 9, 10], target_duration_sec=3.42)
    assert plan is not None
    assert plan.kind == NumericRenderKind.NUMERIC_CADENCE
    assert plan.tokens == ["셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열"]
    assert plan.target_duration_sec == 3.42
    assert plan.text == "셋. 넷. 다섯. 여섯. 일곱. 여덟. 아홉. 열."
    assert plan.text_variant == "native_periods_no_compact"
    assert plan.render_policy == "whole_span_guard120_pad350"
    assert plan.groups == [[3, 4, 5, 6, 7, 8, 9, 10]]


def test_builds_head_single_rest_plan_for_ten_to_one_countdown() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan is not None
    assert plan.kind == NumericRenderKind.COUNTDOWN_10_TO_1
    assert plan.tokens == ["열", "아홉", "여덟", "일곱", "여섯", "다섯", "넷", "셋", "둘", "하나"]
    assert plan.target_duration_sec == 10.0
    assert plan.text == "열, 아홉, 여덟, 일곱, 여섯, 다섯, 넷, 셋, 둘, 하나."
    assert plan.text_variant == "native_countdown"
    assert plan.render_policy == "head_single_rest"
    assert plan.groups == [[10], [9, 8, 7, 6, 5, 4, 3, 2, 1]]


def test_builds_generic_descending_countdown_plan() -> None:
    plan = build_numeric_render_plan([5, 4, 3], target_duration_sec=2.5)
    assert plan is not None
    assert plan.kind == NumericRenderKind.DESCENDING_COUNTDOWN
    assert plan.tokens == ["다섯", "넷", "셋"]
    assert plan.target_duration_sec == 2.5
    assert plan.text == "다섯 넷 셋."
    assert plan.text_variant == "native_spaces"
    assert plan.render_policy == "whole_span_guard120_pad350"
    assert plan.groups == [[5, 4, 3]]


def test_returns_none_for_too_short_input() -> None:
    assert build_numeric_render_plan([3, 4], target_duration_sec=1.0) is None


def test_returns_none_for_unsupported_values() -> None:
    assert build_numeric_render_plan([100, 99, 98], target_duration_sec=1.0) is None


def test_caller_input_mutation_does_not_mutate_plan_values_or_groups() -> None:
    values = [3, 4, 5]
    plan = build_numeric_render_plan(values, target_duration_sec=1.25)
    assert plan is not None

    values.append(6)

    assert plan.values == [3, 4, 5]
    assert plan.groups == [[3, 4, 5]]
