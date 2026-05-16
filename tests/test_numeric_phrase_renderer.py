from dataclasses import replace

import numpy as np
import pytest

from asmr_dub_pipeline.pipeline.stages.numeric_phrase_renderer import (
    RenderedNumericBed,
    build_numeric_phrase_request,
    evaluate_numeric_render_transcript,
    render_from_phrase_candidate,
)
from asmr_dub_pipeline.script.numeric_render_plan import build_numeric_render_plan


def test_countdown_generation_uses_native_text_and_low_randomness() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan is not None
    request = build_numeric_phrase_request(plan, ref={"prompt_lang": "ko"})
    assert request.text == "열, 아홉, 여덟, 일곱, 여섯, 다섯, 넷, 셋, 둘, 하나."
    assert request.text_lang == "all_ko"
    assert request.text_split_method == "cut0"
    assert request.top_k == 5
    assert request.top_p == 0.85
    assert request.temperature == 0.65
    assert request.repetition_penalty == 1.8


def test_numeric_qc_rejects_nineteen_for_ten_nine() -> None:
    qc = evaluate_numeric_render_transcript([10, 9, 8], "19, 8")
    assert qc["gate"] == "fail"
    assert qc["observed_values"] != [10, 9, 8]


def test_whole_span_guard_has_no_internal_cuts() -> None:
    plan = build_numeric_render_plan([3, 4, 5, 6, 7, 8, 9, 10], target_duration_sec=3.42)
    assert plan is not None
    audio = np.ones((48_000 * 3, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 3, "source_start_sec": 0.0, "source_end_sec": 0.3},
        {"value": 10, "source_start_sec": 2.38, "source_end_sec": 2.54},
    ]
    rendered = render_from_phrase_candidate(plan, audio, 48_000, word_timing)
    assert isinstance(rendered, RenderedNumericBed)
    assert rendered.policy == "whole_span_guard120_pad350"
    assert rendered.max_tempo == 1.0
    assert rendered.placements[0]["values"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert len(rendered.placements) == 1


def test_head_single_rest_has_only_one_internal_boundary_after_ten() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan is not None
    phrase = np.ones((48_000 * 9, 2), dtype=np.float32) * 0.01
    head = np.ones((48_000, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 9, "source_start_sec": 0.36, "source_end_sec": 1.38},
        {"value": 1, "source_start_sec": 7.18, "source_end_sec": 7.82},
    ]
    rendered = render_from_phrase_candidate(plan, phrase, 48_000, word_timing, head_audio=head)
    assert rendered.policy == "head_single_rest"
    assert [placement["values"] for placement in rendered.placements] == [[10], [9, 8, 7, 6, 5, 4, 3, 2, 1]]
    assert rendered.max_tempo == 1.0


def test_rendered_bed_is_stereo_float32_with_audio_in_rendered_region() -> None:
    plan = build_numeric_render_plan([3, 4, 5], target_duration_sec=1.4)
    assert plan is not None
    audio = np.ones((48_000, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 3, "source_start_sec": 0.1, "source_end_sec": 0.2},
        {"value": 5, "source_start_sec": 0.5, "source_end_sec": 0.6},
    ]

    rendered = render_from_phrase_candidate(plan, audio, 48_000, word_timing)

    assert rendered.audio.shape == (67_200, 2)
    assert rendered.audio.dtype == np.float32
    placement = rendered.placements[0]
    assert placement["copy_status"] == "copied"
    start, end = placement["target_frame_range"]
    assert np.any(rendered.audio[start:end] != 0.0)


def test_mono_phrase_and_head_inputs_render_to_stereo() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan is not None
    phrase = np.ones(48_000 * 9, dtype=np.float32) * 0.01
    head = np.ones(48_000, dtype=np.float32) * 0.02
    word_timing = [
        {"value": 9, "source_start_sec": 0.36, "source_end_sec": 1.38},
        {"value": 1, "source_start_sec": 7.18, "source_end_sec": 7.82},
    ]

    rendered = render_from_phrase_candidate(plan, phrase, 48_000, word_timing, head_audio=head)

    assert rendered.audio.shape == (480_000, 2)
    assert rendered.audio.dtype == np.float32
    assert np.any(rendered.audio[:, 0] != 0.0)
    assert np.array_equal(rendered.audio[:, 0], rendered.audio[:, 1])


def test_long_source_span_records_truncation_and_required_tempo() -> None:
    plan = build_numeric_render_plan([3, 4, 5], target_duration_sec=1.0)
    assert plan is not None
    audio = np.ones((48_000 * 3, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 3, "source_start_sec": 0.0, "source_end_sec": 0.1},
        {"value": 5, "source_start_sec": 2.1, "source_end_sec": 2.2},
    ]

    rendered = render_from_phrase_candidate(plan, audio, 48_000, word_timing)

    placement = rendered.placements[0]
    assert placement["truncated"] is True
    assert placement["copy_status"] == "copied"
    assert placement["source_frames"] > placement["target_available_frames"]
    assert placement["copied_frames"] == placement["target_available_frames"]
    assert placement["required_tempo"] > 1.0
    assert rendered.max_tempo == placement["required_tempo"]


def test_head_single_rest_requires_head_audio() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan is not None
    phrase = np.ones((48_000 * 9, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 9, "source_start_sec": 0.36, "source_end_sec": 1.38},
        {"value": 1, "source_start_sec": 7.18, "source_end_sec": 7.82},
    ]

    with pytest.raises(ValueError, match="head_audio"):
        render_from_phrase_candidate(plan, phrase, 48_000, word_timing)


def test_unknown_render_policy_raises_value_error() -> None:
    plan = build_numeric_render_plan([3, 4, 5], target_duration_sec=1.4)
    assert plan is not None
    plan = replace(plan, render_policy="bogus")
    audio = np.ones((48_000, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 3, "source_start_sec": 0.1, "source_end_sec": 0.2},
        {"value": 5, "source_start_sec": 0.5, "source_end_sec": 0.6},
    ]

    with pytest.raises(ValueError, match="Unsupported numeric render policy"):
        render_from_phrase_candidate(plan, audio, 48_000, word_timing)


def test_invalid_target_window_records_skipped_metadata() -> None:
    plan = build_numeric_render_plan([3, 4, 5], target_duration_sec=0.2)
    assert plan is not None
    audio = np.ones((48_000, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 3, "source_start_sec": 0.1, "source_end_sec": 0.2},
        {"value": 5, "source_start_sec": 0.5, "source_end_sec": 0.6},
    ]

    rendered = render_from_phrase_candidate(plan, audio, 48_000, word_timing)

    placement = rendered.placements[0]
    assert placement["copy_status"] == "skipped"
    assert placement["skipped_reason"] == "empty_target_window"
    assert placement["copied_frames"] == 0
    assert placement["target_available_frames"] == 0
    assert placement["required_tempo"] == 1.0
    assert placement["truncated"] is False
