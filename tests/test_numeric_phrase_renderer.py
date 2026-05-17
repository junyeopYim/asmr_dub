from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from asmr_dub_pipeline.asr.base import ASRChunk, ASRWord
from asmr_dub_pipeline.audio.features import load_audio, write_audio
from asmr_dub_pipeline.pipeline.stages.numeric_phrase_renderer import (
    RenderedNumericBed,
    build_numeric_phrase_request,
    evaluate_numeric_render_transcript,
    render_from_phrase_candidate,
    render_live_numeric_phrase,
)
from asmr_dub_pipeline.script.numeric_render_plan import build_numeric_render_plan


class FakeGSVClient:
    def __init__(self, *, sample_rate: int = 48_000, duration_sec: float = 1.2) -> None:
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.calls: list[dict[str, Any]] = []

    def synthesize_to_file(self, request: Any, output_path: Path) -> Path:
        self.calls.append({"request": request, "output_path": output_path})
        frames = int(round(self.duration_sec * self.sample_rate))
        signal = np.full((frames, 2), 0.02, dtype=np.float32)
        write_audio(output_path, signal, self.sample_rate)
        return output_path


class FakeASRBackend:
    def __init__(self, responses: list[tuple[str, list[dict[str, Any]]]]) -> None:
        self.responses = list(responses)
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path, _segments: list[Any]) -> list[ASRChunk]:
        self.calls.append(audio_path)
        transcript, words = self.responses.pop(0)
        return [
            ASRChunk(
                start=0.0,
                end=max((float(word["end"]) for word in words), default=0.1),
                text=transcript,
                language="ko",
                words=[
                    ASRWord(
                        start=float(word["start"]),
                        end=float(word["end"]),
                        text=str(word["word"]),
                    )
                    for word in words
                ],
            )
        ]


class FakeASRBackendWithOptions:
    def __init__(self, responses: list[tuple[str, list[dict[str, Any]]]]) -> None:
        self.responses = list(responses)
        self.option_calls: list[dict[str, Any]] = []

    def transcribe_with_options(self, audio_path: Path, segments: list[Any], **options: Any) -> list[ASRChunk]:
        self.option_calls.append({"audio_path": audio_path, "segments": segments, "options": options})
        transcript, words = self.responses.pop(0)
        return [
            ASRChunk(
                start=0.0,
                end=max((float(word["end"]) for word in words), default=0.1),
                text=transcript,
                language="ko",
                words=[
                    ASRWord(
                        start=float(word["start"]),
                        end=float(word["end"]),
                        text=str(word["word"]),
                    )
                    for word in words
                ],
            )
        ]

    def transcribe(self, _audio_path: Path, _segments: list[Any]) -> list[ASRChunk]:
        raise AssertionError("transcribe fallback should not be used when transcribe_with_options exists")


def _words(values: list[int], words: list[str], *, step: float = 0.2) -> list[dict[str, Any]]:
    return [
        {"value": value, "word": word, "start": index * step, "end": index * step + 0.12}
        for index, (value, word) in enumerate(zip(values, words, strict=True))
    ]


def _valid_ref(tmp_path: Path) -> dict[str, str]:
    return {
        "ref_audio_path": str(tmp_path / "ref.wav"),
        "prompt_text": "작게 속삭여요.",
        "prompt_lang": "ko",
    }


def test_countdown_generation_uses_native_text_and_low_randomness() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan is not None
    request = build_numeric_phrase_request(plan, ref={"prompt_lang": "ko"})
    assert request.text == "열, 아홉, 여덟, 일곱, 여섯, 다섯, 넷, 셋, 둘, 하나."
    assert request.text_lang == "all_ko"
    assert request.prompt_lang == "all_ko"
    assert request.as_payload()["prompt_lang"] == "all_ko"
    assert request.text_split_method == "cut0"
    assert request.top_k == 5
    assert request.top_p == 0.85
    assert request.temperature == 0.65
    assert request.repetition_penalty == 1.8


def test_numeric_cadence_generation_uses_native_periods_without_compaction() -> None:
    plan = build_numeric_render_plan([3, 4, 5, 6, 7, 8, 9, 10], target_duration_sec=3.42)
    assert plan is not None

    request = build_numeric_phrase_request(plan, ref={"prompt_lang": "ko"})

    assert plan.text_variant == "native_periods_no_compact"
    assert request.text == "셋. 넷. 다섯. 여섯. 일곱. 여덟. 아홉. 열."
    assert request.text_split_method == "cut0"
    assert request.top_k == 5
    assert request.top_p == 0.85
    assert request.temperature == 0.65
    assert request.repetition_penalty == 1.8


def test_numeric_phrase_request_normalizes_japanese_prompt_language() -> None:
    plan = build_numeric_render_plan([10, 9, 8], target_duration_sec=1.4)
    assert plan is not None
    request = build_numeric_phrase_request(plan, ref={"prompt_lang": "ja"})

    assert request.prompt_lang == "all_ja"
    assert request.text_lang == "all_ko"
    assert request.as_payload()["prompt_lang"] == "all_ja"
    assert request.as_payload()["text_lang"] == "all_ko"


def test_numeric_qc_rejects_nineteen_for_ten_nine() -> None:
    qc = evaluate_numeric_render_transcript([10, 9, 8], "19, 8")
    assert qc["gate"] == "fail"
    assert qc["observed_values"] != [10, 9, 8]


def test_live_whole_span_numeric_phrase_renders_wav_and_payload(tmp_path: Path) -> None:
    plan = build_numeric_render_plan([10, 9, 8], target_duration_sec=1.4)
    assert plan is not None
    client = FakeGSVClient(duration_sec=1.0)
    asr = FakeASRBackend(
        [
            (
                "열 아홉 여덟",
                _words([10, 9, 8], ["열", "아홉", "여덟"]),
            )
        ]
    )

    result = render_live_numeric_phrase(
        plan,
        client,
        tmp_path / "rendered.wav",
        ref=_valid_ref(tmp_path),
        asr_backend=asr,
    )

    assert result["status"] == "rendered"
    assert Path(result["output_path"]).exists()
    rendered_audio, sample_rate = load_audio(result["output_path"])
    assert sample_rate == 48_000
    assert rendered_audio.shape == (67_200, 2)
    assert result["duration_sec"] == pytest.approx(1.4)
    payload = result["payload"]
    assert payload["renderer"] == "numeric_phrase_renderer"
    assert payload["numeric_qc"]["gate"] == "pass"
    assert payload["placements"]
    assert payload["rendered_audio_path"] == result["output_path"]
    assert payload["request"]["text"] == plan.text
    assert payload["candidate_text"] == plan.text
    assert payload["candidate_generation"][0]["path"]


def test_live_numeric_phrase_strict_qc_rejects_collapsed_nineteen(tmp_path: Path) -> None:
    plan = build_numeric_render_plan([10, 9, 8], target_duration_sec=1.0)
    assert plan is not None
    client = FakeGSVClient(duration_sec=1.0)
    asr = FakeASRBackend(
        [
            (
                "19, 8",
                _words([19, 8], ["19", "8"]),
            )
        ]
    )

    result = render_live_numeric_phrase(
        plan,
        client,
        tmp_path / "rendered.wav",
        ref=_valid_ref(tmp_path),
        asr_backend=asr,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "numeric_qc_failed"
    assert result["payload"]["numeric_qc"]["gate"] == "fail"
    assert result["payload"]["numeric_qc"]["observed_values"] == [19, 8]
    assert not Path(result["output_path"]).exists()


def test_live_head_single_rest_synthesizes_head_and_body_and_renders_two_placements(tmp_path: Path) -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan is not None
    client = FakeGSVClient(duration_sec=8.5)
    asr = FakeASRBackend(
        [
            ("열", _words([10], ["열"])),
            (
                "아홉 여덟 일곱 여섯 다섯 넷 셋 둘 하나",
                _words([9, 8, 7, 6, 5, 4, 3, 2, 1], ["아홉", "여덟", "일곱", "여섯", "다섯", "넷", "셋", "둘", "하나"], step=0.8),
            ),
        ]
    )

    result = render_live_numeric_phrase(
        plan,
        client,
        tmp_path / "rendered.wav",
        ref=_valid_ref(tmp_path),
        asr_backend=asr,
    )

    assert result["status"] == "rendered"
    assert len(client.calls) == 2
    payload = result["payload"]
    assert [call["role"] for call in payload["candidate_generation"]] == ["head", "body"]
    assert [placement["values"] for placement in payload["placements"]] == [[10], [9, 8, 7, 6, 5, 4, 3, 2, 1]]
    assert payload["numeric_qc"]["expected_values"] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert payload["numeric_qc"]["gate"] == "pass"


def test_live_numeric_phrase_fails_when_max_tempo_limit_is_exceeded(tmp_path: Path) -> None:
    plan = build_numeric_render_plan([10, 9, 8], target_duration_sec=0.5)
    assert plan is not None
    client = FakeGSVClient(duration_sec=2.0)
    asr = FakeASRBackend(
        [
            (
                "열 아홉 여덟",
                _words([10, 9, 8], ["열", "아홉", "여덟"], step=0.8),
            )
        ]
    )

    result = render_live_numeric_phrase(
        plan,
        client,
        tmp_path / "rendered.wav",
        ref=_valid_ref(tmp_path),
        asr_backend=asr,
        max_tempo_limit=1.1,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "max_tempo_exceeded"
    assert result["payload"]["max_tempo"] > 1.1
    assert result["payload"]["max_tempo_limit"] == 1.1
    assert not Path(result["output_path"]).exists()


def test_live_numeric_phrase_prefers_transcribe_with_word_timestamp_options(tmp_path: Path) -> None:
    plan = build_numeric_render_plan([10, 9, 8], target_duration_sec=1.4)
    assert plan is not None
    client = FakeGSVClient(duration_sec=1.0)
    asr = FakeASRBackendWithOptions(
        [
            (
                "열 아홉 여덟",
                _words([10, 9, 8], ["열", "아홉", "여덟"]),
            )
        ]
    )

    result = render_live_numeric_phrase(
        plan,
        client,
        tmp_path / "rendered.wav",
        ref=_valid_ref(tmp_path),
        asr_backend=asr,
    )

    assert result["status"] == "rendered"
    assert len(asr.option_calls) == 1
    options = asr.option_calls[0]["options"]
    assert options["word_timestamps"] is True
    assert options["language"] == "ko"
    assert options["condition_on_previous_text"] is False
    assert options["vad_filter"] is False


def test_live_numeric_phrase_fails_before_synthesis_when_ref_is_incomplete(tmp_path: Path) -> None:
    plan = build_numeric_render_plan([10, 9, 8], target_duration_sec=1.4)
    assert plan is not None
    client = FakeGSVClient(duration_sec=1.0)
    asr = FakeASRBackend(
        [
            (
                "열 아홉 여덟",
                _words([10, 9, 8], ["열", "아홉", "여덟"]),
            )
        ]
    )

    result = render_live_numeric_phrase(
        plan,
        client,
        tmp_path / "rendered.wav",
        ref={"prompt_lang": "ko"},
        asr_backend=asr,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "missing_ref"
    assert result["payload"]["missing_ref_fields"] == ["ref_audio_path", "prompt_text"]
    assert client.calls == []
    assert not Path(result["output_path"]).exists()


def test_live_numeric_phrase_rejects_repeated_value_when_word_timing_count_is_short(tmp_path: Path) -> None:
    plan = build_numeric_render_plan([1, 1, 2], target_duration_sec=1.4)
    assert plan is not None
    client = FakeGSVClient(duration_sec=1.0)
    asr = FakeASRBackend(
        [
            (
                "하나 하나 둘",
                _words([1, 2], ["하나", "둘"]),
            )
        ]
    )

    result = render_live_numeric_phrase(
        plan,
        client,
        tmp_path / "rendered.wav",
        ref=_valid_ref(tmp_path),
        asr_backend=asr,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "numeric_qc_failed"
    numeric_qc = result["payload"]["numeric_qc"]
    assert numeric_qc["gate"] == "fail"
    assert numeric_qc["observed_values"] == [1, 1, 2]
    assert "repeated_numeric_word_timing_mismatch" in numeric_qc["issues"]
    assert numeric_qc["word_timing_value_counts"][1] == 1
    assert numeric_qc["expected_value_counts"][1] == 2


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


def test_whole_span_repeated_first_value_uses_first_ordered_timing() -> None:
    plan = build_numeric_render_plan([1, 1, 2], target_duration_sec=1.8)
    assert plan is not None
    audio = np.ones((48_000 * 2, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 1, "source_start_sec": 0.10, "source_end_sec": 0.20},
        {"value": 1, "source_start_sec": 0.40, "source_end_sec": 0.50},
        {"value": 2, "source_start_sec": 0.80, "source_end_sec": 0.90},
    ]

    rendered = render_from_phrase_candidate(plan, audio, 48_000, word_timing)

    placement = rendered.placements[0]
    assert placement["requested_source_start_sec"] == pytest.approx(0.02)
    assert placement["requested_source_start_sec"] != pytest.approx(0.32)
    assert placement["requested_source_end_sec"] == pytest.approx(1.25)


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
