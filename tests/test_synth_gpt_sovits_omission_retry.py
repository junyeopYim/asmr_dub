from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import write_tiny_wav

from asmr_dub_pipeline.audio.features import duration_sec, write_audio
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gpt_sovits.client import build_tts_request
from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import (
    _gsv_candidate_selection_score,
    _select_gsv_candidate_for_mix,
)
from asmr_dub_pipeline.pipeline.steps import init_project, synth_step
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
    TTSCandidate,
)


def _write_tone_wav(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    write_audio(path, np.stack([tone, tone], axis=1), sample_rate)
    return path


def test_gsv_candidate_selection_prefers_clean_delivery_over_exact_penalized_duration() -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=5.0,
        duration=5.0,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
        script=JapaneseScript(
            ja_text="ゆっくり聞いてください",
            tts_text="천천히 들어주세요.",
            tts_language="ko",
            ref_style="whisper_close",
            expected_tts_duration_sec=5.0,
        ),
    )
    exact_but_penalized = TTSCandidate(
        candidate_index=0,
        seed=1,
        output_path="exact.wav",
        duration_sec=5.0,
        backend="gpt-sovits",
        duration_ratio=1.0,
        duration_gate="pass",
        acceptable_for_mix=True,
        selection_reason="source_pause_padding_rescue",
        payload={
            "audio_qc": {"gate": "pass"},
            "speed_factor": 1.25,
            "fallback_used": True,
            "requested_ref_style": "whisper_close",
            "resolved_ref_style": "sleepy",
            "postprocess": {
                "edge_silence_trim": {
                    "leading_trim_sec": 0.5,
                    "trailing_trim_sec": 0.5,
                }
            },
        },
    )
    clean_delivery = TTSCandidate(
        candidate_index=1,
        seed=2,
        output_path="clean.wav",
        duration_sec=4.7,
        backend="gpt-sovits",
        duration_ratio=0.94,
        duration_gate="pass",
        acceptable_for_mix=True,
        selection_reason="duration_and_language_contract_pass",
        payload={
            "audio_qc": {"gate": "pass"},
            "speed_factor": 1.0,
            "fallback_used": False,
            "requested_ref_style": "whisper_close",
            "resolved_ref_style": "whisper_close",
            "postprocess": {
                "edge_silence_trim": {
                    "leading_trim_sec": 0.02,
                    "trailing_trim_sec": 0.02,
                }
            },
        },
    )

    assert _gsv_candidate_selection_score(clean_delivery, segment) > _gsv_candidate_selection_score(
        exact_but_penalized,
        segment,
    )
    assert _select_gsv_candidate_for_mix(
        [exact_but_penalized, clean_delivery],
        segment,
    ) is clean_delivery


def test_synth_force_renders_countdown_as_timed_token_candidates(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=3,
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_candidate_count=3,
        gsv_countdown_token_min_sec=0.25,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.85,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)

    segments: list[Segment] = []
    specs = [
        (
            "seg_0095",
            0.0,
            2.0,
            "4 3",
            "넷, 셋",
            1.0,
            [4, 3],
            [("4", "넷", 4, 0.22, 0.54), ("3", "셋", 3, 1.28, 1.6)],
        ),
        (
            "seg_0096",
            2.0,
            4.0,
            "2 1",
            "둘, 하나",
            1.0,
            [2, 1],
            [("2", "둘", 2, 2.18, 2.54), ("1", "하나", 1, 3.24, 3.62)],
        ),
        (
            "seg_0097",
            4.0,
            5.0,
            "0",
            "영",
            0.5,
            [0],
            [("0", "영", 0, 4.42, 4.78)],
        ),
    ]
    for segment_id, start, end, source_text, tts_text, expected, values, timeline in specs:
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        _write_tone_wav(audio, duration=end - start)
        segments.append(
            Segment(
                id=segment_id,
                start=start,
                end=end,
                duration=end - start,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text=source_text,
                    language="ja",
                    backend="mock",
                    start=start,
                    end=end,
                ),
                script=JapaneseScript(
                    literal_ja=source_text,
                    ja_text=source_text,
                    tts_text=tts_text,
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                    expected_tts_duration_sec=expected,
                    ref_style="whisper_close",
                ),
                analysis={
                    "countdown_event": {
                        "kind": "descending_countdown",
                        "values": values,
                        "korean_tokens": [item[1] for item in timeline],
                        "token_timeline": [
                            {
                                "source_text": source,
                                "korean_token": korean_token,
                                "value": value,
                                "start": token_start,
                                "end": token_end,
                            }
                            for source, korean_token, value, token_start, token_end in timeline
                        ],
                    }
                },
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))

    requests: list[str] = []
    token_calls: dict[str, int] = {}

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            token_calls[request.text] = token_calls.get(request.text, 0) + 1
            duration = 2.4 if token_calls[request.text] == 1 else 0.56
            _write_tone_wav(output_path, duration=duration)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert "넷셋둘하나영" not in requests
    assert set(requests) == {"넷", "셋", "둘", "하나", "영"}
    assert all(count == 3 for count in token_calls.values())
    expected_slot_starts = {
        "seg_0095": [0.22, 1.28],
        "seg_0096": [2.18, 3.24],
        "seg_0097": [4.42],
    }
    for segment in manifest.segments:
        assert segment.status == "synthesized"
        assert segment.tts is not None
        assert segment.tts.backend == "gpt-sovits-countdown-renderer"
        assert segment.tts.selected_candidate_path is not None
        assert segment.tts.candidates[0].payload["renderer"] == "countdown_token_timeline"
        assert duration_sec(Path(segment.tts.selected_candidate_path)) == pytest.approx(
            segment.duration,
            abs=0.03,
        )
        assert segment.tts.retry_summary["countdown_renderer"] is True
        assert segment.tts.retry_summary["countdown_renderer_mode"] == "token"
        assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_token_timeline"
        assert [
            placement["slot_start_sec"]
            for placement in segment.analysis["countdown_renderer"]["token_placements"]
        ] == pytest.approx(expected_slot_starts[segment.id], abs=0.001)
        assert all(
            placement["selected_duration_sec"] <= placement["slot_duration_sec"] * 0.85
            for placement in segment.analysis["countdown_renderer"]["token_placements"]
        )


def test_synth_closes_open_korean_text_and_retries_omission_with_static_ref(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        duration_tolerance=0.2,
        gsv_ref_mode="segment",
        gsv_max_attempts_per_candidate=2,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    static_ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(static_ref_audio, duration=4.0)
    segment_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0010_mix.wav"
    write_tiny_wav(segment_audio, duration=8.04)
    original_text = "하지만 눈만은 아직 뜨고 있어야 해요. 멍하니…"
    closed_text = "하지만 눈만은 아직 뜨고 있어야 해요. 멍하니 말이에요."
    segment = Segment(
        id="seg_0010",
        start=77.57,
        end=85.61,
        duration=8.04,
        audio_for_gemma=str(segment_audio),
        audio_for_mix=str(segment_audio),
        source_script=SourceScript(
            text="でも、目だけはまだ、開けたままにしておいてください。ぼんやりと、",
            language="ja",
            backend="mock",
            start=77.57,
            end=85.61,
        ),
        script=JapaneseScript(
            ja_text="でも、目だけはまだ、開けたままにしておいてください。ぼんやりと、",
            tts_text=original_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            ref_style="whisper_close",
            expected_tts_duration_sec=4.5,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            requests.append(
                {
                    "text": request.text,
                    "ref_audio_path": request.ref_audio_path,
                    "prompt_text": request.prompt_text,
                    "seed": request.seed,
                    "repetition_penalty": request.repetition_penalty,
                }
            )
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 0.92 if "work/segments/audio" in request.ref_audio_path else 8.04
            _write_tone_wav(output_path, duration=duration)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.script is not None
    assert segment.script.tts_text == closed_text
    assert requests[0]["text"] == closed_text
    assert "work/segments/audio/seg_0010_mix.wav" in str(requests[0]["ref_audio_path"])
    assert requests[1]["text"] == closed_text
    assert requests[1]["ref_audio_path"] == str(static_ref_audio)
    assert requests[1]["repetition_penalty"] > requests[0]["repetition_penalty"]
    assert segment.tts is not None
    first_candidate = segment.tts.candidates[0]
    assert first_candidate.payload["omission_suspected"] is True
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["omission_retry"]["ref_fallback"] == "static_ref"
    assert selected.payload["omission_retry"]["text_normalization"] == "closed_open_sentence"
