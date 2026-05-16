from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from asmr_dub_pipeline.audio.features import write_audio
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import countdown_synth_step, synth_step
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
)


def _write_refs(project_dir: Path) -> Path:
    ref_audio = project_dir / "refs" / "ref.wav"
    samples = np.zeros((24_000, 2), dtype=np.float32)
    samples[:, 0] = np.sin(np.linspace(0.0, 120.0, len(samples), dtype=np.float32)) * 0.02
    samples[:, 1] = samples[:, 0]
    write_audio(ref_audio, samples, 24_000)
    refs_path = project_dir / "refs" / "refs.json"
    refs_path.write_text(
        json.dumps(
            {
                "whisper_close": {
                    "ref_audio_path": "refs/ref.wav",
                    "prompt_text": "テストです",
                    "prompt_lang": "ja",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        "utf-8",
    )
    return refs_path


def _scripted_segment(
    *,
    segment_id: str,
    tts_text: str,
    source_text: str = "1 2 3",
    duration: float = 3.0,
    analysis: dict[str, Any] | None = None,
) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=duration,
        duration=duration,
        status="scripted",
        audio_for_gemma=f"work/segments/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/{segment_id}_mix.wav",
        source_script=SourceScript(
            text=source_text,
            language="ja",
            backend="mock",
            start=0.0,
            end=duration,
        ),
        script=JapaneseScript(
            literal_ja=source_text,
            ja_text=source_text,
            tts_text=tts_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=duration,
            ref_style="whisper_close",
        ),
        analysis=analysis or {},
    )


def _save_project(project_dir: Path, cfg: ProjectConfig, segment: Segment) -> Path:
    save_project_config(cfg, project_dir / "pipeline.yaml")
    refs_path = _write_refs(project_dir)
    manifest = PipelineManifest(project_config=cfg, segments=[segment])
    manifest.stage_state["korean-script"] = {"status": "completed"}
    save_manifest(project_dir, manifest)
    return refs_path


def _rendered_result(project_dir: Path, segment: Segment) -> dict[str, Any]:
    output_path = project_dir / "work" / "tts" / "numeric_phrase" / f"{segment.id}_fake.wav"
    write_audio(
        output_path,
        np.full((int(round(segment.duration * 48_000)), 2), 0.01, dtype=np.float32),
        48_000,
    )
    return {
        "status": "rendered",
        "output_path": str(output_path),
        "duration_sec": segment.duration,
        "seed": 4242,
        "payload": {"fake_numeric_renderer": True},
        "selection_reason": "test_numeric_phrase_renderer",
    }


def test_numeric_phrase_config_defaults() -> None:
    cfg = ProjectConfig()

    assert cfg.gsv_countdown_renderer == "numeric_phrase"
    assert cfg.gsv_numeric_phrase_renderer_enabled is True
    assert cfg.gsv_numeric_phrase_max_tempo == pytest.approx(1.1)
    assert cfg.gsv_numeric_phrase_failure_fallback == "manual_review"
    assert cfg.gsv_numeric_phrase_whole_lead_in_sec == pytest.approx(0.12)
    assert cfg.gsv_numeric_phrase_tail_guard_sec == pytest.approx(0.16)


def test_numeric_cadence_routes_to_numeric_renderer_before_normal_tts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    calls: list[dict[str, Any]] = []

    def fake_render_numeric_phrase_segment(**kwargs: Any) -> dict[str, Any]:
        calls.append({"segment_id": kwargs["segment"].id, "plan": kwargs["plan"]})
        return _rendered_result(kwargs["project_dir"], kwargs["segment"])

    monkeypatch.setattr(
        synth_stage,
        "render_numeric_phrase_segment",
        fake_render_numeric_phrase_segment,
    )
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="numeric-route"),
        _scripted_segment(
            segment_id="seg_numeric",
            tts_text="하나, 둘, 셋, 넷, 다섯.",
            source_text="1 2 3 4 5",
            duration=3.0,
        ),
    )

    synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    manifest = load_manifest(tmp_path)
    segment = manifest.segments[0]
    assert [call["segment_id"] for call in calls] == ["seg_numeric"]
    assert calls[0]["plan"].values == [1, 2, 3, 4, 5]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    assert segment.tts.backend == "gpt-sovits-countdown-renderer"
    assert segment.tts.candidates[0].payload["renderer"] == "numeric_phrase"
    assert segment.analysis["numeric_phrase_renderer"]["status"] == "rendered"
    assert manifest.stage_state["synth"]["numeric_phrase_rendered_segments"] == ["seg_numeric"]


def test_mixed_korean_prose_with_source_numbers_stays_on_normal_tts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    calls: list[str] = []

    def fake_render_numeric_phrase_segment(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["segment"].id)
        return _rendered_result(kwargs["project_dir"], kwargs["segment"])

    monkeypatch.setattr(
        synth_stage,
        "render_numeric_phrase_segment",
        fake_render_numeric_phrase_segment,
    )
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="mixed-prose-route"),
        _scripted_segment(
            segment_id="seg_mixed",
            tts_text="숨 쉬고 하나, 둘, 셋 해볼게요.",
            source_text="1 2 3",
            duration=3.0,
        ),
    )

    synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    manifest = load_manifest(tmp_path)
    segment = manifest.segments[0]
    assert calls == []
    assert segment.status == "synthesized"
    assert "numeric_phrase_renderer" not in segment.analysis
    assert segment.tts is not None
    assert segment.tts.backend != "gpt-sovits-countdown-renderer"


def test_retry_failed_normal_synth_keeps_countdown_failed_when_countdowns_disabled(
    tmp_path: Path,
) -> None:
    segment = _scripted_segment(
        segment_id="seg_countdown",
        tts_text="삼, 이, 일",
        source_text="3 2 1",
        duration=3.0,
    )
    segment.status = "failed"
    segment.errors = ["No acceptable TTS candidates for mix."]
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="retry-countdown-skip"),
        segment,
    )

    synth_step(
        tmp_path,
        None,
        refs_path,
        mock=True,
        confirm_rights=True,
        retry_failed=True,
        render_countdowns=False,
    )

    manifest = load_manifest(tmp_path)
    segment = manifest.segments[0]
    assert segment.status == "failed"
    assert segment.errors == ["No acceptable TTS candidates for mix."]
    assert segment.tts is None
    assert "numeric_phrase_renderer" not in segment.analysis


def test_countdown_synth_routes_10_to_1_to_head_single_rest_plan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    plans: list[Any] = []

    def fake_render_numeric_phrase_segment(**kwargs: Any) -> dict[str, Any]:
        plans.append(kwargs["plan"])
        return _rendered_result(kwargs["project_dir"], kwargs["segment"])

    monkeypatch.setattr(
        synth_stage,
        "render_numeric_phrase_segment",
        fake_render_numeric_phrase_segment,
    )
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="countdown-route"),
        _scripted_segment(
            segment_id="seg_countdown",
            tts_text="십, 구, 팔, 칠, 육, 오, 사, 삼, 이, 일.",
            source_text="10 9 8 7 6 5 4 3 2 1",
            duration=10.0,
        ),
    )

    countdown_synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    manifest = load_manifest(tmp_path)
    assert len(plans) == 1
    assert plans[0].values == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    assert plans[0].render_policy == "head_single_rest"
    assert plans[0].text_variant == "native_countdown"
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.analysis["numeric_phrase_renderer"]["render_policy"] == "head_single_rest"
    assert manifest.stage_state["countdown-synth"]["numeric_phrase_rendered_segments"] == [
        "seg_countdown"
    ]


def test_numeric_renderer_failure_defaults_to_manual_review_without_normal_tts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    def fail_numeric_phrase_segment(**kwargs: Any) -> dict[str, Any]:
        return {"status": "failed", "reason": "not implemented in test"}

    def fail_normal_mock_tts(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("numeric renderer failure should not fall back to normal TTS")

    monkeypatch.setattr(
        synth_stage,
        "render_numeric_phrase_segment",
        fail_numeric_phrase_segment,
    )
    monkeypatch.setattr(synth_stage, "_mock_synthesize", fail_normal_mock_tts)
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="numeric-failure"),
        _scripted_segment(
            segment_id="seg_numeric",
            tts_text="하나, 둘, 셋, 넷.",
            source_text="1 2 3 4",
            duration=3.0,
        ),
    )

    synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    manifest = load_manifest(tmp_path)
    segment = manifest.segments[0]
    assert segment.status == "needs_manual_review"
    assert "Numeric phrase renderer failed: not implemented in test" in segment.errors
    assert segment.analysis["numeric_phrase_renderer"]["status"] == "failed"
    assert segment.analysis["numeric_phrase_renderer"]["fallback"] == "manual_review"
    assert manifest.stage_state["synth"]["numeric_phrase_failed_segments"] == ["seg_numeric"]
