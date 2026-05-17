from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from asmr_dub_pipeline.audio.features import write_audio
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gpt_sovits.schemas import GPTSoVITSRef
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
    samples = np.zeros((int(24_000 * 3.2), 2), dtype=np.float32)
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


def _rendered_result(
    project_dir: Path,
    segment: Segment,
    *,
    duration: float | None = None,
) -> dict[str, Any]:
    output_path = project_dir / "work" / "tts" / "numeric_phrase" / f"{segment.id}_fake.wav"
    duration = segment.duration if duration is None else duration
    write_audio(
        output_path,
        np.full((int(round(duration * 48_000)), 2), 0.01, dtype=np.float32),
        48_000,
    )
    return {
        "status": "rendered",
        "output_path": str(output_path),
        "duration_sec": duration,
        "seed": 4242,
        "payload": {
            "fake_numeric_renderer": True,
            "max_tempo": 1.0,
            "numeric_qc": {
                "gate": "pass",
                "expected_values": [],
                "observed_values": [],
                "backend": "test",
            },
            "placements": [],
        },
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
    assert segment.tts.candidates[0].payload["counting_compaction_disabled_for_numeric_renderer"] is True
    assert segment.analysis["numeric_phrase_renderer"]["status"] == "rendered"
    assert segment.analysis["numeric_phrase_renderer"]["text_variant"] == "native_periods_no_compact"
    assert segment.analysis["numeric_phrase_renderer"]["candidate_text"] == (
        "하나. 둘. 셋. 넷. 다섯."
    )
    assert (
        segment.analysis["numeric_phrase_renderer"][
            "counting_compaction_disabled_for_numeric_renderer"
        ]
        is True
    )
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
    assert segment.tts is None
    assert segment.analysis["numeric_phrase_renderer"]["status"] == "failed"
    assert segment.analysis["numeric_phrase_renderer"]["fallback"] == "manual_review"
    assert manifest.stage_state["synth"]["numeric_phrase_failed_segments"] == ["seg_numeric"]


def test_numeric_renderer_duration_mismatch_is_timefit_metadata_not_hard_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    def fake_render_numeric_phrase_segment(**kwargs: Any) -> dict[str, Any]:
        return _rendered_result(kwargs["project_dir"], kwargs["segment"], duration=1.0)

    monkeypatch.setattr(
        synth_stage,
        "render_numeric_phrase_segment",
        fake_render_numeric_phrase_segment,
    )
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="numeric-duration-soft", duration_tolerance=0.05),
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
    assert segment.status == "synthesized"
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is not None
    assert segment.tts.candidates[0].duration_gate == "too_short"
    assert segment.tts.candidates[0].acceptable_for_mix is True
    assert segment.analysis["numeric_phrase_renderer"]["duration_gate"] == "too_short"
    assert segment.analysis["numeric_phrase_renderer"]["numeric_qc"]["gate"] == "pass"


def test_numeric_renderer_rendered_result_with_failed_numeric_qc_is_hard_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    def fake_render_numeric_phrase_segment(**kwargs: Any) -> dict[str, Any]:
        result = _rendered_result(kwargs["project_dir"], kwargs["segment"], duration=1.0)
        result["payload"]["numeric_qc"] = {
            "gate": "fail",
            "expected_values": [10, 9, 8],
            "observed_values": [19, 8],
            "issues": ["numeric_sequence_mismatch"],
        }
        return result

    monkeypatch.setattr(
        synth_stage,
        "render_numeric_phrase_segment",
        fake_render_numeric_phrase_segment,
    )
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="numeric-qc-hard-fail", duration_tolerance=0.05),
        _scripted_segment(
            segment_id="seg_countdown",
            tts_text="십, 구, 팔.",
            source_text="10 9 8",
            duration=3.0,
        ),
    )

    countdown_synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    segment = load_manifest(tmp_path).segments[0]
    assert segment.status == "needs_manual_review"
    assert segment.tts is None
    assert segment.analysis["numeric_phrase_renderer"]["reason"] == "numeric_qc_failed"
    assert segment.analysis["numeric_phrase_renderer"]["numeric_qc"]["observed_values"] == [19, 8]


def test_ordinary_short_semantic_speech_stays_on_normal_tts(
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
        ProjectConfig(project_name="ordinary-short-speech"),
        _scripted_segment(
            segment_id="seg_short",
            tts_text="네, 좋아요.",
            source_text="はい、いいです。",
            duration=1.2,
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


def test_non_speech_texture_is_not_routed_to_numeric_renderer(
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
    texture = _scripted_segment(
        segment_id="seg_texture",
        tts_text="하나, 둘, 셋.",
        source_text="texture",
        duration=1.2,
    )
    texture.status = "non_speech_texture"
    texture.keep_original_texture = True
    texture.script = None
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(project_name="texture-skip"),
        texture,
    )

    synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    segment = load_manifest(tmp_path).segments[0]
    assert calls == []
    assert segment.status == "non_speech_texture"
    assert segment.tts is None
    assert "numeric_phrase_renderer" not in segment.analysis


def test_live_numeric_phrase_wrapper_calls_live_renderer_with_asr_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage
    from asmr_dub_pipeline.script.numeric_render_plan import build_numeric_render_plan

    cfg = ProjectConfig(
        project_name="numeric-live-wrapper",
        gsv_numeric_phrase_max_tempo=1.07,
    )
    segment = _scripted_segment(
        segment_id="seg_live",
        tts_text="하나, 둘, 셋.",
        source_text="1 2 3",
        duration=3.0,
    )
    plan = build_numeric_render_plan([1, 2, 3], target_duration_sec=segment.duration)
    ref = GPTSoVITSRef(
        ref_audio_path=str(tmp_path / "refs" / "ref.wav"),
        prompt_text="테스트입니다",
        prompt_lang="ko",
    )
    client = object()
    calls: list[dict[str, Any]] = []
    asr_calls: list[dict[str, Any]] = []

    def fake_create_asr_backend(kind: str, config: dict[str, Any]) -> object:
        asr_calls.append({"kind": kind, "config": config})
        return {"kind": kind, "config": config}

    def fake_render_live_numeric_phrase(*args: Any, **kwargs: Any) -> dict[str, Any]:
        asr_backend = kwargs["asr_backend_factory"]()
        output_path = Path(args[2])
        calls.append(
            {
                "args": args,
                "kwargs": kwargs,
                "asr_backend": asr_backend,
            }
        )
        return {
            "status": "rendered",
            "output_path": str(output_path),
            "duration_sec": segment.duration,
            "payload": {"numeric_qc": {"gate": "pass"}, "placements": [], "max_tempo": 1.0},
            "max_tempo": 1.0,
        }

    monkeypatch.setattr(synth_stage, "create_asr_backend", fake_create_asr_backend)
    monkeypatch.setattr(synth_stage, "render_live_numeric_phrase", fake_render_live_numeric_phrase)

    result = synth_stage.render_numeric_phrase_segment(
        project_dir=tmp_path,
        segment=segment,
        plan=plan,
        cfg=cfg,
        mock=False,
        lane_index=0,
        gsv_url="http://example.invalid",
        ref=ref,
        client=client,
    )

    assert result["status"] == "rendered"
    assert len(calls) == 1
    assert calls[0]["args"][:2] == (plan, client)
    assert calls[0]["kwargs"]["ref"] == ref.model_dump(mode="json")
    assert calls[0]["args"][2] == (
        tmp_path / "work" / "tts" / "numeric_phrase" / "seg_live_numeric_phrase.wav"
    )
    assert calls[0]["kwargs"]["work_dir"] == (
        tmp_path / "work" / "tts" / "numeric_phrase" / "seg_live"
    )
    assert calls[0]["kwargs"]["max_tempo_limit"] == pytest.approx(1.07)
    assert calls[0]["kwargs"]["mock"] is False
    assert calls[0]["asr_backend"] == asr_calls[0]
    assert asr_calls[0]["config"]["language"] == "ko"
    assert asr_calls[0]["config"]["word_timestamps"] is True
    assert asr_calls[0]["config"]["condition_on_previous_text"] is False
    assert asr_calls[0]["config"]["vad_filter"] is True
    assert asr_calls[0]["config"]["vad_parameters"]["min_silence_duration_ms"] <= 120


def test_live_numeric_phrase_wrapper_reports_missing_client_or_ref(tmp_path: Path) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage
    from asmr_dub_pipeline.script.numeric_render_plan import build_numeric_render_plan

    cfg = ProjectConfig(project_name="numeric-live-missing")
    segment = _scripted_segment(
        segment_id="seg_missing",
        tts_text="하나, 둘, 셋.",
        source_text="1 2 3",
        duration=3.0,
    )
    plan = build_numeric_render_plan([1, 2, 3], target_duration_sec=segment.duration)
    ref = GPTSoVITSRef(
        ref_audio_path=str(tmp_path / "refs" / "ref.wav"),
        prompt_text="테스트입니다",
        prompt_lang="ko",
    )

    missing_client = synth_stage.render_numeric_phrase_segment(
        project_dir=tmp_path,
        segment=segment,
        plan=plan,
        cfg=cfg,
        mock=False,
        ref=ref,
        client=None,
    )
    missing_ref = synth_stage.render_numeric_phrase_segment(
        project_dir=tmp_path,
        segment=segment,
        plan=plan,
        cfg=cfg,
        mock=False,
        ref=None,
        client=object(),
    )

    assert missing_client["status"] == "failed"
    assert missing_client["reason"] == "numeric_phrase_client_missing"
    assert missing_ref["status"] == "failed"
    assert missing_ref["reason"] == "numeric_phrase_ref_missing"
    assert "live_numeric_phrase_renderer_not_implemented" not in {
        missing_client["reason"],
        missing_ref["reason"],
    }


def test_live_numeric_phrase_job_resolves_segment_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits as synth_stage

    class FakeGPTSoVITSClient:
        def __init__(self, base_url: str, timeout_sec: float, retries: int) -> None:
            self.base_url = base_url
            self.timeout_sec = timeout_sec
            self.retries = retries

    calls: list[dict[str, Any]] = []

    def fake_create_asr_backend(kind: str, config: dict[str, Any]) -> object:
        return {"kind": kind, "config": config}

    def fake_render_live_numeric_phrase(*args: Any, **kwargs: Any) -> dict[str, Any]:
        plan = args[0]
        output_path = Path(args[2])
        candidate_path = Path(kwargs["work_dir"]) / "seg_live_job_phrase.wav"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        write_audio(
            candidate_path,
            np.full((int(round(plan.target_duration_sec * 24_000)), 2), 0.01, dtype=np.float32),
            24_000,
        )
        write_audio(
            output_path,
            np.full((int(round(plan.target_duration_sec * 48_000)), 2), 0.01, dtype=np.float32),
            48_000,
        )
        calls.append({"args": args, "kwargs": kwargs})
        return {
            "status": "rendered",
            "output_path": str(output_path),
            "duration_sec": plan.target_duration_sec,
            "payload": {
                "numeric_qc": {
                    "gate": "pass",
                    "expected_values": list(plan.values),
                    "observed_values": list(plan.values),
                },
                "placements": [{"values": list(plan.values)}],
                "max_tempo": 1.0,
                "candidate_generation": [
                    {
                        "role": "phrase",
                        "path": str(candidate_path),
                        "request": {"text": plan.text, "text_split_method": "cut0"},
                    }
                ],
            },
            "max_tempo": 1.0,
            "selection_reason": "test_live_numeric_phrase",
        }

    monkeypatch.setattr(synth_stage, "GPTSoVITSClient", FakeGPTSoVITSClient)
    monkeypatch.setattr(synth_stage, "create_asr_backend", fake_create_asr_backend)
    monkeypatch.setattr(synth_stage, "render_live_numeric_phrase", fake_render_live_numeric_phrase)
    refs_path = _save_project(
        tmp_path,
        ProjectConfig(
            project_name="numeric-live-job",
            gsv_gpt_weights_policy="unchanged",
            gsv_sovits_weights_policy="unchanged",
            gsv_numeric_phrase_max_tempo=1.05,
        ),
        _scripted_segment(
            segment_id="seg_live_job",
            tts_text="하나, 둘, 셋.",
            source_text="1 2 3",
            duration=3.0,
        ),
    )

    synth_step(tmp_path, None, refs_path, mock=False, confirm_rights=True)

    manifest = load_manifest(tmp_path)
    segment = manifest.segments[0]
    assert len(calls) == 1
    assert isinstance(calls[0]["args"][1], FakeGPTSoVITSClient)
    assert calls[0]["kwargs"]["ref"]["ref_audio_path"] == str((tmp_path / "refs" / "ref.wav").resolve())
    assert calls[0]["kwargs"]["ref"]["prompt_lang"] == "ja"
    assert calls[0]["kwargs"]["max_tempo_limit"] == pytest.approx(1.05)
    assert segment.status == "synthesized"
    assert segment.analysis["numeric_phrase_renderer"]["status"] == "rendered"
    analysis_generation = segment.analysis["numeric_phrase_renderer"]["candidate_generation"]
    assert isinstance(analysis_generation, list)
    assert analysis_generation[0]["role"] == "phrase"
    assert analysis_generation[0]["path"].endswith("seg_live_job_phrase.wav")
    assert segment.tts is not None
    candidate_generation = segment.tts.candidates[0].payload["candidate_generation"]
    assert isinstance(candidate_generation, list)
    assert candidate_generation[0]["role"] == "phrase"
    assert candidate_generation[0]["path"].endswith("seg_live_job_phrase.wav")
