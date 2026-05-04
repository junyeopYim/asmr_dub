from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import sha256, write_tiny_wav

import asmr_dub_pipeline.cli as cli_module
from asmr_dub_pipeline.audio import preprocess
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import load_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest
from asmr_dub_pipeline.pipeline.steps import extract_step, mix_step, segment_step
from asmr_dub_pipeline.schemas import PipelineManifest


def test_mock_pipeline_e2e(cli_runner, tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    before = sha256(tiny_wav_path)
    result = cli_runner.invoke(
        app,
        [
            "run",
            str(tiny_wav_path),
            "--project",
            str(tmp_project_dir),
            "--confirm-rights",
            "--mock",
        ],
    )
    assert result.exit_code == 0, result.output
    assert sha256(tiny_wav_path) == before
    assert (tmp_project_dir / "work/audio/original_stereo_48k.wav").exists()
    assert (tmp_project_dir / "work/audio/gemma_mono_16k.wav").exists()
    assert (tmp_project_dir / "work/segments/manifests/segments_raw.json").exists()
    assert (tmp_project_dir / "work/rvc/rvc_manifest.json").exists()
    assert (tmp_project_dir / "work/mix/dialogue_stem.wav").exists()
    assert (tmp_project_dir / "work/mix/final_audio.wav").exists()
    manifest = load_manifest(tmp_project_dir)
    assert manifest.rights_audit.confirmed is True
    assert manifest.segments
    assert manifest.segments[0].tts is not None
    assert manifest.segments[0].tts.selected_candidate_path
    assert manifest.segments[0].rvc is not None
    assert manifest.segments[0].rvc.accepted is True
    assert manifest.segments[0].rvc.output_path is not None
    assert "work/tts" in manifest.segments[0].tts.selected_candidate_path
    assert "work/rvc" in manifest.segments[0].rvc.output_path
    assert manifest.stage_state["train-rvc"]["status"] == "completed"
    assert manifest.stage_state["train-rvc"]["backend"] == "mock"
    assert manifest.stage_state["rvc"]["status"] == "completed"
    assert manifest.stage_state["rvc"]["backend"] == "mock"
    assert manifest.artifacts["export"].endswith("_dub.wav")
    assert manifest.artifacts["rvc_manifest"].endswith("rvc_manifest.json")
    assert manifest.artifacts["mix_manifest"].endswith("mix_manifest.json")
    assert manifest.artifacts["export_manifest"].endswith("export_manifest.json")
    assert manifest.segments[0].mix["included"] is True
    assert manifest.segments[0].mix["dialogue_fade_ms"] is None
    assert manifest.segments[0].mix["fade_in_ms"] > 8.0
    mix_manifest = json.loads(Path(manifest.artifacts["mix_manifest"]).read_text("utf-8"))
    assert mix_manifest["config"]["background_bed"] == "preserve_original"
    assert mix_manifest["config"]["loudness_strategy"] == "peak_guard_only"
    assert mix_manifest["config"]["loudness_normalization"] == "disabled"
    assert mix_manifest["background"]["used"] is True
    assert Path(manifest.artifacts["export"]).exists()


def test_extract_merge_parts_creates_canonical_audio_source(
    tmp_path: Path,
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    part_1 = write_tiny_wav(tmp_path / "RJTEST_1.wav", duration=0.7)
    part_2 = write_tiny_wav(tmp_path / "RJTEST_2.wav", duration=0.8)

    def fake_concat_audio_to_wav(
        input_paths: list[Path],
        output_path: Path,
        *,
        sample_rate: int = 48_000,
        channels: int = 2,
    ) -> Path:
        clips = []
        for input_path in input_paths:
            data, sr = sf.read(input_path, always_2d=True, dtype="float32")
            assert sr == sample_rate
            clips.append(data[:, :channels])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, np.concatenate(clips, axis=0), sample_rate)
        return output_path

    monkeypatch.setattr(preprocess.ffmpeg, "concat_audio_to_wav", fake_concat_audio_to_wav)

    manifest = extract_step(part_1, tmp_project_dir, confirm_rights=True, merge_parts=True)

    merged_path = tmp_project_dir / "work" / "input" / "RJTEST_merged_source.wav"
    assert merged_path.exists()
    assert manifest.source_info is not None
    assert manifest.source_info.path == str(merged_path)
    assert manifest.artifacts["merged_source_audio"] == str(merged_path)
    assert Path(manifest.artifacts["input_parts_manifest"]).exists()
    merge = manifest.source_info.raw["input_merge"]
    assert merge["status"] == "merged"
    assert merge["part_count"] == 2
    assert [part["path"] for part in merge["parts"]] == [str(part_1.resolve()), str(part_2.resolve())]
    assert manifest.rights_audit.history[-1]["input_merge"]["status"] == "merged"


def test_extract_folder_input_prefers_clean_asr_parts(
    tmp_path: Path,
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    folder = tmp_path / "RJDIR"
    mix_1 = write_tiny_wav(folder / "本編" / "01 プロローグ.wav", duration=0.7)
    mix_2 = write_tiny_wav(folder / "本編" / "02 催眠誘導.wav", duration=0.8)
    asr_1 = write_tiny_wav(folder / "効果音無し" / "01 プロローグ.wav", duration=0.7)
    asr_2 = write_tiny_wav(folder / "効果音無し" / "02 催眠誘導.wav", duration=0.8)

    def fake_concat_audio_to_wav(
        input_paths: list[Path],
        output_path: Path,
        *,
        sample_rate: int = 48_000,
        channels: int = 2,
    ) -> Path:
        clips = []
        for input_path in input_paths:
            data, sr = sf.read(input_path, always_2d=True, dtype="float32")
            if sr != sample_rate:
                indexes = np.linspace(0, len(data) - 1, max(1, int(len(data) * sample_rate / sr)))
                data = data[indexes.astype(np.int64)]
            if channels == 1:
                data = data.mean(axis=1, keepdims=True)
            elif data.shape[1] < channels:
                data = np.repeat(data[:, :1], channels, axis=1)
            else:
                data = data[:, :channels]
            clips.append(data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, np.concatenate(clips, axis=0), sample_rate)
        return output_path

    monkeypatch.setattr(preprocess.ffmpeg, "concat_audio_to_wav", fake_concat_audio_to_wav)

    manifest = extract_step(folder, tmp_project_dir, confirm_rights=True)

    folder_manifest = Path(manifest.artifacts["folder_input_manifest"])
    metadata = json.loads(folder_manifest.read_text("utf-8"))
    assert folder_manifest.exists()
    assert metadata["status"] == "planned"
    assert metadata["input_kind"] == "folder"
    assert metadata["asr_source_status"] == "separate_asr_parts"
    assert [part["path"] for part in metadata["mix_parts"]] == [str(mix_1.resolve()), str(mix_2.resolve())]
    assert [part["path"] for part in metadata["asr_parts"]] == [str(asr_1.resolve()), str(asr_2.resolve())]
    assert Path(manifest.artifacts["original_stereo_48k"]).exists()
    assert Path(manifest.artifacts["gemma_mono_16k"]).exists()
    assert manifest.source_info is not None
    assert manifest.source_info.raw["folder_input"]["asr_source_status"] == "separate_asr_parts"


def test_extract_folder_input_blocks_minor_sexualized_filenames(
    tmp_path: Path,
    tmp_project_dir: Path,
) -> None:
    folder = tmp_path / "RJDIR"
    write_tiny_wav(folder / "TSSS 完パケ音声 mp3" / "06-2 男の子 H有（SE有）.wav", duration=0.7)

    with pytest.raises(ValueError, match="minor sexualized"):
        extract_step(folder, tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["extract"]["status"] == "failed"
    assert manifest.stage_state["extract"]["safety_blocked"] == 1


def test_folder_input_planner_prefers_main_group_over_diff_tracks(tmp_path: Path) -> None:
    folder = tmp_path / "RJDIFF"
    main_1 = write_tiny_wav(folder / "TSSS 完パケ音声 mp3" / "01 本編.mp3", duration=0.7)
    main_2 = write_tiny_wav(folder / "TSSS 完パケ音声 mp3" / "02 催眠誘導.mp3", duration=0.8)
    write_tiny_wav(folder / "TSSS 完パケ音声 mp3" / "差分トラック mp3" / "04 H有（SE無）.mp3", duration=0.6)
    write_tiny_wav(folder / "TSSS 完パケ音声 mp3" / "差分トラック mp3" / "05 remix.mp3", duration=0.6)
    write_tiny_wav(folder / "TSSS 完パケ音声 mp3" / "差分トラック mp3" / "06 コピーしてお使いください.mp3", duration=0.6)

    plan = preprocess.plan_folder_input(folder)

    assert plan.status == "planned"
    assert plan.mix_parts == (main_1.resolve(), main_2.resolve())


def test_numbered_part_merge_planner_refuses_ambiguous_base_file(tmp_path: Path) -> None:
    write_tiny_wav(tmp_path / "RJAMB.wav")
    part_1 = write_tiny_wav(tmp_path / "RJAMB_1.wav")
    write_tiny_wav(tmp_path / "RJAMB_2.wav")

    plan = preprocess.plan_numbered_part_merge(part_1)

    assert plan.status == "ambiguous_base_file_exists"
    assert not plan.should_merge
    assert plan.parts == (part_1.resolve(),)


def test_full_command_runs_mock_e2e_with_project(cli_runner, tiny_wav_path: Path, tmp_path: Path) -> None:
    project = tmp_path / "full_project"
    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--no-cache-status",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "extract started" in result.output
    assert result.output.index("transcribe started") < result.output.index("segment started")
    assert "translate-ko: 1/" in result.output
    assert "korean-script: 1/" in result.output
    assert "synth: 1/" in result.output
    assert "train-rvc complete" in result.output
    assert "rvc: 1/" in result.output
    assert "mix dialogue: 1/" in result.output
    assert "Pipeline complete" in result.output
    assert "Project:" in result.output
    assert str(project.resolve()) in result.output
    manifest = load_manifest(project)
    assert manifest.rights_audit.confirmed is True
    assert manifest.stage_state["segment"]["source"] == "transcribe"
    assert manifest.artifacts["export"].endswith("_dub.wav")
    assert Path(manifest.artifacts["segments_final"]).exists()
    assert Path(manifest.artifacts["export"]).exists()


def test_full_command_splits_folder_input_outputs(cli_runner, tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "RJFULL"
    write_tiny_wav(folder / "本編" / "01 プロローグ.wav", duration=0.7)
    write_tiny_wav(folder / "本編" / "02 催眠誘導.wav", duration=0.8)
    write_tiny_wav(folder / "効果音無し" / "01 プロローグ.wav", duration=0.7)
    write_tiny_wav(folder / "効果音無し" / "02 催眠誘導.wav", duration=0.8)
    project = tmp_path / "folder_project"

    def fake_concat_audio_to_wav(
        input_paths: list[Path],
        output_path: Path,
        *,
        sample_rate: int = 48_000,
        channels: int = 2,
    ) -> Path:
        clips = []
        for input_path in input_paths:
            data, sr = sf.read(input_path, always_2d=True, dtype="float32")
            if sr != sample_rate:
                indexes = np.linspace(0, len(data) - 1, max(1, int(len(data) * sample_rate / sr)))
                data = data[indexes.astype(np.int64)]
            if channels == 1:
                data = data.mean(axis=1, keepdims=True)
            elif data.shape[1] < channels:
                data = np.repeat(data[:, :1], channels, axis=1)
            else:
                data = data[:, :channels]
            clips.append(data)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, np.concatenate(clips, axis=0), sample_rate)
        return output_path

    monkeypatch.setattr(preprocess.ffmpeg, "concat_audio_to_wav", fake_concat_audio_to_wav)

    result = cli_runner.invoke(
        app,
        [
            "full",
            str(folder),
            "--project",
            str(project),
            "--confirm-rights",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = load_manifest(project)
    output_dir = Path(manifest.artifacts["export"])
    export_manifest = json.loads(Path(manifest.artifacts["export_manifest"]).read_text("utf-8"))
    assert output_dir.is_dir()
    assert export_manifest["folder_input"] is True
    assert [Path(item["output"]).name for item in export_manifest["outputs"]] == [
        "01_프롤로그_dub.wav",
        "02_최면유도_dub.wav",
    ]
    assert all(Path(item["output"]).exists() for item in export_manifest["outputs"])
    assert Path(manifest.artifacts["folder_input_manifest"]).exists()


def test_full_merge_parts_uses_group_run_name_and_passes_flag(
    cli_runner,
    tmp_path: Path,
    monkeypatch,
) -> None:
    part_1 = write_tiny_wav(tmp_path / "RJGROUP_1.wav")
    captured: dict[str, object] = {}
    fake_repo = tmp_path / "repo"

    def fake_run_pipeline(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest(artifacts={"export": "out.wav"})

    monkeypatch.setattr(cli_module, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full",
            str(part_1),
            "--confirm-rights",
            "--merge-parts",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    args = captured["args"]
    kwargs = captured["kwargs"]
    assert Path(args[1]).name.endswith("_RJGROUP")
    assert kwargs["merge_input_parts"] is True


def test_full_real_applies_high_quality_preset_by_default(
    cli_runner,
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "full_real_hq"
    fake_repo = tmp_path / "repo"
    (fake_repo / ".cache/third_party/Retrieval-based-Voice-Conversion-WebUI").mkdir(parents=True)
    captured: dict[str, object] = {}

    def fake_run_pipeline(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest(artifacts={"export": "out.wav"})

    monkeypatch.setattr(cli_module, "REPO_ROOT", fake_repo)
    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--real",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    cfg = load_project_config(project)
    assert cfg.target_language == "ko"
    assert cfg.asr_batched_inference is True
    assert cfg.asr_batch_size == 16
    assert cfg.asr_diagnostics_enabled is True
    assert cfg.asr_resegment_max_sec == 14.0
    assert cfg.asr_repair_enabled is True
    assert cfg.asr_review_enabled is True
    assert cfg.asr_review_generate_candidates is True
    assert cfg.asr_translation_backcheck_enabled is True
    assert cfg.candidate_count == 3
    assert cfg.duration_tolerance == 0.25
    assert cfg.gsv_few_shot_target_sec == 180.0
    assert cfg.gsv_few_shot_min_clip_sec == 2.0
    assert cfg.gsv_few_shot_max_clip_sec == 8.0
    assert cfg.gsv_concurrency == 3
    assert cfg.gsv_tts_min_speed_factor == 0.75
    assert cfg.gsv_tts_max_speed_factor == 1.20
    assert cfg.gsv_top_k == 8
    assert cfg.gsv_top_p == 0.9
    assert cfg.gsv_temperature == 0.7
    assert cfg.gsv_text_split_method == "cut0"
    assert cfg.gsv_parallel_infer is False
    assert cfg.gsv_repetition_penalty == 1.25
    assert cfg.gsv_sample_steps == 32
    assert cfg.gsv_super_sampling is True
    assert cfg.gsv_min_chunk_length == 8
    assert cfg.gemma_llama_cpp_ctx_size == 16384
    assert cfg.gemma_text_batch_size == 1
    assert cfg.gemma_text_context_radius == 8
    assert cfg.gemma_text_concurrency == 4
    assert cfg.gemma_text_n_predict == 2048
    assert cfg.gemma_text_span_size == 12
    assert cfg.gemma_text_span_max_sec == 90.0
    assert cfg.gemma_text_span_max_gap_sec == 3.0
    assert cfg.mix_allow_korean_timing_draft is False
    assert cfg.rvc_required is True
    assert cfg.rvc_backend == "command"
    assert cfg.rvc_train_required is True
    assert cfg.rvc_train_backend == "command"
    assert cfg.rvc_train_timeout_sec == 43200.0
    assert cfg.rvc_train_epochs == 20
    assert cfg.rvc_train_experiment_name == f"asmr-{tiny_wav_path.stem.lower()}-speaker-1"
    assert cfg.rvc_train_command
    assert str(fake_repo / "asmr_dub_pipeline/rvc/webui_train.py") in cfg.rvc_train_command
    assert cfg.rvc_train_batch_size == 0
    assert cfg.rvc_train_preprocess_processes == 0
    assert cfg.rvc_train_f0_workers == 0
    assert cfg.rvc_train_feature_workers == 0
    assert cfg.rvc_train_save_every_epoch == 50
    assert cfg.rvc_train_reuse_intermediate_cache is True
    assert cfg.rvc_concurrency == 4
    assert cfg.rvc_batch_infer is True
    assert cfg.rvc_batch_size == 200
    assert cfg.rvc_batch_concurrency == 2
    assert "{sample_rate}" in cfg.rvc_train_command
    assert "{batch_size}" in cfg.rvc_train_command
    assert "{preprocess_processes}" in cfg.rvc_train_command
    assert "{f0_workers}" in cfg.rvc_train_command
    assert "{feature_workers}" in cfg.rvc_train_command
    assert "{epochs}" in cfg.rvc_train_command
    assert "{save_every_epoch}" in cfg.rvc_train_command
    assert "{reuse_intermediate_cache}" in cfg.rvc_train_command
    assert cfg.rvc_command
    assert str(fake_repo / "asmr_dub_pipeline/rvc/webui_infer.py") in cfg.rvc_command
    assert cfg.rvc_batch_command
    assert str(fake_repo / "asmr_dub_pipeline/rvc/webui_batch_infer.py") in cfg.rvc_batch_command
    assert cfg.rvc_failure_policy == "retry_then_error"
    assert cfg.rvc_allow_pre_rvc_fallback is False
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["mock"] is False
    assert kwargs["gemma_backend"] == "llama_cpp"
    assert kwargs["few_shot"] is True
    assert kwargs["gsv_few_shot_force"] is True
    assert kwargs["regenerate_before_mix"] is True


def test_full_real_use_trained_gpt_flag_passes_to_pipeline(
    cli_runner,
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "full_real_trained_gpt"
    captured: dict[str, object] = {}

    def fake_run_pipeline(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest(artifacts={"export": "out.wav"})

    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--real",
            "--use-trained-gpt",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["use_trained_gpt"] is True


def test_full_real_asr_backend_flag_passes_to_pipeline(
    cli_runner,
    tiny_wav_path: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "full_real_qwen_asr"
    captured: dict[str, object] = {}

    def fake_run_pipeline(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest(artifacts={"export": "out.wav"})

    monkeypatch.setattr(cli_module, "run_pipeline", fake_run_pipeline)

    result = cli_runner.invoke(
        app,
        [
            "full",
            str(tiny_wav_path),
            "--project",
            str(project),
            "--confirm-rights",
            "--real",
            "--asr-backend",
            "qwen_asr",
            "--no-cache-status",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["asr_backend"] == "qwen_asr"


def test_mix_requires_completed_qc(tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    try:
        mix_step(tmp_project_dir, confirm_rights=True)
    except ValueError as exc:
        assert "QC" in str(exc)
    else:
        raise AssertionError("Expected mix to require completed QC")
