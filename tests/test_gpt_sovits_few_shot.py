from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from conftest import write_tiny_wav

from asmr_dub_pipeline.audio.features import duration_sec, write_audio
from asmr_dub_pipeline.audio.quality import AudioQualityMetrics
from asmr_dub_pipeline.config import load_project_config, save_project_config
from asmr_dub_pipeline.gpt_sovits import few_shot
from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSError, build_tts_request
from asmr_dub_pipeline.gpt_sovits.few_shot import (
    build_training_dataset,
    discover_install,
    train_few_shot,
)
from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.stages import gsv_few_shot as gsv_stage
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import _compact_korean_counting_tts_text
from asmr_dub_pipeline.pipeline.steps import init_project, synth_step
from asmr_dub_pipeline.schemas import (
    GSVSpeakerConfig,
    JapaneseScript,
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
    TTSCandidate,
    TTSMetadata,
)
from asmr_dub_pipeline.script.duration_rewrite import estimate_tts_duration, rewrite_for_duration

pytestmark = pytest.mark.regression


def _segment(project_dir: Path, segment_id: str, start: float, duration: float, text: str) -> Segment:
    audio = project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
    write_tiny_wav(audio, duration=duration)
    return Segment(
        id=segment_id,
        speaker_id="speaker_0001",
        start=start,
        end=start + duration,
        duration=duration,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(project_dir)),
        analysis={"speaker_count": 1},
        source_script=SourceScript(
            text=text,
            language="ja",
            backend="mock",
            start=start,
            end=start + duration,
        ),
    )


def _manifest_with_segments(project_dir: Path, durations: list[float]) -> PipelineManifest:
    segments = [
        _segment(project_dir, f"seg_{index:04d}", index * 10.0, duration, f"台詞 {index}")
        for index, duration in enumerate(durations, start=1)
    ]
    return PipelineManifest(segments=segments)


def _write_tone_wav(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    write_audio(path, np.stack([tone, tone], axis=1), sample_rate)
    return path


def _fake_gsv_install(root: Path) -> Path:
    api = root / "api_v2.py"
    configs = root / "GPT_SoVITS" / "configs"
    pretrained = root / "GPT_SoVITS" / "pretrained_models"
    configs.mkdir(parents=True)
    pretrained.mkdir(parents=True)
    api.write_text("", "utf-8")
    (pretrained / "s1v3.ckpt").write_bytes(b"gpt-base")
    (pretrained / "s2Gv4.pth").write_bytes(b"sovits-base")
    (pretrained / "bert").mkdir()
    (pretrained / "hubert").mkdir()
    (configs / "tts_infer.yaml").write_text(
        yaml.safe_dump(
            {
                "custom": {
                    "version": "v4",
                    "bert_base_path": "GPT_SoVITS/pretrained_models/bert",
                    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/hubert",
                    "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
                    "vits_weights_path": "GPT_SoVITS/pretrained_models/s2Gv4.pth",
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    (configs / "s1longer-v2.yaml").write_text(
        yaml.safe_dump({"train": {}, "data": {}, "model": {}}, sort_keys=True),
        "utf-8",
    )
    (configs / "s2.json").write_text(
        json.dumps({"train": {}, "data": {}, "model": {}}, sort_keys=True),
        "utf-8",
    )
    return api


def test_discover_install_uses_tts_config_from_server_command(tmp_project_dir: Path) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    default_config = api.parent / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    default_payload = yaml.safe_load(default_config.read_text("utf-8"))
    default_payload["custom"]["vits_weights_path"] = "GPT_SoVITS/pretrained_models/missing.pth"
    default_config.write_text(yaml.safe_dump(default_payload, sort_keys=True), "utf-8")
    command_config = tmp_project_dir / "tts_infer.local.yaml"
    command_config.write_text(
        yaml.safe_dump(
            {
                "custom": {
                    "version": "v4",
                    "bert_base_path": "GPT_SoVITS/pretrained_models/bert",
                    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/hubert",
                    "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
                    "vits_weights_path": "GPT_SoVITS/pretrained_models/s2Gv4.pth",
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    command = ["python", str(api), "-c", str(command_config)]

    install = discover_install(
        ProjectConfig(project_name="test", gsv_server_command=command),
        command=command,
    )

    assert install.tts_config_path == command_config.resolve()
    assert install.pretrained_sovits_path.name == "s2Gv4.pth"


def test_few_shot_training_python_skips_project_venv_missing_deps(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(project_name="test", gsv_server_command=["python", str(api)])
    install = discover_install(cfg)
    base_python = tmp_project_dir / "conda" / "bin" / "python"
    venv_python = tmp_project_dir / ".venv" / "bin" / "python"
    monkeypatch.setenv("PATH", str(venv_python.parent))
    monkeypatch.setattr(few_shot.sys, "executable", str(venv_python))
    monkeypatch.setattr(few_shot.sys, "base_prefix", str(base_python.parent.parent))

    def fake_missing_imports(python: str, modules) -> list[str] | None:
        if python == str(base_python):
            return []
        if python in {"python", str(venv_python)}:
            if "ffmpeg" in modules:
                return ["ffmpeg"]
            return []
        return None

    monkeypatch.setattr(few_shot, "_python_missing_imports", fake_missing_imports)

    selected = few_shot._select_training_python(cfg, install, None, require_modules=True)

    assert selected == str(base_python)


def test_few_shot_training_python_env_override(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(project_name="test", gsv_server_command=["python", str(api)])
    install = discover_install(cfg)
    override = tmp_project_dir / "gsv_env" / "bin" / "python"
    monkeypatch.setenv("ASMR_DUB_GSV_PYTHON", str(override))
    monkeypatch.setattr(few_shot, "_python_missing_imports", lambda python, modules: [])

    selected = few_shot._select_training_python(cfg, install, None, require_modules=True)

    assert selected == str(override)


def test_few_shot_training_env_includes_python_nvrtc_lib_dir(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(project_name="test", gsv_server_command=["python", str(api)])
    install = discover_install(cfg)
    venv = tmp_project_dir / "venv"
    nvrtc_dir = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib"
    nvrtc_dir.mkdir(parents=True)
    (nvrtc_dir / "libnvrtc-builtins.so.13.0").write_bytes(b"")
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setattr(few_shot.sys, "prefix", str(venv))
    monkeypatch.setattr(few_shot.sys, "base_prefix", str(tmp_project_dir / "base"))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    dataset = few_shot.FewShotDataset(
        items=[],
        wav_dir=tmp_project_dir / "wavs",
        list_path=tmp_project_dir / "dataset.list",
        total_duration_sec=0.0,
    )

    env = few_shot._base_env(cfg, dataset, install, s2_config_path=install.s2_config_path)

    ld_paths = env["LD_LIBRARY_PATH"].split(os.pathsep)
    assert ld_paths[0] == str(nvrtc_dir)
    assert "/existing" in ld_paths


def test_few_shot_training_python_checks_text_and_semantic_deps(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(project_name="test", gsv_server_command=["python", str(api)])
    install = discover_install(cfg)
    base_python = tmp_project_dir / "conda" / "bin" / "python"
    venv_python = tmp_project_dir / ".venv" / "bin" / "python"
    monkeypatch.setenv("PATH", str(venv_python.parent))
    monkeypatch.setattr(few_shot.sys, "executable", str(venv_python))
    monkeypatch.setattr(few_shot.sys, "base_prefix", str(base_python.parent.parent))

    def fake_missing_imports(python: str, modules) -> list[str] | None:
        if python == str(base_python):
            return []
        if python in {"python", str(venv_python)}:
            return [
                module
                for module in ("pyopenjtalk", "x_transformers")
                if module in modules
            ]
        return None

    monkeypatch.setattr(few_shot, "_python_missing_imports", fake_missing_imports)

    selected = few_shot._select_training_python(cfg, install, None, require_modules=True)

    assert selected == str(base_python)


def test_python_missing_imports_rejects_wrong_ffmpeg_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_site = tmp_path / "fake_site"
    fake_site.mkdir()
    (fake_site / "ffmpeg.py").write_text("# wrong ffmpeg package without ffmpeg-python API\n", "utf-8")
    monkeypatch.setenv("PYTHONPATH", str(fake_site))

    missing = few_shot._python_missing_imports(sys.executable, ["ffmpeg"])

    assert missing == ["ffmpeg"]


def test_few_shot_dataset_selects_source_segments(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 0.5, 1.4])
    for segment in manifest.segments:
        segment.speaker_id = "speaker_0001"
        segment.analysis = {"speaker_count": 1}
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == ["seg_0001", "seg_0003"]
    assert dataset.total_duration_sec == pytest.approx(2.4)
    lines = dataset.list_path.read_text("utf-8").splitlines()
    assert lines == [
        "seg_0001.wav|source_voice|ja|せりふ いち",
        "seg_0003.wav|source_voice|ja|せりふ さん",
    ]
    assert dataset.items[0].text == "せりふ いち"
    assert dataset.items[0].source_text_original == "台詞 1"
    assert (dataset.wav_dir / "seg_0001.wav").exists()
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    assert qc["clips"][0]["source_language"] == "ja"
    assert qc["clips"][0]["target_language"] == "ko"
    assert qc["clips"][0]["source_text"] == "せりふ いち"
    assert qc["clips"][0]["source_text_original"] == "台詞 1"
    assert qc["clips"][0]["quality_score"] > 0


def test_few_shot_dataset_uses_all_quality_segments_after_minimum(
    tmp_project_dir: Path,
) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.1, 1.2, 1.3])
    for segment in manifest.segments:
        segment.speaker_id = "speaker_0001"
        segment.analysis = {"speaker_count": 1}
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == [
        "seg_0001",
        "seg_0002",
        "seg_0003",
        "seg_0004",
    ]
    assert dataset.total_duration_sec == pytest.approx(4.6)
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    assert all(clip["selected_for_training"] for clip in qc["clips"])
    assert all("not_selected_target_reached" not in clip["reject_reasons"] for clip in qc["clips"])


def test_few_shot_dataset_rejects_overly_fast_source_speech(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    manifest = PipelineManifest(
        segments=[
            _segment(tmp_project_dir, "seg_0001", 0.0, 3.0, "これはとても速く詰め込まれた長い長い長い台詞です"),
            _segment(tmp_project_dir, "seg_0002", 10.0, 4.0, "ゆっくり話すね"),
            _segment(tmp_project_dir, "seg_0003", 20.0, 4.0, "静かに続けるね"),
        ]
    )
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=7.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
        gsv_few_shot_preferred_chars_per_sec=4.5,
        gsv_few_shot_max_chars_per_sec=5.2,
    )
    scores = {"seg_0001": 1.0, "seg_0002": 0.82, "seg_0003": 0.81}

    def fake_evaluate_voice_training_candidate(project_dir, segment, *args, **kwargs):
        audio_path = Path(segment.audio_for_mix)
        if not audio_path.is_absolute():
            audio_path = project_dir / audio_path
        metrics = AudioQualityMetrics(
            duration_sec=segment.duration,
            peak_dbfs=-6.0,
            rms_dbfs=-24.0,
            clipping_ratio=0.0,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            active_ratio=0.9,
            silence_ratio=0.1,
            estimated_snr_db=30.0,
            score=scores[segment.id],
            issues=[],
        )
        return few_shot.VoiceTrainingCandidateCheck(
            accepted=True,
            source_audio_path=audio_path,
            metrics=metrics,
            reject_reasons=(),
        )

    monkeypatch.setattr(few_shot, "evaluate_voice_training_candidate", fake_evaluate_voice_training_candidate)

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == ["seg_0002", "seg_0003"]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    by_segment = {clip["segment_id"]: clip for clip in qc["clips"]}
    assert by_segment["seg_0001"]["source_chars_per_sec"] > cfg.gsv_few_shot_max_chars_per_sec
    assert by_segment["seg_0001"]["reject_reasons"] == [
        "source_chars_per_sec_above_max:8.000>5.200"
    ]
    assert by_segment["seg_0002"]["training_selection_score"] > by_segment["seg_0001"]["training_selection_score"]


def test_few_shot_dataset_rejects_asr_risk_segments_for_training(
    tmp_project_dir: Path,
) -> None:
    init_project(tmp_project_dir)
    manifest = PipelineManifest(
        segments=[
            _segment(tmp_project_dir, "seg_0001", 0.0, 2.0, "これは使わない"),
            _segment(tmp_project_dir, "seg_0002", 10.0, 2.0, "これも使わない"),
            _segment(tmp_project_dir, "seg_0003", 20.0, 2.0, "これは使う"),
        ]
    )
    transcribe_dir = tmp_project_dir / "work" / "transcribe"
    transcribe_dir.mkdir(parents=True, exist_ok=True)
    (transcribe_dir / "asr_high_risk_report.json").write_text(
        json.dumps(
            {
                "schema_version": "asr-high-risk-1.0",
                "summary": {},
                "items": [
                    {
                        "segment_id": "seg_0001",
                        "severity": "warning",
                        "reasons": ["asr_countdown_unverified"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        "utf-8",
    )
    (transcribe_dir / "asr_postprocess_review.json").write_text(
        json.dumps(
            {
                "schema_version": "asr-postprocess-review-1.0",
                "summary": {},
                "items": [
                    {
                        "segment_id": "seg_0002",
                        "action": "manual_review",
                        "reasons": ["asr_review_required"],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        "utf-8",
    )
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
        gsv_few_shot_asr_risk_filter=True,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == ["seg_0003"]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected["seg_0001"] == ["asr_high_risk_warning:asr_countdown_unverified"]
    assert rejected["seg_0002"] == ["asr_postprocess_manual_review:asr_review_required"]


def test_few_shot_dataset_applies_selection_score_and_max_total(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    manifest = PipelineManifest(
        segments=[
            _segment(tmp_project_dir, "seg_best", 0.0, 3.0, "静かに話す"),
            _segment(tmp_project_dir, "seg_good", 10.0, 3.0, "やさしく話す"),
            _segment(tmp_project_dir, "seg_extra", 20.0, 3.0, "少し強く話す"),
            _segment(tmp_project_dir, "seg_low", 30.0, 3.0, "低い点数"),
        ]
    )
    scores = {
        "seg_best": 0.91,
        "seg_good": 0.82,
        "seg_extra": 0.78,
        "seg_low": 0.55,
    }

    def fake_evaluate_voice_training_candidate(project_dir, segment, *args, **kwargs):
        audio_path = Path(segment.audio_for_mix)
        if not audio_path.is_absolute():
            audio_path = project_dir / audio_path
        metrics = AudioQualityMetrics(
            duration_sec=segment.duration,
            peak_dbfs=-6.0,
            rms_dbfs=-24.0,
            clipping_ratio=0.0,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            active_ratio=0.9,
            silence_ratio=0.1,
            estimated_snr_db=30.0,
            score=scores[segment.id],
            issues=[],
        )
        return few_shot.VoiceTrainingCandidateCheck(
            accepted=True,
            source_audio_path=audio_path,
            metrics=metrics,
            reject_reasons=(),
        )

    monkeypatch.setattr(few_shot, "evaluate_voice_training_candidate", fake_evaluate_voice_training_candidate)
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=6.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
        gsv_few_shot_min_selection_score=0.65,
        gsv_few_shot_max_total_sec=6.0,
        gsv_few_shot_prefer_plain_text=False,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == ["seg_best", "seg_good"]
    assert dataset.total_duration_sec == pytest.approx(6.0)
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected["seg_low"] == ["training_selection_score_below_min:0.550<0.650"]
    assert rejected["seg_extra"] == ["max_total_sec_trimmed:6.000"]


def test_few_shot_dataset_keeps_all_quality_timing_buckets_after_target_duration(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    manifest = PipelineManifest(
        segments=[
            _segment(tmp_project_dir, "seg_fast_1", 0.0, 2.0, "長長長長長長"),
            _segment(tmp_project_dir, "seg_fast_2", 10.0, 2.0, "声声声声声声"),
            _segment(tmp_project_dir, "seg_fast_3", 20.0, 2.0, "音音音音音音"),
            _segment(tmp_project_dir, "seg_normal", 30.0, 2.0, "あいうえおかきく"),
            _segment(tmp_project_dir, "seg_slow", 40.0, 2.0, "あいうえ"),
        ]
    )
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=6.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
        gsv_few_shot_max_chars_per_sec=20.0,
    )
    scores = {
        "seg_fast_1": 0.99,
        "seg_fast_2": 0.98,
        "seg_fast_3": 0.97,
        "seg_normal": 0.81,
        "seg_slow": 0.80,
    }

    def fake_evaluate_voice_training_candidate(project_dir, segment, *args, **kwargs):
        audio_path = Path(segment.audio_for_mix)
        if not audio_path.is_absolute():
            audio_path = project_dir / audio_path
        metrics = AudioQualityMetrics(
            duration_sec=segment.duration,
            peak_dbfs=-6.0,
            rms_dbfs=-24.0,
            clipping_ratio=0.0,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            active_ratio=0.9,
            silence_ratio=0.1,
            estimated_snr_db=30.0,
            score=scores[segment.id],
            issues=[],
        )
        return few_shot.VoiceTrainingCandidateCheck(
            accepted=True,
            source_audio_path=audio_path,
            metrics=metrics,
            reject_reasons=(),
        )

    monkeypatch.setattr(few_shot, "evaluate_voice_training_candidate", fake_evaluate_voice_training_candidate)

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == [
        "seg_fast_1",
        "seg_fast_2",
        "seg_fast_3",
        "seg_normal",
        "seg_slow",
    ]
    assert dataset.total_duration_sec == pytest.approx(10.0)
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    by_segment = {clip["segment_id"]: clip for clip in qc["clips"]}
    assert by_segment["seg_fast_1"]["timing_bucket"] == "fast"
    assert by_segment["seg_normal"]["timing_bucket"] == "normal"
    assert by_segment["seg_slow"]["timing_bucket"] == "very_slow"
    assert all(clip["selected_for_training"] for clip in qc["clips"])
    assert all("not_selected_target_reached" not in clip["reject_reasons"] for clip in qc["clips"])


def test_few_shot_dataset_prefers_source_pacing_near_korean_target(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    segments = [
        _segment(tmp_project_dir, "seg_fast_1", 0.0, 2.0, "あいうえおかきくけこ"),
        _segment(tmp_project_dir, "seg_fast_2", 10.0, 2.0, "さしすせそたちつてと"),
        _segment(tmp_project_dir, "seg_fast_3", 20.0, 2.0, "なにぬねのはひふへほ"),
        _segment(tmp_project_dir, "seg_target_1", 30.0, 2.0, "あいうえお"),
        _segment(tmp_project_dir, "seg_target_2", 40.0, 2.0, "かきくけこ"),
        _segment(tmp_project_dir, "seg_target_3", 50.0, 2.0, "さしすせそ"),
    ]
    for segment in segments:
        segment.script = JapaneseScript(
            ja_text=segment.source_script.text,
            tts_text="가나다라마",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=segment.duration,
        )
    manifest = PipelineManifest(segments=segments)
    cfg = ProjectConfig(
        project_name="test",
        target_language="ko",
        gsv_few_shot_min_total_sec=6.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
        gsv_few_shot_max_chars_per_sec=20.0,
    )
    scores = {
        "seg_fast_1": 0.99,
        "seg_fast_2": 0.98,
        "seg_fast_3": 0.97,
        "seg_target_1": 0.80,
        "seg_target_2": 0.79,
        "seg_target_3": 0.78,
    }

    def fake_evaluate_voice_training_candidate(project_dir, segment, *args, **kwargs):
        audio_path = Path(segment.audio_for_mix)
        if not audio_path.is_absolute():
            audio_path = project_dir / audio_path
        metrics = AudioQualityMetrics(
            duration_sec=segment.duration,
            peak_dbfs=-6.0,
            rms_dbfs=-24.0,
            clipping_ratio=0.0,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            active_ratio=0.9,
            silence_ratio=0.1,
            estimated_snr_db=30.0,
            score=scores[segment.id],
            issues=[],
        )
        return few_shot.VoiceTrainingCandidateCheck(
            accepted=True,
            source_audio_path=audio_path,
            metrics=metrics,
            reject_reasons=(),
        )

    monkeypatch.setattr(few_shot, "evaluate_voice_training_candidate", fake_evaluate_voice_training_candidate)

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert {item.segment_id for item in dataset.items} == {
        "seg_fast_1",
        "seg_fast_2",
        "seg_fast_3",
        "seg_target_1",
        "seg_target_2",
        "seg_target_3",
    }
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    by_segment = {clip["segment_id"]: clip for clip in qc["clips"]}
    assert all(clip["selected_for_training"] for clip in qc["clips"])
    assert by_segment["seg_target_1"]["target_pacing_score"] > by_segment["seg_fast_1"]["target_pacing_score"]
    assert by_segment["seg_target_1"]["training_selection_score"] > by_segment["seg_fast_1"]["training_selection_score"]
    assert by_segment["seg_target_1"]["target_pacing_ratio"] == pytest.approx(1.0)
    assert by_segment["seg_fast_1"]["target_pacing_ratio"] == pytest.approx(0.5)


def test_few_shot_dataset_rejects_clean_source_failures_without_pacing_relaxation(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    segments = [
        _segment(tmp_project_dir, "seg_target_1", 0.0, 4.0, "あいうえおかきくけこさしすせそた"),
        _segment(tmp_project_dir, "seg_target_2", 10.0, 4.0, "なにぬねのはひふへほまみむめもや"),
        _segment(tmp_project_dir, "seg_target_3", 20.0, 4.0, "らりるれろわをんあいうえおかきく"),
        _segment(tmp_project_dir, "seg_slow_soft", 30.0, 4.0, "あいうえおかきく"),
        _segment(tmp_project_dir, "seg_fast_soft", 40.0, 4.0, "あいうえおかきくけこさしすせそたちつてとなにぬね"),
    ]
    for segment in segments:
        segment.script = JapaneseScript(
            ja_text=segment.source_script.text,
            tts_text="가나다라마바사아자차카타파하가나",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=segment.duration,
        )
    manifest = PipelineManifest(segments=segments)
    cfg = ProjectConfig(
        project_name="test",
        target_language="ko",
        gsv_few_shot_min_total_sec=12.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
        gsv_few_shot_max_chars_per_sec=20.0,
    )
    scores = {
        "seg_target_1": 0.90,
        "seg_target_2": 0.89,
        "seg_target_3": 0.88,
        "seg_slow_soft": 0.76,
        "seg_fast_soft": 0.77,
    }

    def fake_evaluate_voice_training_candidate(project_dir, segment, *args, **kwargs):
        audio_path = Path(segment.audio_for_mix)
        if not audio_path.is_absolute():
            audio_path = project_dir / audio_path
        metrics = AudioQualityMetrics(
            duration_sec=segment.duration,
            peak_dbfs=-6.0,
            rms_dbfs=-24.0,
            clipping_ratio=0.0,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            active_ratio=0.9,
            silence_ratio=0.1,
            estimated_snr_db=30.0,
            score=scores[segment.id],
            issues=[],
        )
        reject_reasons = (
            ("side_to_mid_db_above_max:-2.000>-6.000",)
            if segment.id.endswith("_soft")
            else ()
        )
        return few_shot.VoiceTrainingCandidateCheck(
            accepted=not reject_reasons,
            source_audio_path=audio_path,
            metrics=metrics,
            reject_reasons=reject_reasons,
            clean_source_metrics={"side_to_mid_db": -2.0} if reject_reasons else {},
        )

    monkeypatch.setattr(few_shot, "evaluate_voice_training_candidate", fake_evaluate_voice_training_candidate)

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    selected_ids = {item.segment_id for item in dataset.items}
    assert selected_ids == {"seg_target_1", "seg_target_2", "seg_target_3"}
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected == {
        "seg_slow_soft": ["side_to_mid_db_above_max:-2.000>-6.000"],
        "seg_fast_soft": ["side_to_mid_db_above_max:-2.000>-6.000"],
    }


def test_few_shot_dataset_prefers_plain_source_text_over_effect_like_lines(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    manifest = PipelineManifest(
        segments=[
            _segment(tmp_project_dir, "seg_0001", 0.0, 4.0, "2 ぎゅるぎゅるっと強く響いている"),
            _segment(tmp_project_dir, "seg_0002", 10.0, 4.0, "今は少しだけ静かに話しますね。"),
            _segment(tmp_project_dir, "seg_0003", 20.0, 4.0, "3 ぷしゃっと音が続いている"),
        ]
    )
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=12.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )
    scores = {"seg_0001": 0.96, "seg_0002": 0.84, "seg_0003": 0.95}

    def fake_evaluate_voice_training_candidate(project_dir, segment, *args, **kwargs):
        audio_path = Path(segment.audio_for_mix)
        if not audio_path.is_absolute():
            audio_path = project_dir / audio_path
        metrics = AudioQualityMetrics(
            duration_sec=segment.duration,
            peak_dbfs=-6.0,
            rms_dbfs=-24.0,
            clipping_ratio=0.0,
            leading_silence_sec=0.0,
            trailing_silence_sec=0.0,
            active_ratio=0.9,
            silence_ratio=0.1,
            estimated_snr_db=30.0,
            score=scores[segment.id],
            issues=[],
        )
        return few_shot.VoiceTrainingCandidateCheck(
            accepted=True,
            source_audio_path=audio_path,
            metrics=metrics,
            reject_reasons=(),
        )

    monkeypatch.setattr(few_shot, "evaluate_voice_training_candidate", fake_evaluate_voice_training_candidate)

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == ["seg_0002", "seg_0001", "seg_0003"]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    by_segment = {clip["segment_id"]: clip for clip in qc["clips"]}
    assert by_segment["seg_0001"]["selection_penalties"]
    assert by_segment["seg_0003"]["selection_penalties"]
    assert by_segment["seg_0002"]["training_selection_score"] > by_segment["seg_0001"]["training_selection_score"]
    assert by_segment["seg_0002"]["training_selection_score"] > by_segment["seg_0003"]["training_selection_score"]


def test_few_shot_dataset_ignores_skipped_status_segments(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [2.0, 7.0])
    manifest.segments[0].speaker_id = "speaker_0001"
    manifest.segments[1].speaker_id = "speaker_0002"
    manifest.segments[1].status = "needs_manual_review"
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    speaker_ids = few_shot.select_training_speaker_ids(tmp_project_dir, manifest, cfg)
    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert speaker_ids == ["speaker_0001"]
    assert [item.segment_id for item in dataset.items] == ["seg_0001"]


def test_few_shot_training_speaker_ids_skip_model_fallback_speakers(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [2.0, 1.0])
    manifest.segments[0].speaker_id = "speaker_0001"
    manifest.segments[1].speaker_id = "speaker_0002"
    manifest.segments[1].analysis["source_speaker_model_fallback"] = {
        "speaker_id": "speaker_0001",
        "reason": "insufficient_distinct_speaker_training_data",
    }
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    speaker_ids = few_shot.select_training_speaker_ids(tmp_project_dir, manifest, cfg)
    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert speaker_ids == ["speaker_0001"]
    assert [item.segment_id for item in dataset.items] == ["seg_0001"]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected["seg_0002"] == ["source_speaker_model_fallback:speaker_0001"]


def test_few_shot_training_speaker_ids_skip_low_dominant_source_overlap(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [2.0, 2.0])
    manifest.segments[0].speaker_id = "speaker_0001"
    manifest.segments[1].speaker_id = "speaker_0002"
    manifest.segments[1].analysis["source_speaker_assignment"] = {
        "speaker_id": "speaker_0002",
        "speaker_count": 1,
        "dominant_overlap_ratio": 0.708205,
        "overlaps": {"speaker_0002": 1.41641},
    }
    manifest.segments[1].analysis["voice_training"] = {
        "exclude": True,
        "reason": "low_dominant_source_speaker_overlap",
    }
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    speaker_ids = few_shot.select_training_speaker_ids(tmp_project_dir, manifest, cfg)
    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert speaker_ids == ["speaker_0001"]
    assert [item.segment_id for item in dataset.items] == ["seg_0001"]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected["seg_0002"] == ["manual_training_exclude"]


def test_few_shot_dataset_keeps_downstream_failed_segments_with_source_audio(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [2.0, 2.0])
    manifest.segments[0].status = "failed"
    manifest.segments[0].errors.append("No acceptable TTS candidates for mix.")
    manifest.segments[1].status = "needs_manual_review"
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == ["seg_0001"]


def test_train_gsv_reports_suspicious_small_speaker_bucket_before_training(
    tmp_project_dir: Path,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
        gsv_few_shot_insufficient_policy="error",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    manifest = _manifest_with_segments(tmp_project_dir, [2.0, 1.0])
    manifest.segments[0].speaker_id = "speaker_0001"
    manifest.segments[1].speaker_id = "speaker_0002"
    save_manifest(tmp_project_dir, manifest)

    with pytest.raises(
        GPTSoVITSError,
        match=r"source speaker sanity check failed.*speaker_0002.*1\.00s.*2\.00s",
    ):
        steps.gsv_few_shot_step(tmp_project_dir, confirm_rights=True)


def test_train_gsv_registers_speaker_models_for_mixed_speakers(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=1.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_ref_min_sec=1.0,
        gsv_ref_max_sec=10.0,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    manifest = _manifest_with_segments(tmp_project_dir, [1.2, 1.2])
    manifest.segments[0].speaker_id = "speaker_0001"
    manifest.segments[1].speaker_id = "speaker_0002"
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str | None, Path | None]] = []

    def fake_train_few_shot(project_dir: Path, manifest: PipelineManifest, cfg: ProjectConfig, **kwargs):
        speaker_id = kwargs["speaker_id"]
        work_dir = kwargs["work_dir"]
        assert isinstance(speaker_id, str)
        assert isinstance(work_dir, Path)
        calls.append((speaker_id, work_dir))
        gpt = work_dir / "weights" / "gpt" / "final.ckpt"
        sovits = work_dir / "weights" / "sovits" / "final.pth"
        metadata = work_dir / "training_manifest.json"
        for path, payload in ((gpt, b"gpt"), (sovits, b"sovits"), (metadata, b"{}")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        item = type("Item", (), {"segment_id": f"{speaker_id}_seg", "speaker_id": speaker_id})()
        dataset = type(
            "Dataset",
            (),
            {
                "items": [item],
                "list_path": work_dir / "dataset.list",
                "wav_dir": work_dir / "wavs",
                "total_duration_sec": 1.2,
            },
        )()
        install = type("Install", (), {"root": project_dir / "gsv", "checkout": None, "version": "v4"})()
        return type(
            "Result",
            (),
            {
                "status": "completed",
                "fingerprint": f"fp-{speaker_id}",
                "dataset": dataset,
                "install": install,
                "metadata_path": metadata,
                "gpt_weights_path": gpt,
                "sovits_weights_path": sovits,
                "gpt_weights_sha256": "gpt-sha",
                "sovits_weights_sha256": "sovits-sha",
                "reused_existing": False,
                "log_path": work_dir / "logs" / "train.log",
            },
        )()

    monkeypatch.setattr(gsv_stage, "train_few_shot", fake_train_few_shot)

    trained = steps.gsv_few_shot_step(tmp_project_dir, confirm_rights=True)

    assert [speaker_id for speaker_id, _ in calls] == ["speaker_0001", "speaker_0002"]
    assert sorted(trained.project_config.gsv_speaker_models) == ["speaker_0001", "speaker_0002"]
    saved_cfg = load_project_config(tmp_project_dir)
    assert sorted(saved_cfg.gsv_speaker_models) == ["speaker_0001", "speaker_0002"]
    for speaker_id, speaker_cfg in saved_cfg.gsv_speaker_models.items():
        assert Path(speaker_cfg.gpt_weights_path or "").exists()
        assert Path(speaker_cfg.sovits_weights_path).exists()
        assert Path(speaker_cfg.refs_path).exists()
        assert f"refs/speakers/{speaker_id}/refs.json" in speaker_cfg.refs_path


def test_train_gsv_keeps_single_model_when_other_speaker_is_not_training_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=1.0,
        gsv_few_shot_min_clip_sec=1.0,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    manifest = _manifest_with_segments(tmp_project_dir, [1.2, 1.2])
    manifest.segments[0].speaker_id = "speaker_0001"
    manifest.segments[1].speaker_id = "speaker_0002"
    manifest.segments[1].analysis = {"speaker_count": 2}
    save_manifest(tmp_project_dir, manifest)
    calls: list[str | None] = []

    def fake_train_few_shot(project_dir: Path, manifest: PipelineManifest, cfg: ProjectConfig, **kwargs):
        calls.append(kwargs.get("speaker_id"))
        base_dir = project_dir / "work" / "gpt_sovits" / "few_shot"
        gpt = base_dir / "weights" / "gpt" / "final.ckpt"
        sovits = base_dir / "weights" / "sovits" / "final.pth"
        metadata = base_dir / "training_manifest.json"
        for path, payload in ((gpt, b"gpt"), (sovits, b"sovits"), (metadata, b"{}")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        item = type("Item", (), {"segment_id": "seg_0001", "speaker_id": "speaker_0001"})()
        dataset = type(
            "Dataset",
            (),
            {
                "items": [item],
                "list_path": base_dir / "dataset.list",
                "wav_dir": base_dir / "wavs",
                "total_duration_sec": 1.2,
            },
        )()
        install = type("Install", (), {"root": project_dir / "gsv", "checkout": None, "version": "v4"})()
        return type(
            "Result",
            (),
            {
                "status": "completed",
                "fingerprint": "fp-single",
                "dataset": dataset,
                "install": install,
                "metadata_path": metadata,
                "gpt_weights_path": gpt,
                "sovits_weights_path": sovits,
                "gpt_weights_sha256": "gpt-sha",
                "sovits_weights_sha256": "sovits-sha",
                "reused_existing": False,
                "log_path": base_dir / "logs" / "train.log",
            },
        )()

    monkeypatch.setattr(gsv_stage, "train_few_shot", fake_train_few_shot)

    trained = steps.gsv_few_shot_step(tmp_project_dir, confirm_rights=True)

    assert calls == [None]
    assert trained.project_config.gsv_speaker_models == {}
    assert "gsv_few_shot_speaker_models" not in trained.artifacts


def test_few_shot_dataset_filters_effected_and_multi_speaker_segments(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.0, 1.2, 1.2])
    clean_one, effected, clean_two, multi_speaker = manifest.segments
    for segment in manifest.segments:
        segment.speaker_id = "speaker_0001"
        segment.analysis = {"speaker_count": 1}
    effected.analysis = {"speaker_count": 1, "style_tags": ["whisper", "reverb"]}
    multi_speaker.analysis = {"speaker_count": 2}
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == [clean_one.id, clean_two.id]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected[effected.id] == ["disallowed_training_tag:reverb"]
    assert rejected[multi_speaker.id] == ["speaker_count_not_one:2"]


def test_few_shot_dataset_requires_none_effect_tag_for_training(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.0, 1.2])
    clean_one, effected, clean_two = manifest.segments
    for segment in manifest.segments:
        segment.speaker_id = "speaker_0001"
        segment.analysis = {
            "speaker_count": 1,
            "voice_training": {
                "clean_voice": True,
                "eligible": True,
                "effect_tags": ["none"],
            },
        }
    effected.analysis["voice_training"]["effect_tags"] = ["pitch_shift"]
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == [clean_one.id, clean_two.id]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected[effected.id] == ["voice_training_effect_tag_not_none:pitch_shift"]


def test_few_shot_dataset_rejects_background_bleed_for_tts_quality(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    sample_rate = 48_000
    duration = 1.2
    manifest = _manifest_with_segments(tmp_project_dir, [duration, duration, duration])
    contaminated, clean_one, clean_two = manifest.segments
    for segment in manifest.segments:
        segment.speaker_id = "speaker_0001"
        segment.analysis = {"speaker_count": 1}
        t = np.linspace(0.0, segment.duration, int(sample_rate * segment.duration), endpoint=False)
        voice = (0.08 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        write_audio(Path(segment.audio_for_mix), np.stack([voice, voice], axis=1), sample_rate)

    background_duration = max(segment.end for segment in manifest.segments) + 1.0
    background = np.zeros((int(sample_rate * background_duration), 2), dtype=np.float32)
    start = int(round(contaminated.start * sample_rate))
    end = start + int(round(contaminated.duration * sample_rate))
    t = np.linspace(0.0, contaminated.duration, end - start, endpoint=False)
    bleed = (0.08 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    background[start:end] = np.stack([bleed, bleed], axis=1)
    write_audio(tmp_project_dir / "work" / "audio" / "background_only_48k.wav", background, sample_rate)
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == [clean_one.id, clean_two.id]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert any(reason.startswith("background_bleed_db_above_max:") for reason in rejected[contaminated.id])


def test_few_shot_dataset_ranks_mildly_contaminated_source_lower(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    sample_rate = 48_000
    duration = 1.2
    manifest = _manifest_with_segments(tmp_project_dir, [duration, duration])
    contaminated, clean = manifest.segments
    for segment in manifest.segments:
        segment.speaker_id = "speaker_0001"
        segment.analysis = {"speaker_count": 1}
        t = np.linspace(0.0, segment.duration, int(sample_rate * segment.duration), endpoint=False)
        voice = (0.08 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        write_audio(Path(segment.audio_for_mix), np.stack([voice, voice], axis=1), sample_rate)

    background_duration = max(segment.end for segment in manifest.segments) + 1.0
    background = np.zeros((int(sample_rate * background_duration), 2), dtype=np.float32)
    start = int(round(contaminated.start * sample_rate))
    end = start + int(round(contaminated.duration * sample_rate))
    t = np.linspace(0.0, contaminated.duration, end - start, endpoint=False)
    mild_bleed = (0.08 * (10 ** (-30.0 / 20.0)) * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    background[start:end] = np.stack([mild_bleed, mild_bleed], axis=1)
    write_audio(tmp_project_dir / "work" / "audio" / "background_only_48k.wav", background, sample_rate)
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.4,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == [clean.id, contaminated.id]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    scores = {clip["segment_id"]: clip["quality_score"] for clip in qc["clips"]}
    assert scores[clean.id] > scores[contaminated.id]


def test_few_shot_dataset_rejects_duplicate_source_audio_segments(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.0, 1.0])
    first, duplicate, unique = manifest.segments
    duplicate.audio_for_mix = first.audio_for_mix
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == [first.id, unique.id]
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    rejected = {clip["segment_id"]: clip["reject_reasons"] for clip in qc["clips"] if not clip["selected_for_training"]}
    assert rejected[duplicate.id] == [f"duplicate_source_audio:{first.id}"]


def test_few_shot_dataset_rejects_too_little_source_voice(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0])
    cfg = ProjectConfig(project_name="test", gsv_few_shot_min_total_sec=2.0)

    with pytest.raises(Exception, match="Not enough source voice data"):
        build_training_dataset(tmp_project_dir, manifest, cfg)


def test_gsv_few_shot_zero_shot_policy_skips_insufficient_training_data(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_min_total_sec=2.0,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    save_manifest(tmp_project_dir, _manifest_with_segments(tmp_project_dir, [1.0]))

    assert cfg.gsv_few_shot_insufficient_policy == "zero_shot"
    skipped = steps.gsv_few_shot_step(tmp_project_dir, confirm_rights=True)

    assert skipped.stage_state["gsv-few-shot"]["status"] == "skipped_insufficient_training_data"
    assert skipped.stage_state["gsv-few-shot"]["policy"] == "zero_shot"
    assert "gsv_few_shot_gpt_weights" not in skipped.artifacts
    assert "gsv_few_shot_sovits_weights" not in skipped.artifacts


def test_few_shot_training_runs_commands_and_reuses_matching_weights(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(
        project_name="test",
        gsv_url="http://127.0.0.1:9880",
        gsv_server_command=["python", str(api)],
        gsv_few_shot_min_total_sec=2.0,
    )
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.2])
    commands: list[list[str]] = []
    progress_events: list[tuple[str, str, int, int, str | None]] = []

    def runner(command, **kwargs):
        commands.append(list(command))
        env_paths = kwargs["env"]["PYTHONPATH"].split(":")
        assert str(api.parent) in env_paths
        assert str(api.parent / "GPT_SoVITS") in env_paths
        if any("s2_train_v3_lora.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/sovits/final.pth"
            weights.write_bytes(b"sovits-trained")
        if any("s1_train.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/gpt/final.ckpt"
            weights.write_bytes(b"gpt-trained")
        return subprocess.CompletedProcess(command, 0)

    def record_progress(event) -> None:
        progress_events.append((event.phase, event.status, event.index, event.total, event.detail))

    first = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner, progress_callback=record_progress)
    second = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner, progress_callback=record_progress)
    forced = train_few_shot(
        tmp_project_dir,
        manifest,
        cfg,
        force=True,
        runner=runner,
        progress_callback=record_progress,
    )

    assert first.status == "completed"
    assert second.status == "skipped"
    assert second.reused_existing is True
    assert forced.status == "completed"
    assert [Path(command[2]).name for command in commands[:4]] == [
        "1-get-text.py",
        "2-get-hubert-wav32k.py",
        "3-get-semantic.py",
        "s2_train_v3_lora.py",
    ]
    assert sum(any("s1_train.py" in part for part in command) for command in commands) == 2
    assert ("dataset", "completed", 0, 5, "selected=2 duration=2.20s") in progress_events
    assert any(event[:4] == ("fine-tune-sovits", "started", 4, 5) for event in progress_events)
    assert any(event[:4] == ("fine-tune-gpt", "completed", 5, 5) for event in progress_events)
    assert any(event[:2] == ("reuse", "skipped") for event in progress_events)


def test_run_logged_streams_training_output(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    events = []
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('epoch 1\\r'); sys.stdout.flush(); "
        "sys.stdout.write('saving final\\n'); sys.stdout.flush()",
    ]

    few_shot._run_logged(
        subprocess.run,
        command,
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=log_path,
        phase="fine-tune-gpt",
        index=1,
        total=1,
        progress_callback=events.append,
    )

    output_events = [event.detail for event in events if event.status == "output"]
    assert "epoch 1" in output_events
    assert "saving final" in output_events
    assert "saving final" in log_path.read_text("utf-8")


def test_few_shot_fingerprint_changes_when_training_audio_changes(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(
        project_name="test",
        gsv_url="http://127.0.0.1:9880",
        gsv_server_command=["python", str(api)],
        gsv_few_shot_min_total_sec=2.0,
    )
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.2])

    def runner(command, **_kwargs):
        if any("s2_train_v3_lora.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/sovits/final.pth"
            weights.write_bytes(b"sovits-trained")
        if any("s1_train.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/gpt/final.ckpt"
            weights.write_bytes(b"gpt-trained")
        return subprocess.CompletedProcess(command, 0)

    first = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner)
    changed_audio = tmp_project_dir / manifest.segments[0].audio_for_mix
    write_tiny_wav(changed_audio, duration=1.0, sample_rate=44_100)
    second = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner)

    assert first.fingerprint != second.fingerprint
    assert second.reused_existing is False


def test_synth_rejects_korean_few_shot_gpt_policy_from_manifest(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", gsv_gpt_weights_policy="few_shot")
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    gpt = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "gpt" / "final.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    gpt.parent.mkdir(parents=True, exist_ok=True)
    sovits.parent.mkdir(parents=True, exist_ok=True)
    gpt.write_bytes(b"gpt")
    sovits.write_bytes(b"sovits")
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(text="テスト", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="テスト",
            ja_text="テスト",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    manifest = PipelineManifest(
        segments=[segment],
        artifacts={
            "gsv_few_shot_gpt_weights": str(gpt),
            "gsv_few_shot_sovits_weights": str(sovits),
        },
    )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    with pytest.raises(GPTSoVITSError, match="base_for_korean"):
        synth_step(
            tmp_project_dir,
            gsv_url="http://gsv.local",
            refs_path=tmp_project_dir / "refs" / "refs.json",
            mock=False,
            confirm_rights=True,
        )

    assert calls == []


def test_resolve_gpt_weights_rejects_project_few_shot_gpt_for_korean_tts(
    tmp_project_dir: Path,
) -> None:
    gpt = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "gpt" / "final.ckpt"
    external_gpt = tmp_project_dir / "models" / "korean-compatible.ckpt"
    manifest = PipelineManifest(
        artifacts={"gsv_few_shot_gpt_weights": str(gpt)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
                script=JapaneseScript(ja_text="こんにちは", tts_text="안녕하세요", tts_language="ko"),
            )
        ],
    )

    cfg = ProjectConfig(project_name="test", gsv_gpt_weights_policy="few_shot")
    with pytest.raises(GPTSoVITSError, match="base_for_korean"):
        steps._resolve_gpt_weights_for_tts(tmp_project_dir, manifest, cfg, None, {})

    explicit_cfg = ProjectConfig(project_name="test", gsv_gpt_weights_path=str(gpt))
    with pytest.raises(GPTSoVITSError, match="base_for_korean"):
        steps._resolve_gpt_weights_for_tts(tmp_project_dir, manifest, explicit_cfg, None, {})

    external_cfg = ProjectConfig(project_name="test", gsv_gpt_weights_path=str(external_gpt))
    model_switch: dict[str, str] = {}
    assert (
        steps._resolve_gpt_weights_for_tts(tmp_project_dir, manifest, external_cfg, None, model_switch)
        == str(external_gpt)
    )
    assert model_switch["gpt_weights_mode"] == "explicit"


@pytest.mark.parametrize("use_trained_gpt", [False, True])
def test_synth_gpt_selection_for_korean_few_shot_voice(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_trained_gpt: bool,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test")
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    gpt = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "gpt" / "final.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    training_manifest = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "training_manifest.json"
    for path, payload in (
        (base_gpt, b"base-gpt"),
        (gpt, b"few-shot-gpt"),
        (sovits, b"few-shot-sovits"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_manifest.write_text(
        json.dumps(
            {
                "fingerprint_payload": {
                    "gpt_sovits": {
                        "pretrained_gpt_path": str(base_gpt),
                    }
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    manifest = PipelineManifest(
        segments=[segment],
        artifacts={
            "gsv_few_shot_gpt_weights": str(gpt),
            "gsv_few_shot_sovits_weights": str(sovits),
            "gsv_few_shot_manifest": str(training_manifest),
        },
    )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, str]] = []
    payloads: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payloads.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    if use_trained_gpt:
        with pytest.raises(GPTSoVITSError, match="base_for_korean"):
            synth_step(
                tmp_project_dir,
                gsv_url="http://gsv.local",
                refs_path=tmp_project_dir / "refs" / "refs.json",
                mock=False,
                confirm_rights=True,
                use_trained_gpt=use_trained_gpt,
            )
        assert calls == []
        assert payloads == []
        return

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        use_trained_gpt=use_trained_gpt,
    )

    assert calls[:2] == [("gpt", str(base_gpt)), ("sovits", str(sovits))]
    assert payloads[0]["text"] == "안녕하세요"
    assert payloads[0]["text_lang"] == "all_ko"
    assert payloads[0]["prompt_lang"] == "all_ja"
    assert payloads[0]["prompt_text"]
    manifest = load_manifest(tmp_project_dir)
    candidate_payload = manifest.segments[0].tts.candidates[0].payload
    assert candidate_payload["text_lang"] == "all_ko"
    assert candidate_payload["prompt_lang"] == "all_ja"
    assert candidate_payload["speed_factor"] == payloads[0]["speed_factor"]
    assert candidate_payload["top_k"] == payloads[0]["top_k"]


def test_synth_normalizes_korean_tts_text_before_gsv_request(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", target_language="ko", candidate_count=1, gsv_concurrency=1)
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.2)
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(text="ちりん、ちりん。", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="ちりん、ちりん。",
            ja_text="ちりん、ちりん。",
            tts_text="。징, 징…",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payloads: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payloads.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    assert payloads[0]["text"] == "징, 징…"
    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].script.tts_text == "징, 징…"
    assert manifest.segments[0].analysis["pre_synth_tts_text_normalization"]["before"] == "。징, 징…"


def test_korean_segment_ref_can_be_disabled_for_pronunciation_priority(
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        project_name="test",
        gsv_ref_mode="segment",
        gsv_ref_min_sec=1.0,
        gsv_ref_max_sec=4.0,
        gsv_ref_min_quality_score=0.0,
        gsv_korean_segment_ref_enabled=False,
    )
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=3.2,
        duration=3.2,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        analysis={"speaker_count": 1},
        source_script=SourceScript(
            text="これは参照に使う日本語です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.2,
        ),
        script=JapaneseScript(
            literal_ja="これは参照に使う日本語です。",
            ja_text="これは参照に使う日本語です。",
            tts_text="한국어 대사입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )

    ref, metadata = steps._segment_source_ref_for_gsv(
        tmp_project_dir,
        segment,
        cfg,
        [segment],
    )

    assert ref is None
    assert metadata["used"] is False
    assert metadata["reject_reasons"] == [
        "korean_segment_ref_disabled_for_pronunciation_priority"
    ]


def test_synth_extends_short_source_segment_reference_with_neighbor(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    static_ref = tmp_project_dir / "refs" / "whisper_close.wav"
    _write_tone_wav(static_ref, duration=3.2)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_ref_mode="segment",
        gsv_ref_min_sec=1.0,
        gsv_ref_max_sec=4.0,
        gsv_ref_min_quality_score=0.0,
        gsv_korean_segment_ref_enabled=True,
        gsv_concurrency=1,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    short_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    next_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0002_mix.wav"
    _write_tone_wav(short_audio, duration=1.2)
    _write_tone_wav(next_audio, duration=2.0)
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            segments=[
                Segment(
                    id="seg_0001",
                    speaker_id="speaker_0001",
                    start=0.0,
                    end=1.2,
                    duration=1.2,
                    audio_for_gemma=str(short_audio),
                    audio_for_mix=str(short_audio.relative_to(tmp_project_dir)),
                    analysis={"speaker_count": 1},
                    source_script=SourceScript(
                        text="短い参照です。",
                        language="ja",
                        backend="mock",
                        start=0.0,
                        end=1.2,
                    ),
                    script=JapaneseScript(
                        literal_ja="短い参照です。",
                        ja_text="短い参照です。",
                        tts_text="짧은 대사입니다",
                        tts_language="ko",
                        source_language="ja",
                        target_language="ko",
                    ),
                ),
                Segment(
                    id="seg_0002",
                    speaker_id="speaker_0001",
                    start=1.2,
                    end=3.2,
                    duration=2.0,
                    audio_for_gemma=str(next_audio),
                    audio_for_mix=str(next_audio.relative_to(tmp_project_dir)),
                    analysis={"speaker_count": 1},
                    source_script=SourceScript(
                        text="続きです。",
                        language="ja",
                        backend="mock",
                        start=1.2,
                        end=3.2,
                    ),
                ),
            ]
        ),
    )
    payloads: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payloads.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    assert payloads[0]["prompt_text"] == "みじかいさんしょーです。つずきです。"
    ref_path = Path(str(payloads[0]["ref_audio_path"]))
    assert ref_path.name == "seg_0001_ref.wav"
    assert duration_sec(ref_path) == pytest.approx(3.2)
    manifest = load_manifest(tmp_project_dir)
    candidate_payload = manifest.segments[0].tts.candidates[0].payload
    assert candidate_payload["prompt_text_policy"] == "use_segment_source_reference"
    assert candidate_payload["segment_ref"]["used"] is True
    assert candidate_payload["segment_ref"]["expanded_with_neighbors"] is True
    assert candidate_payload["segment_ref"]["span_segment_ids"] == ["seg_0001", "seg_0002"]


def test_synth_falls_back_to_static_ref_below_gsv_api_reference_minimum(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    static_ref = tmp_project_dir / "refs" / "whisper_close.wav"
    _write_tone_wav(static_ref, duration=3.2)
    cfg = ProjectConfig(
        project_name="test",
        gsv_ref_mode="segment",
        gsv_ref_min_sec=1.0,
        gsv_ref_max_sec=10.0,
        gsv_ref_min_quality_score=0.0,
        gsv_korean_segment_ref_enabled=True,
        gsv_concurrency=1,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=2.5)
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=2.5,
        duration=2.5,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        analysis={"speaker_count": 1},
        source_script=SourceScript(text="短い参照です。", language="ja", backend="mock", start=0.0, end=2.5),
        script=JapaneseScript(
            literal_ja="短い参照です。",
            ja_text="短い参照です。",
            tts_text="짧은 대사입니다",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payloads: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payloads.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=2.5)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    assert payloads[0]["prompt_text"] == "それじゃー……みみもとで、ゆっくりささやいていきますね。"
    assert Path(str(payloads[0]["ref_audio_path"])).name == "whisper_close.wav"
    manifest = load_manifest(tmp_project_dir)
    candidate_payload = manifest.segments[0].tts.candidates[0].payload
    assert candidate_payload["prompt_text_policy"] == "use_source_reference_prompt"
    assert candidate_payload["segment_ref"]["used"] is False
    assert candidate_payload["segment_ref"]["reject_reasons"] == [
        "duration_below_gsv_api_ref_min:2.500<3.000"
    ]


def test_synth_retry_failed_reprocesses_previous_tts_failures(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio, duration=1.2)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.2)
    save_project_config(
        ProjectConfig(project_name="test", candidate_count=1, gsv_concurrency=1),
        tmp_project_dir / "pipeline.yaml",
    )
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        status="failed",
        errors=["No acceptable TTS candidates for mix."],
        analysis={"speaker_count": 1},
        source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            calls.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        retry_failed=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert calls == ["안녕하세요"]
    assert manifest.segments[0].status == "synthesized"
    assert "No acceptable TTS candidates for mix." not in manifest.segments[0].errors


def test_synth_reprocesses_previous_tts_failures_by_default(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio, duration=1.2)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.2)
    save_project_config(
        ProjectConfig(
            project_name="test",
            candidate_count=1,
            gsv_concurrency=1,
            gsv_initial_candidate_count=1,
            gsv_korean_clarity_retry_enabled=False,
            gsv_low_temperature_retry_enabled=False,
            gsv_terminal_failure_policy="fail",
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        status="failed",
        errors=["No acceptable TTS candidates for mix."],
        analysis={"speaker_count": 1},
        source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def build_payload(self, text: str, ref: object, options: object = None) -> object:
            calls.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request: object, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
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
    assert calls == ["안녕하세요"]
    assert manifest.segments[0].status == "synthesized"
    assert "No acceptable TTS candidates for mix." not in manifest.segments[0].errors


def test_synth_repairs_truncated_korean_script_before_preflight_block(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio, duration=1.5)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.5)
    save_project_config(
        ProjectConfig(
            project_name="test",
            candidate_count=1,
            gsv_concurrency=1,
            gsv_initial_candidate_count=1,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=1.5,
        duration=1.5,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        status="scripted",
        analysis={"speaker_count": 1},
        source_script=SourceScript(
            text="少し近づきますね",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.5,
        ),
        script=JapaneseScript(
            literal_ja="少し近づきますね",
            ja_text="少し近づきますね",
            tts_text="그럼 다음 푸",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def build_payload(self, text: str, ref: object, options: object = None) -> object:
            calls.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request: object, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.5)
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
    assert calls
    assert set(calls) == {"그럼 다음 푸..."}
    assert segment.status == "synthesized"
    assert segment.script is not None
    assert segment.script.tts_text == "그럼 다음 푸..."
    assert segment.analysis["pre_synth_text_qc_recovery"]["action"] == (
        "repaired_truncated_sentence"
    )


def test_synth_retry_failed_only_attempts_failed_non_countdown_segments(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio, duration=3.2)
    save_project_config(
        ProjectConfig(project_name="test", candidate_count=1, gsv_concurrency=1),
        tmp_project_dir / "pipeline.yaml",
    )
    old_final = tmp_project_dir / "work" / "tts" / "seg_0002_final.wav"
    write_tiny_wav(old_final, duration=1.2)

    failed = _segment(tmp_project_dir, "seg_0001", 0.0, 1.2, "こんにちは")
    failed.status = "failed"
    failed.errors = ["No acceptable TTS candidates for mix."]
    failed.script = JapaneseScript(
        literal_ja="こんにちは",
        ja_text="こんにちは",
        tts_text="안녕하세요",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    synthesized = _segment(tmp_project_dir, "seg_0002", 2.0, 1.2, "成功済み")
    synthesized.status = "synthesized"
    synthesized.script = JapaneseScript(
        literal_ja="成功済み",
        ja_text="成功済み",
        tts_text="이미 성공했어요.",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    synthesized.tts = TTSMetadata(
        backend="gpt-sovits",
        candidate_count=1,
        selected_candidate_path=str(old_final),
        candidates=[
            TTSCandidate(
                candidate_index=0,
                seed=1,
                output_path=str(old_final),
                duration_sec=1.2,
                backend="gpt-sovits",
                selected=True,
                duration_ratio=1.0,
                duration_gate="pass",
                acceptable_for_mix=True,
                payload={"text": "이미 성공했어요."},
            )
        ],
    )
    scripted = _segment(tmp_project_dir, "seg_0003", 4.0, 1.2, "未処理")
    scripted.status = "scripted"
    scripted.script = JapaneseScript(
        literal_ja="未処理",
        ja_text="未処理",
        tts_text="아직 처리 전이에요.",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    countdown_failed = _segment(tmp_project_dir, "seg_0004", 6.0, 1.2, "3 2 1")
    countdown_failed.status = "failed"
    countdown_failed.errors = ["No acceptable TTS candidates for mix."]
    countdown_failed.script = JapaneseScript(
        literal_ja="3 2 1",
        ja_text="3 2 1",
        tts_text="삼, 이, 일",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(segments=[failed, synthesized, scripted, countdown_failed]),
    )
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            calls.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        retry_failed=True,
        render_countdowns=False,
    )

    manifest = load_manifest(tmp_project_dir)
    assert calls == ["안녕하세요"]
    assert manifest.segments[0].status == "synthesized"
    assert manifest.segments[1].status == "synthesized"
    assert manifest.segments[1].tts.selected_candidate_path == str(old_final)
    assert manifest.segments[1].tts.candidates[0].payload["text"] == "이미 성공했어요."
    assert manifest.segments[2].status == "scripted"
    assert manifest.segments[2].tts is None
    assert manifest.segments[3].status == "failed"
    assert manifest.segments[3].errors == ["No acceptable TTS candidates for mix."]


def test_synth_retries_fine_tuned_failures_then_zero_shot_fallback(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_max_attempts_per_candidate=1,
        gsv_retry_candidate_count=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    training_manifest = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "training_manifest.json"
    for path, payload in ((base_gpt, b"base-gpt"), (sovits, b"few-shot-sovits")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_manifest.write_text(
        json.dumps(
            {
                "fingerprint_payload": {
                    "gpt_sovits": {
                        "pretrained_gpt_path": str(base_gpt),
                    }
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(text="テストです", language="ja", backend="mock", start=0.0, end=3.0),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            segments=[segment],
            artifacts={
                "gsv_few_shot_sovits_weights": str(sovits),
                "gsv_few_shot_manifest": str(training_manifest),
            },
        ),
    )
    weight_calls: list[tuple[str, str]] = []
    synth_calls: list[int] = []

    class FallbackClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            weight_calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            weight_calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            synth_calls.append(len(synth_calls) + 1)
            _write_tone_wav(output_path, duration=0.2 if len(synth_calls) < 3 else 3.0)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FallbackClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert synth_calls == [1, 2, 3]
    assert weight_calls == [("sovits", str(sovits))]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["synth_pass"] == "zero_shot_fallback"
    assert manifest.stage_state["synth"]["fine_tuned_retry"]["attempted_segments"] == ["seg_0001"]
    assert manifest.stage_state["synth"]["zero_shot_fallback"]["attempted_segments"] == ["seg_0001"]


def test_synth_uses_progressive_candidate_counts_and_low_temperature_before_zero_shot(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=5,
        gsv_concurrency=1,
        gsv_max_attempts_per_candidate=1,
        gsv_duration_rewrite_backend="none",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    training_manifest = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "training_manifest.json"
    for path, payload in ((base_gpt, b"base-gpt"), (sovits, b"few-shot-sovits")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_manifest.write_text(
        json.dumps(
            {
                "fingerprint_payload": {
                    "gpt_sovits": {
                        "pretrained_gpt_path": str(base_gpt),
                    }
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(text="テストです", language="ja", backend="mock", start=0.0, end=3.0),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            segments=[segment],
            artifacts={
                "gsv_few_shot_sovits_weights": str(sovits),
                "gsv_few_shot_manifest": str(training_manifest),
            },
        ),
    )
    temperatures: list[float] = []

    class ProgressiveClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            temperatures.append(request.temperature)
            _write_tone_wav(output_path, duration=3.0 if len(temperatures) == 31 else 0.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ProgressiveClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert temperatures == [1.0] * 3 + [1.0] * 7 + [0.3] * 20 + [1.0] * 5
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["synth_pass"] == "zero_shot_fallback"
    assert selected.payload["candidate_count_used"] == 5
    assert selected.payload["temperature_used"] == pytest.approx(1.0)
    assert manifest.stage_state["synth"]["fine_tuned_retry"]["attempted_segments"] == ["seg_0001"]
    assert manifest.stage_state["synth"]["low_temperature_retry"]["attempted_segments"] == [
        "seg_0001"
    ]
    assert manifest.stage_state["synth"]["zero_shot_fallback"]["attempted_segments"] == ["seg_0001"]


def test_synth_runs_duration_rewrite_before_zero_shot_when_configured(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_max_attempts_per_candidate=1,
        gsv_retry_candidate_count=1,
        gsv_duration_rewrite_backend="gemma",
        gsv_duration_rewrite_timing="before_zero_shot",
        gsv_low_temperature_retry_enabled=False,
        gsv_terminal_failure_policy="fail",
        gemma_text_server_auto_start=False,
        duration_tolerance=0.2,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    training_manifest = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "training_manifest.json"
    for path, payload in ((base_gpt, b"base-gpt"), (sovits, b"few-shot-sovits")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_manifest.write_text(
        json.dumps(
            {
                "fingerprint_payload": {
                    "gpt_sovits": {
                        "pretrained_gpt_path": str(base_gpt),
                    }
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=4.0)
    original_text = "여기는 지금의 세계와는 다른, 조용하지만 아주 낯설고 이상한 평행세계예요."
    rewritten_text = "여기는 낯설고 이상한 평행세계예요."
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="ここは今の世界とは違う、平行世界です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            literal_ja="ここは今の世界とは違う、平行世界です。",
            ja_text="ここは今の世界とは違う、平行世界です。",
            tts_text=original_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            segments=[segment],
            artifacts={
                "gsv_few_shot_sovits_weights": str(sovits),
                "gsv_few_shot_manifest": str(training_manifest),
            },
        ),
    )
    payload_texts: list[str] = []
    rewrite_calls: list[dict[str, object]] = []

    class FallbackRewriteClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            payload_texts.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 4.0 if request.text == rewritten_text else 8.0
            _write_tone_wav(output_path, duration=duration)
            return output_path

    class FakeGemmaClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def rewrite_tts_for_duration(self, **kwargs):
            rewrite_calls.append(kwargs)
            return KoreanTranslation(
                ko_literal=rewritten_text,
                ko_natural=rewritten_text,
                notes=[],
                confidence=0.99,
                model="fake-gemma",
                batch_id=str(kwargs["batch_id"]),
            )

    monkeypatch.setattr(steps, "GPTSoVITSClient", FallbackRewriteClient)
    monkeypatch.setattr(steps, "LlamaServerTranslationClient", FakeGemmaClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    scripted = segment.script
    assert scripted is not None
    assert scripted.tts_text == rewritten_text
    assert payload_texts == [original_text, original_text, rewritten_text]
    assert len(rewrite_calls) == 1
    assert rewrite_calls[0]["reason"] == "too_long"
    assert segment.status == "synthesized"
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["synth_pass"] == "before_zero_shot_duration_rewrite"
    assert selected.payload["duration_rewrite"]["reason"] == "too_long"
    assert manifest.stage_state["synth"]["fine_tuned_retry"]["attempted_segments"] == ["seg_0001"]
    assert manifest.stage_state["synth"]["duration_rewrite_before_zero_shot"]["attempted_segments"] == [
        "seg_0001"
    ]
    assert "zero_shot_fallback" not in manifest.stage_state["synth"]


def test_synth_retries_static_ref_before_zero_shot_when_segment_ref_fails(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_max_attempts_per_candidate=1,
        gsv_ref_mode="segment",
        gsv_ref_min_sec=1.0,
        gsv_ref_max_sec=4.0,
        gsv_ref_min_quality_score=0.0,
        gsv_korean_segment_ref_enabled=True,
        gsv_retry_candidate_count=1,
        gsv_duration_rewrite_backend="none",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    static_ref = tmp_project_dir / "refs" / "whisper_close.wav"
    _write_tone_wav(static_ref, duration=3.2)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    training_manifest = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "training_manifest.json"
    for path, payload in ((base_gpt, b"base-gpt"), (sovits, b"few-shot-sovits")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_manifest.write_text(
        json.dumps(
            {
                "fingerprint_payload": {
                    "gpt_sovits": {
                        "pretrained_gpt_path": str(base_gpt),
                    }
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.2)
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=3.2,
        duration=3.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        analysis={"speaker_count": 1},
        source_script=SourceScript(
            text="これは参照に使う日本語です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.2,
        ),
        script=JapaneseScript(
            literal_ja="これは参照に使う日本語です。",
            ja_text="これは参照に使う日本語です。",
            tts_text="한국어 대사입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            segments=[segment],
            artifacts={
                "gsv_few_shot_sovits_weights": str(sovits),
                "gsv_few_shot_manifest": str(training_manifest),
            },
        ),
    )
    weight_calls: list[tuple[str, str]] = []
    prompt_policies: list[str] = []
    ref_names: list[str] = []
    synth_passes: list[str] = []

    class StaticRefFallbackClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            weight_calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            weight_calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            ref_name = Path(request.ref_audio_path).name
            ref_names.append(ref_name)
            _write_tone_wav(
                output_path,
                duration=3.2 if ref_name == "whisper_close.wav" else 0.2,
            )
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", StaticRefFallbackClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert weight_calls == [("sovits", str(sovits))]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    for candidate in segment.tts.candidates:
        prompt_policies.append(str(candidate.payload["prompt_text_policy"]))
        synth_passes.append(str(candidate.payload["synth_pass"]))
    assert ref_names == ["seg_0001_mix.wav", "seg_0001_mix.wav", "whisper_close.wav"]
    assert prompt_policies == ["use_static_reference_retry"]
    assert synth_passes == ["static_ref_retry"]
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["synth_pass"] == "static_ref_retry"
    assert manifest.stage_state["synth"]["fine_tuned_retry"]["attempted_segments"] == ["seg_0001"]
    assert manifest.stage_state["synth"]["static_ref_retry"]["attempted_segments"] == ["seg_0001"]
    assert "zero_shot_fallback" not in manifest.stage_state["synth"]


def test_synth_force_reprocesses_previous_successes(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio, duration=3.2)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.2)
    old_final = tmp_project_dir / "work" / "tts" / "seg_0001_final.wav"
    write_tiny_wav(old_final, duration=1.2)
    save_project_config(
        ProjectConfig(project_name="test", candidate_count=1, gsv_concurrency=1),
        tmp_project_dir / "pipeline.yaml",
    )
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        status="synthesized",
        errors=["All TTS candidates failed."],
        analysis={"speaker_count": 1},
        source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
        tts=TTSMetadata(
            backend="gpt-sovits",
            candidate_count=1,
            selected_candidate_path=str(old_final),
            candidates=[
                TTSCandidate(
                    candidate_index=0,
                    seed=1,
                    payload={"text": "old"},
                    output_path=str(old_final),
                    duration_sec=1.2,
                    backend="gpt-sovits",
                    selected=True,
                    duration_ratio=1.0,
                    duration_gate="pass",
                    acceptable_for_mix=True,
                )
            ],
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            calls.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert calls == ["안녕하세요"]
    assert manifest.segments[0].status == "synthesized"
    assert manifest.segments[0].tts.candidates[0].payload["text"] == "안녕하세요"
    assert "All TTS candidates failed." not in manifest.segments[0].errors


def test_synth_uses_project_gsv_pronunciation_options(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_tts_min_speed_factor=0.92,
        gsv_tts_max_speed_factor=1.0,
        gsv_top_k=8,
        gsv_top_p=0.9,
        gsv_temperature=0.7,
        gsv_text_split_method="cut0",
        gsv_parallel_infer=False,
        gsv_repetition_penalty=1.25,
        gsv_sample_steps=32,
        gsv_super_sampling=True,
        gsv_overlap_length=2,
        gsv_min_chunk_length=8,
        gsv_fragment_interval=0.3,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.2,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payloads: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payloads.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    assert payloads
    payload = payloads[0]
    assert payload["top_k"] == 8
    assert payload["top_p"] == 0.9
    assert payload["temperature"] == 0.7
    assert payload["text_split_method"] == "cut0"
    assert payload["parallel_infer"] is False
    assert payload["repetition_penalty"] == 1.25
    assert payload["sample_steps"] == 32
    assert payload["super_sampling"] is True
    assert payload["overlap_length"] == 2
    assert payload["min_chunk_length"] == 8
    assert payload["fragment_interval"] == 0.3
    assert payload["speed_factor"] <= cfg.gsv_tts_max_speed_factor


def test_synth_switches_gpt_and_sovits_weights_per_voice_bank_speaker(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    base_gpt.parent.mkdir(parents=True, exist_ok=True)
    base_gpt.write_bytes(b"base-gpt")
    speaker_models: dict[str, GSVSpeakerConfig] = {}
    for speaker_id, prompt in (("speaker_0001", "こんにちは"), ("speaker_0002", "おやすみ")):
        speaker_dir = tmp_project_dir / "voice_bank" / "speakers" / speaker_id
        gpt = speaker_dir / "gsv" / "v001" / "gpt.ckpt"
        sovits = speaker_dir / "gsv" / "v001" / "final.pth"
        ref_audio = speaker_dir / "refs" / "whisper_close.wav"
        refs_json = speaker_dir / "refs" / "refs.json"
        sovits.parent.mkdir(parents=True, exist_ok=True)
        gpt.write_bytes(f"{speaker_id}-gpt".encode())
        sovits.write_bytes(speaker_id.encode())
        write_tiny_wav(ref_audio)
        refs_json.write_text(
            json.dumps(
                {
                    "whisper_close": {
                        "ref_audio_path": str(ref_audio.relative_to(tmp_project_dir)),
                        "prompt_text": prompt,
                        "prompt_lang": "ja",
                    }
                }
            ),
            "utf-8",
        )
        speaker_models[speaker_id] = GSVSpeakerConfig(
            gpt_weights_path=str(gpt.relative_to(tmp_project_dir)),
            sovits_weights_path=str(sovits.relative_to(tmp_project_dir)),
            refs_path=str(refs_json.relative_to(tmp_project_dir)),
        )
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_gpt_weights_path=str(base_gpt),
        gsv_speaker_models=speaker_models,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    segments: list[Segment] = []
    spoken_numbers = ["일", "이", "삼"]
    for index, speaker_id in enumerate(("speaker_0001", "speaker_0002", "speaker_0001"), start=1):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"seg_{index:04d}_mix.wav"
        write_tiny_wav(audio)
        segments.append(
            Segment(
                id=f"seg_{index:04d}",
                speaker_id=speaker_id,
                start=(index - 1) * 1.2,
                end=index * 1.2,
                duration=1.2,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
                script=JapaneseScript(
                    literal_ja="こんにちは",
                    ja_text="こんにちは",
                    tts_text=f"안녕하세요 {spoken_numbers[index - 1]}",
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                ),
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))
    calls: list[tuple[str, str]] = []
    prompt_texts: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            prompt_texts.append(ref.prompt_text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=Path("refs/refs.json"),
        mock=False,
        confirm_rights=True,
    )

    assert calls == [
        ("gpt", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "gpt.ckpt")),
        ("sovits", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "final.pth")),
        ("gpt", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0002" / "gsv" / "v001" / "gpt.ckpt")),
        ("sovits", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0002" / "gsv" / "v001" / "final.pth")),
        ("gpt", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "gpt.ckpt")),
        ("sovits", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "final.pth")),
    ]
    assert prompt_texts == ["こんにちわ", "おやすみ", "こんにちわ"]
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["synth"]["model_switch"]["gpt_weights_mode"] == "speaker_voice_bank"
    assert manifest.stage_state["synth"]["model_switch"]["sovits_weights_mode"] == "speaker_voice_bank"


def test_synth_uses_model_fallback_for_distinct_insufficient_source_speaker(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    speaker_dir = tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001"
    gpt = speaker_dir / "gsv" / "v001" / "gpt.ckpt"
    sovits = speaker_dir / "gsv" / "v001" / "final.pth"
    ref_audio = speaker_dir / "refs" / "whisper_close.wav"
    refs_json = speaker_dir / "refs" / "refs.json"
    sovits.parent.mkdir(parents=True, exist_ok=True)
    gpt.write_bytes(b"speaker_0001-gpt")
    sovits.write_bytes(b"speaker_0001-sovits")
    write_tiny_wav(ref_audio)
    refs_json.write_text(
        json.dumps(
            {
                "whisper_close": {
                    "ref_audio_path": str(ref_audio.relative_to(tmp_project_dir)),
                    "prompt_text": "こんにちは",
                    "prompt_lang": "ja",
                }
            }
        ),
        "utf-8",
    )
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_speaker_models={
            "speaker_0001": GSVSpeakerConfig(
                gpt_weights_path=str(gpt.relative_to(tmp_project_dir)),
                sovits_weights_path=str(sovits.relative_to(tmp_project_dir)),
                refs_path=str(refs_json.relative_to(tmp_project_dir)),
            )
        },
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0002",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        analysis={
            "source_speaker_model_fallback": {
                "speaker_id": "speaker_0001",
                "reason": "insufficient_distinct_speaker_training_data",
            }
        },
        source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    calls: list[tuple[str, str]] = []
    payload_speaker_ids: list[str | None] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            payload_speaker_ids.append(load_manifest(tmp_project_dir).segments[0].speaker_id)
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=Path("refs/refs.json"),
        mock=False,
        confirm_rights=True,
    )

    assert calls[:2] == [
        ("gpt", str(gpt)),
        ("sovits", str(sovits)),
    ]
    assert payload_speaker_ids == ["speaker_0002"]


def test_synth_does_not_locally_rewrite_korean_duration_when_gemma_disabled(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_tts_max_speed_factor=1.05,
        gsv_micro_segment_unfit_policy="manual_review",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    long_text = "수고하셨습니다 오늘도 정말 잘하셨어요"
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=0.2,
        duration=0.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        script=JapaneseScript(ja_text="原文", tts_text=long_text, tts_language="ko"),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payload_texts: list[str] = []
    payload_speeds: list[float] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payload_texts.append(request.text)
            payload_speeds.append(request.speed_factor)
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            duration = 1.2 if request.text == long_text else 0.2
            t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
            tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
            write_audio(output_path, np.stack([tone, tone], axis=1), sample_rate)
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
    scripted = manifest.segments[0].script
    assert scripted is not None
    assert scripted.tts_text == long_text
    assert not any("duration_rewrite" in flag for flag in scripted.risk_flags)
    assert payload_texts == [long_text, long_text, long_text]
    assert all(speed <= cfg.gsv_tts_max_speed_factor for speed in payload_speeds)
    assert manifest.segments[0].status == "needs_manual_review"


def test_korean_duration_rewrite_keeps_realistic_paced_korean_line() -> None:
    script = JapaneseScript(
        ja_text="ここは今の世界とは違う、平行世界です。",
        tts_text="여기는 지금의 세계와는 다른, 평행세계예요.",
        tts_language="ko",
    )

    rewritten = rewrite_for_duration(script, target_sec=4.04, tolerance=0.25)

    assert rewritten.tts_text == script.tts_text
    assert "duration_rewrite_shortened" not in rewritten.risk_flags
    assert estimate_tts_duration(rewritten.tts_text, "ko") <= 4.04 * 1.25


def test_korean_duration_rewrite_avoids_tiny_leading_sentence_only_result() -> None:
    script = JapaneseScript(
        ja_text="そうでしょう。水着に黒いストッキングの店員がいたら誰でも驚きます。",
        tts_text="그렇겠죠. 수영복에 검은 스타킹을 신은 점원이 있으면 누구라도 놀랄 거예요.",
        tts_language="ko",
    )

    rewritten = rewrite_for_duration(script, target_sec=6.0, tolerance=0.25)

    assert rewritten.tts_text != "그렇겠죠."
    assert "놀랄 거예요" in rewritten.tts_text
    assert 6.0 * 0.5 <= estimate_tts_duration(rewritten.tts_text, "ko") <= 6.0 * 1.25


def test_synth_uses_gemma_duration_rewrite_for_long_korean_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_tts_min_speed_factor=0.85,
        gsv_tts_max_speed_factor=1.2,
        gsv_duration_rewrite_backend="gemma",
        gsv_duration_rewrite_max_attempts=1,
        gemma_text_server_auto_start=False,
        duration_tolerance=0.2,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=4.04)
    original_text = "여기는 지금의 세계와는 다른, 조용하지만 아주 낯설고 이상한 평행세계예요."
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.04,
        duration=4.04,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        script=JapaneseScript(
            ja_text="ここは今の世界とは違う、平行世界です。",
            tts_text=original_text,
            tts_language="ko",
            expected_tts_duration_sec=4.04,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payload_texts: list[str] = []
    payload_speeds: list[float] = []
    rewritten_text = "여기는 낯설고 이상한 평행세계예요."

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payload_texts.append(request.text)
            payload_speeds.append(request.speed_factor)
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 8.0 if request.text == original_text else 5.0 / request.speed_factor
            sample_rate = 48_000
            t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
            tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
            write_audio(output_path, np.stack([tone, tone], axis=1), sample_rate)
            return output_path

    class FakeGemmaClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def rewrite_tts_for_duration(self, **kwargs):
            return KoreanTranslation(
                ko_literal=rewritten_text,
                ko_natural=rewritten_text,
                notes=[],
                confidence=0.99,
                model="fake-gemma",
                batch_id=str(kwargs["batch_id"]),
            )

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "LlamaServerTranslationClient", FakeGemmaClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    scripted = manifest.segments[0].script
    assert scripted is not None
    assert scripted.tts_text == rewritten_text
    assert "gemma_duration_rewrite_too_long" in scripted.risk_flags
    assert payload_texts == [original_text, rewritten_text, rewritten_text]
    assert payload_speeds == [1.0, 1.0, 1.2]
    selected = next(candidate for candidate in manifest.segments[0].tts.candidates if candidate.selected)
    assert selected.payload["duration_rewrite"]["reason"] == "too_long"


def test_synth_uses_gemma_duration_rewrite_for_short_korean_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        duration_tolerance=0.2,
        gsv_max_attempts_per_candidate=3,
        gsv_duration_rewrite_backend="gemma",
        gsv_duration_rewrite_max_attempts=1,
        gemma_text_server_auto_start=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=8.0)
    original_text = "인간의 말로 하면 은마라고나 할까."
    rewritten_text = "인간의 말로 하면 은마, 흔히 말하는 서큐버스라고나 할까."
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=8.0,
        duration=8.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="私は、人間の言葉で言うなら、隠魔。そう、俗に言うサキバスってやつね。",
            language="ja",
            backend="mock",
            start=0.0,
            end=8.0,
        ),
        script=JapaneseScript(
            ja_text="私は、人間の言葉で言うなら、隠魔。そう、俗に言うサキバスってやつね。",
            tts_text=original_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payload_texts: list[str] = []
    rewrite_calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payload_texts.append(request.text)
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 8.0 if request.text == rewritten_text else 4.0
            _write_tone_wav(output_path, duration=duration)
            return output_path

    class FakeGemmaClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def rewrite_tts_for_duration(self, **kwargs):
            rewrite_calls.append(kwargs)
            return KoreanTranslation(
                ko_literal=rewritten_text,
                ko_natural=rewritten_text,
                notes=[],
                confidence=0.99,
                model="fake-gemma",
                batch_id=str(kwargs["batch_id"]),
            )

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "LlamaServerTranslationClient", FakeGemmaClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    scripted = manifest.segments[0].script
    assert scripted is not None
    assert scripted.tts_text == rewritten_text
    assert "괜찮아요" not in scripted.tts_text
    assert payload_texts == [original_text, rewritten_text]
    assert len(rewrite_calls) == 1
    assert rewrite_calls[0]["reason"] == "too_short"
    selected = next(candidate for candidate in manifest.segments[0].tts.candidates if candidate.selected)
    assert selected.payload["text"] == rewritten_text
    assert selected.payload["duration_rewrite"]["backend"] == "gemma_text"
    assert selected.payload["duration_rewrite"]["accepted"] is True


def test_synth_uses_gemma_timing_expansion_for_persistently_short_korean_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        duration_tolerance=0.2,
        gsv_max_attempts_per_candidate=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_korean_clarity_retry_enabled=False,
        gsv_timing_expansion_enabled=True,
        gsv_timing_expansion_max_attempts=1,
        gsv_terminal_failure_policy="fail",
        gemma_text_server_auto_start=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=4.0)
    original_text = "어떠니?"
    expanded_text = "어떠니… 지금 이 느낌은 어떠니?"
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="はどう?",
            language="ja",
            backend="mock",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            ja_text="はどう?",
            tts_text=original_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payload_texts: list[str] = []
    expansion_calls: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payload_texts.append(request.text)
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(
                output_path,
                duration=4.0 if request.text == expanded_text else 1.0,
            )
            return output_path

    class FakeGemmaClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def rewrite_tts_for_timing_expansion(self, **kwargs):
            expansion_calls.append(kwargs)
            return KoreanTranslation(
                ko_literal=expanded_text,
                ko_natural=expanded_text,
                notes=[],
                confidence=0.99,
                model="fake-gemma",
                batch_id=str(kwargs["batch_id"]),
            )

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "LlamaServerTranslationClient", FakeGemmaClient)

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
    assert segment.script.tts_text == expanded_text
    assert "gemma_timing_expansion_too_short" in segment.script.risk_flags
    assert payload_texts[0] == original_text
    assert set(payload_texts[1:]) == {expanded_text}
    assert len(expansion_calls) == 1
    assert expansion_calls[0]["reason"] == "too_short"
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["text"] == expanded_text
    assert selected.payload["timing_expansion"]["backend"] == "gemma_text"
    assert selected.payload["timing_expansion"]["accepted"] is True
    assert manifest.stage_state["synth"]["timing_expansion"]["attempted_segments"] == [
        "seg_0001"
    ]
    assert manifest.stage_state["synth"]["timing_expansion"]["succeeded_segments"] == [
        "seg_0001"
    ]


def test_synth_keeps_original_when_terminal_tts_failure_policy_allows_fallback(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        duration_tolerance=0.2,
        gsv_max_attempts_per_candidate=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_korean_clarity_retry_enabled=False,
        gsv_timing_expansion_enabled=False,
        gsv_terminal_failure_policy="keep_original",
        rvc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="わかるよねじ",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="わかるよねじ",
            ja_text="わかるよねじ",
            tts_text="알겠지?",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class ShortClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.0)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ShortClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert manifest.stage_state["synth"]["status"] == "completed"
    assert segment.status == "absorbed"
    assert segment.keep_original_texture is True
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is not None
    selected_path = Path(segment.tts.selected_candidate_path)
    assert selected_path.exists()
    assert selected_path.name == "seg_0001_final.wav"
    assert [candidate.selected for candidate in segment.tts.candidates] == [True]
    fallback = segment.analysis["synth_keep_original_fallback"]
    assert fallback["action"] == "keep_original_after_tts_failure"
    assert fallback["terminal_failure_policy"] == "keep_original"
    assert fallback["previous_status"] == "failed"
    assert fallback["selected_candidate_path"] == str(selected_path)
    assert fallback["selected_duration_gate"] == "too_short"
    assert manifest.stage_state["synth"]["keep_original_fallback"] == {
        "attempted_segments": ["seg_0001"],
        "succeeded_segments": ["seg_0001"],
        "failed_segments": [],
    }

    manifest = steps.qc_step(tmp_project_dir, "mock", confirm_rights=True)
    segment = manifest.segments[0]
    assert segment.status == "absorbed"
    assert "Cannot QC without selected TTS and script." not in segment.errors


def test_synth_defers_gemma_duration_rewrite_until_after_initial_gsv_pass(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_auto_start=True,
        gsv_server_command=["fake-gsv"],
        gsv_max_attempts_per_candidate=3,
        gsv_duration_rewrite_backend="gemma",
        gsv_duration_rewrite_max_attempts=1,
        gsv_duration_rewrite_pre_candidate_count=2,
        gemma_text_server_auto_start=True,
        gemma_text_server_command=["fake-gemma"],
        duration_tolerance=0.2,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=4.0)
    original_text = "여기는 지금의 세계와는 다른, 조용하지만 아주 낯설고 이상한 평행세계예요."
    rewritten_text = "여기는 낯설고 이상한 평행세계예요."
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="ここは今の世界とは違う、平行世界です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            ja_text="ここは今の世界とは違う、平行世界です。",
            tts_text=original_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    events: list[str] = []
    active = {"gsv": 0, "gemma": 0}

    class FakeGSVManager:
        def __init__(self, **kwargs) -> None:
            self.base_url = kwargs["base_url"]
            self.log_path = kwargs.get("log_path")
            self.started = False
            self.reused_existing = False

        def start(self):
            assert active["gemma"] == 0
            events.append("gsv_start")
            active["gsv"] += 1
            self.started = True
            return self

        def stop(self) -> None:
            events.append("gsv_stop")
            active["gsv"] = max(0, active["gsv"] - 1)

    class FakeGemmaManager:
        def __init__(self, **kwargs) -> None:
            self.base_url = kwargs["base_url"]
            self.log_path = kwargs.get("log_path")
            self.started = False
            self.reused_existing = False

        def start(self):
            assert active["gsv"] == 0
            events.append("gemma_start")
            active["gemma"] += 1
            self.started = True
            return self

        def stop(self) -> None:
            events.append("gemma_stop")
            active["gemma"] = max(0, active["gemma"] - 1)

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            assert active["gsv"] > 0
            assert active["gemma"] == 0
            events.append(f"tts:{text}")
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 4.0 if request.text == rewritten_text else 8.0
            _write_tone_wav(output_path, duration=duration)
            return output_path

    class FakeGemmaClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def rewrite_tts_for_duration(self, **kwargs):
            assert active["gsv"] == 0
            assert active["gemma"] > 0
            events.append("gemma_rewrite")
            return KoreanTranslation(
                ko_literal=rewritten_text,
                ko_natural=rewritten_text,
                notes=[],
                confidence=0.99,
                model="fake-gemma",
                batch_id=str(kwargs["batch_id"]),
            )

    monkeypatch.setattr(steps, "ManagedGPTSoVITSServer", FakeGSVManager)
    monkeypatch.setattr(steps, "ManagedGemmaTextServer", FakeGemmaManager)
    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "LlamaServerTranslationClient", FakeGemmaClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        auto_gsv_server=True,
    )

    assert events == [
        "gsv_start",
        f"tts:{original_text}",
        f"tts:{original_text}",
        "gsv_stop",
        "gemma_start",
        "gemma_rewrite",
        "gemma_stop",
        "gsv_start",
        f"tts:{rewritten_text}",
        "gsv_stop",
    ]


def test_synth_skips_deferred_duration_rewrite_retry_when_gemma_rejects(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_duration_rewrite_backend="gemma",
        gsv_duration_rewrite_max_attempts=1,
        gsv_duration_rewrite_pre_candidate_count=2,
        gsv_low_temperature_retry_enabled=False,
        gsv_terminal_failure_policy="fail",
        gemma_text_server_auto_start=False,
        duration_tolerance=0.2,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=4.0)
    original_text = "괜찮아요."
    rejected_text = "짧아요."
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="大丈夫です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            ja_text="大丈夫です。",
            tts_text=original_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payload_texts: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def build_payload(self, text, ref, options=None):
            payload_texts.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.0)
            return output_path

    class FakeGemmaClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def rewrite_tts_for_duration(self, **kwargs):
            return KoreanTranslation(
                ko_literal=rejected_text,
                ko_natural=rejected_text,
                notes=[],
                confidence=0.99,
                model="fake-gemma",
                batch_id=str(kwargs["batch_id"]),
            )

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "LlamaServerTranslationClient", FakeGemmaClient)

    with pytest.raises(GPTSoVITSError, match="seg_0001"):
        synth_step(
            tmp_project_dir,
            gsv_url="http://gsv.local",
            refs_path=tmp_project_dir / "refs" / "refs.json",
            mock=False,
            confirm_rights=True,
        )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    skipped = segment.analysis["duration_rewrite_retry_skipped"]
    assert payload_texts == [original_text, original_text]
    assert segment.status == "failed"
    assert segment.tts is not None
    assert skipped["accepted"] is False
    assert skipped["retry_scheduled"] is False
    assert "pending_duration_rewrite" not in segment.analysis
    assert segment.analysis["duration_rewrite_history"][0]["retry_scheduled"] is False


def test_synth_uses_three_gsv_lanes_by_segment_id(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", gsv_concurrency=3)
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    segments: list[Segment] = []
    spoken_numbers = ["일", "이", "삼", "사", "오", "육"]
    for index in range(1, 7):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"seg_{index:04d}_mix.wav"
        write_tiny_wav(audio)
        source_text = f"テスト {index}"
        tts_text = f"테스트 {spoken_numbers[index - 1]}"
        segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=(index - 1) * 1.2,
                end=index * 1.2,
                duration=1.2,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text=source_text,
                    language="ja",
                    backend="mock",
                    start=(index - 1) * 1.2,
                    end=index * 1.2,
                ),
                script=JapaneseScript(
                    literal_ja=source_text,
                    ja_text=source_text,
                    tts_text=tts_text,
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                ),
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, base_url: str, *args: object, **kwargs: object) -> None:
            self.base_url = base_url

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            calls.append((request.text, self.base_url))
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    by_text = dict(calls)
    assert by_text["테스트 일"] == "http://127.0.0.1:9880"
    assert by_text["테스트 이"] == "http://127.0.0.1:9881"
    assert by_text["테스트 삼"] == "http://127.0.0.1:9882"
    assert by_text["테스트 사"] == "http://127.0.0.1:9880"
    assert by_text["테스트 오"] == "http://127.0.0.1:9881"
    assert by_text["테스트 육"] == "http://127.0.0.1:9882"
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["synth"]["concurrency"] == 3


def test_synth_limits_gsv_lanes_to_only_segments(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", gsv_concurrency=3, candidate_count=1)
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    segments: list[Segment] = []
    spoken_numbers = ["일", "이", "삼"]
    for index in range(1, 4):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"seg_{index:04d}_mix.wav"
        write_tiny_wav(audio)
        source_text = f"テスト {index}"
        segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=(index - 1) * 1.2,
                end=index * 1.2,
                duration=1.2,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text=source_text,
                    language="ja",
                    backend="mock",
                    start=(index - 1) * 1.2,
                    end=index * 1.2,
                ),
                script=JapaneseScript(
                    literal_ja=source_text,
                    ja_text=source_text,
                    tts_text=f"테스트 {spoken_numbers[index - 1]}",
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                ),
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, base_url: str, *args: object, **kwargs: object) -> None:
            self.base_url = base_url

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            calls.append((request.text, self.base_url))
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        only_segment_ids={"seg_0002"},
    )

    assert calls == [("테스트 이", "http://127.0.0.1:9880")]
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["synth"]["concurrency"] == 1


def test_synth_time_fits_audible_candidate_when_duration_gate_fails(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", gsv_concurrency=1, duration_tolerance=0.05)
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="テストです",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class ShortClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            duration = 2.84
            t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
            tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
            write_audio(output_path, np.stack([tone, tone], axis=1), sample_rate)
            return output_path

    def fake_fit_audio_duration(
        input_path: Path,
        output_path: Path,
        *,
        target_duration_sec: float,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, sample_rate, channels
        sr = sample_rate or 48_000
        edge = int(sr * 0.8)
        tone_frames = int(sr * max(0.1, target_duration_sec - 1.6))
        t = np.arange(tone_frames, dtype=np.float32) / sr
        tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
        silence = np.zeros(edge, dtype=np.float32)
        signal = np.concatenate([silence, tone, silence])
        write_audio(output_path, np.stack([signal, signal], axis=1), sr)
        return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ShortClient)
    monkeypatch.setattr(steps.ffmpeg, "fit_audio_duration", fake_fit_audio_duration)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    tts = manifest.segments[0].tts
    assert tts is not None
    selected = next(candidate for candidate in tts.candidates if candidate.selected)
    assert selected.duration_gate == "pass"
    assert selected.acceptable_for_mix is True
    assert selected.payload["time_fit"]["source_duration_sec"] == pytest.approx(2.84, abs=0.01)
    assert selected.payload["time_fit"]["target_duration_sec"] == 3.0
    assert selected.payload["time_fit"]["stretch"] == pytest.approx(3.0 / 2.84)
    assert Path(tts.selected_candidate_path).exists()


def test_synth_rejects_excessive_time_fit_stretch(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_concurrency=1,
        duration_tolerance=0.2,
        gsv_terminal_failure_policy="fail",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="テストです",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class ShortClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.0)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ShortClient)

    with pytest.raises(GPTSoVITSError, match="seg_0001"):
        synth_step(
            tmp_project_dir,
            gsv_url="http://127.0.0.1:9880",
            refs_path=tmp_project_dir / "refs" / "refs.json",
            mock=False,
            confirm_rights=True,
        )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "failed"
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is None
    assert segment.tts.candidates[0].payload["time_fit"]["rejected_reason"] == (
        "stretch_above_max:2.609>1.080"
    )


def test_synth_rejects_excessive_time_fit_compression(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_concurrency=1,
        duration_tolerance=0.2,
        gsv_timefit_max_tempo=1.35,
        gsv_terminal_failure_policy="fail",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="テストです",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class LongClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            t = np.arange(sample_rate * 8, dtype=np.float32) / sample_rate
            tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
            write_audio(output_path, np.stack([tone, tone], axis=1), sample_rate)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", LongClient)

    with pytest.raises(GPTSoVITSError, match="seg_0001"):
        synth_step(
            tmp_project_dir,
            gsv_url="http://127.0.0.1:9880",
            refs_path=tmp_project_dir / "refs" / "refs.json",
            mock=False,
            confirm_rights=True,
        )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "failed"
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is None
    rejected = [
        candidate
        for candidate in segment.tts.candidates
        if candidate.payload.get("time_fit", {}).get("rejected_reason")
    ]
    assert rejected
    assert rejected[0].payload["time_fit"]["rejected_reason"] == (
        "tempo_above_max:2.667>1.350"
    )


def test_synth_rescues_existing_audible_candidate_with_relaxed_duration_gate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_max_attempts_per_candidate=1,
        gsv_micro_segment_unfit_policy="manual_review",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=10.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=10.0,
        duration=10.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="長い台詞です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=10.0,
        ),
        script=JapaneseScript(
            literal_ja="長い台詞です。",
            ja_text="長い台詞です。",
            tts_text="긴 대사예요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class ShortClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=7.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ShortClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    strict_candidate = segment.tts.candidates[0]
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert strict_candidate.duration_gate == "too_short"
    assert strict_candidate.acceptable_for_mix is False
    assert selected.output_path == strict_candidate.output_path
    assert selected.duration_gate == "pass"
    assert selected.acceptable_for_mix is True
    assert selected.selection_reason == "duration_relaxed_rescue"
    assert selected.payload["rescue"]["tier"] == "relaxed_duration_gate"
    assert selected.payload["rescue"]["strict_duration_gate"] == "too_short"
    assert selected.payload["rescue"]["duration_tolerance_used"] == pytest.approx(0.35)


def test_synth_pads_short_candidate_when_speech_budget_matches_long_asmr_pause(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_max_attempts_per_candidate=1,
        gsv_micro_segment_unfit_policy="manual_review",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=11.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=11.0,
        duration=11.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="長めの休止を含む台詞です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=11.0,
        ),
        script=JapaneseScript(
            literal_ja="長めの休止を含む台詞です。",
            ja_text="長めの休止を含む台詞です。",
            tts_text="긴 쉼을 포함한 대사예요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=6.0,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class ShortClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=6.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ShortClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    strict_candidate = segment.tts.candidates[0]
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert strict_candidate.duration_gate == "too_short"
    assert strict_candidate.acceptable_for_mix is False
    assert selected.selection_reason == "source_pause_padding_rescue"
    assert selected.duration_gate == "pass"
    assert selected.acceptable_for_mix is True
    assert duration_sec(Path(selected.output_path)) == pytest.approx(segment.duration, abs=0.02)
    assert selected.payload["pause_padding"]["tier"] == "source_pause_padding"
    assert selected.payload["pause_padding"]["speech_duration_sec"] == pytest.approx(6.2, abs=0.02)
    assert selected.payload["pause_padding"]["padding_sec"] == pytest.approx(4.8, abs=0.02)
    assert selected.payload["pause_padding"]["expected_tts_duration_sec"] == pytest.approx(6.0)


def test_synth_pads_short_candidate_when_omission_signal_is_only_long_pause_ratio(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_max_attempts_per_candidate=1,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=11.8)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=11.8,
        duration=11.8,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="スーッと全身の力を抜いて、",
            language="ja",
            backend="mock",
            start=0.0,
            end=11.8,
        ),
        script=JapaneseScript(
            literal_ja="スーッと全身の力を抜いて、",
            ja_text="スーッと全身の力を抜いて、",
            tts_text="몸의 힘을 스르르 빼고",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=2.25,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class ShortClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=2.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ShortClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    strict_candidate = segment.tts.candidates[0]
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert strict_candidate.selection_reason == "omission_suspected"
    assert strict_candidate.payload["omission_detection"]["reasons"] == [
        "duration_below_segment_ratio:0.186<0.250"
    ]
    assert selected.selection_reason == "source_pause_padding_rescue"
    assert selected.payload["pause_padding"]["allowed_omission_detection_reasons"] == [
        "duration_below_segment_ratio:0.186<0.250"
    ]
    assert duration_sec(Path(selected.output_path)) == pytest.approx(segment.duration, abs=0.02)


def test_synth_periodizes_korean_counting_runs_and_skips_compaction_before_gsv_request(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_max_attempts_per_candidate=1,
        gsv_numeric_sequence_qc_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=2.45)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=2.45,
        duration=2.45,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="1,2,3,4,5,6,7,8,9,10",
            language="ja",
            backend="mock",
            start=0.0,
            end=2.45,
        ),
        script=JapaneseScript(
            literal_ja="1,2,3,4,5,6,7,8,9,10",
            ja_text="1,2,3,4,5,6,7,8,9,10",
            tts_text="일, 이, 삼, 사, 오, 육, 칠, 팔, 구, 십 말이에요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=2.5,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    request_texts: list[str] = []

    class RecordingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            request_texts.append(request.text)
            _write_tone_wav(output_path, duration=2.4)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", RecordingClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert request_texts == ["하나. 둘. 셋. 넷. 다섯. 여섯. 일곱. 여덟. 아홉. 열."]
    assert segment.script is not None
    assert segment.script.tts_text == "하나. 둘. 셋. 넷. 다섯. 여섯. 일곱. 여덟. 아홉. 열."
    assert "pre_synth_tts_counting_compaction" not in segment.analysis
    assert segment.analysis["pre_synth_tts_numeric_cadence_periodization"]["before"] == (
        "일, 이, 삼, 사, 오, 육, 칠, 팔, 구, 십 말이에요."
    )
    assert segment.analysis["pre_synth_tts_numeric_cadence_periodization"]["variant"] == (
        "native_periods_no_compact"
    )


def test_korean_counting_compaction_does_not_cross_sentence_boundary_before_measure() -> None:
    compacted, metadata = _compact_korean_counting_tts_text(
        "일곱, 여덟, 아홉, 열. 여덟 배의 절정 쾌감."
    )

    assert compacted == "일곱여덟아홉열. 여덟 배의 절정 쾌감."
    assert metadata is not None
    assert metadata["runs"] == [
        {
            "before": "일곱, 여덟, 아홉, 열",
            "after": "일곱여덟아홉열",
            "token_count": 4,
        }
    ]


def test_synth_time_fits_compacted_counting_candidate_with_counting_rescue_tempo(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.05,
        gsv_max_attempts_per_candidate=1,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=11.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=11.0,
        duration=11.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="7、8、9、10。8倍の絶頂快感。1、3、4、5、6、7、8、9、10。",
            language="ja",
            backend="mock",
            start=0.0,
            end=11.0,
        ),
        script=JapaneseScript(
            literal_ja="7、8、9、10。8倍の絶頂快感。1、3、4、5、6、7、8、9、10。",
            ja_text="7、8、9、10。8倍の絶頂快感。1、3、4、5、6、7、8、9、10。",
            tts_text=(
                "일곱, 여덟, 아홉, 열. 여덟 배의 절정 쾌감. "
                "하나, 셋, 넷, 다섯, 여섯, 일곱, 여덟, 아홉, 열."
            ),
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=13.5,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class LongCountingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=16.1)
            return output_path

    def fake_fit_audio_duration(
        input_path: Path,
        output_path: Path,
        *,
        target_duration_sec: float,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, channels
        _write_tone_wav(output_path, duration=target_duration_sec, sample_rate=sample_rate or 48_000)
        return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", LongCountingClient)
    monkeypatch.setattr(steps.ffmpeg, "fit_audio_duration", fake_fit_audio_duration)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.selection_reason == "duration_relaxed_timefit_rescue"
    assert selected.payload["time_fit"]["max_tempo"] == pytest.approx(1.6)
    assert selected.payload["rescue"]["counting_compaction_timefit"] is True
    assert "열. 여덟 배" in segment.script.tts_text


def test_synth_marks_micro_segment_manual_review_after_rescue_exhausted(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_max_attempts_per_candidate=1,
        gsv_micro_segment_unfit_policy="manual_review",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=0.3)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=0.3,
        duration=0.3,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="うん",
            language="ja",
            backend="mock",
            start=0.0,
            end=0.3,
        ),
        script=JapaneseScript(
            literal_ja="うん",
            ja_text="うん",
            tts_text="응.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class LongClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.0)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", LongClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "needs_manual_review"
    assert manifest.stage_state["synth"]["status"] == "completed"
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is None
    assert segment.tts.retry_summary["rescue_status"] == "micro_segment_manual_review"
    assert segment.tts.retry_summary["micro_segment_max_sec"] == pytest.approx(0.6)
    assert "Micro segment too short for Korean TTS." in segment.errors


def test_synth_keeps_original_for_unfit_micro_segment_by_default(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_max_attempts_per_candidate=1,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=0.3)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=0.3,
        duration=0.3,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="うん",
            language="ja",
            backend="mock",
            start=0.0,
            end=0.3,
        ),
        script=JapaneseScript(
            literal_ja="うん",
            ja_text="うん",
            tts_text="응.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class LongClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.0)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", LongClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "absorbed"
    assert segment.keep_original_texture is True
    assert manifest.stage_state["synth"]["status"] == "completed"
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is None
    assert segment.tts.retry_summary["rescue_status"] == "micro_segment_keep_original"
    assert segment.analysis["micro_segment_auto_fallback"]["action"] == "keep_original_micro_segment"
    assert segment.analysis["micro_segment_auto_fallback"]["reason"] == "tts_duration_unfit"


def test_synth_time_fits_micro_segment_with_relaxed_compression(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_concurrency=1,
        duration_tolerance=0.05,
        gsv_timefit_max_tempo=1.18,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.8)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.8,
        duration=1.8,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="重い",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.8,
        ),
        script=JapaneseScript(
            literal_ja="重い",
            ja_text="重い",
            tts_text="무거워.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class LongClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=2.3)
            return output_path

    def fake_fit_audio_duration(
        input_path: Path,
        output_path: Path,
        *,
        target_duration_sec: float,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, channels
        _write_tone_wav(output_path, duration=target_duration_sec, sample_rate=sample_rate or 48_000)
        return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", LongClient)
    monkeypatch.setattr(steps.ffmpeg, "fit_audio_duration", fake_fit_audio_duration)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.acceptable_for_mix is True
    assert selected.payload["time_fit"]["policy"] == "micro_segment_relaxed"
    assert selected.payload["time_fit"]["max_tempo"] == pytest.approx(1.3)


def test_synth_time_fits_mix_pass_candidate_outside_timing_quality_tolerance(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_timing_quality_tolerance=0.10,
        gsv_timefit_max_tempo=1.18,
        gsv_max_attempts_per_candidate=1,
        gsv_pronunciation_qc_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=4.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="少し長いです",
            language="ja",
            backend="mock",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            literal_ja="少し長いです",
            ja_text="少し長いです",
            tts_text="조금 길어요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class SlightlyLongClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=4.44)
            return output_path

    fit_calls: list[Path] = []

    def fake_fit_audio_duration(
        input_path: Path,
        output_path: Path,
        *,
        target_duration_sec: float,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = channels
        fit_calls.append(input_path)
        _write_tone_wav(output_path, duration=target_duration_sec, sample_rate=sample_rate or 48_000)
        return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", SlightlyLongClient)
    monkeypatch.setattr(steps.ffmpeg, "fit_audio_duration", fake_fit_audio_duration)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert fit_calls
    assert selected.selection_reason == "timing_quality_timefit"
    assert selected.duration_ratio == pytest.approx(1.0, abs=0.02)
    assert selected.timing_quality_gate == "good"
    assert selected.payload["timing_quality"]["gate"] == "good"
    assert selected.payload["time_fit"]["source_timing_quality_gate"] == "warn"
    assert segment.tts.retry_summary["selected_timing_quality_gate"] == "good"


def test_synth_records_timing_warning_when_mix_pass_candidate_cannot_be_safely_timefit(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.25,
        gsv_timing_quality_tolerance=0.10,
        gsv_timefit_max_tempo=1.18,
        gsv_max_attempts_per_candidate=1,
        gsv_pronunciation_qc_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=4.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="かなり長いです",
            language="ja",
            backend="mock",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            literal_ja="かなり長いです",
            ja_text="かなり長いです",
            tts_text="꽤 길어요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class TooLongButMixPassClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=4.84)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", TooLongButMixPassClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.acceptable_for_mix is True
    assert selected.duration_gate == "pass"
    assert selected.timing_quality_gate == "warn"
    assert selected.payload["timing_quality"]["gate"] == "warn"
    assert selected.payload["time_fit"]["rejected_reason"].startswith("tempo_above_max")
    assert segment.tts.retry_summary["selected_timing_quality_gate"] == "warn"


def test_synth_time_fits_long_segment_with_relaxed_stretch(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_concurrency=1,
        duration_tolerance=0.05,
        gsv_timefit_max_stretch=1.12,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=11.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=11.0,
        duration=11.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="長い台詞です。もう一文あります。",
            language="ja",
            backend="mock",
            start=0.0,
            end=11.0,
        ),
        script=JapaneseScript(
            literal_ja="長い台詞です。もう一文あります。",
            ja_text="長い台詞です。もう一文あります。",
            tts_text="긴 대사예요. 문장이 하나 더 있어요.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class ShortClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=9.6)
            return output_path

    def fake_fit_audio_duration(
        input_path: Path,
        output_path: Path,
        *,
        target_duration_sec: float,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, channels
        _write_tone_wav(output_path, duration=target_duration_sec, sample_rate=sample_rate or 48_000)
        return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ShortClient)
    monkeypatch.setattr(steps.ffmpeg, "fit_audio_duration", fake_fit_audio_duration)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.acceptable_for_mix is True
    assert selected.payload["time_fit"]["policy"] == "long_segment_relaxed"
    assert selected.payload["time_fit"]["max_stretch"] == pytest.approx(1.15)


def test_synth_uses_configured_max_attempts_per_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        duration_tolerance=0.05,
        gsv_timefit_max_tempo=1.05,
        gsv_max_attempts_per_candidate=5,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="テストです",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    durations = [5.0, 4.4, 4.2, 4.0, 3.0]
    calls: list[int] = []
    seeds: list[int] = []

    class LatePassingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            attempt_index = len(calls)
            calls.append(attempt_index)
            seeds.append(request.seed)
            _write_tone_wav(output_path, duration=durations[attempt_index])
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", LatePassingClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert calls == [0, 1, 2, 3, 4]
    assert len(set(seeds)) >= 3
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.retry_summary["attempt"] == 4
    assert selected.retry_summary["max_attempts"] == 5


def test_synth_retries_silent_candidate_even_when_duration_matches(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", candidate_count=1, gsv_concurrency=1)
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="テストです",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class SilentThenToneClient:
        calls = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            type(self).calls += 1
            if type(self).calls == 1:
                write_audio(output_path, np.zeros((16000, 1), dtype=np.float32), 16000)
            else:
                write_tiny_wav(output_path, duration=1.0)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", SilentThenToneClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    tts = manifest.segments[0].tts
    assert tts is not None
    assert SilentThenToneClient.calls == 2
    assert tts.selected_candidate_path is not None
    selected = next(candidate for candidate in tts.candidates if candidate.selected)
    assert selected.payload["audio_qc"]["gate"] == "pass"
    assert selected.acceptable_for_mix is True
    silent = tts.candidates[0]
    assert silent.payload["audio_qc"]["gate"] == "silent"
    assert silent.payload["retry"]["next_action"] == "seed_changed"


def test_synth_fails_segment_when_all_gsv_candidates_are_silent(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        gsv_concurrency=1,
        gsv_terminal_failure_policy="fail",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        source_script=SourceScript(
            text="テストです",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="テストです",
            ja_text="テストです",
            tts_text="테스트입니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class SilentClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_audio(output_path, np.zeros((16000, 1), dtype=np.float32), 16000)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", SilentClient)

    with pytest.raises(GPTSoVITSError, match="seg_0001"):
        synth_step(
            tmp_project_dir,
            gsv_url="http://127.0.0.1:9880",
            refs_path=tmp_project_dir / "refs" / "refs.json",
            mock=False,
            confirm_rights=True,
        )

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["synth"]["status"] == "failed"
    assert manifest.segments[0].status == "failed"
    assert "No acceptable TTS candidates for mix." in manifest.segments[0].errors
    assert manifest.segments[0].tts is not None
    assert manifest.segments[0].tts.selected_candidate_path is None
    assert all(not candidate.selected for candidate in manifest.segments[0].tts.candidates)


def test_semantic_parts_merge_with_upstream_header(tmp_project_dir: Path) -> None:
    dataset_dir = tmp_project_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "6-name2semantic-0.tsv").write_text("seg_0001\t1 2 3\n", "utf-8")
    (dataset_dir / "6-name2semantic-1.tsv").write_text("seg_0002\t4 5 6\n", "utf-8")

    from asmr_dub_pipeline.gpt_sovits import few_shot

    few_shot._merge_dataset_part_outputs(dataset_dir)

    assert (dataset_dir / "6-name2semantic.tsv").read_text("utf-8").splitlines() == [
        "item_name\tsemantic_audio",
        "seg_0001\t1 2 3",
        "seg_0002\t4 5 6",
    ]


def test_pandas_shim_default_header_matches_gpt_sovits_semantics(tmp_project_dir: Path) -> None:
    from asmr_dub_pipeline.gpt_sovits.shims import pandas

    tmp_project_dir.mkdir(parents=True)
    csv_path = tmp_project_dir / "semantic.tsv"
    csv_path.write_text("item_name\tsemantic_audio\nseg_0001\t1 2 3\n", "utf-8")

    default_frame = pandas.read_csv(csv_path, delimiter="\t")
    raw_frame = pandas.read_csv(csv_path, delimiter="\t", header=None)

    assert len(default_frame) == 1
    assert default_frame.iloc[0, 0] == "seg_0001"
    assert len(raw_frame) == 2
