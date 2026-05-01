from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from conftest import write_tiny_wav

import asmr_dub_pipeline.cli as cli_module
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import load_project_config
from asmr_dub_pipeline.schemas import (
    GSVSpeakerConfig,
    PipelineManifest,
    ProjectConfig,
    RVCSpeakerConfig,
    Segment,
    SourceInfo,
    VoiceBankManifest,
    VoiceBankSourceSegment,
    VoiceBankSpeaker,
)
from asmr_dub_pipeline.voice_bank import (
    DiarizationTurn,
    VoiceBankError,
    assign_speakers_to_manifest,
    build_voice_bank,
    cluster_turns,
    load_voice_bank,
    save_voice_bank,
    validate_voice_bank_models,
)
from asmr_dub_pipeline.voice_bank import manager as voice_bank_manager


def _voice_bank_speaker(project_dir: Path, source_path: Path) -> VoiceBankSpeaker:
    speaker_dir = project_dir / "voice_bank" / "speakers" / "speaker_0001"
    gpt = speaker_dir / "gsv" / "v001" / "gpt.ckpt"
    gsv = speaker_dir / "gsv" / "v001" / "final.pth"
    refs = speaker_dir / "refs" / "refs.json"
    ref_audio = speaker_dir / "refs" / "whisper_close.wav"
    rvc = speaker_dir / "rvc" / "v001" / "model.pth"
    index = speaker_dir / "rvc" / "v001" / "added.index"
    for path, payload in (
        (gpt, b"gpt"),
        (gsv, b"sovits"),
        (ref_audio, b"wav"),
        (rvc, b"rvc"),
        (index, b"index"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    refs.write_text(
        json.dumps(
            {
                "whisper_close": {
                    "ref_audio_path": str(ref_audio.relative_to(project_dir)),
                    "prompt_text": "こんにちは",
                    "prompt_lang": "ja",
                }
            }
        ),
        "utf-8",
    )
    return VoiceBankSpeaker(
        speaker_id="speaker_0001",
        source_segments=[
            VoiceBankSourceSegment(
                source_id="src_0001",
                source_path=str(source_path.resolve()),
                local_speaker_label="SPEAKER_00",
                segment_id="src_0001_SPEAKER_00_0001",
                speaker_id="speaker_0001",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_path="voice_bank/speakers/speaker_0001/clips/src_0001_SPEAKER_00_0001.wav",
            )
        ],
        gsv=GSVSpeakerConfig(
            gpt_weights_path=str(gpt.relative_to(project_dir)),
            sovits_weights_path=str(gsv.relative_to(project_dir)),
            refs_path=str(refs.relative_to(project_dir)),
        ),
        rvc=RVCSpeakerConfig(
            model_path=str(rvc.relative_to(project_dir)),
            index_path=str(index.relative_to(project_dir)),
        ),
        dataset_fingerprint="fingerprint",
    )


def test_voice_bank_manifest_round_trip_and_validation(tmp_project_dir: Path, tiny_wav_path: Path) -> None:
    bank_path = tmp_project_dir / "voice_bank" / "voice_bank_manifest.json"
    bank = VoiceBankManifest(
        speakers={"speaker_0001": _voice_bank_speaker(tmp_project_dir, tiny_wav_path)},
        source_paths=[str(tiny_wav_path.resolve())],
        backend="mock",
    )

    save_voice_bank(bank_path, bank)
    loaded = load_voice_bank(tmp_project_dir, ProjectConfig())

    assert loaded.speakers["speaker_0001"].speaker_id == "speaker_0001"
    validate_voice_bank_models(tmp_project_dir, loaded)

    (tmp_project_dir / loaded.speakers["speaker_0001"].gsv.sovits_weights_path).unlink()
    with pytest.raises(VoiceBankError, match="SoVITS weights missing"):
        validate_voice_bank_models(tmp_project_dir, loaded)


def test_file_safe_segment_and_speaker_ids_are_enforced() -> None:
    with pytest.raises(ValueError, match="file-safe"):
        Segment(
            id="../seg",
            start=0.0,
            end=1.0,
            duration=1.0,
            audio_for_gemma="a.wav",
            audio_for_mix="b.wav",
        )
    with pytest.raises(ValueError, match="file-safe"):
        VoiceBankSourceSegment(
            source_id="src_0001",
            source_path="input.wav",
            local_speaker_label="SPEAKER_00",
            segment_id="bad/segment",
            speaker_id="speaker_0001",
            start=0.0,
            end=1.0,
            duration=1.0,
            audio_path="clip.wav",
        )


def test_cross_source_clustering_merges_same_embedding_and_splits_different() -> None:
    turns = [
        DiarizationTurn("src_1", Path("a.wav"), "A", 0.0, 1.0, embedding=np.array([1.0, 0.0])),
        DiarizationTurn("src_2", Path("b.wav"), "B", 0.0, 1.0, embedding=np.array([0.99, 0.01])),
        DiarizationTurn("src_3", Path("c.wav"), "C", 0.0, 1.0, embedding=np.array([0.0, 1.0])),
    ]

    clustered = cluster_turns(turns, threshold=0.95)

    assert clustered[0].speaker_id == clustered[1].speaker_id
    assert clustered[2].speaker_id != clustered[0].speaker_id


def test_pyannote_diarize_skips_subwindow_turns_before_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSegment:
        def __init__(self, start: float, end: float) -> None:
            self.start = start
            self.end = end

    class FakeAnnotation:
        def itertracks(self, yield_label: bool):
            assert yield_label is True
            yield FakeSegment(0.000, 0.017), None, "SPEAKER_00"
            yield FakeSegment(0.300, 1.000), None, "SPEAKER_00"

    class FakePipeline:
        def __call__(self, audio_path: str, **kwargs):
            _ = audio_path, kwargs
            return FakeAnnotation()

    backend = object.__new__(voice_bank_manager.PyannoteDiarizationBackend)
    backend.pipeline = FakePipeline()
    embedded: list[tuple[float, float]] = []

    def fake_embed_excerpt(audio_path: Path, start: float, end: float) -> np.ndarray:
        _ = audio_path
        embedded.append((start, end))
        return np.array([1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(backend, "embed_excerpt", fake_embed_excerpt)

    turns = backend.diarize(tmp_path / "source.wav", "src_0001", ProjectConfig())

    assert embedded == [(0.3, 1.0)]
    assert len(turns) == 1
    assert turns[0].start == pytest.approx(0.3)
    assert turns[0].end == pytest.approx(1.0)


def test_pyannote_diarize_reports_when_all_turns_are_too_short(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSegment:
        start = 0.0
        end = 0.017

    class FakeAnnotation:
        def itertracks(self, yield_label: bool):
            assert yield_label is True
            yield FakeSegment(), None, "SPEAKER_00"

    class FakePipeline:
        def __call__(self, audio_path: str, **kwargs):
            _ = audio_path, kwargs
            return FakeAnnotation()

    backend = object.__new__(voice_bank_manager.PyannoteDiarizationBackend)
    backend.pipeline = FakePipeline()
    monkeypatch.setattr(
        backend,
        "embed_excerpt",
        lambda audio_path, start, end: pytest.fail("short turns should not be embedded"),
    )

    with pytest.raises(VoiceBankError, match=r"shorter than 0\.250s"):
        backend.diarize(tmp_path / "source.wav", "src_0001", ProjectConfig())


def test_pyannote_tf32_flags_are_disabled_before_cuda_inference() -> None:
    class FakeMatmul:
        allow_tf32 = True

    class FakeCudaBackend:
        matmul = FakeMatmul()

    class FakeCudnnBackend:
        allow_tf32 = True

    class FakeBackends:
        cuda = FakeCudaBackend()
        cudnn = FakeCudnnBackend()

    class FakeTorch:
        backends = FakeBackends()

    voice_bank_manager._disable_pyannote_tf32(FakeTorch)

    assert FakeTorch.backends.cuda.matmul.allow_tf32 is False
    assert FakeTorch.backends.cudnn.allow_tf32 is False


def test_embedding_crop_bounds_expands_short_turn_to_stable_window(tiny_wav_path: Path) -> None:
    start, end = voice_bank_manager._embedding_crop_bounds(tiny_wav_path, 0.02, 0.05)

    assert start == pytest.approx(0.0)
    assert end - start >= 0.5


def test_mock_voice_bank_build_creates_project_models_and_config(tmp_project_dir: Path, tmp_path: Path) -> None:
    first = write_tiny_wav(tmp_path / "first.wav")
    second = write_tiny_wav(tmp_path / "second.wav")

    bank = build_voice_bank(
        [first, second],
        tmp_project_dir,
        confirm_rights=True,
        backend_kind="mock",
        mock_training=True,
    )
    cfg = load_project_config(tmp_project_dir)

    assert sorted(bank.speakers) == ["speaker_0001"]
    assert len(bank.speakers["speaker_0001"].source_segments) == 2
    assert {segment.source_path for segment in bank.speakers["speaker_0001"].source_segments} == {
        str(first.resolve()),
        str(second.resolve()),
    }
    assert sorted(cfg.gsv_speaker_models) == ["speaker_0001"]
    assert sorted(cfg.rvc_speaker_models) == ["speaker_0001"]
    assert cfg.rvc_train_required is False
    validate_voice_bank_models(tmp_project_dir, bank)


def test_voice_bank_build_reuses_unchanged_speaker_artifacts(
    tmp_project_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = write_tiny_wav(tmp_path / "first.wav")
    second = write_tiny_wav(tmp_path / "second.wav")

    class TwoSpeakerBackend:
        name = "fake"

        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = cfg
            embedding = np.array([1.0, 0.0]) if source_id.startswith("src_0001") else np.array([0.0, 1.0])
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="SPEAKER_00",
                    start=0.0,
                    end=1.0,
                    embedding=embedding,
                )
            ]

        def embed_clip(self, audio_path: Path) -> np.ndarray:
            _ = audio_path
            return np.array([1.0, 0.0])

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: TwoSpeakerBackend())
    first_bank = build_voice_bank(
        [first, second],
        tmp_project_dir,
        confirm_rights=True,
        backend_kind="fake",
        mock_training=True,
    )
    speaker_1_weights = tmp_project_dir / first_bank.speakers["speaker_0001"].gsv.sovits_weights_path
    speaker_2_weights = tmp_project_dir / first_bank.speakers["speaker_0002"].gsv.sovits_weights_path
    speaker_1_mtime = speaker_1_weights.stat().st_mtime_ns
    speaker_2_mtime = speaker_2_weights.stat().st_mtime_ns

    second_bank = build_voice_bank(
        [first, second],
        tmp_project_dir,
        confirm_rights=True,
        backend_kind="fake",
        mock_training=True,
    )

    assert second_bank.speakers["speaker_0001"].dataset_fingerprint == first_bank.speakers["speaker_0001"].dataset_fingerprint
    assert speaker_1_weights.stat().st_mtime_ns == speaker_1_mtime
    assert speaker_2_weights.stat().st_mtime_ns == speaker_2_mtime

    time.sleep(0.01)
    replacement = np.stack([np.zeros(24_000, dtype=np.float32), np.zeros(24_000, dtype=np.float32)], axis=1)
    sf.write(second, replacement, 48_000)
    third_bank = build_voice_bank(
        [first, second],
        tmp_project_dir,
        confirm_rights=True,
        backend_kind="fake",
        mock_training=True,
    )

    assert third_bank.speakers["speaker_0001"].dataset_fingerprint == first_bank.speakers["speaker_0001"].dataset_fingerprint
    assert third_bank.speakers["speaker_0002"].dataset_fingerprint != first_bank.speakers["speaker_0002"].dataset_fingerprint
    assert speaker_1_weights.stat().st_mtime_ns == speaker_1_mtime
    assert speaker_2_weights.stat().st_mtime_ns > speaker_2_mtime


def test_assign_speakers_from_voice_bank_overlap_early_fails_missing_model(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
) -> None:
    bank = VoiceBankManifest(
        speakers={"speaker_0001": _voice_bank_speaker(tmp_project_dir, tiny_wav_path)},
        source_paths=[str(tiny_wav_path.resolve())],
        backend="mock",
    )
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="a.wav",
        audio_for_mix="b.wav",
    )
    manifest = PipelineManifest(
        project_config=ProjectConfig(),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=1.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        segments=[segment],
    )

    assigned = assign_speakers_to_manifest(tmp_project_dir, manifest, bank)
    assert assigned.segments[0].speaker_id == "speaker_0001"

    assigned.segments[0].speaker_id = None
    assigned.source_info = assigned.source_info.model_copy(update={"path": str((tmp_project_dir / "other.wav").resolve())})
    with pytest.raises(VoiceBankError, match="speaker assignment failed"):
        assign_speakers_to_manifest(tmp_project_dir, assigned, bank)


def test_pyannote_embedding_assignment_requires_configured_threshold(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    centroid_path = tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "embedding.npy"
    centroid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(centroid_path), np.array([1.0, 0.0], dtype=np.float32))
    speaker = _voice_bank_speaker(tmp_project_dir, tiny_wav_path).model_copy(
        update={"embedding_centroid_path": str(centroid_path.relative_to(tmp_project_dir))}
    )
    bank = VoiceBankManifest(
        speakers={"speaker_0001": speaker},
        source_paths=[str(tiny_wav_path.resolve())],
        backend="mock",
    )
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="a.wav",
        audio_for_mix="b.wav",
    )
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.78),
        source_info=SourceInfo(
            path=str((tmp_project_dir / "unseen.wav").resolve()),
            duration_sec=1.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        segments=[segment],
    )

    class FakeEmbeddingBackend:
        def __init__(self, embedding: np.ndarray) -> None:
            self.embedding = embedding

        def embed_clip(self, audio_path: Path) -> np.ndarray:
            _ = audio_path
            return self.embedding

    monkeypatch.setattr(
        voice_bank_manager,
        "create_diarization_backend",
        lambda kind, cfg: FakeEmbeddingBackend(np.array([0.7, 0.714], dtype=np.float32)),
    )
    with pytest.raises(VoiceBankError, match="speaker assignment failed"):
        assign_speakers_to_manifest(tmp_project_dir, manifest, bank, backend_kind="pyannote")

    monkeypatch.setattr(
        voice_bank_manager,
        "create_diarization_backend",
        lambda kind, cfg: FakeEmbeddingBackend(np.array([0.9, 0.436], dtype=np.float32)),
    )
    assigned = assign_speakers_to_manifest(tmp_project_dir, manifest, bank, backend_kind="pyannote")

    assert assigned.segments[0].speaker_id == "speaker_0001"


def test_pyannote_model_resolution_uses_local_cache_then_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_home = tmp_path / "hf"
    cached = (
        hf_home
        / "hub"
        / "models--pyannote--embedding"
        / "snapshots"
        / "cached123"
    )
    cached.mkdir(parents=True)
    (cached / "config.yaml").write_text("pipeline:\n  name: cached\n", "utf-8")
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    resolved = voice_bank_manager._resolve_pyannote_model(
        "pyannote/embedding",
        ProjectConfig(),
        label="speaker embedding",
        token=None,
    )

    assert resolved == str(cached.resolve())

    downloaded = tmp_path / "downloaded"
    calls: list[tuple[str, str | None]] = []

    def fake_download(model_id: str, token: str | None) -> Path:
        calls.append((model_id, token))
        downloaded.mkdir()
        return downloaded

    monkeypatch.setattr(voice_bank_manager, "_download_hf_snapshot", fake_download)
    resolved_missing = voice_bank_manager._resolve_pyannote_model(
        "pyannote/missing-model",
        ProjectConfig(diarization_auto_download=True),
        label="diarization",
        token="token",
    )

    assert resolved_missing == str(downloaded)
    assert calls == [("pyannote/missing-model", "token")]


def test_pyannote_pipeline_dependencies_are_resolved_before_offline_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hf_home = tmp_path / "hf"
    pipeline_snapshot = (
        hf_home
        / "hub"
        / "models--pyannote--speaker-diarization-3.1"
        / "snapshots"
        / "pipe123"
    )
    segmentation_snapshot = (
        hf_home
        / "hub"
        / "models--pyannote--segmentation-3.0"
        / "snapshots"
        / "seg123"
    )
    embedding_snapshot = (
        hf_home
        / "hub"
        / "models--pyannote--wespeaker-voxceleb-resnet34-LM"
        / "snapshots"
        / "emb123"
    )
    for snapshot in (pipeline_snapshot, segmentation_snapshot, embedding_snapshot):
        snapshot.mkdir(parents=True)
        (snapshot / "config.yaml").write_text("model: cached\n", "utf-8")
    (pipeline_snapshot / "config.yaml").write_text(
        """
pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    segmentation: pyannote/segmentation-3.0
    embedding: pyannote/wespeaker-voxceleb-resnet34-LM
    clustering: AgglomerativeClustering
""".lstrip(),
        "utf-8",
    )
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    dependencies = voice_bank_manager._ensure_pyannote_pipeline_dependencies(
        str(pipeline_snapshot),
        ProjectConfig(),
        token=None,
    )

    assert dependencies == [str(segmentation_snapshot.resolve()), str(embedding_snapshot.resolve())]
    assert voice_bank_manager._all_local_paths([str(pipeline_snapshot), *dependencies])
    voice_bank_manager._force_hf_offline_for_local_cache()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_voice_bank_build_cli_mock(cli_runner, tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    result = cli_runner.invoke(
        app,
        [
            "voice-bank-build",
            str(tiny_wav_path),
            "--project",
            str(tmp_project_dir),
            "--confirm-rights",
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_project_dir / "voice_bank" / "voice_bank_manifest.json").exists()
    cfg = load_project_config(tmp_project_dir)
    assert cfg.speaker_assignment_backend == "pyannote"


def test_voice_bank_build_audio_discovers_audio_dir_and_applies_personal_defaults(
    cli_runner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    monkeypatch.setattr(cli_module, "REPO_ROOT", repo)
    audio_dir = tmp_path / "audio"
    write_tiny_wav(audio_dir / "b.wav")
    (audio_dir / "ignore.txt").write_text("nope", "utf-8")
    project = tmp_path / "voice_bank_project"

    result = cli_runner.invoke(
        app,
        [
            "voice-bank-build-audio",
            "--audio-dir",
            str(audio_dir),
            "--project",
            str(project),
            "--confirm-rights",
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "discovered 1 file" in result.output
    assert (project / "voice_bank" / "voice_bank_manifest.json").exists()
    cfg = load_project_config(project)
    assert cfg.speaker_assignment_backend == "pyannote"
    assert cfg.diarization_auto_download is True
    assert cfg.diarization_embedding_match_threshold == 0.78
