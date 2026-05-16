from __future__ import annotations

import json
import os
import sys
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
    SourceScript,
    VoiceBankManifest,
    VoiceBankSourceSegment,
    VoiceBankSpeaker,
)
from asmr_dub_pipeline.voice_bank import (
    DiarizationTurn,
    VoiceBankError,
    assign_source_speakers_to_manifest,
    assign_speakers_to_manifest,
    build_voice_bank,
    cluster_turns,
    load_voice_bank,
    save_voice_bank,
    validate_voice_bank_models,
)
from asmr_dub_pipeline.voice_bank import manager as voice_bank_manager

pytestmark = pytest.mark.regression


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


def test_voice_bank_speaker_refs_write_kana_normalized_prompt_text(tmp_project_dir: Path) -> None:
    speaker_dir = tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001"
    audio_path = tmp_project_dir / "clips" / "turn.wav"
    write_tiny_wav(audio_path, duration=1.0)
    turn = DiarizationTurn(
        source_id="src_0001",
        source_path=audio_path,
        local_speaker_label="SPEAKER_00",
        start=0.0,
        end=1.0,
        text="耳元で囁きますね。",
        language="ja",
        audio_path=audio_path,
    )

    refs_path = voice_bank_manager._write_speaker_refs(
        tmp_project_dir,
        speaker_dir,
        [turn],
        ProjectConfig(),
    )

    refs = json.loads(refs_path.read_text("utf-8"))
    assert refs["whisper_close"]["prompt_text"] == "みみもとでささやきますね。"
    assert refs["whisper_close"]["prompt_text_original"] == "耳元で囁きますね。"


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


def test_pyannote_diarize_can_skip_turn_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSegment:
        start = 0.300
        end = 1.000

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
        lambda audio_path, start, end: pytest.fail("source-speakers should skip turn embeddings"),
    )

    turns = backend.diarize(
        tmp_path / "source.wav",
        "source",
        ProjectConfig(),
        include_embeddings=False,
    )

    assert len(turns) == 1
    assert turns[0].embedding is None
    assert turns[0].local_speaker_label == "SPEAKER_00"


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


def test_pyannote_from_pretrained_falls_back_for_legacy_token_keyword() -> None:
    class LegacyLoader:
        calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> str:
            _ = model_id
            cls.calls.append(kwargs)
            if "token" in kwargs:
                raise TypeError("from_pretrained() got an unexpected keyword argument 'token'")
            return "loaded"

    result = voice_bank_manager._pyannote_from_pretrained(
        LegacyLoader,
        "pyannote/legacy",
        token="hf-token",
    )

    assert result == "loaded"
    assert LegacyLoader.calls == [
        {"token": "hf-token", "cache_dir": str(voice_bank_manager._hf_hub_cache())},
        {"use_auth_token": "hf-token"},
    ]


def test_pyannote_from_pretrained_does_not_hide_internal_typeerror() -> None:
    class BrokenLoader:
        calls = 0

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> str:
            _ = model_id, kwargs
            cls.calls += 1
            raise TypeError("weights_only must be a supertype of typing.Optional[bool]")

    with pytest.raises(TypeError, match="weights_only"):
        voice_bank_manager._pyannote_from_pretrained(BrokenLoader, "pyannote/broken", token=None)

    assert BrokenLoader.calls == 1


def test_pyannote_optional_nemo_block_restores_existing_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_nemo = object()
    fake_collection = object()
    monkeypatch.setitem(sys.modules, "nemo", fake_nemo)
    monkeypatch.setitem(sys.modules, "nemo.collections", fake_collection)

    with voice_bank_manager._block_broken_pyannote_optional_nemo():
        assert sys.modules["nemo"] is None
        assert "nemo.collections" not in sys.modules

    assert sys.modules["nemo"] is fake_nemo
    assert sys.modules["nemo.collections"] is fake_collection


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


def test_cluster_turns_defaults_to_075_cosine_threshold(tmp_project_dir: Path) -> None:
    turns = [
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="SPEAKER_00",
            start=0.0,
            end=1.0,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="SPEAKER_01",
            start=1.0,
            end=2.0,
            embedding=np.array([0.76, 0.649923], dtype=np.float32),
        ),
    ]

    cluster_turns(turns)

    assert [turn.speaker_id for turn in turns] == ["speaker_0001", "speaker_0001"]


def test_source_turn_clustering_uses_part_label_centroids_for_noisy_embeddings(
    tmp_project_dir: Path,
) -> None:
    turns = [
        DiarizationTurn(
            source_id="source_part_0001",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="SPEAKER_00",
            start=0.0,
            end=1.0,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        DiarizationTurn(
            source_id="source_part_0001",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="SPEAKER_00",
            start=1.0,
            end=2.0,
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        ),
        DiarizationTurn(
            source_id="source_part_0002",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="SPEAKER_09",
            start=2.0,
            end=3.0,
            embedding=np.array([0.72, 0.69], dtype=np.float32),
        ),
    ]

    voice_bank_manager._cluster_source_turns(turns, threshold=0.95)

    assert [turn.speaker_id for turn in turns] == ["speaker_0001", "speaker_0001", "speaker_0001"]


def test_source_speaker_assignment_marks_clean_and_overlapped_segments(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    write_tiny_wav(audio_path, duration=2.0)
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.78),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        artifacts={"source_vocals_48k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=0.7,
                duration=0.7,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
                analysis={"voice_training": {"exclude": True, "reason": "minor_bucket_auto_merged"}},
            ),
            Segment(
                id="seg_0002",
                start=1.3,
                end=2.0,
                duration=0.7,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
            ),
            Segment(
                id="seg_0003",
                start=0.8,
                end=1.2,
                duration=0.4,
                audio_for_gemma="work/segments/audio/seg_0003_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0003_mix.wav",
            ),
        ],
    )

    class TwoSpeakerBackend:
        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = cfg
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=0.0,
                    end=1.0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="B",
                    start=1.0,
                    end=2.0,
                    embedding=np.array([0.0, 1.0], dtype=np.float32),
                ),
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: TwoSpeakerBackend())

    assigned = assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    assert assigned.segments[0].speaker_id == "speaker_0001"
    assert "voice_training" not in assigned.segments[0].analysis
    assert assigned.segments[1].speaker_id == "speaker_0002"
    overlapped = assigned.segments[2]
    assert overlapped.speaker_id is None
    assert overlapped.analysis["speaker_count"] == 2
    assert overlapped.analysis["voice_training"] == {
        "exclude": True,
        "reason": "multi_speaker_overlap",
    }


def test_source_speakers_marks_short_no_overlap_interjection_as_texture(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    write_tiny_wav(audio_path, duration=1.0)
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
        artifacts={"source_vocals_48k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0252",
                start=0.1,
                end=0.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0252_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0252_mix.wav",
                source_script=SourceScript(
                    text="あれ",
                    language="ja",
                    backend="faster_whisper",
                    confidence=0.94,
                    start=0.1,
                    end=0.9,
                ),
            )
        ],
    )

    class NoSpeechBackend:
        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = audio_path, source_id, cfg
            return []

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: NoSpeechBackend())

    assigned = assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    segment = assigned.segments[0]
    assert segment.status == "non_speech_texture"
    assert segment.speaker_id is None
    assert segment.keep_original_texture is True
    assert segment.errors == ["source_speaker_no_overlap_short_texture"]
    assert segment.analysis["asr_quality_gate"] == {
        "decision": "texture",
        "reasons": ["source_speaker_no_overlap_short_texture"],
        "tts_blocked": True,
    }
    assert segment.analysis["voice_training"] == {
        "exclude": True,
        "reason": "source_speaker_no_overlap_short_texture",
    }


def test_source_speakers_routes_borderline_single_overlap_and_excludes_training(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    write_tiny_wav(audio_path, duration=1.0)
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
        artifacts={"source_vocals_48k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
                source_script=SourceScript(
                    text="こんにちは",
                    language="ja",
                    backend="faster_whisper",
                    start=0.0,
                    end=1.0,
                ),
            )
        ],
    )

    class BorderlineBackend:
        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = cfg
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=0.0,
                    end=0.49,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                )
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: BorderlineBackend())

    assigned = assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    segment = assigned.segments[0]
    assert segment.speaker_id == "speaker_0001"
    assert segment.analysis["source_speaker_assignment"]["dominant_overlap_ratio"] == 0.49
    assert segment.analysis["source_speaker_assignment"]["routing_speaker_id"] == "speaker_0001"
    assert segment.analysis["source_speaker_routing"]["reason"] == "borderline_single_speaker_overlap"
    assert segment.analysis["voice_training"] == {
        "exclude": True,
        "reason": "borderline_single_speaker_overlap",
    }


def test_source_speakers_routes_low_single_overlap_for_tts_only(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    write_tiny_wav(audio_path, duration=2.0)
    manifest = PipelineManifest(
        project_config=ProjectConfig(),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        artifacts={"source_vocals_48k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=2.0,
                duration=2.0,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
                source_script=SourceScript(
                    text="五四三二一",
                    language="ja",
                    backend="faster_whisper",
                    start=0.0,
                    end=2.0,
                ),
            )
        ],
    )

    class LowOverlapBackend:
        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = cfg
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=0.0,
                    end=0.5,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                )
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: LowOverlapBackend())

    assigned = assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    segment = assigned.segments[0]
    assert segment.speaker_id == "speaker_0001"
    assert segment.analysis["source_speaker_routing"]["reason"] == "single_speaker_overlap_tts_routing"
    assert segment.analysis["voice_training"] == {
        "exclude": True,
        "reason": "single_speaker_overlap_tts_routing",
    }


def test_source_speakers_excludes_low_dominant_single_speaker_from_training_qc(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    write_tiny_wav(audio_path, duration=6.0)
    low_clip = tmp_project_dir / "work" / "segments" / "audio" / "seg_0378_mix.wav"
    clean_clip = tmp_project_dir / "work" / "segments" / "audio" / "seg_0400_mix.wav"
    write_tiny_wav(low_clip, duration=3.9)
    write_tiny_wav(clean_clip, duration=2.0)
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            gsv_few_shot_min_clip_sec=1.0,
            gsv_few_shot_max_clip_sec=10.0,
        ),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=6.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        artifacts={"source_vocals_48k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0378",
                start=0.0,
                end=3.9,
                duration=3.9,
                audio_for_gemma="work/segments/audio/seg_0378_gemma.wav",
                audio_for_mix=str(low_clip.relative_to(tmp_project_dir)),
                source_script=SourceScript(
                    text="同時に話しているように聞こえる台詞",
                    language="ja",
                    backend="faster_whisper",
                    start=0.0,
                    end=3.9,
                ),
            ),
            Segment(
                id="seg_0400",
                start=4.0,
                end=6.0,
                duration=2.0,
                audio_for_gemma="work/segments/audio/seg_0400_gemma.wav",
                audio_for_mix=str(clean_clip.relative_to(tmp_project_dir)),
                source_script=SourceScript(
                    text="これはきれいな単独話者です",
                    language="ja",
                    backend="faster_whisper",
                    start=4.0,
                    end=6.0,
                ),
            ),
        ],
    )

    class LowDominantBackend:
        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = cfg
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=0.0,
                    end=2.762,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=4.0,
                    end=6.0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: LowDominantBackend())

    assigned = assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    low, clean = assigned.segments
    assert low.speaker_id == "speaker_0001"
    assert low.analysis["source_speaker_assignment"]["dominant_overlap_ratio"] == pytest.approx(0.708205)
    assert low.analysis["voice_training"] == {
        "exclude": True,
        "reason": "low_dominant_source_speaker_overlap",
    }
    assert clean.speaker_id == "speaker_0001"
    assert "voice_training" not in clean.analysis
    assert "source_speaker_training_qc" in assigned.artifacts

    qc_path = Path(assigned.artifacts["source_speaker_training_qc"])
    qc = json.loads(qc_path.read_text("utf-8"))
    speakers = {row["speaker_id"]: row for row in qc["speakers"]}
    speaker = speakers["speaker_0001"]
    assert speaker["eligible_training_segment_ids"] == ["seg_0400"]
    assert speaker["excluded_segment_ids"] == ["seg_0378"]
    assert speaker["exclude_reason_counts"] == {"low_dominant_source_speaker_overlap": 1}
    assert [row["segment_id"] for row in speaker["representative_wav_candidates"]] == ["seg_0400"]
    assert assigned.stage_state["source-speakers"]["training_eligible_counts"] == {"speaker_0001": 1}
    assert assigned.stage_state["source-speakers"]["training_excluded_counts"] == {"speaker_0001": 1}
    assert assigned.stage_state["source-speakers"]["possible_overlap_counts"] == {"speaker_0001": 1}


def test_source_speaker_resolver_marks_unroutable_multi_overlap_keep_original() -> None:
    manifest = PipelineManifest(
        segments=[
            Segment(
                id="seg_overlap",
                start=0.0,
                end=2.0,
                duration=2.0,
                audio_for_gemma="work/segments/audio/seg_overlap_gemma.wav",
                audio_for_mix="work/segments/audio/seg_overlap_mix.wav",
                analysis={
                    "source_speaker_assignment": {
                        "speaker_id": None,
                        "speaker_count": 2,
                        "overlaps": {
                            "speaker_0001": 0.7,
                            "speaker_0002": 0.6,
                        },
                        "dominant_overlap_ratio": 0.35,
                    },
                    "voice_training": {"exclude": True, "reason": "multi_speaker_overlap"},
                },
                source_script=SourceScript(
                    text="二人が重なって話す",
                    language="ja",
                    backend="faster_whisper",
                    start=0.0,
                    end=2.0,
                ),
            )
        ],
    )
    bucket_normalization = {"merges": [], "summary": {}}

    summary = voice_bank_manager._resolve_source_speaker_null_segments(manifest, bucket_normalization)

    segment = manifest.segments[0]
    assert summary["keep_original_segments"] == 1
    assert summary["unresolved_segments"] == 0
    assert segment.status == "absorbed"
    assert segment.speaker_id is None
    assert segment.keep_original_texture is True
    assert segment.analysis["source_speaker_routing"] == {
        "decision": "keep_original_texture",
        "reason": "multi_speaker_overlap",
        "tts_blocked": True,
    }
    assert segment.analysis["voice_training"] == {
        "exclude": True,
        "reason": "multi_speaker_overlap",
    }


def test_source_speaker_resolver_collapses_merged_overlap_candidates_for_routing() -> None:
    manifest = PipelineManifest(
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=2.0,
                duration=2.0,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
                analysis={
                    "source_speaker_assignment": {
                        "speaker_id": None,
                        "speaker_count": 2,
                        "overlaps": {
                            "speaker_0001": 1.1,
                            "speaker_0004": 0.7,
                        },
                        "dominant_overlap_ratio": 0.55,
                    }
                },
                source_script=SourceScript(
                    text="四三二一",
                    language="ja",
                    backend="faster_whisper",
                    start=0.0,
                    end=2.0,
                ),
            )
        ],
    )
    bucket_normalization = {
        "merges": [
            {
                "original_speaker_id": "speaker_0004",
                "merged_into_speaker_id": "speaker_0001",
            }
        ],
        "summary": {},
    }

    summary = voice_bank_manager._resolve_source_speaker_null_segments(manifest, bucket_normalization)

    segment = manifest.segments[0]
    assert summary["routed_segments"] == 1
    assert segment.speaker_id == "speaker_0001"
    assert segment.analysis["source_speaker_routing"]["reason"] == "merged_overlap_candidates_tts_routing"
    assert segment.analysis["voice_training"] == {
        "exclude": True,
        "reason": "merged_overlap_candidates_tts_routing",
    }


def test_source_speakers_neighbor_smooths_no_overlap_between_same_speaker(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    write_tiny_wav(audio_path, duration=3.0)

    def segment(segment_id: str, start: float, end: float, text: str) -> Segment:
        return Segment(
            id=segment_id,
            start=start,
            end=end,
            duration=end - start,
            audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
            audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
            source_script=SourceScript(
                text=text,
                language="ja",
                backend="faster_whisper",
                start=start,
                end=end,
            ),
        )

    manifest = PipelineManifest(
        project_config=ProjectConfig(),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=3.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        artifacts={"source_vocals_48k": str(audio_path)},
        segments=[
            segment("seg_0001", 0.0, 1.0, "こんにちは"),
            segment("seg_0002", 1.05, 1.25, "大丈夫"),
            segment("seg_0003", 1.3, 2.3, "おやすみ"),
        ],
    )

    class NeighborBackend:
        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = cfg
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=0.0,
                    end=1.0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=1.3,
                    end=2.3,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: NeighborBackend())

    assigned = assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    middle = assigned.segments[1]
    assert [segment.speaker_id for segment in assigned.segments] == [
        "speaker_0001",
        "speaker_0001",
        "speaker_0001",
    ]
    assert middle.analysis["source_speaker_routing"]["reason"] == "neighbor_same_speaker_context"
    assert middle.analysis["voice_training"] == {
        "exclude": True,
        "reason": "neighbor_same_speaker_context",
    }


def test_source_speakers_neighbor_smooths_long_no_overlap_countdown_between_same_speaker() -> None:
    manifest = PipelineManifest(
        segments=[
            Segment(
                id="seg_0001",
                speaker_id="speaker_0001",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            ),
            Segment(
                id="seg_0002",
                start=4.0,
                end=12.0,
                duration=8.0,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
                analysis={
                    "source_speaker_assignment": {
                        "speaker_id": None,
                        "speaker_count": 0,
                        "overlaps": {},
                        "dominant_overlap_ratio": 0.0,
                    }
                },
                source_script=SourceScript(
                    text="3 5 4 3 2 1",
                    language="ja",
                    backend="faster_whisper",
                    start=4.0,
                    end=12.0,
                ),
            ),
            Segment(
                id="seg_0003",
                speaker_id="speaker_0001",
                start=16.0,
                end=17.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_0003_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0003_mix.wav",
            ),
        ],
    )

    summary = voice_bank_manager._resolve_source_speaker_null_segments(
        manifest,
        {"merges": [], "summary": {}},
    )

    middle = manifest.segments[1]
    assert summary["unresolved_segments"] == 0
    assert middle.speaker_id == "speaker_0001"
    assert middle.analysis["source_speaker_routing"]["reason"] == "neighbor_same_speaker_context"
    assert middle.analysis["voice_training"] == {
        "exclude": True,
        "reason": "neighbor_same_speaker_context",
    }


def test_source_speakers_auto_merges_minor_bucket_to_trainable_major(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    write_tiny_wav(audio_path, duration=3.4)

    def segment(segment_id: str, start: float, end: float) -> Segment:
        clip = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        write_tiny_wav(clip, duration=end - start)
        return Segment(
            id=segment_id,
            start=start,
            end=end,
            duration=end - start,
            audio_for_gemma=str(clip.relative_to(tmp_project_dir)),
            audio_for_mix=str(clip.relative_to(tmp_project_dir)),
            source_script=SourceScript(
                text=f"台詞 {segment_id}",
                language="ja",
                backend="mock",
                start=start,
                end=end,
            ),
        )

    manifest = PipelineManifest(
        project_config=ProjectConfig(
            gsv_few_shot_min_total_sec=2.0,
            gsv_few_shot_min_clip_sec=0.5,
            gsv_few_shot_max_clip_sec=10.0,
        ),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=3.4,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        artifacts={"source_vocals_48k": str(audio_path)},
        segments=[
            segment("seg_0001", 0.0, 1.2),
            segment("seg_0002", 1.2, 2.4),
            segment("seg_0003", 2.4, 3.4),
        ],
    )

    class MinorBucketBackend:
        def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
            _ = cfg
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=0.0,
                    end=2.4,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                ),
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="B",
                    start=2.4,
                    end=3.4,
                    embedding=np.array([0.0, 1.0], dtype=np.float32),
                ),
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: MinorBucketBackend())

    assigned = assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    assert [segment.speaker_id for segment in assigned.segments] == [
        "speaker_0001",
        "speaker_0001",
        "speaker_0001",
    ]
    normalization = assigned.segments[2].analysis["source_speaker_bucket_normalization"]
    assert normalization["original_speaker_id"] == "speaker_0002"
    assert normalization["merged_into_speaker_id"] == "speaker_0001"
    assert assigned.segments[2].analysis["voice_training"] == {
        "exclude": True,
        "reason": "minor_bucket_auto_merged",
    }
    assert assigned.stage_state["source-speakers"]["bucket_normalization"]["merged_bucket_count"] == 1
    assert "source_speaker_bucket_qc" in assigned.artifacts


def test_source_speaker_bucket_merge_can_use_effect_augmented_prototype(
    tmp_project_dir: Path,
) -> None:
    def segment(segment_id: str, speaker_id: str, duration: float) -> Segment:
        clip = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        write_tiny_wav(clip, duration=duration)
        return Segment(
            id=segment_id,
            speaker_id=speaker_id,
            start=0.0,
            end=duration,
            duration=duration,
            audio_for_gemma=str(clip.relative_to(tmp_project_dir)),
            audio_for_mix=str(clip.relative_to(tmp_project_dir)),
            source_script=SourceScript(
                text=f"台詞 {segment_id}",
                language="ja",
                backend="mock",
                start=0.0,
                end=duration,
            ),
        )

    manifest = PipelineManifest(
        project_config=ProjectConfig(
            gsv_few_shot_min_total_sec=2.0,
            gsv_few_shot_min_clip_sec=0.5,
            gsv_few_shot_max_clip_sec=10.0,
        ),
        segments=[
            segment("seg_major_1", "speaker_0001", 1.1),
            segment("seg_major_2", "speaker_0001", 1.1),
            segment("seg_minor", "speaker_0002", 1.0),
        ],
    )
    turns = [
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="A",
            start=0.0,
            end=2.2,
            speaker_id="speaker_0001",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="B",
            start=2.2,
            end=3.2,
            speaker_id="speaker_0002",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]

    class EffectEmbeddingBackend:
        def embed_clip(self, audio_path: Path) -> np.ndarray:
            if "sfx_center" in audio_path.name:
                return np.array([0.0, 1.0], dtype=np.float32)
            return np.array([1.0, 0.0], dtype=np.float32)

    payload = voice_bank_manager._normalize_source_speaker_buckets(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        turns,
        embedding_backend_factory=lambda: EffectEmbeddingBackend(),
    )

    merge = payload["merges"][0]
    assert merge["original_speaker_id"] == "speaker_0002"
    assert merge["merged_into_speaker_id"] == "speaker_0001"
    assert merge["match_basis"] == "effect_augmented"
    assert merge["effect_profile"] == "sfx_center"
    assert merge["effect_augmented_similarity"] == pytest.approx(1.0)
    assert merge["clean_centroid_similarity"] == pytest.approx(0.0)
    assert manifest.segments[2].speaker_id == "speaker_0001"


def test_source_speaker_bucket_merge_records_rejected_effect_augmented_candidate(
    tmp_project_dir: Path,
) -> None:
    def segment(segment_id: str, speaker_id: str, duration: float) -> Segment:
        clip = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        write_tiny_wav(clip, duration=duration)
        return Segment(
            id=segment_id,
            speaker_id=speaker_id,
            start=0.0,
            end=duration,
            duration=duration,
            audio_for_gemma=str(clip.relative_to(tmp_project_dir)),
            audio_for_mix=str(clip.relative_to(tmp_project_dir)),
            source_script=SourceScript(
                text=f"台詞 {segment_id}",
                language="ja",
                backend="mock",
                start=0.0,
                end=duration,
            ),
        )

    manifest = PipelineManifest(
        project_config=ProjectConfig(
            gsv_few_shot_min_total_sec=2.0,
            gsv_few_shot_min_clip_sec=0.5,
            gsv_few_shot_max_clip_sec=10.0,
        ),
        segments=[
            segment("seg_major_1", "speaker_0001", 1.1),
            segment("seg_major_2", "speaker_0001", 1.1),
            segment("seg_minor", "speaker_0002", 1.0),
        ],
    )
    turns = [
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="A",
            start=0.0,
            end=2.2,
            speaker_id="speaker_0001",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="B",
            start=2.2,
            end=3.2,
            speaker_id="speaker_0002",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]

    class WeakEffectEmbeddingBackend:
        def embed_clip(self, audio_path: Path) -> np.ndarray:
            if "sfx_center" in audio_path.name:
                return np.array([0.8, 0.5], dtype=np.float32)
            return np.array([1.0, 0.0], dtype=np.float32)

    payload = voice_bank_manager._normalize_source_speaker_buckets(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        turns,
        embedding_backend_factory=lambda: WeakEffectEmbeddingBackend(),
    )

    merge = payload["merges"][0]
    assert merge["match_basis"] == "major_bucket_fallback"
    assert merge["effect_augmented_similarity"] == pytest.approx(0.529999, abs=1e-6)
    assert merge["effect_profile"] == "sfx_center"


def test_source_speaker_bucket_preserves_substantial_low_confidence_minor_bucket(
    tmp_project_dir: Path,
) -> None:
    def segment(segment_id: str, speaker_id: str, duration: float) -> Segment:
        clip = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        write_tiny_wav(clip, duration=duration)
        return Segment(
            id=segment_id,
            speaker_id=speaker_id,
            start=0.0,
            end=duration,
            duration=duration,
            audio_for_gemma=str(clip.relative_to(tmp_project_dir)),
            audio_for_mix=str(clip.relative_to(tmp_project_dir)),
            source_script=SourceScript(
                text=f"台詞 {segment_id}",
                language="ja",
                backend="mock",
                start=0.0,
                end=duration,
            ),
        )

    manifest = PipelineManifest(
        project_config=ProjectConfig(
            gsv_few_shot_min_total_sec=10.0,
            gsv_few_shot_min_clip_sec=0.5,
            gsv_few_shot_max_clip_sec=20.0,
        ),
        segments=[
            segment("seg_major_1", "speaker_0001", 5.1),
            segment("seg_major_2", "speaker_0001", 5.1),
            segment("seg_minor", "speaker_0002", 3.5),
        ],
    )
    turns = [
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="A",
            start=0.0,
            end=10.2,
            speaker_id="speaker_0001",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        DiarizationTurn(
            source_id="source",
            source_path=tmp_project_dir / "source.wav",
            local_speaker_label="B",
            start=10.2,
            end=13.7,
            speaker_id="speaker_0002",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]

    payload = voice_bank_manager._normalize_source_speaker_buckets(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        turns,
    )

    assert payload["merges"] == []
    preserve = payload["preserves"][0]
    assert preserve["speaker_id"] == "speaker_0002"
    assert preserve["model_fallback_speaker_id"] == "speaker_0001"
    assert preserve["reason"] == "distinct_minor_bucket_insufficient_training_data"
    assert manifest.segments[2].speaker_id == "speaker_0002"
    assert manifest.segments[2].analysis["source_speaker_model_fallback"]["speaker_id"] == "speaker_0001"
    assert manifest.segments[2].analysis["voice_training"] == {
        "exclude": True,
        "reason": "insufficient_distinct_speaker_training_data",
    }


def test_source_speakers_prefers_mono_analysis_audio(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
) -> None:
    stereo_path = tmp_project_dir / "work" / "audio" / "source_vocals_48k.wav"
    mono_path = tmp_project_dir / "work" / "audio" / "source_vocals_mono_16k.wav"
    write_tiny_wav(stereo_path, sample_rate=48_000)
    write_tiny_wav(mono_path, sample_rate=16_000)
    manifest = PipelineManifest(
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=1.2,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        artifacts={
            "source_vocals_48k": str(stereo_path),
            "source_vocals_mono_16k": str(mono_path),
        },
    )

    assert voice_bank_manager._source_speaker_audio_path(tmp_project_dir, manifest) == mono_path.resolve()


def test_source_speakers_reuses_cached_diarization_turns(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_mono_16k.wav"
    write_tiny_wav(audio_path, sample_rate=16_000, duration=1.2)
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.78),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=1.2,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
        ),
        artifacts={"source_vocals_mono_16k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            )
        ],
    )
    calls: list[tuple[Path, str, bool | None]] = []

    class CachedBackend:
        def diarize(
            self,
            audio_path: Path,
            source_id: str,
            cfg: ProjectConfig,
            *,
            include_embeddings: bool | None = None,
        ) -> list[DiarizationTurn]:
            _ = cfg
            calls.append((audio_path, source_id, include_embeddings))
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="A",
                    start=0.0,
                    end=1.0,
                )
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: CachedBackend())

    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")
    manifest.segments[0].speaker_id = None
    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote")

    assert calls == [(audio_path.resolve(), "source", False)]
    assert manifest.segments[0].speaker_id == "speaker_0001"
    assert (tmp_project_dir / "work" / "diarization" / "source_turns.json").exists()


def test_source_speakers_parallelizes_folder_tracks_and_reuses_part_caches(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_mono_16k.wav"
    write_tiny_wav(audio_path, sample_rate=16_000, duration=2.0)
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.78),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
            raw={
                "folder_input": {
                    "input_kind": "folder",
                    "asr_parts": [
                        {"part_index": 1, "path": "part1.wav", "start_sec": 0.0, "end_sec": 1.0},
                        {"part_index": 2, "path": "part2.wav", "start_sec": 1.0, "end_sec": 2.0},
                    ],
                }
            },
        ),
        artifacts={"source_vocals_mono_16k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.1,
                end=0.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            ),
            Segment(
                id="seg_0002",
                start=1.1,
                end=1.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
            ),
        ],
    )
    calls: list[tuple[str, str, bool | None]] = []

    class TrackBackend:
        def diarize(
            self,
            audio_path: Path,
            source_id: str,
            cfg: ProjectConfig,
            *,
            include_embeddings: bool | None = None,
        ) -> list[DiarizationTurn]:
            _ = cfg
            calls.append((audio_path.name, source_id, include_embeddings))
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="SPEAKER_00",
                    start=0.0,
                    end=1.0,
                )
            ]

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", lambda kind, cfg: TrackBackend())

    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote", jobs=2)
    manifest.segments[0].speaker_id = None
    manifest.segments[1].speaker_id = None
    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote", jobs=2)

    assert sorted(calls) == [
        ("part_0001.wav", "source_part_0001", True),
        ("part_0002.wav", "source_part_0002", True),
    ]
    assert [segment.speaker_id for segment in manifest.segments] == ["speaker_0001", "speaker_0001"]
    assert manifest.stage_state["source-speakers"]["parallel_tracks"] == 2
    assert manifest.stage_state["source-speakers"]["diarization_cache"] == "hit"
    assert (
        tmp_project_dir / "work" / "diarization" / "source_parts" / "part_0001_turns.json"
    ).exists()
    assert (
        tmp_project_dir / "work" / "diarization" / "source_parts" / "part_0002_turns.json"
    ).exists()


def test_source_speakers_parallel_skips_silent_folder_tracks(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_mono_16k.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    silence = np.zeros(sample_rate, dtype=np.float32)
    sf.write(str(audio_path), np.concatenate([tone, silence])[:, None], sample_rate)
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.78),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
            raw={
                "folder_input": {
                    "input_kind": "folder",
                    "asr_parts": [
                        {"part_index": 1, "path": "spoken.wav", "start_sec": 0.0, "end_sec": 1.0},
                        {"part_index": 2, "path": "silent.wav", "start_sec": 1.0, "end_sec": 2.0},
                    ],
                }
            },
        ),
        artifacts={"source_vocals_mono_16k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.1,
                end=0.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            ),
            Segment(
                id="seg_0002",
                start=1.1,
                end=1.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
            ),
        ],
    )
    calls: list[str] = []

    class TrackBackend:
        def diarize(
            self,
            audio_path: Path,
            source_id: str,
            cfg: ProjectConfig,
            *,
            include_embeddings: bool | None = None,
        ) -> list[DiarizationTurn]:
            _ = cfg, include_embeddings
            calls.append(audio_path.name)
            if source_id.endswith("0002"):
                raise AssertionError("silent source part should not be diarized")
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="SPEAKER_00",
                    start=0.0,
                    end=1.0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                )
            ]

    def create_backend(
        kind: str,
        cfg: ProjectConfig,
        *,
        load_embedding_model: bool = True,
    ) -> TrackBackend:
        _ = kind, cfg, load_embedding_model
        return TrackBackend()

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", create_backend)

    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote", jobs=2)

    assert calls == ["part_0001.wav"]
    assert manifest.segments[0].speaker_id == "speaker_0001"
    assert manifest.segments[1].speaker_id is None
    assert manifest.segments[1].analysis["voice_training"] == {
        "exclude": True,
        "reason": "no_source_speaker_match",
    }
    assert manifest.stage_state["source-speakers"]["skipped_silent_parts"] == 1


def test_source_speakers_parallel_skips_effect_only_folder_tracks(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_mono_16k.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(audio_path), tone[:, None], sample_rate)
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.78),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
            raw={
                "folder_input": {
                    "input_kind": "folder",
                    "asr_parts": [
                        {"part_index": 1, "path": "spoken.wav", "start_sec": 0.0, "end_sec": 1.0},
                        {
                            "part_index": 2,
                            "path": "93_おまけ（耳責めの効果音のみ抜粋）/効果音トラック02e.mp3",
                            "start_sec": 1.0,
                            "end_sec": 2.0,
                            "asr_silenced": False,
                        },
                    ],
                }
            },
        ),
        artifacts={"source_vocals_mono_16k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.1,
                end=0.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            ),
            Segment(
                id="seg_0002",
                start=1.1,
                end=1.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
            ),
        ],
    )
    calls: list[str] = []

    class TrackBackend:
        def diarize(
            self,
            audio_path: Path,
            source_id: str,
            cfg: ProjectConfig,
            *,
            include_embeddings: bool | None = None,
        ) -> list[DiarizationTurn]:
            _ = cfg, include_embeddings
            calls.append(source_id)
            if source_id.endswith("0002"):
                raise AssertionError("effect-only source part should not be diarized")
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="SPEAKER_00",
                    start=0.0,
                    end=1.0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                )
            ]

    monkeypatch.setattr(
        voice_bank_manager,
        "create_diarization_backend",
        lambda *_args, **_kwargs: TrackBackend(),
    )

    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote", jobs=2)

    assert calls == ["source_part_0001"]
    assert manifest.segments[0].speaker_id == "speaker_0001"
    assert manifest.segments[1].speaker_id is None
    assert manifest.stage_state["source-speakers"]["skipped_effect_only_parts"] == 1


def test_source_speakers_parallel_skips_pyannote_parts_with_no_turns(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_mono_16k.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16_000
    t = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    sf.write(str(audio_path), tone[:, None], sample_rate)
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.78),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
            raw={
                "folder_input": {
                    "input_kind": "folder",
                    "asr_parts": [
                        {"part_index": 1, "path": "spoken.wav", "start_sec": 0.0, "end_sec": 1.0},
                        {"part_index": 2, "path": "texture.wav", "start_sec": 1.0, "end_sec": 2.0},
                    ],
                }
            },
        ),
        artifacts={"source_vocals_mono_16k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.1,
                end=0.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            ),
            Segment(
                id="seg_0002",
                start=1.1,
                end=1.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
            ),
        ],
    )

    class TrackBackend:
        def diarize(
            self,
            audio_path: Path,
            source_id: str,
            cfg: ProjectConfig,
            *,
            include_embeddings: bool | None = None,
        ) -> list[DiarizationTurn]:
            _ = cfg, include_embeddings
            if source_id.endswith("0002"):
                raise VoiceBankError(f"pyannote produced no diarization turns for {audio_path}")
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="SPEAKER_00",
                    start=0.0,
                    end=1.0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32),
                )
            ]

    monkeypatch.setattr(
        voice_bank_manager,
        "create_diarization_backend",
        lambda *_args, **_kwargs: TrackBackend(),
    )

    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote", jobs=2)

    assert manifest.segments[0].speaker_id == "speaker_0001"
    assert manifest.segments[1].speaker_id is None
    assert manifest.stage_state["source-speakers"]["skipped_no_diarization_parts"] == 1


def test_source_speakers_parallel_merges_part_labels_with_cached_embeddings(
    tmp_project_dir: Path,
    tiny_wav_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "audio" / "source_vocals_mono_16k.wav"
    write_tiny_wav(audio_path, sample_rate=16_000, duration=2.0)
    manifest = PipelineManifest(
        project_config=ProjectConfig(diarization_embedding_match_threshold=0.75),
        source_info=SourceInfo(
            path=str(tiny_wav_path.resolve()),
            duration_sec=2.0,
            sample_rate=48_000,
            channels=2,
            has_video=False,
            format_name="wav",
            raw={
                "folder_input": {
                    "input_kind": "folder",
                    "asr_parts": [
                        {"part_index": 1, "path": "part1.wav", "start_sec": 0.0, "end_sec": 1.0},
                        {"part_index": 2, "path": "part2.wav", "start_sec": 1.0, "end_sec": 2.0},
                    ],
                }
            },
        ),
        artifacts={"source_vocals_mono_16k": str(audio_path)},
        segments=[
            Segment(
                id="seg_0001",
                start=0.1,
                end=0.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0001_mix.wav",
            ),
            Segment(
                id="seg_0002",
                start=1.1,
                end=1.9,
                duration=0.8,
                audio_for_gemma="work/segments/audio/seg_0002_gemma.wav",
                audio_for_mix="work/segments/audio/seg_0002_mix.wav",
            ),
        ],
    )
    calls: list[tuple[str, str, bool | None, bool]] = []

    class TrackBackend:
        def __init__(self, load_embedding_model: bool) -> None:
            self.load_embedding_model = load_embedding_model

        def diarize(
            self,
            audio_path: Path,
            source_id: str,
            cfg: ProjectConfig,
            *,
            include_embeddings: bool | None = None,
        ) -> list[DiarizationTurn]:
            _ = cfg
            calls.append((audio_path.name, source_id, include_embeddings, self.load_embedding_model))
            is_first_part = source_id.endswith("0001")
            return [
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label="SPEAKER_00" if is_first_part else "SPEAKER_09",
                    start=0.0,
                    end=1.0,
                    embedding=np.array([1.0, 0.0], dtype=np.float32)
                    if is_first_part and include_embeddings
                    else np.array([0.99, 0.01], dtype=np.float32)
                    if include_embeddings
                    else None,
                )
            ]

    def create_backend(
        kind: str,
        cfg: ProjectConfig,
        *,
        load_embedding_model: bool = True,
    ) -> TrackBackend:
        _ = kind, cfg
        return TrackBackend(load_embedding_model)

    monkeypatch.setattr(voice_bank_manager, "create_diarization_backend", create_backend)

    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote", jobs=2)

    assert [segment.speaker_id for segment in manifest.segments] == ["speaker_0001", "speaker_0001"]
    assert sorted(calls) == [
        ("part_0001.wav", "source_part_0001", True, True),
        ("part_0002.wav", "source_part_0002", True, True),
    ]
    cached = json.loads(
        (tmp_project_dir / "work" / "diarization" / "source_parts" / "part_0002_turns.json").read_text("utf-8")
    )
    assert cached["turns"][0]["embedding"] == pytest.approx([0.99, 0.01])

    calls.clear()
    manifest.segments[0].speaker_id = None
    manifest.segments[1].speaker_id = None
    assign_source_speakers_to_manifest(tmp_project_dir, manifest, backend_kind="pyannote", jobs=2)

    assert calls == []
    assert [segment.speaker_id for segment in manifest.segments] == ["speaker_0001", "speaker_0001"]


def test_source_speakers_cli_invokes_stage(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_source_speakers_step(
        project_dir: Path,
        backend_kind: str | None = None,
        confirm_rights: bool = False,
        jobs: int = 1,
    ):
        seen["project_dir"] = project_dir
        seen["backend_kind"] = backend_kind
        seen["confirm_rights"] = confirm_rights
        seen["jobs"] = jobs
        return PipelineManifest(stage_state={"source-speakers": {"assigned_segments": 1}})

    monkeypatch.setattr(cli_module, "source_speakers_step", fake_source_speakers_step)

    result = cli_runner.invoke(
        app,
        [
            "source-speakers",
            "--project",
            str(tmp_project_dir),
            "--backend",
            "mock",
            "--jobs",
            "2",
            "--confirm-rights",
        ],
    )

    assert result.exit_code == 0
    assert seen == {
        "project_dir": tmp_project_dir.resolve(),
        "backend_kind": "mock",
        "confirm_rights": True,
        "jobs": 2,
    }
    assert "Source speaker assignment complete" in result.output


def test_source_speakers_cli_defaults_to_four_jobs(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    def fake_source_speakers_step(
        project_dir: Path,
        backend_kind: str | None = None,
        confirm_rights: bool = False,
        jobs: int = 1,
    ):
        seen["project_dir"] = project_dir
        seen["backend_kind"] = backend_kind
        seen["confirm_rights"] = confirm_rights
        seen["jobs"] = jobs
        return PipelineManifest(stage_state={"source-speakers": {"assigned_segments": 1}})

    monkeypatch.setattr(cli_module, "source_speakers_step", fake_source_speakers_step)

    result = cli_runner.invoke(
        app,
        [
            "source-speakers",
            "--project",
            str(tmp_project_dir),
            "--backend",
            "mock",
            "--confirm-rights",
        ],
    )

    assert result.exit_code == 0
    assert seen == {
        "project_dir": tmp_project_dir.resolve(),
        "backend_kind": "mock",
        "confirm_rights": True,
        "jobs": 4,
    }
    assert "Source speaker assignment complete" in result.output


def test_project_config_uses_default_diarization_threshold() -> None:
    assert ProjectConfig().diarization_embedding_match_threshold == 0.75


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
    assert cfg.diarization_embedding_match_threshold == 0.75
