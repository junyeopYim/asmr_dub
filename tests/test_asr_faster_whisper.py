from __future__ import annotations

from pathlib import Path

from asmr_dub_pipeline.asr.base import create_asr_backend
from asmr_dub_pipeline.asr.faster_whisper import (
    FasterWhisperASRBackend,
    _ctranslate2_snapshot_for_model,
)


def test_ctranslate2_snapshot_for_model_accepts_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo"
    snapshot = cache_root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (cache_root / "refs").mkdir()
    (cache_root / "refs" / "main").write_text("abc123", "utf-8")
    (snapshot / "model.bin").write_bytes(b"fake model")

    assert _ctranslate2_snapshot_for_model(str(cache_root)) == snapshot.resolve()


def test_ctranslate2_snapshot_for_model_accepts_direct_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.bin").write_bytes(b"fake model")

    assert _ctranslate2_snapshot_for_model(str(snapshot)) == snapshot.resolve()


def test_create_faster_whisper_backend_passes_asr_options() -> None:
    backend = create_asr_backend(
        "faster_whisper",
        {
            "model_id": "Systran/faster-whisper-large-v3",
            "language": "ja",
            "local_files_only": True,
            "device": "cuda",
            "compute_type": "float16",
            "batched_inference": True,
            "batch_size": 16,
            "beam_size": 8,
            "best_of": 8,
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": 250},
            "word_timestamps": True,
            "hallucination_silence_threshold": 0.8,
            "initial_prompt": "絶頂 媚薬",
            "hotwords": "絶頂 媚薬 耳舐め",
        },
    )

    assert isinstance(backend, FasterWhisperASRBackend)
    assert backend.device == "cuda"
    assert backend.compute_type == "float16"
    assert backend.batched_inference is True
    assert backend.batch_size == 16
    assert backend.beam_size == 8
    assert backend.best_of == 8
    assert backend.condition_on_previous_text is False
    assert backend.vad_filter is True
    assert backend.vad_parameters == {"min_silence_duration_ms": 250}
    assert backend.word_timestamps is True
    assert backend.hallucination_silence_threshold == 0.8
    assert backend.initial_prompt == "絶頂 媚薬"
    assert backend.hotwords == "絶頂 媚薬 耳舐め"


def test_faster_whisper_backend_uses_batched_pipeline_options() -> None:
    class FakeInfo:
        language = "ja"

    class FakeSegment:
        start = 0.0
        end = 1.0
        text = "テスト"
        avg_logprob = -0.25

    class FakeBatchedModel:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        def transcribe(self, _audio_path: str, **kwargs: object):
            self.kwargs = kwargs
            return [FakeSegment()], FakeInfo()

    fake_batched = FakeBatchedModel()
    backend = FasterWhisperASRBackend(
        model_id="Systran/faster-whisper-large-v3",
        batched_inference=True,
        batch_size=16,
    )
    backend._model = object()
    backend._batched_model = fake_batched

    chunks = backend.transcribe_with_options(Path("audio.wav"), [])

    assert len(chunks) == 1
    assert fake_batched.kwargs is not None
    assert fake_batched.kwargs["batch_size"] == 16
    assert fake_batched.kwargs["beam_size"] == 5
    assert fake_batched.kwargs["without_timestamps"] is False


def test_asr_whisper_preset_updates_effective_backend_options() -> None:
    from asmr_dub_pipeline.pipeline import steps as pipeline_steps
    from asmr_dub_pipeline.schemas import ProjectConfig

    cfg = pipeline_steps._effective_asr_config(ProjectConfig(asr_preset="whisper"))
    backend_config = pipeline_steps._asr_backend_config(cfg)

    assert backend_config["vad_filter"] is True
    assert backend_config["vad_parameters"]["threshold"] <= 0.3
    assert backend_config["vad_parameters"]["speech_pad_ms"] >= 500
    assert backend_config["vad_parameters"]["min_silence_duration_ms"] >= 700
    assert backend_config["vad_parameters"]["max_speech_duration_s"] <= 24
    assert backend_config["word_timestamps"] is True
    assert backend_config["hallucination_silence_threshold"] is not None
