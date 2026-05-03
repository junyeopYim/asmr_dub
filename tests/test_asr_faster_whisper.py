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
    assert backend.beam_size == 8
    assert backend.best_of == 8
    assert backend.condition_on_previous_text is False
    assert backend.vad_filter is True
    assert backend.vad_parameters == {"min_silence_duration_ms": 250}
    assert backend.word_timestamps is True
    assert backend.hallucination_silence_threshold == 0.8
    assert backend.initial_prompt == "絶頂 媚薬"
    assert backend.hotwords == "絶頂 媚薬 耳舐め"
