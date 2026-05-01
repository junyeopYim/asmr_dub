from __future__ import annotations

from pathlib import Path

from asmr_dub_pipeline.asr.faster_whisper import _ctranslate2_snapshot_for_model


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
