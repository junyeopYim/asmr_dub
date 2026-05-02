from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from asmr_dub_pipeline.asr import qwen_asr as qwen_module
from asmr_dub_pipeline.asr.base import ASRChunk, create_asr_backend
from asmr_dub_pipeline.asr.qwen_asr import (
    QwenASRBackend,
    _coalesce_qwen_timestamp_chunks,
    _hf_snapshot_for_model,
    _qwen_language,
)
from asmr_dub_pipeline.schemas import Segment


def test_qwen_language_maps_iso_codes_to_qwen_names() -> None:
    assert _qwen_language("ja") == "Japanese"
    assert _qwen_language("ko") == "Korean"
    assert _qwen_language("auto") is None
    assert _qwen_language("Japanese") == "Japanese"


def test_hf_snapshot_for_qwen_model_accepts_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "models--Qwen--Qwen3-ASR-1.7B"
    snapshot = cache_root / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (cache_root / "refs").mkdir()
    (cache_root / "refs" / "main").write_text("abc123", "utf-8")
    (snapshot / "config.json").write_text("{}", "utf-8")
    (snapshot / "model.safetensors").write_bytes(b"fake model")

    assert _hf_snapshot_for_model(str(cache_root)) == snapshot.resolve()


def test_create_asr_backend_can_build_qwen_backend() -> None:
    backend = create_asr_backend(
        "qwen_asr",
        {
            "model_id": "Systran/faster-whisper-large-v3",
            "qwen_model_id": "Qwen/Qwen3-ASR-1.7B",
            "qwen_forced_aligner_model_id": None,
        },
    )

    assert isinstance(backend, QwenASRBackend)
    assert backend.model_id == "Qwen/Qwen3-ASR-1.7B"


def test_qwen_asr_backend_converts_timestamps_to_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeQwenModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeQwenModel:
            calls.append(("from_pretrained", model_id))
            calls.append(("kwargs", kwargs))
            return cls()

        def transcribe(self, **kwargs: object) -> list[object]:
            calls.append(("transcribe", kwargs))
            return [
                SimpleNamespace(
                    language="Japanese",
                    text="いらっしゃいませ お兄さん",
                    time_stamps=[
                        SimpleNamespace(text="いらっしゃいませ", start_time=0.0, end_time=0.8),
                        SimpleNamespace(text="お兄さん", start_time=0.8, end_time=1.4),
                    ],
                ),
                SimpleNamespace(
                    language="Japanese",
                    text="またね",
                    time_stamps=[],
                )
            ]

    monkeypatch.setitem(
        sys.modules,
        "qwen_asr",
        SimpleNamespace(Qwen3ASRModel=FakeQwenModel),
    )
    monkeypatch.setattr(qwen_module, "_audio_duration_sec", lambda _audio_path: 1.8)
    backend = QwenASRBackend(
        model_id="local-or-remote/qwen",
        local_files_only=False,
        forced_aligner_model_id=None,
        return_timestamps=True,
        language="ja",
        context="固有名詞: 鳥桜",
        max_inference_batch_size=4,
        max_new_tokens=512,
    )
    segments = [
        Segment(
            id="seg_0001",
            start=0.0,
            end=1.5,
            duration=1.5,
            audio_for_gemma="gemma.wav",
            audio_for_mix="mix.wav",
        )
    ]

    chunks = backend.transcribe(Path("dummy.wav"), segments)

    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (0.0, 1.4, "いらっしゃいませお兄さん"),
        (0.0, 1.8, "またね"),
    ]
    assert chunks[0].language == "Japanese"
    transcribe_call = next(value for name, value in calls if name == "transcribe")
    assert isinstance(transcribe_call, dict)
    assert transcribe_call["language"] == "Japanese"
    assert transcribe_call["context"] == "固有名詞: 鳥桜"
    assert transcribe_call["return_time_stamps"] is True


def test_qwen_asr_coalesces_japanese_word_timestamps() -> None:
    chunks = _coalesce_qwen_timestamp_chunks(
        [
            ASRChunk(start=1.52, end=1.84, text="深", language="Japanese"),
            ASRChunk(start=2.0, end=2.4, text="呼吸", language="Japanese"),
            ASRChunk(start=2.4, end=2.48, text="や", language="Japanese"),
            ASRChunk(start=2.72, end=3.52, text="リラックス", language="Japanese"),
            ASRChunk(start=3.52, end=3.6, text="で", language="Japanese"),
            ASRChunk(start=3.84, end=4.48, text="も", language="Japanese"),
        ],
        language="Japanese",
    )

    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (1.52, 4.48, "深呼吸やリラックスでも")
    ]


def test_qwen_asr_coalescing_splits_on_large_gaps() -> None:
    chunks = _coalesce_qwen_timestamp_chunks(
        [
            ASRChunk(start=0.0, end=0.5, text="hello", language="English"),
            ASRChunk(start=0.6, end=1.0, text="there", language="English"),
            ASRChunk(start=4.0, end=4.5, text="again", language="English"),
        ],
        language="English",
    )

    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (0.0, 1.0, "hello there"),
        (4.0, 4.5, "again"),
    ]
