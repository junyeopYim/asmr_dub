from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from asmr_dub_pipeline import cli as cli_module
from asmr_dub_pipeline import orchestrator
from asmr_dub_pipeline.asr.base import ASRChunk, map_chunks_to_segments
from asmr_dub_pipeline.audio.features import duration_sec, write_audio
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gemma.text_translate import (
    LlamaServerTranslationClient,
    parse_asr_review_response,
    parse_translation_response,
)
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import (
    extract_step,
    korean_script_step,
    prepare_source_voice_refs_step,
    segment_step,
    transcribe_step,
    translate_ko_step,
)
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    QCMetadata,
    Segment,
    SourceScript,
    TTSCandidate,
    TTSMetadata,
)
from asmr_dub_pipeline.script.korean_colloquial import (
    COLLOQUIAL_REWRITE_NOTE,
    colloquialize_korean_text,
)


def sample_segment(
    segment_id: str = "seg_0001",
    *,
    start: float = 0.0,
    end: float = 1.0,
) -> Segment:
    return Segment(
        id=segment_id,
        start=start,
        end=end,
        duration=end - start,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
    )


def test_asr_chunks_map_to_segments_by_overlap() -> None:
    segments = [
        sample_segment("seg_0001", start=0.0, end=1.0),
        sample_segment("seg_0002", start=1.0, end=2.0),
        sample_segment("seg_0003", start=2.0, end=3.0),
    ]
    chunks = [
        ASRChunk(start=0.0, end=0.5, text="おはよう", language="ja", confidence=0.8),
        ASRChunk(start=0.5, end=1.0, text="ございます", language="ja", confidence=0.4),
        ASRChunk(start=1.2, end=1.8, text="耳元です", language="ja", confidence=0.9),
    ]

    mapped = map_chunks_to_segments(segments, chunks, backend="mock")

    assert mapped["seg_0001"] is not None
    assert mapped["seg_0001"].text == "おはよう ございます"
    assert mapped["seg_0001"].confidence == pytest.approx(0.6)
    assert mapped["seg_0001"].language == "ja"
    assert mapped["seg_0002"] is not None
    assert mapped["seg_0002"].text == "耳元です"
    assert mapped["seg_0003"] is None


def test_asr_chunks_do_not_duplicate_long_chunk_across_micro_segments() -> None:
    segments = [
        sample_segment("seg_0001", start=0.0, end=1.0),
        sample_segment("seg_0002", start=1.0, end=2.0),
        sample_segment("seg_0003", start=2.0, end=3.0),
    ]
    chunks = [
        ASRChunk(
            start=0.2,
            end=2.8,
            text="初めて催眠音声に出演させていただきました",
            language="ja",
            confidence=0.7,
        )
    ]

    mapped = map_chunks_to_segments(segments, chunks, backend="mock")

    assert mapped["seg_0001"] is None
    assert mapped["seg_0002"] is not None
    assert mapped["seg_0002"].text == "初めて催眠音声に出演させていただきました"
    assert mapped["seg_0003"] is None


def test_asr_resegment_groups_short_adjacent_chunks_for_tts(tmp_project_dir: Path) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(start=0.0, end=0.3, text="お疲れ様でした", language="ja", confidence=0.8),
            ASRChunk(start=0.5, end=1.1, text="初めまして", language="ja", confidence=0.6),
            ASRChunk(start=2.0, end=3.1, text="ゆっくり聞いてください", language="ja", confidence=0.9),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=4.0,
        min_segment_sec=0.8,
        merge_gap_sec=0.45,
    )

    assert [segment.duration for segment in segments] == [1.1, 1.1]
    assert segments[0].source_script is not None
    assert segments[0].source_script.text == "お疲れ様でした 初めまして"
    assert segments[1].source_script is not None
    assert segments[1].source_script.text == "ゆっくり聞いてください"


def test_asr_resegment_drops_sparse_hallucinated_long_chunks(tmp_project_dir: Path) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(start=0.0, end=4.0, text="おやすみなさい", language="ja", confidence=0.9),
            ASRChunk(start=10.0, end=80.0, text="気持ちいいですね", language="ja", confidence=0.95),
            ASRChunk(start=81.0, end=86.0, text="ゆっくり聞いてください", language="ja", confidence=0.9),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=90.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
        sparse_chunk_max_sec=30.0,
        sparse_chunk_min_chars_per_sec=0.5,
    )

    assert [segment.source_script.text for segment in segments if segment.source_script] == [
        "おやすみなさい",
        "ゆっくり聞いてください",
    ]


def test_transcribe_asr_review_mock_selects_domain_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    manifest.artifacts["source_vocals_mono_16k"] = manifest.artifacts["gemma_mono_16k"]
    manifest.artifacts["source_vocals_48k"] = manifest.artifacts["original_stereo_48k"]
    save_manifest(tmp_project_dir, manifest)
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
            asr_review_enabled=True,
            asr_review_backend="mock",
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text="もっと大きな手帳が来る",
                    language="ja",
                    confidence=0.99,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")

    assert manifest.segments[0].source_script is not None
    assert manifest.segments[0].source_script.text == "もっと大きな絶頂が来る"
    summary = json.loads(Path(manifest.artifacts["asr_review_summary"]).read_text("utf-8"))
    assert summary["attempted"] == 1
    assert summary["replaced"] == 1
    assert summary["items"][0]["candidates"] == [
        {"candidate_id": "original", "text": "もっと大きな手帳が来る"},
        {"candidate_id": "domain_replacement", "text": "もっと大きな絶頂が来る"},
    ]


def test_transcribe_asr_audio_review_passes_clip_to_client(
    monkeypatch: pytest.MonkeyPatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    manifest.artifacts["source_vocals_mono_16k"] = manifest.artifacts["gemma_mono_16k"]
    manifest.artifacts["source_vocals_48k"] = manifest.artifacts["original_stereo_48k"]
    save_manifest(tmp_project_dir, manifest)
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
            asr_review_enabled=True,
            asr_review_backend="llama_server_audio",
            asr_review_audio_padding_sec=0.05,
            gemma_text_server_auto_start=False,
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text="もっと大きな手帳が来る",
                    language="ja",
                    confidence=0.99,
                )
            ]

    review_calls: list[tuple[str, Path, list[dict[str, object]]]] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def review_asr_candidates_with_audio(
            self,
            items: list[dict[str, object]],
            batch_id: str,
            audio_path: Path,
        ) -> dict[str, dict[str, object]]:
            assert audio_path.exists()
            review_calls.append((batch_id, audio_path, items))
            return {
                "chunk_0001": {
                    "chunk_id": "chunk_0001",
                    "decision": "replace",
                    "selected_candidate_id": "domain_replacement",
                    "confidence": 0.99,
                    "reason": "audio and context support 絶頂.",
                    "risk_terms": ["手帳"],
                }
            }

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper", confirm_rights=True)

    assert review_calls
    assert manifest.segments[0].source_script is not None
    assert manifest.segments[0].source_script.text == "もっと大きな絶頂が来る"
    summary = json.loads(Path(manifest.artifacts["asr_review_summary"]).read_text("utf-8"))
    assert summary["backend"] == "llama_server_audio"
    assert summary["audio_input"]["enabled"] is True
    assert summary["audio_input"]["created"] == 1
    assert summary["items"][0]["audio_clip"]["path"] == str(review_calls[0][1])


def test_asr_review_generates_retranscribe_candidates(
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_review_candidate_padding_sec=[0.1],
        asr_review_initial_prompt="絶頂 媚薬 耳舐め",
    )
    item = {
        "chunk_id": "chunk_0001",
        "start": 0.0,
        "end": 0.8,
        "candidates": [{"candidate_id": "original", "text": "もっと大きな手帳が来る"}],
    }
    calls: list[dict[str, object]] = []

    class FakeASRBackend:
        def transcribe_with_options(
            self,
            _audio_path: Path,
            _segments: list[Segment],
            **overrides: object,
        ) -> list[ASRChunk]:
            calls.append(overrides)
            text = "もっと大きな絶頂が来る" if "initial_prompt" in overrides else "もっと大きな手帳が来る"
            return [ASRChunk(start=0.0, end=0.8, text=text, language="ja", confidence=0.95)]

    generated = pipeline_steps._add_generated_asr_review_candidates(
        [item],
        backend=FakeASRBackend(),
        project_dir=tmp_project_dir,
        review_audio_path=tiny_wav_path,
        audio_duration_sec=duration_sec(tiny_wav_path),
        cfg=cfg,
    )

    assert generated == 1
    assert len(calls) == 2
    assert any(candidate["text"] == "もっと大きな絶頂が来る" for candidate in item["candidates"])


def test_asr_review_default_candidates_are_unprompted() -> None:
    cfg = ProjectConfig(project_name="test-project")

    option_rows = pipeline_steps._generated_asr_review_candidate_options(cfg)

    assert option_rows
    assert all("prompted" not in option_id for option_id, _padding, _overrides in option_rows)
    assert all(overrides["initial_prompt"] is None for _option_id, _padding, overrides in option_rows)
    assert all(overrides["hotwords"] is None for _option_id, _padding, overrides in option_rows)


def test_asr_prompt_leak_filter_rejects_video_outro_text() -> None:
    cfg = ProjectConfig(project_name="test-project")

    leaked = pipeline_steps._asr_candidate_looks_prompt_leaked(
        "次の動画でお会いしましょう。",
        cfg,
    )

    assert leaked is True


def test_asr_review_replacements_include_observed_adult_asr_artifacts() -> None:
    cfg = ProjectConfig(project_name="test-project")

    assert cfg.asr_review_candidate_replacements["めず行きセックス"] == "メスイキセックス"
    assert cfg.asr_review_candidate_replacements["薄引き"] == "メスイキ"
    assert cfg.asr_review_candidate_replacements["グリドリス"] == "クリトリス"
    assert cfg.asr_review_candidate_replacements["お孫"] == "おまんこ"
    assert "めず行き" in cfg.asr_review_suspicious_text_patterns


def test_translation_asr_backcheck_flags_suspicious_korean_smell() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text="会館に飲み込まれて",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=1.0,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="회관에 삼켜져",
        ko_natural="회관에 삼켜져요.",
        model="mock",
        batch_id="batch_0001",
    )

    items = pipeline_steps._apply_translation_asr_backcheck(
        [segment],
        ProjectConfig(project_name="test"),
    )

    assert items
    assert items[0]["source_hits"] == ["会館"]
    assert items[0]["translation_hits"] == ["회관"]
    assert segment.status == "raw"
    assert any("ASR translation backcheck" in error for error in segment.errors)


def test_asr_resegment_splits_dense_long_chunks(tmp_project_dir: Path) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(
                start=0.0,
                end=31.0,
                text="催眠じゃないなんて思う人もいるかもしれませんが",
                language="ja",
                confidence=0.9,
            ),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=40.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
        max_segment_sec=20.0,
        sparse_chunk_max_sec=30.0,
        sparse_chunk_min_chars_per_sec=0.5,
    )

    assert len(segments) == 2
    assert all(segment.duration <= 20.0 for segment in segments)
    assert " ".join(segment.source_script.text for segment in segments if segment.source_script) == (
        "催眠じゃないなんて思う人 もいるかもしれませんが"
    )


def test_asr_repair_flags_low_confidence_and_sparse_chunks() -> None:
    assert pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=2.0, text="少し近づきますね", language="ja", confidence=0.91),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )
    assert pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=10.0, end=30.0, text="鉄柱", language="ja", confidence=0.98),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )
    assert pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=3.0, text="もちなとい", language="ja", confidence=0.99),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
        suspicious_text_patterns=["もちなとい"],
    )
    assert not pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=0.8, text="10", language="ja", confidence=0.7),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )
    assert not pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=3.0, text="ゆっくり聞いてください", language="ja", confidence=0.98),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )


def test_asr_repair_replaces_suspicious_chunk_with_no_vad_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_audio = tmp_project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    samples = np.zeros((16_000, 1), dtype=np.float32)
    write_audio(repair_audio, samples, 16_000)
    sliced_paths: list[Path] = []

    def fake_slice_audio(
        input_path: Path,
        start: float,
        end: float,
        output_path: Path,
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        assert input_path == repair_audio
        assert start == pytest.approx(9.0)
        assert end == pytest.approx(16.0)
        assert sample_rate == 16_000
        assert channels == 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"repair clip")
        sliced_paths.append(output_path)
        return output_path

    class FakeBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, dict[str, object]]] = []

        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            self.calls.append((audio_path, kwargs))
            assert segments == []
            return [
                ASRChunk(
                    start=1.0,
                    end=5.0,
                    text="ゾクゾクしますね",
                    language="ja",
                    confidence=0.93,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    backend = FakeBackend()
    repaired, summary = pipeline_steps._repair_asr_chunks(
        [
            ASRChunk(
                start=10.0,
                end=15.0,
                text="付属しますね",
                language="ja",
                confidence=0.83,
            )
        ],
        backend=backend,
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=30.0,
        cfg=ProjectConfig(project_name=tmp_project_dir.name),
    )

    assert summary["attempted"] == 1
    assert summary["repaired"] == 1
    assert summary["items"][0]["accepted"] is True
    assert repaired[0].text == "ゾクゾクしますね"
    assert repaired[0].start == pytest.approx(10.0)
    assert repaired[0].end == pytest.approx(14.0)
    assert sliced_paths
    assert backend.calls[0][1]["vad_filter"] is False
    assert backend.calls[0][1]["vad_parameters"] is None
    assert backend.calls[0][1]["initial_prompt"] is None
    assert backend.calls[0][1]["hotwords"] is None


def test_asr_repair_tries_next_candidate_when_first_candidate_prompt_leaks(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_audio = tmp_project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    samples = np.zeros((24_000, 1), dtype=np.float32)
    write_audio(repair_audio, samples, 16_000)

    def fake_slice_audio(
        input_path: Path,
        start: float,
        end: float,
        output_path: Path,
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, start, end, sample_rate, channels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"repair clip")
        return output_path

    class FakeBackend:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return [
                    ASRChunk(
                        start=0.0,
                        end=1.0,
                        text="ご視聴ありがとうございました",
                        language="ja",
                        confidence=0.99,
                    )
                ]
            return [
                ASRChunk(
                    start=0.2,
                    end=1.3,
                    text="ゾクゾクしますね",
                    language="ja",
                    confidence=0.95,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    backend = FakeBackend()
    repaired, summary = pipeline_steps._repair_asr_chunks(
        [
            ASRChunk(
                start=4.0,
                end=5.4,
                text="付属しますね",
                language="ja",
                confidence=0.82,
            )
        ],
        backend=backend,
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=8.0,
        cfg=ProjectConfig(project_name=tmp_project_dir.name),
    )

    assert len(backend.calls) >= 2
    assert summary["repaired"] == 1
    assert summary["items"][0]["accepted_candidate_id"] != summary["items"][0]["attempts"][0]["candidate_id"]
    assert summary["items"][0]["attempts"][0]["prompt_leaked"] is True
    assert repaired[0].text == "ゾクゾクしますね"


def test_asr_repair_splits_long_suspicious_group_before_retranscribe(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_audio = tmp_project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    samples = np.zeros((80 * 16_000, 1), dtype=np.float32)
    write_audio(repair_audio, samples, 16_000)
    slices: list[tuple[float, float]] = []

    def fake_slice_audio(
        input_path: Path,
        start: float,
        end: float,
        output_path: Path,
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, sample_rate, channels
        slices.append((start, end))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"repair clip")
        return output_path

    class FakeBackend:
        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments, kwargs
            return [
                ASRChunk(
                    start=0.0,
                    end=12.0,
                    text="絶頂が来る " * 8,
                    language="ja",
                    confidence=0.99,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    original = ASRChunk(
        start=0.0,
        end=65.0,
        text="釣りが来る " * 40,
        language="ja",
        confidence=0.7,
    )
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_resegment_max_sec=20.0,
        asr_repair_padding_sec=1.0,
        asr_repair_max_chunks=10,
    )

    _repaired, summary = pipeline_steps._repair_asr_chunks(
        [original],
        backend=FakeBackend(),
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=80.0,
        cfg=cfg,
    )

    assert summary["attempted"] == 4
    assert len(slices) == 4
    assert all(end - start <= cfg.asr_resegment_max_sec + cfg.asr_repair_padding_sec * 2 for start, end in slices)


def test_asr_repair_rejects_prompt_leaked_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_audio = tmp_project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    samples = np.zeros((16_000, 1), dtype=np.float32)
    write_audio(repair_audio, samples, 16_000)

    def fake_slice_audio(
        input_path: Path,
        start: float,
        end: float,
        output_path: Path,
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, start, end, sample_rate, channels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"repair clip")
        return output_path

    class FakeBackend:
        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments, kwargs
            return [
                ASRChunk(
                    start=0.0,
                    end=8.0,
                    text="気持ちいい イっちゃう 飛んじゃってください さくら ジンジン 痺れる",
                    language="ja",
                    confidence=0.99,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    original = ASRChunk(start=10.0, end=18.0, text="釣りが来ちゃう", language="ja", confidence=0.8)
    repaired, summary = pipeline_steps._repair_asr_chunks(
        [original],
        backend=FakeBackend(),
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=30.0,
        cfg=ProjectConfig(project_name=tmp_project_dir.name),
    )

    assert summary["attempted"] == 1
    assert summary["repaired"] == 0
    assert summary["items"][0]["accepted"] is False
    assert summary["items"][0]["prompt_leaked"] is True
    assert repaired == [original]


def test_asr_repair_rejects_generic_hallucination_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repair_audio = tmp_project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    samples = np.zeros((16_000, 1), dtype=np.float32)
    write_audio(repair_audio, samples, 16_000)

    def fake_slice_audio(
        input_path: Path,
        start: float,
        end: float,
        output_path: Path,
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
    ) -> Path:
        _ = input_path, start, end, sample_rate, channels
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"repair clip")
        return output_path

    class FakeBackend:
        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments, kwargs
            return [
                ASRChunk(
                    start=0.0,
                    end=12.0,
                    text="ご視聴ありがとうございました おやすみ なさい おやすみ なさい",
                    language="ja",
                    confidence=0.99,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    original = ASRChunk(start=10.0, end=22.0, text="強くたま", language="ja", confidence=0.8)
    repaired, summary = pipeline_steps._repair_asr_chunks(
        [original],
        backend=FakeBackend(),
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=30.0,
        cfg=ProjectConfig(project_name=tmp_project_dir.name),
    )

    assert summary["attempted"] == 1
    assert summary["repaired"] == 0
    assert summary["items"][0]["accepted"] is False
    assert summary["items"][0]["prompt_leaked"] is True
    assert repaired == [original]


def test_asr_text_replacements_normalize_known_domain_mishears() -> None:
    chunks, replaced = pipeline_steps._apply_asr_text_replacements_to_chunks(
        [
            ASRChunk(
                start=0.0,
                end=3.0,
                text="あっという間に釣りが来ちゃう",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=3.0,
                end=6.0,
                text="媚薬を塗り込んで",
                language="ja",
                confidence=0.95,
            ),
        ],
        ProjectConfig().asr_text_replacements,
    )

    assert replaced == 1
    assert chunks[0].text == "あっという間に絶頂が来ちゃう"
    assert chunks[1].text == "媚薬を塗り込んで"


def test_asr_text_replacements_include_observed_midcheck_mishears() -> None:
    replacements = ProjectConfig().asr_text_replacements
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="女体科の薬で全身生還体になる",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="君の志士は拘束されている",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=8.0,
                end=12.0,
                text="簡易版最終マシーンでドリーム最終を継続",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=12.0,
                end=16.0,
                text="エネルギー速化しています ああ 速化",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=16.0,
                end=20.0,
                text="雨宿りをするために駆け込んだ親城",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=20.0,
                end=24.0,
                text="愛 催眠へと引きずり込む 血のお耳も敏感になる",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=24.0,
                end=28.0,
                text="いいえ 気が揺らぐ ぶり気持ちよくなろうね",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=28.0,
                end=32.0,
                text="全身が薄いて熱くなってくる",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=32.0,
                end=36.0,
                text="発症中 発症した 発症してる 発症する 発症しちゃう",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=36.0,
                end=40.0,
                text="お泣きして待っててね 尿位が強くなる",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=40.0,
                end=44.0,
                text="私は中旬なメスになります 巣に侵されることを考えて",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=44.0,
                end=48.0,
                text="お巣に侵されることを考えて 速速が止まらない",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=48.0,
                end=52.0,
                text="貧乱なメス犬 鼻ならしてかきなさい",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=52.0,
                end=56.0,
                text="端となく鼻ならして",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=56.0,
                end=60.0,
                text="もっと大きな手帳が来る",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=60.0,
                end=64.0,
                text="媚薬スプレー 豆腐",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=64.0,
                end=68.0,
                text="お耳ジュガジュガピスタンされて",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=68.0,
                end=72.0,
                text="魅力まで触手が入ってくる",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=72.0,
                end=76.0,
                text="ウニオクまで触手が入ってくる",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=76.0,
                end=80.0,
                text="この音声は中八菌催眠音声です",
                language="ja",
                confidence=0.92,
            ),
        ],
        replacements,
    )

    assert summary["chunks_changed"] == 20
    assert summary["total_replacements"] == 34
    assert chunks[0].text == "女体化の薬で全身性感帯になる"
    assert chunks[1].text == "君の四肢は拘束されている"
    assert chunks[2].text == "簡易版採集マシーンでドリーム採集を継続"
    assert chunks[3].text == "エネルギー不足しています ああ 不足"
    assert chunks[4].text == "雨宿りをするために駆け込んだ神社"
    assert chunks[5].text == "甘い催眠へと引きずり込む 右のお耳も敏感になる"
    assert chunks[6].text == "意識が揺らぐ たっぷり気持ちよくなろうね"
    assert chunks[7].text == "全身が疼いて熱くなってくる"
    assert chunks[8].text == "発情中 発情した 発情してる 発情する 発情しちゃう"
    assert chunks[9].text == "オナニーして待っててね 尿意が強くなる"
    assert chunks[10].text == "私は従順なメスになります オスに犯されることを考えて"
    assert chunks[11].text == "オスに犯されることを考えて ゾクゾクが止まらない"
    assert chunks[12].text == "淫乱なメス犬 鼻鳴らして嗅ぎなさい"
    assert chunks[13].text == "はしたなく鼻鳴らして"
    assert chunks[14].text == "もっと大きな絶頂が来る"
    assert chunks[15].text == "媚薬スプレー投与"
    assert chunks[16].text == "お耳グチュグチュピストンされて"
    assert chunks[17].text == "耳奥まで触手が入ってくる"
    assert chunks[18].text == "耳奥まで触手が入ってくる"
    assert chunks[19].text == "この音声は18禁催眠音声です"


def test_asr_text_replacements_handle_cascaded_yaml_order() -> None:
    replacements = dict(sorted(ProjectConfig().asr_text_replacements.items()))
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="媚薬スプレー 豆腐",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="美薬スプレー 豆腐",
                language="ja",
                confidence=0.92,
            ),
        ],
        replacements,
    )

    assert summary["chunks_changed"] == 2
    assert summary["total_replacements"] == 3
    assert chunks[0].text == "媚薬スプレー投与"
    assert chunks[1].text == "媚薬スプレー投与"


def test_transcribe_writes_unified_asr_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_repair_enabled=False,
            asr_review_enabled=False,
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text="釣りが来ちゃう",
                    language="ja",
                    confidence=0.74,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper", confirm_rights=True)

    diagnostics_path = Path(manifest.artifacts["asr_diagnostics"])
    summary_path = Path(manifest.artifacts["asr_diagnostics_summary"])
    diagnostics = json.loads(diagnostics_path.read_text("utf-8"))
    summary = json.loads(summary_path.read_text("utf-8"))
    assert diagnostics["raw_asr_chunks"][0]["text"] == "釣りが来ちゃう"
    assert diagnostics["repaired_asr_chunks"][0]["text"] == "釣りが来ちゃう"
    assert diagnostics["final_asr_chunks"][0]["text"] == "絶頂が来ちゃう"
    assert diagnostics["final_asr_chunks"][0]["text_density"] > 0
    assert diagnostics["final_asr_chunks"][0]["replacement_hits"][0]["source"] == "釣りが来ちゃう"
    assert diagnostics["vad"]["vad_filter"] is True
    assert summary["raw_asr_chunk_count"] == 1
    assert summary["final_asr_chunk_count"] == 1
    assert summary["text_replacements"]["total_replacements"] == 1
    assert manifest.artifacts["asr_input_diagnostics"]


def test_rejected_asr_repair_marks_overlapping_text_for_manual_review() -> None:
    source_script = SourceScript(
        text="私はあなたを愛しています すよ お耳 私に食べら",
        language="ja",
        confidence=0.96,
        backend="faster_whisper",
        start=3177.56,
        end=3195.86,
    )
    repair_summary = {
        "items": [
            {
                "start": 3180.41,
                "end": 3195.86,
                "accepted": False,
                "attempts": [
                    {
                        "candidate_id": "no_vad_clean",
                        "reason": "prompt_or_hallucination_leak",
                        "candidate_text": "ご視聴ありがとうございました",
                    }
                ],
            }
        ]
    }

    assert pipeline_steps._source_script_rejected_repair_reasons(
        source_script,
        repair_summary,
    ) == ["asr_repair_rejected:prompt_or_hallucination_leak"]


def test_transcribe_qwen_repair_fallback_skips_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            source_separation_backend="none",
            asr_resegment_from_chunks=False,
            asr_qwen_repair_fallback_enabled=True,
            asr_repair_enabled=True,
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class FakeASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(start=0.0, end=2.0, text="付属しますね", language="ja", confidence=0.7)
            ]

        def transcribe_with_options(
            self,
            _audio_path: Path,
            _segments: list[Segment],
            **_kwargs: object,
        ) -> list[ASRChunk]:
            return [
                ASRChunk(start=0.0, end=2.0, text="付属しますね", language="ja", confidence=0.7)
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())
    monkeypatch.setattr(pipeline_steps, "_qwen_asr_dependency_available", lambda: False)

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper", confirm_rights=True)

    diagnostics = json.loads(Path(manifest.artifacts["asr_diagnostics"]).read_text("utf-8"))
    assert diagnostics["qwen_repair_fallback"]["enabled"] is True
    assert diagnostics["qwen_repair_fallback"]["available"] is False
    assert any("qwen-asr" in warning for warning in manifest.warnings)


def test_source_script_and_korean_translation_schema_round_trip() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text="少し近づきますね",
        language="ja",
        confidence=0.91,
        backend="faster_whisper",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="조금 가까이 갈게요.",
        ko_natural="조금 더 가까이 갈게요.",
        notes=["gentle ASMR tone"],
        confidence=0.87,
        model="gemma4",
        batch_id="batch_0001",
    )

    restored = Segment.model_validate(segment.model_dump(mode="json"))

    assert restored.source_script is not None
    assert restored.source_script.text == "少し近づきますね"
    assert restored.translation_ko is not None
    assert restored.translation_ko.ko_natural == "조금 더 가까이 갈게요."


def test_project_config_defaults_translate_ko_uses_single_server_slots() -> None:
    assert ProjectConfig().gemma_text_batch_size == 12
    assert ProjectConfig().gemma_text_context_radius == 4
    assert ProjectConfig().gemma_text_span_size == 4
    assert ProjectConfig().gemma_text_span_max_sec == pytest.approx(18.0)
    assert ProjectConfig().gemma_text_span_max_gap_sec == pytest.approx(1.2)
    assert ProjectConfig().gemma_text_two_pass is True
    assert ProjectConfig().gemma_text_concurrency == 4
    assert ProjectConfig().gsv_concurrency == 3
    assert ProjectConfig().source_language == "ja"
    assert ProjectConfig().target_language == "ko"
    assert ProjectConfig(target_language="kr").target_language == "ko"
    assert ProjectConfig().asr_resegment_from_chunks is True
    assert ProjectConfig().asr_resegment_min_sec == pytest.approx(3.0)
    assert ProjectConfig().asr_resegment_max_sec == pytest.approx(20.0)
    assert ProjectConfig().asr_resegment_merge_gap_sec == pytest.approx(1.0)
    assert ProjectConfig().asr_word_timestamps is False
    assert ProjectConfig().asr_hallucination_silence_threshold is None
    assert ProjectConfig().asr_sparse_chunk_max_sec == pytest.approx(30.0)
    assert ProjectConfig().asr_sparse_chunk_min_chars_per_sec == pytest.approx(0.5)
    assert ProjectConfig().asr_repair_enabled is True
    assert ProjectConfig().asr_repair_confidence_threshold == pytest.approx(0.94)
    assert ProjectConfig().asr_repair_sparse_min_sec == pytest.approx(12.0)
    assert ProjectConfig().asr_repair_sparse_min_chars_per_sec == pytest.approx(1.0)
    assert ProjectConfig().asr_repair_padding_sec == pytest.approx(1.0)
    assert ProjectConfig().asr_repair_max_chunks == 160
    assert "もちなとい" in ProjectConfig().asr_repair_suspicious_text_patterns
    assert "ご処生" in ProjectConfig().asr_repair_suspicious_text_patterns
    assert ProjectConfig(asr_review_backend="llama_server_audio").asr_review_backend == "llama_server_audio"
    assert ProjectConfig().asr_review_audio_padding_sec == pytest.approx(0.4)
    assert "女体化" in ProjectConfig().asr_hotwords
    assert "性感帯" in ProjectConfig().asr_hotwords
    assert "採集マシーン" in ProjectConfig().asr_hotwords
    assert "発情" in ProjectConfig().asr_hotwords
    assert "尿意" in ProjectConfig().asr_hotwords
    assert "オナニー" in ProjectConfig().asr_hotwords
    assert "18禁" in ProjectConfig().asr_hotwords
    assert "投与" in ProjectConfig().asr_hotwords
    assert "耳奥" in ProjectConfig().asr_hotwords
    assert "ピストン" in ProjectConfig().asr_hotwords
    assert ProjectConfig().asr_text_replacements["釣りが来ちゃう"] == "絶頂が来ちゃう"
    assert ProjectConfig().asr_text_replacements["女体科"] == "女体化"
    assert ProjectConfig().asr_text_replacements["生還体"] == "性感帯"
    assert ProjectConfig().asr_text_replacements["薄いて"] == "疼いて"
    assert ProjectConfig().asr_text_replacements["尿位"] == "尿意"
    assert ProjectConfig().asr_text_replacements["中八菌催眠音声"] == "18禁催眠音声"
    assert ProjectConfig().asr_text_replacements["手帳が来る"] == "絶頂が来る"
    assert ProjectConfig().asr_text_replacements["ピスタン"] == "ピストン"
    assert ProjectConfig().asr_text_replacements["ウニアクナで触手"] == "耳奥まで触手"
    assert ProjectConfig().source_separation_backend == "demucs"
    assert ProjectConfig().source_separation_model == "htdemucs"
    assert ProjectConfig().gsv_trim_edge_silence is True
    assert ProjectConfig().gsv_ref_min_sec == pytest.approx(3.0)
    assert ProjectConfig().gsv_ref_max_sec == pytest.approx(10.0)
    assert ProjectConfig().gsv_tts_min_speed_factor == pytest.approx(0.85)
    assert ProjectConfig().gsv_tts_max_speed_factor == pytest.approx(1.12)
    assert ProjectConfig().gsv_top_k == 15
    assert ProjectConfig().gsv_top_p == pytest.approx(1.0)
    assert ProjectConfig().gsv_temperature == pytest.approx(1.0)
    assert ProjectConfig().gsv_text_split_method == "cut5"
    assert ProjectConfig().gsv_parallel_infer is True
    assert ProjectConfig().gsv_repetition_penalty == pytest.approx(1.35)
    assert ProjectConfig().gsv_sample_steps == 32
    assert ProjectConfig().gsv_super_sampling is False
    assert ProjectConfig().gsv_overlap_length == 2
    assert ProjectConfig().gsv_min_chunk_length == 16
    assert ProjectConfig().gsv_fragment_interval == pytest.approx(0.3)


def test_source_voice_ref_selection_extends_short_candidate_to_duration_window(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_short",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_short_gemma.wav",
                audio_for_mix="work/segments/audio/seg_short_mix.wav",
                source_script=SourceScript(
                    text="短すぎます。",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=1.0,
                ),
            ),
            Segment(
                id="seg_valid",
                start=1.0,
                end=5.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_valid_gemma.wav",
                audio_for_mix="work/segments/audio/seg_valid_mix.wav",
                source_script=SourceScript(
                    text="参照音声に使える長さです。",
                    language="ja",
                    backend="mock",
                    start=1.0,
                    end=5.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(tmp_project_dir, manifest, manifest.project_config)

    assert [[segment.id for segment in span.segments] for span in selected] == [
        ["seg_short", "seg_valid"]
    ]
    assert selected[0].duration == pytest.approx(5.0)


def test_prepare_source_voice_refs_writes_combined_short_reference_span(
    tmp_project_dir: Path,
) -> None:
    save_project_config(ProjectConfig(project_name=tmp_project_dir.name), tmp_project_dir / "pipeline.yaml")
    sample_rate = 48_000
    segments = []
    for segment_id, start, end, text, frequency in [
        ("seg_short", 0.0, 1.0, "短いです。", 440.0),
        ("seg_next", 1.0, 3.2, "続きです。", 660.0),
    ]:
        t = np.arange(int(sample_rate * (end - start)), dtype=np.float32) / sample_rate
        tone = 0.1 * np.sin(2 * np.pi * frequency * t)
        audio = np.stack([tone, tone * 0.8], axis=1)
        mix_path = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        gemma_path = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_gemma.wav"
        write_audio(mix_path, audio, sample_rate)
        write_audio(gemma_path, audio[:, :1], sample_rate)
        segments.append(
            Segment(
                id=segment_id,
                start=start,
                end=end,
                duration=end - start,
                audio_for_gemma=str(gemma_path),
                audio_for_mix=str(mix_path),
                source_script=SourceScript(
                    text=text,
                    language="ja",
                    backend="mock",
                    start=start,
                    end=end,
                ),
            )
        )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            project_config=ProjectConfig(project_name=tmp_project_dir.name),
            segments=segments,
        ),
    )

    manifest = prepare_source_voice_refs_step(tmp_project_dir, confirm_rights=True)

    refs = json.loads((tmp_project_dir / "refs" / "refs.json").read_text("utf-8"))
    ref_path = tmp_project_dir / refs["whisper_close"]["ref_audio_path"]
    assert duration_sec(ref_path) == pytest.approx(3.2)
    assert refs["whisper_close"]["prompt_text"] == "短いです。 続きです。"
    ref_qc = json.loads(Path(manifest.artifacts["source_voice_ref_qc"]).read_text("utf-8"))
    assert ref_qc["refs"][0]["selected_segment_ids"] == ["seg_short", "seg_next"]


def _force_single_translation_lane(project_dir: Path) -> None:
    save_project_config(
        ProjectConfig(
            project_name=project_dir.name,
            gemma_text_batch_size=40,
            gemma_text_concurrency=1,
        ),
        project_dir / "pipeline.yaml",
    )


def test_translate_ko_passes_neighbor_context_to_context_aware_clients(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_batch_size=1,
            gemma_text_span_size=1,
            gemma_text_concurrency=1,
            gemma_text_context_radius=1,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    for index in range(2, 4):
        start = base.end + (index - 2) * base.duration
        manifest.segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text=f"前後の台詞です {index}",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=start,
                    end=start + base.duration,
                ),
            )
        )
    save_manifest(tmp_project_dir, manifest)
    contexts: list[tuple[str, list[str], list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
            context_segments: list[Segment] | None = None,
        ) -> dict[str, KoreanTranslation]:
            contexts.append(
                (
                    batch_id,
                    [segment.id for segment in segments],
                    [segment.id for segment in (context_segments or [])],
                )
            )
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    assert contexts == [
        ("batch_0001", ["seg_0001"], ["seg_0001", "seg_0002"]),
        ("batch_0002", ["seg_0002"], ["seg_0001", "seg_0002", "seg_0003"]),
        ("batch_0003", ["seg_0003"], ["seg_0002", "seg_0003"]),
    ]
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["translate-ko"]["context_radius"] == 1
    assert manifest.stage_state["translate-ko"]["two_pass"] is True


def test_translate_ko_groups_adjacent_segments_into_contextual_spans(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_batch_size=1,
            gemma_text_span_size=3,
            gemma_text_span_max_sec=30.0,
            gemma_text_span_max_gap_sec=1.0,
            gemma_text_concurrency=1,
            gemma_text_context_radius=1,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    for index in range(2, 4):
        start = base.end + (index - 2) * 0.2
        manifest.segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text=f"続きの台詞です {index}",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=start,
                    end=start + base.duration,
                ),
            )
        )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str], list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
            context_segments: list[Segment] | None = None,
        ) -> dict[str, KoreanTranslation]:
            calls.append(
                (
                    batch_id,
                    [segment.id for segment in segments],
                    [segment.id for segment in (context_segments or [])],
                )
            )
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    assert calls == [
        (
            "batch_0001",
            ["seg_0001", "seg_0002", "seg_0003"],
            ["seg_0001", "seg_0002", "seg_0003"],
        )
    ]
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["translate-ko"]["span_size"] == 3
    assert manifest.stage_state["translate-ko"]["span_count"] == 1


def test_translate_ko_batches_numeric_source_segments_with_context(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_span_size=3,
            gemma_text_span_max_sec=30.0,
            gemma_text_span_max_gap_sec=1.0,
            gemma_text_concurrency=1,
            gemma_text_context_radius=1,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    manifest.segments = [
        Segment(
            id="seg_0001",
            start=0.0,
            end=1.0,
            duration=1.0,
            audio_for_gemma=base.audio_for_gemma,
            audio_for_mix=base.audio_for_mix,
            source_script=SourceScript(
                text="2014",
                language="ja",
                confidence=0.99,
                backend="mock",
                start=0.0,
                end=1.0,
            ),
        ),
        Segment(
            id="seg_0002",
            start=1.0,
            end=2.0,
            duration=1.0,
            audio_for_gemma=base.audio_for_gemma,
            audio_for_mix=base.audio_for_mix,
            source_script=SourceScript(
                text="耳元です",
                language="ja",
                confidence=0.99,
                backend="mock",
                start=1.0,
                end=2.0,
            ),
        ),
        Segment(
            id="seg_0003",
            start=2.0,
            end=3.0,
            duration=1.0,
            audio_for_gemma=base.audio_for_gemma,
            audio_for_mix=base.audio_for_mix,
            source_script=SourceScript(
                text="2",
                language="ja",
                confidence=0.99,
                backend="mock",
                start=2.0,
                end=3.0,
            ),
        ),
    ]
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str], list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
            context_segments: list[Segment] | None = None,
        ) -> dict[str, KoreanTranslation]:
            calls.append(
                (
                    batch_id,
                    [segment.id for segment in segments],
                    [segment.id for segment in (context_segments or [])],
                )
            )
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert calls == [
        (
            "batch_0001",
            ["seg_0001", "seg_0002", "seg_0003"],
            ["seg_0001", "seg_0002", "seg_0003"],
        )
    ]
    assert all(segment.translation_ko is not None for segment in manifest.segments)
    assert all(
        "deterministic_numeric_source" not in segment.translation_ko.notes
        for segment in manifest.segments
        if segment.translation_ko is not None
    )
    assert manifest.stage_state["translate-ko"]["translated"] == 3
    assert manifest.stage_state["translate-ko"]["span_count"] == 1


def test_numeric_counting_postprocess_normalizes_counting_runs() -> None:
    def translated_segment(segment_id: str, source: str, ko: str, start: float) -> Segment:
        segment = sample_segment(segment_id, start=start, end=start + 1.0)
        segment.source_script = SourceScript(
            text=source,
            language="ja",
            confidence=0.99,
            backend="mock",
            start=segment.start,
            end=segment.end,
        )
        segment.translation_ko = KoreanTranslation(
            ko_literal=ko,
            ko_natural=ko,
            notes=[],
            confidence=0.9,
            model="fake",
            batch_id="batch_0001",
        )
        return segment

    segments = [
        translated_segment("seg_0001", "4 5", "네, 다섯", 0.0),
        translated_segment("seg_0002", "6 7", "육, 칠", 1.0),
        translated_segment("seg_0003", "8", "여덟", 2.0),
        translated_segment("seg_0004", "808", "팔공팔", 20.0),
        translated_segment("seg_0005", "80", "팔십", 21.0),
        translated_segment("seg_0006", "60", "육십", 22.0),
        translated_segment("seg_0007", "80", "팔십", 23.0),
    ]

    rewritten = pipeline_steps._apply_korean_numeric_counting_postprocess(segments)

    assert rewritten == 2
    assert segments[0].translation_ko is not None
    assert segments[0].translation_ko.ko_natural == "넷, 다섯"
    assert segments[1].translation_ko is not None
    assert segments[1].translation_ko.ko_natural == "여섯, 일곱"
    assert segments[3].translation_ko is not None
    assert segments[3].translation_ko.ko_natural == "팔공팔"
    assert segments[4].translation_ko is not None
    assert segments[4].translation_ko.ko_natural == "팔십"
    assert "numeric_counting_postprocess" in segments[0].translation_ko.notes


def test_numeric_counting_postprocess_handles_interleaved_countdown() -> None:
    def translated_segment(segment_id: str, source: str, ko: str, start: float) -> Segment:
        segment = sample_segment(segment_id, start=start, end=start + 1.0)
        segment.source_script = SourceScript(
            text=source,
            language="ja",
            confidence=0.99,
            backend="mock",
            start=segment.start,
            end=segment.end,
        )
        segment.translation_ko = KoreanTranslation(
            ko_literal=ko,
            ko_natural=ko,
            notes=[],
            confidence=0.9,
            model="fake",
            batch_id="batch_0001",
        )
        return segment

    phrase = sample_segment("seg_0004", start=3.0, end=4.0)
    phrase.source_script = SourceScript(
        text="気持ちいいのが止まらない",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=3.0,
        end=4.0,
    )
    phrase.translation_ko = KoreanTranslation(
        ko_literal="기분 좋은 게 멈추질 않아요",
        ko_natural="기분 좋은 게 멈추질 않아요",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )
    segments = [
        translated_segment("seg_0001", "10", "십", 0.0),
        translated_segment("seg_0002", "9", "아홉", 1.0),
        translated_segment("seg_0003", "8", "여덟", 2.0),
        phrase,
        translated_segment("seg_0005", "7 6", "칠, 육", 4.0),
        translated_segment("seg_0006", "5", "다섯", 5.0),
        translated_segment("seg_0007", "4", "사", 6.0),
        translated_segment("seg_0008", "3", "셋", 7.0),
        translated_segment("seg_0009", "2", "둘", 8.0),
        translated_segment("seg_0010", "1", "일", 9.0),
        translated_segment("seg_0011", "0", "영", 10.0),
    ]

    rewritten = pipeline_steps._apply_korean_numeric_counting_postprocess(segments)

    assert rewritten == 4
    assert segments[0].translation_ko is not None
    assert segments[0].translation_ko.ko_natural == "열"
    assert segments[4].translation_ko is not None
    assert segments[4].translation_ko.ko_natural == "일곱, 여섯"
    assert segments[6].translation_ko is not None
    assert segments[6].translation_ko.ko_natural == "넷"
    assert segments[9].translation_ko is not None
    assert segments[9].translation_ko.ko_natural == "하나"


def test_translate_ko_retranslates_legacy_deterministic_numeric_results(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    segment.source_script = SourceScript(
        text="1",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="일",
        ko_natural="일",
        notes=["deterministic_numeric_source"],
        confidence=1.0,
        model="deterministic:numeric-source",
        batch_id=f"numeric_{segment.id}",
    )
    save_manifest(tmp_project_dir, manifest)
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append(batch_id)
            return {
                segment.id: KoreanTranslation(
                    ko_literal="하나",
                    ko_natural="하나",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    translation = manifest.segments[0].translation_ko
    assert calls == ["batch_0001"]
    assert translation is not None
    assert translation.ko_natural == "하나"
    assert "deterministic_numeric_source" not in translation.notes


def test_translate_ko_force_retranslate_ignores_resumed_translations(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append(batch_id)
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {len(calls)}",
                    ko_natural=f"자연 {len(calls)}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)
    first_manifest = load_manifest(tmp_project_dir)
    first_translation = first_manifest.segments[0].translation_ko
    assert first_translation is not None
    assert first_translation.ko_natural == "자연 1"

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")
    assert calls == ["batch_0001"]

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", force_retranslate=True)

    manifest = load_manifest(tmp_project_dir)
    translation = manifest.segments[0].translation_ko
    assert translation is not None
    assert translation.ko_natural == "자연 2"
    assert manifest.stage_state["translate-ko"]["force_retranslate"] is True


def test_korean_draft_mix_allows_timing_only_qc_regeneration() -> None:
    segment = sample_segment()
    segment.status = "needs_regeneration"
    segment.script = JapaneseScript(ja_text="안녕하세요.", tts_text="안녕하세요.", tts_language="ko")
    segment.tts = TTSMetadata(selected_candidate_path="work/tts/seg_0001_final.wav")
    segment.qc = QCMetadata(
        recommendation="regenerate",
        status="needs_regeneration",
        issues=["duration_ratio_out_of_range"],
    )

    assert pipeline_steps._include_segment_in_mix(segment, allow_korean_timing_draft=True)
    segment.qc.issues.append("clipping_detected")
    assert not pipeline_steps._include_segment_in_mix(segment, allow_korean_timing_draft=True)


def test_mix_blocks_selected_candidate_with_failed_duration_gate() -> None:
    segment = sample_segment()
    segment.status = "ok"
    segment.script = JapaneseScript(ja_text="こんにちは", tts_text="안녕하세요.", tts_language="ko")
    segment.tts = TTSMetadata(
        selected_candidate_path="work/tts/seg_0001_final.wav",
        candidates=[
            TTSCandidate(
                candidate_index=0,
                seed=1,
                output_path="work/tts/seg_0001_final.wav",
                duration_sec=0.2,
                selected=True,
                duration_ratio=0.2,
                duration_gate="too_short",
                acceptable_for_mix=False,
            )
        ],
    )
    segment.qc = QCMetadata(recommendation="pass", status="ok")

    assert not pipeline_steps._include_segment_in_mix(segment, allow_korean_timing_draft=False)


def test_llama_server_translation_client_repairs_invalid_json() -> None:
    requests: list[dict[str, object]] = []
    repaired = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "직역",
            "ko_natural": "자연",
            "notes": [],
            "confidence": 0.9,
            "model": "gemma4",
            "batch_id": "batch_0001",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = "not json" if len(requests) == 1 else json.dumps(repaired, ensure_ascii=False)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="こんにちは",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "자연"
    assert len(requests) == 3
    second_messages = requests[1]["messages"]
    assert isinstance(second_messages, list)
    assert "Repair the previous response" in second_messages[0]["content"]


def test_translation_parser_accepts_label_confidence_and_string_notes() -> None:
    parsed = parse_translation_response(
        json.dumps(
            [
                {
                    "segment_id": "seg_0001",
                    "ko_literal": "직역",
                    "ko_natural": "자연",
                    "notes": "tone preserved",
                    "confidence": "High",
                },
                {
                    "segment_id": "seg_0002",
                    "ko_literal": "직역2",
                    "ko_natural": "자연2",
                    "notes": "none",
                    "confidence": "87%",
                },
            ]
        ),
        batch_id="batch_0001",
        model="gemma4",
    )

    assert parsed["seg_0001"].confidence == pytest.approx(0.9)
    assert parsed["seg_0001"].notes == ["tone preserved"]
    assert parsed["seg_0002"].confidence == pytest.approx(0.87)
    assert parsed["seg_0002"].notes == []


def test_translation_parser_accepts_minimal_model_output() -> None:
    parsed = parse_translation_response(
        json.dumps([{"segment_id": "seg_0001", "ko_natural": "안녕하세요."}]),
        batch_id="batch_0001",
        model="gemma4",
    )

    translation = parsed["seg_0001"]
    assert translation.ko_natural == "안녕하세요."
    assert translation.ko_literal == "안녕하세요."
    assert translation.notes == []
    assert translation.confidence is None
    assert translation.model == "gemma4"
    assert translation.batch_id == "batch_0001"


def test_asr_review_parser_accepts_candidate_selection() -> None:
    parsed = parse_asr_review_response(
        json.dumps(
            [
                {
                    "chunk_id": "chunk_0001",
                    "heard_text": "もっと大きな絶頂が来る",
                    "decision": "replace",
                    "selected_candidate_id": "domain_replacement",
                    "confidence": "92%",
                    "reason": "ASMR context points to 絶頂.",
                    "risk_terms": ["手帳"],
                }
            ],
            ensure_ascii=False,
        ),
        batch_id="asr_review_0001",
        model="gemma4",
    )

    assert parsed["chunk_0001"]["decision"] == "replace"
    assert parsed["chunk_0001"]["selected_candidate_id"] == "domain_replacement"
    assert parsed["chunk_0001"]["confidence"] == pytest.approx(0.92)
    assert parsed["chunk_0001"]["heard_text"] == "もっと大きな絶頂が来る"
    assert parsed["chunk_0001"]["risk_terms"] == ["手帳"]


def test_llama_server_translation_client_reviews_asr_candidates_with_audio(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []
    audio_path = tmp_path / "chunk_0001.wav"
    audio_path.write_bytes(b"RIFFmock-wav-data")
    response = [
        {
            "chunk_id": "chunk_0001",
            "decision": "replace",
            "selected_candidate_id": "domain_replacement",
            "confidence": 0.94,
            "reason": "Attached audio supports 絶頂.",
            "risk_terms": ["手帳"],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]},
        )

    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
    )
    reviews = client.review_asr_candidates_with_audio(
        [
            {
                "chunk_id": "chunk_0001",
                "audio_clip_path": str(audio_path),
                "context_before": [{"text": "もっと大きな"}],
                "context_after": [{"text": "行く"}],
                "candidates": [
                    {"candidate_id": "original", "text": "もっと大きな手帳が来る"},
                    {"candidate_id": "domain_replacement", "text": "もっと大きな絶頂が来る"},
                ],
            }
        ],
        "asr_review_0001",
        audio_path,
    )

    assert reviews["chunk_0001"]["selected_candidate_id"] == "domain_replacement"
    content = requests[0]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "japanese_asr_audio_candidate_review" in content[0]["text"]
    assert "You cannot hear the audio" not in content[0]["text"]
    assert "audio_clip_path" not in content[0]["text"]
    assert content[1]["type"] == "input_audio"
    assert content[1]["input_audio"]["format"] == "wav"
    assert base64.b64decode(content[1]["input_audio"]["data"]) == audio_path.read_bytes()


def test_llama_server_translation_client_audio_review_does_not_fallback_to_text_repair(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []
    audio_path = tmp_path / "chunk_0001.wav"
    audio_path.write_bytes(b"RIFFmock-wav-data")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            500,
            json={
                "error": {
                    "message": "audio input is not supported",
                }
            },
        )

    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    with pytest.raises(Exception, match="audio input is not supported"):
        client.review_asr_candidates_with_audio(
            [
                {
                    "chunk_id": "chunk_0001",
                    "candidates": [
                        {"candidate_id": "original", "text": "手帳が来る"},
                        {"candidate_id": "domain_replacement", "text": "絶頂が来る"},
                    ],
                }
            ],
            "asr_review_0001",
            audio_path,
        )

    assert len(requests) == 2
    for request in requests:
        content = request["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[1]["type"] == "input_audio"


def test_llama_server_translation_client_repairs_audio_review_decision_mismatch(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []
    audio_path = tmp_path / "chunk_0001.wav"
    audio_path.write_bytes(b"RIFFmock-wav-data")
    responses = [
        [
            {
                "chunk_id": "chunk_0001",
                "decision": "manual_review",
                "selected_candidate_id": "domain_replacement",
                "confidence": 0.95,
                "reason": "best candidate but wrong decision label",
                "risk_terms": [],
            }
        ],
        [
            {
                "chunk_id": "chunk_0001",
                "decision": "replace",
                "selected_candidate_id": "domain_replacement",
                "confidence": 0.95,
                "reason": "fixed decision label",
                "risk_terms": [],
            }
        ],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(responses[len(requests) - 1], ensure_ascii=False)
                        }
                    }
                ]
            },
        )

    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    reviews = client.review_asr_candidates_with_audio(
        [
            {
                "chunk_id": "chunk_0001",
                "candidates": [
                    {"candidate_id": "original", "text": "私の声は 弟兄"},
                    {"candidate_id": "domain_replacement", "text": "息を深く吐くたびに体から嫌な力が抜けて"},
                ],
            }
        ],
        "asr_review_0001",
        audio_path,
    )

    assert reviews["chunk_0001"]["decision"] == "replace"
    assert len(requests) == 2
    first_content = requests[0]["messages"][0]["content"]
    second_content = requests[1]["messages"][0]["content"]
    assert isinstance(first_content, list)
    assert "Repair the previous ASR review response" in second_content


def test_llama_server_translation_client_aligns_audio_review_with_heard_text(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "chunk_0001.wav"
    audio_path.write_bytes(b"RIFFmock-wav-data")

    def handler(_request: httpx.Request) -> httpx.Response:
        response = [
            {
                "chunk_id": "chunk_0001",
                "heard_text": "20 そろそろきつくなってきた",
                "decision": "manual_review",
                "selected_candidate_id": "original",
                "confidence": 0.95,
                "reason": "heard the leading count but chose the wrong id",
                "risk_terms": [],
            }
        ]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]},
        )

    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
    )

    reviews = client.review_asr_candidates_with_audio(
        [
            {
                "chunk_id": "chunk_0001",
                "candidates": [
                    {"candidate_id": "original", "text": "そろそろきつくなってきた?"},
                    {"candidate_id": "repair_no_vad", "text": "20 そろそろきつくなってきた"},
                ],
            }
        ],
        "asr_review_0001",
        audio_path,
    )

    assert reviews["chunk_0001"]["decision"] == "replace"
    assert reviews["chunk_0001"]["selected_candidate_id"] == "repair_no_vad"


def test_llama_server_translation_client_rejects_non_candidate_asr_text(tmp_path: Path) -> None:
    audio_path = tmp_path / "chunk_0001.wav"
    audio_path.write_bytes(b"RIFFmock-wav-data")

    def handler(_request: httpx.Request) -> httpx.Response:
        response = [
            {
                "chunk_id": "chunk_0001",
                "decision": "replace",
                "selected_candidate_id": "invented_text",
                "confidence": 0.99,
                "reason": "bad",
                "risk_terms": [],
            }
        ]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response)}}]},
        )

    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retries=0,
        n_predict=128,
    )

    with pytest.raises(Exception, match="invalid selected_candidate_id"):
        client.review_asr_candidates_with_audio(
            [
                {
                    "chunk_id": "chunk_0001",
                    "candidates": [
                        {"candidate_id": "original", "text": "手帳が来る"},
                        {"candidate_id": "domain_replacement", "text": "絶頂が来る"},
                    ],
                }
            ],
            "asr_review_0001",
            audio_path,
        )


def test_llama_server_translation_client_uses_literal_then_natural_pass() -> None:
    requests: list[dict[str, object]] = []
    responses = [
        [{"segment_id": "seg_0001", "ko_literal": "어서 오세요, 오빠."}],
        [{"segment_id": "seg_0001", "ko_natural": "어서 와요, 오빠."}],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(responses[len(requests) - 1], ensure_ascii=False)}}
                ]
            },
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="いらっしゃいませお兄さん",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_literal == "어서 오세요, 오빠."
    assert translations["seg_0001"].ko_natural == "어서 와요, 오빠."
    assert len(requests) == 2
    first_prompt = requests[0]["messages"][0]["content"]
    second_prompt = requests[1]["messages"][0]["content"]
    assert "First pass" in first_prompt
    assert "Second pass" in second_prompt
    assert "ko_literal" in second_prompt
    assert "target_span" in first_prompt
    assert "combined_source_text" in first_prompt


def test_llama_server_translation_client_repairs_low_quality_translation() -> None:
    requests: list[dict[str, object]] = []
    bad = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "だって",
            "ko_natural": "だって",
            "notes": [],
            "confidence": 0.98,
            "model": "gemma4",
            "batch_id": "batch_0001",
        }
    ]
    repaired = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "왜냐하면.",
            "ko_natural": "왜냐하면요.",
            "notes": [],
            "confidence": 0.98,
            "model": "gemma4",
            "batch_id": "batch_0001",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = bad if len(requests) == 1 else repaired
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]},
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="だって",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "왜냐하면요."
    assert len(requests) == 3
    second_prompt = requests[1]["messages"][0]["content"]
    assert "Original input" in second_prompt
    assert "だって" in second_prompt


def test_llama_server_translation_client_repairs_numeric_raw_digits() -> None:
    requests: list[dict[str, object]] = []
    responses = [
        [{"segment_id": "seg_0001", "ko_literal": "2014"}],
        [{"segment_id": "seg_0001", "ko_natural": "2014"}],
        [{"segment_id": "seg_0001", "ko_natural": "이천십사"}],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(responses[len(requests) - 1], ensure_ascii=False)}}
                ]
            },
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="2014",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "이천십사"
    assert len(requests) == 3
    repair_prompt = requests[2]["messages"][0]["content"]
    assert "numeric-only or digit-heavy" in repair_prompt
    assert "context" in repair_prompt


def test_llama_server_translation_client_repairs_tts_unsafe_korean_punctuation() -> None:
    requests: list[dict[str, object]] = []
    responses = [
        [{"segment_id": "seg_0001", "ko_literal": "지금은-"}],
        [{"segment_id": "seg_0001", "ko_natural": "지금은요—"}],
        [{"segment_id": "seg_0001", "ko_natural": "지금은요..."}],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(responses[len(requests) - 1], ensure_ascii=False)}}
                ]
            },
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="今はー",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "지금은요..."
    assert len(requests) == 3
    assert "TTS-unsafe punctuation" in requests[2]["messages"][0]["content"]


def test_llama_server_translation_client_keeps_valid_items_from_mixed_batch() -> None:
    mixed = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "안녕하세요.",
            "ko_natural": "안녕하세요.",
            "notes": [],
            "confidence": 0.95,
            "model": "gemma4",
            "batch_id": "batch_0001",
        },
        {
            "segment_id": "seg_0002",
            "ko_literal": "だって",
            "ko_natural": "だって",
            "notes": [],
            "confidence": 0.95,
            "model": "gemma4",
            "batch_id": "batch_0001",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(mixed, ensure_ascii=False)}}]},
        )

    segments = [sample_segment("seg_0001"), sample_segment("seg_0002")]
    for segment, text in zip(segments, ["こんにちは", "だって"], strict=True):
        segment.source_script = SourceScript(
            text=text,
            language="ja",
            confidence=0.9,
            backend="mock",
            start=segment.start,
            end=segment.end,
        )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
    )

    translations = client.translate_batch(segments, "batch_0001")

    assert set(translations) == {"seg_0001"}
    assert translations["seg_0001"].ko_natural == "안녕하세요."


def test_korean_colloquializer_rewrites_stiff_polite_forms() -> None:
    text = "저는 괜찮습니다. 이것은 좋은 것입니다. 천천히 하겠습니다."

    assert colloquialize_korean_text(text) == "전 괜찮아요. 이건 좋은 거예요. 천천히 할게요."


def test_translate_ko_colloquializes_finished_translations(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            return {
                segment.id: KoreanTranslation(
                    ko_literal="저는 괜찮습니다.",
                    ko_natural="저는 괜찮습니다. 이것은 좋은 것입니다. 천천히 하겠습니다.",
                    notes=["model_note"],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")

    manifest = load_manifest(tmp_project_dir)
    translation = manifest.segments[0].translation_ko
    assert translation is not None
    assert translation.ko_natural == "전 괜찮아요. 이건 좋은 거예요. 천천히 할게요."
    assert translation.notes == ["model_note", COLLOQUIAL_REWRITE_NOTE]
    assert manifest.stage_state["translate-ko"]["colloquialized"] == 1

    rows = [
        json.loads(line)
        for line in Path(manifest.artifacts["translation_bundles"]).read_text().splitlines()
    ]
    assert rows[0]["colloquialized"] is True
    assert rows[0]["translation_ko"]["ko_natural"] == translation.ko_natural


def test_transcribe_and_translate_mock_steps_write_artifacts(
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)

    transcribe_step(tmp_project_dir, asr_backend="mock")
    segment_step(tmp_project_dir)
    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")
    korean_script_step(tmp_project_dir, confirm_rights=True)
    save_project_config(
        ProjectConfig(project_name=tmp_project_dir.name, gsv_ref_min_sec=0.1),
        tmp_project_dir / "pipeline.yaml",
    )
    prepare_source_voice_refs_step(tmp_project_dir)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["transcribe"]["status"] == "completed"
    assert manifest.stage_state["transcribe-seed"]["status"] == "completed"
    assert manifest.stage_state["segment"]["source"] == "transcribe"
    assert manifest.stage_state["translate-ko"]["status"] == "completed"
    assert manifest.stage_state["korean-script"]["status"] == "completed"
    assert manifest.stage_state["prepare-refs"]["status"] == "completed"
    assert manifest.segments
    assert manifest.segments[0].source_script is not None
    assert manifest.segments[0].translation_ko is not None
    assert manifest.segments[0].script is not None
    assert manifest.segments[0].script.tts_language == "ko"
    assert manifest.segments[0].script.source_language == "ja"
    assert manifest.segments[0].script.target_language == "ko"
    assert manifest.segments[0].script.ja_text.startswith("mock source script")
    assert manifest.segments[0].script.tts_text.startswith("자연 번역,")
    assert Path(manifest.artifacts["source_segments"]).exists()
    assert Path(manifest.artifacts["segments_transcribe_seed"]).exists()
    assert Path(manifest.artifacts["segments_transcribed"]).exists()
    assert Path(manifest.artifacts["segments_final"]).exists()
    assert Path(manifest.artifacts["translation_bundles"]).exists()
    assert Path(manifest.artifacts["translation_summary"]).exists()
    assert Path(manifest.artifacts["segments_ko_script"]).exists()
    refs = json.loads((tmp_project_dir / "refs" / "refs.json").read_text("utf-8"))
    assert refs["whisper_close"]["prompt_text"].startswith("mock source script")
    assert refs["whisper_close"]["prompt_lang"] == "ja"
    assert refs["whisper_close"]["target_language"] == "ko"
    assert (tmp_project_dir / refs["whisper_close"]["ref_audio_path"]).exists()
    ref_qc = json.loads(Path(manifest.artifacts["source_voice_ref_qc"]).read_text("utf-8"))
    assert ref_qc["refs"][0]["source_language"] == "ja"
    assert ref_qc["refs"][0]["target_language"] == "ko"
    summary = json.loads(Path(manifest.artifacts["translation_summary"]).read_text("utf-8"))
    assert summary["backend"] == "mock"
    assert summary["translated"] == len(manifest.segments)


def test_translate_ko_skips_manual_review_segments_even_with_source_text(
    tmp_project_dir: Path,
) -> None:
    segment = sample_segment(start=0.0, end=18.0)
    segment.status = "needs_manual_review"
    segment.errors.append("asr_sparse_text_density")
    segment.source_script = SourceScript(
        text="ム",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=18.0,
        confidence=0.95,
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["transcribe"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].translation_ko is None
    assert manifest.stage_state["translate-ko"]["translated"] == 0
    assert manifest.stage_state["translate-ko"]["needs_manual_review"] == 1
    rows = [
        json.loads(line)
        for line in Path(manifest.artifacts["translation_bundles"]).read_text().splitlines()
    ]
    assert rows == [
        {
            "segment_id": "seg_0001",
            "status": "needs_manual_review",
            "reason": "segment status is needs_manual_review",
            "source_text": "ム",
            "translation_ko": None,
        }
    ]


def test_korean_script_blocks_japanese_fallback_text(tmp_project_dir: Path) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        source_script=SourceScript(
            text="少し近づきますね",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="少し近づきますね",
            ko_natural="少し近づきますね",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].status == "needs_manual_review"
    assert "korean_tts_contains_kana" in manifest.segments[0].errors[-1]


def test_transcribe_and_translate_mock_cli(
    cli_runner,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)

    transcribe_result = cli_runner.invoke(
        app,
        ["transcribe", "-p", str(tmp_project_dir), "--asr-backend", "mock"],
    )
    translate_result = cli_runner.invoke(
        app,
        ["translate-ko", "-p", str(tmp_project_dir), "--gemma-text-backend", "mock"],
    )

    assert transcribe_result.exit_code == 0, transcribe_result.output
    assert translate_result.exit_code == 0, translate_result.output
    assert "- 원문:" in translate_result.output
    assert "- 번역문:" in translate_result.output
    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].source_script is not None
    assert manifest.segments[0].translation_ko is not None


def test_transcribe_cli_accepts_asr_debug_options(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_transcribe_step(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest()

    monkeypatch.setattr(cli_module, "transcribe_step", fake_transcribe_step)

    result = cli_runner.invoke(
        app,
        [
            "transcribe",
            "-p",
            str(tmp_project_dir),
            "--asr-backend",
            "faster_whisper",
            "--asr-preset",
            "whisper",
            "--asr-vad-off",
            "--asr-diagnostics",
            "--asr-device",
            "cuda",
            "--asr-compute-type",
            "float16",
            "--asr-batched",
            "--asr-batch-size",
            "16",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["asr_preset"] == "whisper"
    assert kwargs["asr_vad_off"] is True
    assert kwargs["asr_diagnostics"] is True
    assert kwargs["asr_device"] == "cuda"
    assert kwargs["asr_compute_type"] == "float16"
    assert kwargs["asr_batched_inference"] is True
    assert kwargs["asr_batch_size"] == 16


def test_translate_ko_requires_transcribe_stage(tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)

    with pytest.raises(ValueError, match="transcribe"):
        translate_ko_step(tmp_project_dir, gemma_text_backend="mock")


def test_translate_ko_retries_missing_model_items(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    if len(manifest.segments) == 1:
        base = manifest.segments[0]
        manifest.segments.append(
            Segment(
                id="seg_0002",
                start=base.end,
                end=base.end + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text="追加の台詞です",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=base.end,
                    end=base.end + base.duration,
                ),
            )
        )
        save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, [segment.id for segment in segments]))
            selected = segments if len(segments) == 1 else segments[:1]
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in selected
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert all(segment.translation_ko is not None for segment in manifest.segments)
    assert any(batch_id.startswith("batch_0001_single_") for batch_id, _ in calls)


def test_translate_ko_falls_back_to_single_segments_when_batch_fails(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    manifest.segments.append(
        Segment(
            id="seg_0002",
            start=base.end,
            end=base.end + base.duration,
            duration=base.duration,
            audio_for_gemma=base.audio_for_gemma,
            audio_for_mix=base.audio_for_mix,
            source_script=SourceScript(
                text="追加の台詞です",
                language="ja",
                confidence=0.99,
                backend="mock",
                start=base.end,
                end=base.end + base.duration,
            ),
        )
    )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, [segment.id for segment in segments]))
            if len(segments) > 1:
                raise RuntimeError("bad batch json")
            segment = segments[0]
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert all(segment.translation_ko is not None for segment in manifest.segments)
    assert all(
        not any(error.startswith("Korean translation batch failed") for error in segment.errors)
        for segment in manifest.segments
    )
    assert calls[0] == ("batch_0001", ["seg_0001", "seg_0002"])
    assert {tuple(ids) for _, ids in calls[1:]} == {("seg_0001",), ("seg_0002",)}


def test_translate_ko_fails_fast_when_llama_server_connection_is_refused(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    state = {"stopped": False}

    class FakeServer:
        started = True
        reused_existing = False
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            state["stopped"] = True

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            raise RuntimeError("Gemma text translation failed: [Errno 111] Connection refused")

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    with pytest.raises(RuntimeError, match="Connection refused"):
        translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].translation_ko is None
    assert state["stopped"] is True
    assert not (tmp_project_dir / "work" / "translate_ko" / "translation_bundles.jsonl").exists()


def test_translate_ko_splits_failed_batches_before_single_retries(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    for index in range(2, 5):
        start = base.end + (index - 2) * base.duration
        manifest.segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text=f"追加の台詞です {index}",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=start,
                    end=start + base.duration,
                ),
            )
        )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, [segment.id for segment in segments]))
            if len(segments) > 2:
                raise RuntimeError("batch too large")
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert all(segment.translation_ko is not None for segment in manifest.segments)
    assert calls[0] == ("batch_0001", ["seg_0001", "seg_0002", "seg_0003", "seg_0004"])
    assert [len(ids) for _, ids in calls[1:]] == [2, 2]


def test_translate_ko_uses_single_llama_server_slot_workers(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_batch_size=1,
            gemma_text_span_size=1,
            gemma_text_concurrency=2,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    for index in range(2, 5):
        start = base.end + (index - 2) * base.duration
        manifest.segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text=f"追加の台詞です {index}",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=start,
                    end=start + base.duration,
                ),
            )
        )
    save_manifest(tmp_project_dir, manifest)
    server_base_urls: list[str] = []
    calls: list[tuple[str, str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def __init__(self, **kwargs: object) -> None:
            self.base_url = str(kwargs["base_url"])
            self.log_path = kwargs.get("log_path")
            server_base_urls.append(self.base_url)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, base_url: str, *args: object, **kwargs: object) -> None:
            self.base_url = base_url

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, self.base_url, [segment.id for segment in segments]))
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", FakeServer)
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    assert server_base_urls == ["http://127.0.0.1:8080"]
    assert {base_url for _, base_url, _ in calls} == {"http://127.0.0.1:8080"}
    calls_by_batch = {batch_id: ids for batch_id, _, ids in calls}
    assert calls_by_batch == {
        "batch_0001": ["seg_0001"],
        "batch_0002": ["seg_0002"],
        "batch_0003": ["seg_0003"],
        "batch_0004": ["seg_0004"],
    }
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["translate-ko"]["concurrency"] == 2
    assert manifest.stage_state["translate-ko"]["server"]["server_count"] == 1
    assert manifest.stage_state["translate-ko"]["server"]["mode"] == "single_server_slots"


def test_target_language_ko_full_uses_text_only_korean_lane(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def step(name: str):
        def inner(*args: object, **kwargs: object) -> PipelineManifest:
            calls.append((name, args, kwargs))
            return PipelineManifest()

        return inner

    monkeypatch.setattr(orchestrator, "init_project", step("init"))
    monkeypatch.setattr(orchestrator, "extract_step", step("extract"))
    monkeypatch.setattr(orchestrator, "source_separation_step", step("source-separation"))
    monkeypatch.setattr(orchestrator, "segment_step", step("segment"))
    monkeypatch.setattr(orchestrator, "analyze_step", step("analyze"))
    monkeypatch.setattr(orchestrator, "script_step", step("script"))
    monkeypatch.setattr(orchestrator, "transcribe_step", step("transcribe"))
    monkeypatch.setattr(orchestrator, "translate_ko_step", step("translate-ko"))
    monkeypatch.setattr(orchestrator, "korean_script_step", step("korean-script"))
    monkeypatch.setattr(orchestrator, "prepare_source_voice_refs_step", step("prepare-refs"))
    monkeypatch.setattr(orchestrator, "gsv_few_shot_step", step("gsv-few-shot"))
    monkeypatch.setattr(orchestrator, "synth_step", step("synth"))
    monkeypatch.setattr(orchestrator, "rvc_train_step", step("train-rvc"))
    monkeypatch.setattr(orchestrator, "rvc_step", step("rvc"))
    monkeypatch.setattr(orchestrator, "qc_step", step("qc"))
    monkeypatch.setattr(orchestrator, "regenerate_needs_step", step("regenerate"))
    monkeypatch.setattr(orchestrator, "mix_step", step("mix"))
    monkeypatch.setattr(orchestrator, "export_step", step("export"))
    monkeypatch.setattr(orchestrator, "validate_rvc_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "validate_rvc_training_config", lambda *args, **kwargs: None)

    orchestrator.run_pipeline(
        tmp_path / "input.wav",
        tmp_path / "project",
        confirm_rights=True,
        mock=False,
        gemma_backend="hf",
        target_language="kr",
        asr_backend="qwen_asr",
    )

    assert [name for name, _, _ in calls] == [
        "init",
        "extract",
        "source-separation",
        "transcribe",
        "segment",
        "translate-ko",
        "korean-script",
        "prepare-refs",
        "gsv-few-shot",
        "synth",
        "train-rvc",
        "rvc",
        "qc",
        "mix",
        "export",
    ]
    assert calls[3][2]["asr_backend"] == "qwen_asr"
    assert calls[5][1][1] == "llama_server"
    assert calls[9][2]["mock"] is False
    assert calls[10][2]["mock"] is False
    assert calls[11][2]["mock"] is False
    assert calls[12][1][1] == "mock"

    calls.clear()
    orchestrator.run_pipeline(
        tmp_path / "input.wav",
        tmp_path / "project_regen",
        confirm_rights=True,
        mock=False,
        gemma_backend="hf",
        target_language="kr",
        regenerate_before_mix=True,
    )

    names = [name for name, _, _ in calls]
    assert names[names.index("qc") + 1] == "regenerate"
    assert names[names.index("regenerate") + 1] == "mix"
