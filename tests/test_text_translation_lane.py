from __future__ import annotations

import base64
import json
import logging as py_logging
from collections import Counter
from pathlib import Path
from threading import Event, Lock

import httpx
import numpy as np
import pytest

from asmr_dub_pipeline import cli as cli_module
from asmr_dub_pipeline import orchestrator
from asmr_dub_pipeline.asr.base import ASRChunk, ASRWord, map_chunks_to_segments
from asmr_dub_pipeline.audio.features import duration_sec, write_audio
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gemma.text_translate import (
    LlamaServerTranslationClient,
    build_translate_ko_prompt,
    parse_asr_review_response,
    parse_translation_response,
)
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import (
    audio_style_step,
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
    RVCMetadata,
    Segment,
    SourceInfo,
    SourceScript,
    TTSCandidate,
    TTSMetadata,
)
from asmr_dub_pipeline.script.countdown import (
    source_countdown_token_matches,
    source_countdown_values,
)
from asmr_dub_pipeline.script.duration_rewrite import (
    estimate_tts_duration,
    japanese_pronunciation_count,
    korean_tts_speech_char_count,
    korean_tts_timing_budget,
)
from asmr_dub_pipeline.script.korean_colloquial import (
    COLLOQUIAL_REWRITE_NOTE,
    colloquialize_korean_text,
)

pytestmark = pytest.mark.regression


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


def test_asr_resegment_absorbs_trailing_micro_group_for_tts(tmp_project_dir: Path) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(
                start=0.0,
                end=3.2,
                text="すっごく濡れて やらしい音しちゃってる",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=3.25,
                end=3.45,
                text="でも、こんなにゴリゴリになっちゃってるし",
                language="ja",
                confidence=0.89,
            ),
            ASRChunk(start=5.0, end=7.4, text="待ってー", language="ja", confidence=0.9),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=8.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
    )

    assert [segment.duration for segment in segments] == [3.45, 2.4]
    assert segments[0].source_script is not None
    assert segments[0].source_script.text == (
        "すっごく濡れて やらしい音しちゃってる "
        "でも、こんなにゴリゴリになっちゃってるし"
    )


def test_asr_resegment_absorbs_nearby_micro_group_beyond_normal_gap(
    tmp_project_dir: Path,
) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(start=0.0, end=3.2, text="ちゃんと聞いて", language="ja", confidence=0.95),
            ASRChunk(start=4.45, end=4.87, text="いい?", language="ja", confidence=0.96),
            ASRChunk(start=6.08, end=9.0, text="準備していくからね", language="ja", confidence=0.9),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=10.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
    )

    assert [segment.duration for segment in segments] == [3.2, 4.55]
    assert segments[1].source_script is not None
    assert segments[1].source_script.text == "いい? 準備していくからね"


def test_asr_resegment_absorbs_short_group_within_merge_gap_for_tts(
    tmp_project_dir: Path,
) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(start=0.0, end=3.4, text="まずはゆっくり息を吸って", language="ja", confidence=0.95),
            ASRChunk(start=4.2, end=6.0, text="そう", language="ja", confidence=0.94),
            ASRChunk(start=7.3, end=10.4, text="次は力を抜いて", language="ja", confidence=0.9),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=12.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
        max_segment_sec=10.0,
    )

    assert [segment.duration for segment in segments] == [6.0, 3.1]
    assert segments[0].source_script is not None
    assert segments[0].source_script.text == "まずはゆっくり息を吸って そう"
    assert all(segment.duration >= 3.0 for segment in segments)


def test_asr_sparse_edge_speech_moves_to_neighbor_and_silence_becomes_texture(
    tmp_project_dir: Path,
) -> None:
    sr = 16_000
    tone = np.full(int(sr * 0.45), 0.08, dtype=np.float32)
    silence = np.zeros(int(sr * 11.55), dtype=np.float32)
    sparse_clip = tmp_project_dir / "seg_0002_gemma.wav"
    write_audio(sparse_clip, np.concatenate([tone, silence])[:, None], sr)
    previous_clip = tmp_project_dir / "seg_0001_gemma.wav"
    write_audio(previous_clip, np.full((int(sr * 2.0), 1), 0.08, dtype=np.float32), sr)
    project_audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    segments = [
        Segment(
            id="seg_0001",
            start=0.0,
            end=2.0,
            duration=2.0,
            audio_for_gemma=str(previous_clip),
            audio_for_mix=str(previous_clip),
            source_script=SourceScript(
                text="前の言葉",
                language="ja",
                backend="faster_whisper",
                start=0.0,
                end=2.0,
                confidence=0.95,
            ),
        ),
        Segment(
            id="seg_0002",
            start=2.2,
            end=14.2,
            duration=12.0,
            audio_for_gemma=str(sparse_clip),
            audio_for_mix=str(sparse_clip),
            source_script=SourceScript(
                text="鉛筆 バス",
                language="ja",
                backend="faster_whisper",
                start=2.2,
                end=14.2,
                confidence=0.95,
            ),
        ),
    ]

    split = pipeline_steps._split_sparse_edge_segments_by_audio(
        segments,
        project_dir=tmp_project_dir,
        cfg=ProjectConfig(),
        merge_gap_sec=0.5,
    )

    assert [segment.id for segment in split] == ["seg_0001", "seg_0002"]
    assert split[0].source_script is not None
    assert split[0].source_script.text == "前の言葉 鉛筆 バス"
    assert split[0].end == pytest.approx(2.95, abs=0.08)
    assert split[1].status == "non_speech_texture"
    assert split[1].source_script is not None
    assert split[1].source_script.text == "…"
    assert split[1].start == pytest.approx(split[0].end)
    assert split[1].end == pytest.approx(14.2)
    assert split[1].audio_for_gemma == str(project_audio_dir / "seg_0002_gemma.wav")


def test_asr_resegment_preserves_countdown_wall_clock_span(tmp_project_dir: Path) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(
                start=863.298,
                end=864.338,
                text="4 3",
                language="ja",
                confidence=0.8,
                words=[
                    ASRWord(start=863.298, end=863.798, text="4", confidence=0.9),
                    ASRWord(start=863.9, end=864.338, text="3", confidence=0.88),
                ],
            ),
            ASRChunk(
                start=865.615,
                end=866.655,
                text="2 1",
                language="ja",
                confidence=0.8,
                words=[
                    ASRWord(start=865.615, end=866.08, text="2", confidence=0.91),
                    ASRWord(start=866.17, end=866.655, text="1", confidence=0.87),
                ],
            ),
            ASRChunk(
                start=868.606,
                end=869.128,
                text="0",
                language="ja",
                confidence=0.8,
                words=[ASRWord(start=868.606, end=869.128, text="0", confidence=0.92)],
            ),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=900.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
        countdown_merge_enabled=True,
        countdown_merge_gap_sec=2.5,
        countdown_merge_max_span_sec=14.0,
    )

    assert len(segments) == 1
    segment = segments[0]
    assert segment.start == pytest.approx(863.298)
    assert segment.end == pytest.approx(869.128)
    assert segment.duration == pytest.approx(5.83)
    assert segment.source_script is not None
    assert segment.source_script.text == "4 3 2 1 0"
    assert segment.analysis["countdown_event"]["values"] == [4, 3, 2, 1, 0]
    assert segment.analysis["countdown_event"]["source_chunk_texts"] == ["4 3", "2 1", "0"]
    assert segment.analysis["countdown_event"]["token_timeline"] == [
        {
            "value": 4,
            "source_text": "4",
            "korean_token": "사",
            "start": 863.298,
            "end": 863.798,
            "confidence": 0.9,
        },
        {
            "value": 3,
            "source_text": "3",
            "korean_token": "삼",
            "start": 863.9,
            "end": 864.338,
            "confidence": 0.88,
        },
        {
            "value": 2,
            "source_text": "2",
            "korean_token": "이",
            "start": 865.615,
            "end": 866.08,
            "confidence": 0.91,
        },
        {
            "value": 1,
            "source_text": "1",
            "korean_token": "일",
            "start": 866.17,
            "end": 866.655,
            "confidence": 0.87,
        },
        {
            "value": 0,
            "source_text": "0",
            "korean_token": "영",
            "start": 868.606,
            "end": 869.128,
            "confidence": 0.92,
        },
    ]


def test_source_countdown_values_accepts_spoken_japanese_number_readings() -> None:
    assert source_countdown_values("じゅうさん なな ろく ゼロ") == [13, 7, 6, 0]
    assert source_countdown_values("ジュウイチ ナナ ロク") == [11, 7, 6]


def test_embedded_countdown_tokens_ignore_kana_number_inside_japanese_word_without_translation_route() -> None:
    segment = sample_segment(start=0.0, end=7.89)
    segment.source_script = SourceScript(
        text="9 8 7 6 5 4 3 2 1 絶頂します 0",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=7.89,
    )

    assert [raw for _value, raw, _start, _end in source_countdown_token_matches(segment.source_script.text)] == [
        "9",
        "8",
        "7",
        "6",
        "5",
        "4",
        "3",
        "2",
        "1",
        "0",
    ]
    assert pipeline_steps._countdown_values_for_segment(segment) is None


def test_asr_resegment_derives_countdown_timeline_from_chunk_spans_when_words_missing(
    tmp_project_dir: Path,
) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(start=863.298, end=864.338, text="4 3", language="ja", confidence=0.8),
            ASRChunk(start=865.615, end=866.655, text="2 1", language="ja", confidence=0.8),
            ASRChunk(start=868.606, end=869.128, text="0", language="ja", confidence=0.8),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=900.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
        countdown_merge_enabled=True,
        countdown_merge_gap_sec=2.5,
        countdown_merge_max_span_sec=14.0,
    )

    timeline = segments[0].analysis["countdown_event"]["token_timeline"]
    assert [item["value"] for item in timeline] == [4, 3, 2, 1, 0]
    assert [item["korean_token"] for item in timeline] == ["사", "삼", "이", "일", "영"]
    assert [item["start"] for item in timeline] == pytest.approx(
        [863.298, 863.818, 865.615, 866.135, 868.606],
        abs=0.001,
    )
    assert [item["end"] for item in timeline] == pytest.approx(
        [863.818, 864.338, 866.135, 866.655, 869.128],
        abs=0.001,
    )


def test_asr_resegment_marks_embedded_countdown_with_equal_slot_timeline(
    tmp_project_dir: Path,
) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(
                start=1293.84,
                end=1308.67,
                text="じゃあ5 4 3 2 1 ゼロ スーッと",
                language="ja",
                confidence=0.98,
            ),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=1400.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
        countdown_merge_enabled=True,
        countdown_merge_gap_sec=2.5,
        countdown_merge_max_span_sec=20.0,
    )

    event = segments[0].analysis["countdown_event"]
    timeline = event["token_timeline"]
    assert event["values"] == [5, 4, 3, 2, 1, 0]
    assert event["source_chunk_texts"] == ["じゃあ5 4 3 2 1 ゼロ スーッと"]
    assert [item["value"] for item in timeline] == [5, 4, 3, 2, 1, 0]
    assert [item["source_text"] for item in timeline] == ["5", "4", "3", "2", "1", "ゼロ"]
    assert timeline[0]["start"] == pytest.approx(1293.84, abs=0.001)
    assert timeline[-1]["end"] == pytest.approx(1308.67, abs=0.001)


def test_asr_resegment_tightens_sparse_chunk_timing_from_word_spans(
    tmp_project_dir: Path,
) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(
                start=220.0,
                end=235.0,
                text="今の姿勢のまま",
                language="ja",
                confidence=0.97,
                words=[
                    ASRWord(start=226.2, end=226.7, text="今の", confidence=0.96),
                    ASRWord(start=226.8, end=227.4, text="姿勢", confidence=0.96),
                    ASRWord(start=227.5, end=228.0, text="のまま", confidence=0.96),
                ],
            ),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=300.0,
        min_segment_sec=3.0,
        merge_gap_sec=1.0,
        max_segment_sec=14.0,
        sparse_chunk_max_sec=30.0,
        sparse_chunk_min_chars_per_sec=1.0,
    )

    assert len(segments) == 1
    assert segments[0].start == pytest.approx(225.8, abs=0.001)
    assert segments[0].end == pytest.approx(228.4, abs=0.001)
    assert segments[0].source_script is not None
    assert segments[0].source_script.start == pytest.approx(225.8, abs=0.001)
    assert segments[0].source_script.end == pytest.approx(228.4, abs=0.001)


def test_segment_stage_records_countdown_timeline_summary(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(project_name="test")
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    segment = sample_segment("seg_0001", start=0.0, end=3.0)
    segment.status = "transcribed"
    segment.source_script = SourceScript(
        text="3 2 1",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=3.0,
    )
    segment.analysis["countdown_event"] = {
        "kind": "descending_countdown",
        "values": [3, 2, 1],
        "korean_text": "삼, 이, 일",
        "korean_tokens": ["삼", "이", "일"],
        "preserve_wall_clock_span": True,
    }
    manifest = PipelineManifest(project_config=cfg, segments=[segment])
    manifest.stage_state["transcribe"] = {
        "status": "completed",
        "asr_word_timestamps": True,
    }
    save_manifest(tmp_project_dir, manifest)

    finalized = segment_step(tmp_project_dir, confirm_rights=True)

    segment_state = finalized.stage_state["segment"]
    assert segment_state["countdown_event_count"] == 1
    assert segment_state["countdown_token_timeline_count"] == 0
    assert segment_state["countdown_token_timeline_missing"] == 1
    assert segment_state["countdown_token_timeline_missing_ids"] == ["seg_0001"]
    assert any("countdown token timeline missing" in warning for warning in finalized.warnings)


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


def test_transcribe_marks_empty_real_asr_as_no_speech(
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
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class EmptyASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return []

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: EmptyASRBackend())

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")

    assert manifest.segments[0].status == "no_speech_detected"
    assert manifest.segments[0].analysis["asr_quality_gate"] == {
        "decision": "no_speech",
        "reasons": ["no_speech_detected"],
        "tts_blocked": True,
    }
    assert "missing_asr_text" not in manifest.segments[0].errors
    rows = [
        json.loads(line)
        for line in Path(manifest.artifacts["source_segments"]).read_text("utf-8").splitlines()
    ]
    assert rows[0]["status"] == "no_speech_detected"
    summary = json.loads(Path(manifest.artifacts["asr_diagnostics_summary"]).read_text("utf-8"))
    assert summary["no_speech_detected"] == 1
    assert summary["needs_manual_review"] == 0
    assert manifest.stage_state["transcribe"]["no_speech_detected"] == 1


def test_transcribe_marks_vocal_texture_as_non_speech(
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
            asr_repair_enabled=True,
            asr_review_enabled=True,
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class TextureASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=3.0,
                    text="ぃぃぃぃ",
                    language="ja",
                    confidence=0.98,
                )
            ]

        def transcribe_with_options(
            self,
            _audio_path: Path,
            _segments: list[Segment],
            **_kwargs: object,
        ) -> list[ASRChunk]:
            raise AssertionError("non-speech texture should not consume repair/review candidates")

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: TextureASRBackend())

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")

    segment = manifest.segments[0]
    assert segment.status == "non_speech_texture"
    assert segment.keep_original_texture is True
    assert segment.errors == ["asr_non_speech_texture"]
    assert segment.analysis["asr_quality_gate"] == {
        "decision": "texture",
        "reasons": ["asr_non_speech_texture"],
        "tts_blocked": True,
    }
    rows = [
        json.loads(line)
        for line in Path(manifest.artifacts["source_segments"]).read_text("utf-8").splitlines()
    ]
    assert rows[0]["status"] == "non_speech_texture"
    summary = json.loads(Path(manifest.artifacts["asr_diagnostics_summary"]).read_text("utf-8"))
    assert summary["non_speech_texture"] == 1
    assert summary["no_speech_detected"] == 1
    assert summary["needs_manual_review"] == 0


def test_transcribe_retries_sparse_segment_locally_before_manual_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    input_path = tmp_project_dir.parent / "long_input.wav"
    sample_rate = 48_000
    t = np.arange(int(sample_rate * 9.0), dtype=np.float32) / sample_rate
    tone = 0.08 * np.sin(2 * np.pi * 220.0 * t)
    write_audio(input_path, np.stack([tone, tone], axis=1), sample_rate)
    extract_step(input_path, tmp_project_dir, confirm_rights=True)
    manifest = load_manifest(tmp_project_dir)
    manifest.artifacts["source_vocals_mono_16k"] = manifest.artifacts["gemma_mono_16k"]
    manifest.artifacts["source_vocals_48k"] = manifest.artifacts["original_stereo_48k"]
    save_manifest(tmp_project_dir, manifest)
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            asr_resegment_from_chunks=False,
            asr_repair_enabled=True,
            asr_review_enabled=False,
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class SparseThenLocalASRBackend:
        name = "faster_whisper"

        def __init__(self) -> None:
            self.retry_paths: list[Path] = []

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=8.4, text="五", language="ja", confidence=0.96)]

        def transcribe_with_options(
            self,
            audio_path: Path,
            _segments: list[Segment],
            **_kwargs: object,
        ) -> list[ASRChunk]:
            self.retry_paths.append(audio_path)
            return [
                ASRChunk(
                    start=0.2,
                    end=7.8,
                    text="ゆっくり息をして耳元で話しています",
                    language="ja",
                    confidence=0.97,
                )
            ]

    backend = SparseThenLocalASRBackend()
    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: backend)

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper", confirm_rights=True)

    segment = manifest.segments[0]
    assert segment.status == "transcribed"
    assert segment.source_script is not None
    assert segment.source_script.text == "ゆっくり息をして耳元で話しています"
    assert segment.errors == []
    assert segment.analysis["asr_quality_gate"] == {
        "decision": "pass",
        "reasons": [],
        "tts_blocked": False,
    }
    assert backend.retry_paths
    assert all(path.parent.name == "asr_segment_retry_clips" for path in backend.retry_paths)
    summary = json.loads(Path(manifest.artifacts["asr_segment_retry_summary"]).read_text("utf-8"))
    assert summary["attempted"] == 1
    assert summary["repaired"] == 1
    assert summary["items"][0]["segment_id"] == "seg_0001"
    assert summary["items"][0]["original_text"] == "五"
    assert summary["items"][0]["accepted"] is True
    assert summary["items"][0]["accepted_text"] == "ゆっくり息をして耳元で話しています"
    assert summary["items"][0]["accepted_vote_count"] >= 2


def test_sparse_single_character_source_script_is_texture_not_manual_review() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="声",
        language="ja",
        backend="faster_whisper",
        start=120.0,
        end=133.0,
        confidence=0.98,
    )

    assert pipeline_steps._source_script_non_speech_texture_reason(source_script) == "asr_non_speech_texture"
    assert pipeline_steps._source_script_asr_review_reasons(source_script, cfg) == []


def test_countdown_words_are_not_treated_as_texture() -> None:
    source_script = SourceScript(
        text="ゼロ",
        language="ja",
        backend="faster_whisper",
        start=120.0,
        end=133.0,
        confidence=0.98,
    )

    assert pipeline_steps._source_script_non_speech_texture_reason(source_script) is None


def test_short_filler_keep_original_candidate_uses_conservative_allowlist() -> None:
    candidate = SourceScript(
        text="あの",
        language="ja",
        backend="faster_whisper",
        start=10.0,
        end=10.36,
        confidence=0.98,
    )
    repeated = SourceScript(
        text="うんうん、",
        language="ja",
        backend="faster_whisper",
        start=11.0,
        end=11.74,
        confidence=0.98,
    )
    meaningful = SourceScript(
        text="今日は",
        language="ja",
        backend="faster_whisper",
        start=12.0,
        end=12.52,
        confidence=0.98,
    )
    numeric = SourceScript(
        text="9",
        language="ja",
        backend="faster_whisper",
        start=13.0,
        end=13.5,
        confidence=0.98,
    )
    long_filler = SourceScript(
        text="ほら",
        language="ja",
        backend="faster_whisper",
        start=14.0,
        end=15.3,
        confidence=0.98,
    )

    payload = pipeline_steps._source_script_keep_original_texture_candidate(candidate)

    assert payload == {
        "action": "keep_original_texture",
        "reason": "asr_short_filler_keep_original_texture",
        "duration_sec": 0.36,
        "source_text": "あの",
        "normalized_source_text": "あの",
        "policy": "conservative_short_filler",
    }
    assert (
        pipeline_steps._source_script_keep_original_texture_candidate(repeated)["reason"]
        == "asr_short_filler_keep_original_texture"
    )
    assert pipeline_steps._source_script_keep_original_texture_candidate(meaningful) is None
    assert pipeline_steps._source_script_keep_original_texture_candidate(numeric) is None
    assert pipeline_steps._source_script_keep_original_texture_candidate(long_filler) is None


def test_transcribe_marks_short_filler_as_keep_original_texture(
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
            asr_repair_enabled=True,
            asr_review_enabled=True,
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class ShortFillerASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=0.36,
                    text="あの",
                    language="ja",
                    confidence=0.98,
                )
            ]

        def transcribe_with_options(
            self,
            _audio_path: Path,
            _segments: list[Segment],
            **_kwargs: object,
        ) -> list[ASRChunk]:
            raise AssertionError("short filler texture should not consume repair/review candidates")

    monkeypatch.setattr(
        pipeline_steps,
        "create_asr_backend",
        lambda *_args, **_kwargs: ShortFillerASRBackend(),
    )

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")

    segment = manifest.segments[0]
    assert segment.status == "non_speech_texture"
    assert segment.keep_original_texture is True
    assert segment.errors == ["asr_short_filler_keep_original_texture"]
    keep_original_payload = segment.analysis["candidate_keep_original_texture"]
    assert keep_original_payload["action"] == "keep_original_texture"
    assert keep_original_payload["reason"] == "asr_short_filler_keep_original_texture"
    assert keep_original_payload["source_text"] == "あの"
    assert keep_original_payload["normalized_source_text"] == "あの"
    assert keep_original_payload["policy"] == "conservative_short_filler"
    assert keep_original_payload["duration_sec"] == pytest.approx(
        segment.source_script.end - segment.source_script.start
    )
    assert segment.analysis["asr_quality_gate"] == {
        "decision": "texture",
        "reasons": ["asr_short_filler_keep_original_texture"],
        "tts_blocked": True,
    }
    rows = [
        json.loads(line)
        for line in Path(manifest.artifacts["source_segments"]).read_text("utf-8").splitlines()
    ]
    assert rows[0]["status"] == "non_speech_texture"
    summary = json.loads(Path(manifest.artifacts["asr_diagnostics_summary"]).read_text("utf-8"))
    assert summary["non_speech_texture"] == 1
    assert summary["no_speech_detected"] == 1
    assert summary["needs_manual_review"] == 0


def test_transcribe_records_asr_quality_gate_for_suspicious_text(
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
            asr_review_enabled=False,
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class SuspiciousASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text="チンジンの先頭を撫でます",
                    language="ja",
                    confidence=0.98,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: SuspiciousASRBackend())

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")

    assert manifest.segments[0].status == "needs_manual_review"
    gate = manifest.segments[0].analysis["asr_quality_gate"]
    assert gate["decision"] == "block_tts"
    assert gate["tts_blocked"] is True
    assert gate["reasons"] == ["asr_suspicious_pattern:チンジン"]


def test_transcribe_rerun_clears_resolved_asr_quality_gate(
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
            asr_review_enabled=False,
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class SequenceASRBackend:
        name = "faster_whisper"
        calls = 0

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            self.calls += 1
            text = "チンジンの先頭を撫でます" if self.calls == 1 else "ジンジンの先端を撫でます"
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text=text,
                    language="ja",
                    confidence=0.98,
                )
            ]

    backend = SequenceASRBackend()
    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: backend)

    first = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")
    assert first.segments[0].status == "needs_manual_review"
    assert any(error.startswith("asr_suspicious_pattern:") for error in first.segments[0].errors)

    second = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")

    assert second.segments[0].status == "transcribed"
    assert second.segments[0].analysis["asr_quality_gate"] == {
        "decision": "pass",
        "reasons": [],
        "tts_blocked": False,
    }
    assert not any(error.startswith("asr_suspicious_pattern:") for error in second.segments[0].errors)
    rows = [
        json.loads(line)
        for line in Path(second.artifacts["source_segments"]).read_text("utf-8").splitlines()
    ]
    assert rows[0]["status"] == "transcribed"


def test_transcribe_blocks_observed_garbled_domain_phrase_before_tts(
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
            asr_review_enabled=False,
            source_separation_backend="none",
        ),
        tmp_project_dir / "pipeline.yaml",
    )

    class GarbledASRBackend:
        name = "faster_whisper"

        def transcribe(self, _audio_path: Path, _segments: list[Segment]) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text="意識が揺らぐ マンクロイプ 暗",
                    language="ja",
                    confidence=0.96,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: GarbledASRBackend())

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper")

    assert manifest.segments[0].status == "needs_manual_review"
    gate = manifest.segments[0].analysis["asr_quality_gate"]
    assert gate["decision"] == "block_tts"
    assert gate["tts_blocked"] is True
    assert gate["reasons"] == ["asr_suspicious_pattern:マンクロイプ"]


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


def test_asr_audio_review_continues_after_invalid_item(
    monkeypatch: pytest.MonkeyPatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_review_enabled=True,
        asr_review_backend="llama_server_audio",
        asr_review_audio_padding_sec=0.05,
        gemma_text_server_auto_start=False,
    )
    chunks = [
        ASRChunk(
            start=0.0,
            end=0.4,
            text="もっと大きな手帳が来る",
            language="ja",
            confidence=0.99,
        ),
        ASRChunk(
            start=0.5,
            end=0.9,
            text="もっと大きな手帳が来る",
            language="ja",
            confidence=0.99,
        ),
    ]
    review_calls: list[str] = []

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
            chunk_id = str(items[0]["chunk_id"])
            review_calls.append(chunk_id)
            if chunk_id == "chunk_0001":
                raise RuntimeError(
                    "Gemma ASR audio review failed: "
                    "chunk_0001: decision manual_review requires selected_candidate_id original"
                )
            return {
                chunk_id: {
                    "chunk_id": chunk_id,
                    "decision": "replace",
                    "selected_candidate_id": "domain_replacement",
                    "confidence": 0.99,
                    "reason": "audio supports 絶頂.",
                    "risk_terms": ["手帳"],
                }
            }

    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    reviewed_chunks, summary = pipeline_steps._review_asr_chunks_with_model(
        chunks,
        backend=object(),
        project_dir=tmp_project_dir,
        review_audio_path=tiny_wav_path,
        audio_duration_sec=1.0,
        cfg=cfg,
    )

    assert review_calls == ["chunk_0001", "chunk_0002"]
    assert summary["attempted"] == 2
    assert summary["failed"] == 1
    assert summary["manual_review"] == 1
    assert summary["replaced"] == 1
    assert "1 ASR review item(s) failed" in summary["error"]
    assert summary["items"][0]["decision"] == "manual_review"
    assert summary["items"][0]["selected_candidate_id"] == "original"
    assert summary["items"][0]["accepted"] is False
    assert "decision manual_review requires selected_candidate_id original" in summary["items"][0]["reason"]
    assert reviewed_chunks[0].text == "もっと大きな手帳が来る"
    assert reviewed_chunks[1].text == "もっと大きな絶頂が来る"


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


def test_asr_prompt_leak_filter_rejects_prompt_term_list() -> None:
    cfg = ProjectConfig(
        project_name="test-project",
        asr_initial_prompt=(
            "Japanese ASMR domain terms: 快感 快感蓄積 快感増幅 快感の波 "
            "気持ちいい レーザー 子宮 悪夢ノイド"
        ),
    )

    assert (
        pipeline_steps._asr_candidate_looks_prompt_leaked(
            "気持ちいい レーザー 子宮 悪夢ノイド",
            cfg,
        )
        is True
    )
    assert (
        pipeline_steps._asr_candidate_looks_prompt_leaked(
            "すごく気持ちいいね 体がぽかぽかしてきた",
            cfg,
        )
        is False
    )


def test_asr_prompt_leak_filter_rejects_qwen_context_text() -> None:
    cfg = ProjectConfig(project_name="test-project")
    text = cfg.asr.correction_profile.qwen_context

    assert pipeline_steps._asr_candidate_looks_prompt_leaked(text, cfg) is True
    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text=text,
            language="ja",
            backend="qwen_asr",
            start=1370.915,
            end=1384.353,
        ),
        cfg,
    ) == ["asr_prompt_or_hallucination_leak"]


def test_asr_prompt_leak_filter_rejects_compacted_qwen_context_fragment() -> None:
    cfg = ProjectConfig(project_name="test-project")

    assert (
        pipeline_steps._asr_candidate_looks_prompt_leaked(
            "Japanesewhisperingdialogueintimateadultoverdomainassumptions",
            cfg,
        )
        is True
    )


def test_asr_prompt_leak_filter_allows_dialogue_thanks_followed_by_story_text() -> None:
    cfg = ProjectConfig(project_name="test-project")
    text = "ありがとうございましたよしお前たち明日は今日よりもっと可愛くていやらしい女の子うん"

    assert pipeline_steps._asr_candidate_looks_prompt_leaked(text, cfg) is False
    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text=text,
            language="ja",
            backend="faster_whisper",
            start=4554.03,
            end=4565.755,
        ),
        cfg,
    ) == []


def test_asr_hallucination_filter_keeps_dialogue_thanks_but_drops_video_outro() -> None:
    cfg = ProjectConfig(project_name="test-project")
    chunks = [
        ASRChunk(
            start=0.0,
            end=8.0,
            text="ありがとうございましたよしお前たち明日は今日よりもっと可愛くて",
            language="ja",
            confidence=0.98,
        ),
        ASRChunk(
            start=8.0,
            end=14.0,
            text="ご視聴ありがとうございました チャンネル登録 高評価お願いします",
            language="ja",
            confidence=0.98,
        ),
        ASRChunk(
            start=14.0,
            end=28.0,
            text="最後までご覧いただきありがとうございます。",
            language="ja",
            confidence=0.98,
        ),
    ]

    kept, dropped = pipeline_steps._filter_final_asr_chunks_for_hallucinations(chunks, cfg=cfg)

    assert kept == [chunks[0]]
    assert len(dropped) == 2
    assert {item["reason"] for item in dropped} == {"repeated_outro_hallucination"}


def test_asr_review_replacements_include_observed_adult_asr_artifacts() -> None:
    cfg = ProjectConfig(project_name="test-project")

    assert cfg.asr_review_candidate_replacements["めず行きセックス"] == "メスイキセックス"
    assert cfg.asr_review_candidate_replacements["薄引き"] == "メスイキ"
    assert cfg.asr_review_candidate_replacements["グリドリス"] == "クリトリス"
    assert cfg.asr_review_candidate_replacements["お孫"] == "おまんこ"
    assert cfg.asr_review_candidate_replacements["生体コアの高校"] == "生体コアの口腔"
    assert cfg.asr_review_candidate_replacements["高校に放出"] == "口腔に放出"
    assert cfg.asr_review_candidate_replacements["静観体"] == "性感帯"
    assert "めず行き" in cfg.asr_review_suspicious_text_patterns
    assert "生体コアの高校" in cfg.asr_review_suspicious_text_patterns
    assert "静観体" in cfg.asr_review_suspicious_text_patterns


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


def test_translation_asr_backcheck_allows_legitimate_biyakjeok_translation() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text="飛躍的な快楽",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=1.0,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="비약적인 쾌락",
        ko_natural="비약적인 쾌락이에요.",
        model="mock",
        batch_id="batch_0001",
    )

    items = pipeline_steps._apply_translation_asr_backcheck(
        [segment],
        ProjectConfig(project_name="test"),
    )

    assert items == []
    assert not any("ASR translation backcheck" in error for error in segment.errors)


def test_translation_asr_backcheck_allows_legitimate_biyaku_translation() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text="メス媚薬も効いてきています",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=1.0,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="암컷 미약도 듣고 있습니다",
        ko_natural="암컷 미약도 듣고 있어요.",
        model="mock",
        batch_id="batch_0001",
    )

    items = pipeline_steps._apply_translation_asr_backcheck(
        [segment],
        ProjectConfig(project_name="test"),
    )

    assert items == []


def test_translation_asr_backcheck_allows_legitimate_taikan_translation() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text="体感レベル150 生体コアの限界をはるかに超えていますが",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=1.0,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="체감 레벨 백오십 생체 코어의 한계를 훨씬 넘어서고 있습니다만",
        ko_natural="체감 레벨 일백오십, 생체 코어의 한계를 훨씬 넘었지만,",
        model="mock",
        batch_id="batch_0001",
    )

    items = pipeline_steps._apply_translation_asr_backcheck(
        [segment],
        ProjectConfig(project_name="test"),
    )

    assert items == []
    assert not any("ASR translation backcheck" in error for error in segment.errors)


def test_translation_acceptance_warns_on_japanese_word_internal_segment_split() -> None:
    left = sample_segment("seg_0001", start=0.0, end=1.0)
    left.status = "needs_manual_review"
    left.errors.append("Korean translation rejected before TTS: source_split_inside_japanese_word")
    left.source_script = SourceScript(
        text="この世は恥",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=1.0,
    )
    left.translation_ko = KoreanTranslation(
        ko_literal="이 세상은 부끄",
        ko_natural="이 세상은 온통 부끄",
        model="mock",
        batch_id="batch_0001",
    )
    right = sample_segment("seg_0002", start=1.0, end=2.0)
    right.status = "needs_manual_review"
    right.errors.append("Korean translation rejected before TTS: source_split_inside_japanese_word")
    right.source_script = SourceScript(
        text="ずかしいことでいっぱい",
        language="ja",
        backend="faster_whisper",
        start=1.0,
        end=2.0,
    )
    right.translation_ko = KoreanTranslation(
        ko_literal="러운 일로 가득",
        ko_natural="러운 일들로 가득해요.",
        model="mock",
        batch_id="batch_0001",
    )
    rows = [
        {
            "batch_id": "batch_0001",
            "segment_id": left.id,
            "status": "translated",
            "translation_ko": left.translation_ko.model_dump(mode="json"),
        },
        {
            "batch_id": "batch_0001",
            "segment_id": right.id,
            "status": "translated",
            "translation_ko": right.translation_ko.model_dump(mode="json"),
        },
    ]
    quality_counters: Counter[str] = Counter()

    pipeline_steps._finalize_translation_acceptance(
        rows,
        [left, right],
        [],
        quality_counters,
    )

    assert [row["status"] for row in rows] == ["translated", "translated"]
    assert quality_counters["source_split_inside_japanese_word"] == 2
    assert rows[0]["quality_issues"] == []
    assert rows[1]["quality_issues"] == []
    assert rows[0]["quality_warnings"] == ["source_split_inside_japanese_word"]
    assert rows[1]["quality_warnings"] == ["source_split_inside_japanese_word"]
    assert left.status == "raw"
    assert right.status == "raw"
    assert left.errors == []
    assert right.errors == []


def test_source_split_inside_japanese_word_ignores_common_sentence_boundaries() -> None:
    assert pipeline_steps._source_split_inside_japanese_word(
        "もちろん逆もしかり あなたはとても優しい人",
        "だからこそ 誰かがそばにいてあげないと",
    ) is False
    assert pipeline_steps._source_split_inside_japanese_word(
        "その変態的状況にさらに興奮していた",
        "気持ちいい",
    ) is False
    assert pipeline_steps._source_split_inside_japanese_word(
        "この世は恥",
        "ずかしいことでいっぱいよ",
    ) is True


def test_translation_acceptance_rejects_natural_pass_that_omits_source_repetition() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text=(
            "一生懸命締め付けますので私の穴を使って気持ちよくなってください"
            "一生懸命締め付けますので私の穴を使って気持ちよくなってください"
        ),
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=1.0,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal=(
            "열심히 조여드릴 테니까 제 구멍을 써서 기분 좋아져 주세요. "
            "열심히 조여드릴 테니까 제 구멍을 써서 기분 좋아져 주세요."
        ),
        ko_natural="열심히 조여드릴 테니까 제 구멍을 써서 기분 좋아져 주세요.",
        model="mock",
        batch_id="batch_0001",
    )
    rows = [
        {
            "batch_id": "batch_0001",
            "segment_id": segment.id,
            "status": "translated",
            "translation_ko": segment.translation_ko.model_dump(mode="json"),
        }
    ]
    quality_counters: Counter[str] = Counter()

    pipeline_steps._finalize_translation_acceptance(
        rows,
        [segment],
        [],
        quality_counters,
        cfg=ProjectConfig(gemma_text_repetition_omission_policy="manual_review"),
    )

    assert rows[0]["status"] == "needs_manual_review"
    assert rows[0]["quality_issues"] == ["natural_repetition_omission"]
    assert quality_counters["natural_repetition_omission"] == 1


def test_translation_acceptance_warns_repetition_omission_by_default() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text=(
            "一生懸命締め付けますので私の穴を使って気持ちよくなってください"
            "一生懸命締め付けますので私の穴を使って気持ちよくなってください"
        ),
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=1.0,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal=(
            "열심히 조여드릴 테니까 제 구멍을 써서 기분 좋아져 주세요. "
            "열심히 조여드릴 테니까 제 구멍을 써서 기분 좋아져 주세요."
        ),
        ko_natural="열심히 조여드릴 테니까 제 구멍을 써서 기분 좋아져 주세요.",
        model="mock",
        batch_id="batch_0001",
    )
    rows = [
        {
            "batch_id": "batch_0001",
            "segment_id": segment.id,
            "status": "translated",
            "translation_ko": segment.translation_ko.model_dump(mode="json"),
        }
    ]
    quality_counters: Counter[str] = Counter()

    pipeline_steps._finalize_translation_acceptance(rows, [segment], [], quality_counters)

    assert segment.status != "needs_manual_review"
    assert rows[0]["status"] == "translated"
    assert rows[0]["quality_issues"] == []
    assert rows[0]["quality_warnings"] == ["natural_repetition_omission"]
    assert rows[0]["accepted"] is True
    assert segment.analysis["translation_auto_fallback"]["reason"] == (
        "natural_repetition_omission_warn_only"
    )
    assert quality_counters["natural_repetition_omission"] == 1


def test_translation_acceptance_allows_parallel_question_repetition() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text=(
            "液体の浮力があなたを包み込んで 上を向いているのか下を向いているのか "
            "全くわからないほど ふわふわ ふわふわと漂っています"
        ),
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=13.4,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal=(
            "액체의 부력이 당신을 감싸 안아 위를 향하고 있는지 아래를 향하고 있는지 "
            "전혀 알 수 없을 정도로 둥실둥실 둥실둥실 떠다니고 있습니다"
        ),
        ko_natural="액체의 부력이 당신을 감싸 안아서, 위인지 아래인지 알 수 없을 만큼 둥실둥실 떠다니고 있어요.",
        model="mock",
        batch_id="batch_0001",
    )
    rows = [
        {
            "batch_id": "batch_0001",
            "segment_id": segment.id,
            "status": "translated",
            "translation_ko": segment.translation_ko.model_dump(mode="json"),
        }
    ]
    quality_counters: Counter[str] = Counter()

    pipeline_steps._finalize_translation_acceptance(rows, [segment], [], quality_counters)

    assert rows[0]["status"] == "translated"
    assert rows[0]["quality_issues"] == []
    assert quality_counters["natural_repetition_omission"] == 0


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


def test_asr_resegment_prefers_japanese_text_boundaries_for_long_chunks(
    tmp_project_dir: Path,
) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(
                start=0.0,
                end=29.18,
                text=(
                    "君は女の子なんだもの 男性なら平気なことでも"
                    "可愛い女の子の君にとってはこの世は恥ずかしいことでいっぱいよ"
                    "さあ 全てを受け入れて生まれ変わった自分を楽しみましょう"
                ),
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

    texts = [segment.source_script.text for segment in segments if segment.source_script]

    assert len(texts) == 2
    assert all(text != "ずかしいことでいっぱいよさあ 全てを受け入れて生まれ変わった自分を楽しみましょう" for text in texts)
    assert all(not text.endswith("恥") for text in texts)
    assert "恥ずかしいことでいっぱい" in " ".join(texts)


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


def test_asr_repair_flags_long_numeric_only_and_numeric_runaway_chunks() -> None:
    assert pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=18.0, text="10 9 8 7 6 5 4 3 2 1 0", language="ja", confidence=0.98),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )
    assert pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=2664.718, end=2690.518, text="五 " * 48, language="ja", confidence=0.96),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )
    assert pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=6.0, text="五 五 五 五 五 五", language="ja", confidence=0.98),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )
    assert not pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=0.6, text="三", language="ja", confidence=0.2),
        confidence_threshold=0.94,
        sparse_min_sec=12.0,
        sparse_min_chars_per_sec=1.0,
    )
    assert not pipeline_steps._asr_chunk_needs_repair(
        ASRChunk(start=0.0, end=0.9, text="ゼロ", language="ja", confidence=0.2),
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
        cfg=ProjectConfig(
            project_name=tmp_project_dir.name,
            asr_repair_confidence_threshold=0.94,
        ),
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


def test_asr_repair_rejects_short_text_change_without_candidate_vote(
    tmp_project_dir: Path,
) -> None:
    repair_audio = tmp_project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    write_audio(repair_audio, np.zeros((12 * 16_000, 1), dtype=np.float32), 16_000)
    original = ASRChunk(start=10.0, end=10.56, text="って", language="ja", confidence=0.5)

    class FakeBackend:
        def transcribe_with_options(
            self,
            audio_path: Path,
            _segments: list[Segment],
            **_kwargs: object,
        ) -> list[ASRChunk]:
            if "vad_no_prompt" in audio_path.name:
                return [ASRChunk(start=1.0, end=1.56, text="ステーク", language="ja", confidence=0.95)]
            return [ASRChunk(start=1.0, end=1.56, text="で", language="ja", confidence=0.95)]

    repaired, summary = pipeline_steps._repair_asr_chunks(
        [original],
        backend=FakeBackend(),
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=12.0,
        cfg=ProjectConfig(project_name=tmp_project_dir.name),
    )

    assert repaired == [original]
    assert summary["repaired"] == 0
    assert summary["items"][0]["accepted"] is False
    assert summary["items"][0]["reject_reason"] == "candidate_vote_required"


def test_asr_repair_uses_plain_transcribe_backend_when_option_overrides_are_unavailable(
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

    class PlainBackend:
        name = "qwen_asr"

        def __init__(self) -> None:
            self.calls: list[Path] = []

        def transcribe(self, audio_path: Path, segments: list[Segment]) -> list[ASRChunk]:
            self.calls.append(audio_path)
            assert segments == []
            return [
                ASRChunk(
                    start=1.0,
                    end=5.0,
                    text="ゾクゾクしますね",
                    language="ja",
                    confidence=None,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    backend = PlainBackend()
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
    assert summary["skipped"] == 0
    assert summary["repaired"] == 1
    assert summary["items"][0]["accepted_candidate_id"] == "plain_transcribe"
    assert repaired[0].text == "ゾクゾクしますね"
    assert repaired[0].start == pytest.approx(10.0)
    assert repaired[0].end == pytest.approx(14.0)
    assert sliced_paths
    assert len(backend.calls) == 1


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


def test_asr_repair_skips_qwen_fallback_when_local_candidate_is_accepted(
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
        name = "faster_whisper"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments, kwargs
            self.calls += 1
            return [
                ASRChunk(
                    start=1.0,
                    end=2.0,
                    text="ゾクゾクしますね",
                    language="ja",
                    confidence=0.89,
                )
            ]

    class FakeQwenFallback:
        name = "qwen_asr"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path: Path, segments: list[Segment]) -> list[ASRChunk]:
            _ = audio_path, segments
            self.calls += 1
            return [
                ASRChunk(
                    start=1.0,
                    end=2.0,
                    text="ゾクゾクして止まりません",
                    language="ja",
                    confidence=None,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    backend = FakeBackend()
    qwen_fallback = FakeQwenFallback()
    repaired, summary = pipeline_steps._repair_asr_chunks(
        [
            ASRChunk(
                start=4.0,
                end=5.0,
                text="付属しますね",
                language="ja",
                confidence=0.90,
            )
        ],
        backend=backend,
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=8.0,
        cfg=ProjectConfig(
            project_name=tmp_project_dir.name,
            asr_repair_confidence_threshold=0.94,
        ),
        qwen_fallback_backend=qwen_fallback,
    )

    assert qwen_fallback.calls == 0
    assert summary["items"][0]["accepted_candidate_id"] == "no_vad_clean"
    assert repaired[0].text == "ゾクゾクしますね"


def test_asr_repair_rejects_qwen_fallback_that_remains_suspicious(
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

    class EmptyLocalBackend:
        name = "faster_whisper"

        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments, kwargs
            return []

    class SuspiciousQwenFallback:
        name = "qwen_asr"

        def transcribe(self, audio_path: Path, segments: list[Segment]) -> list[ASRChunk]:
            _ = audio_path, segments
            return [
                ASRChunk(
                    start=1.0,
                    end=4.0,
                    text="悪夢し続けます",
                    language="ja",
                    confidence=None,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    original = ASRChunk(start=4.0, end=8.0, text="悪夢していいですよ", language="ja", confidence=0.82)
    repaired, summary = pipeline_steps._repair_asr_chunks(
        [original],
        backend=EmptyLocalBackend(),
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=12.0,
        cfg=ProjectConfig(project_name=tmp_project_dir.name),
        qwen_fallback_backend=SuspiciousQwenFallback(),
    )

    assert repaired == [original]
    assert summary["repaired"] == 0
    assert summary["items"][0]["accepted"] is False
    assert summary["items"][0]["attempts"][-1]["candidate_id"] == "qwen_asr_fallback"
    assert summary["items"][0]["attempts"][-1]["reason"] == "qwen_fallback_still_suspicious"


def test_asr_repair_accepts_local_candidate_after_domain_replacement_review(
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

    class SuspiciousLocalBackend:
        name = "faster_whisper"

        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments, kwargs
            return [
                ASRChunk(
                    start=1.0,
                    end=4.0,
                    text="悪夢し続けます",
                    language="ja",
                    confidence=0.95,
                )
            ]

    monkeypatch.setattr(pipeline_steps.ffmpeg, "slice_audio", fake_slice_audio)
    original = ASRChunk(start=4.0, end=8.0, text="悪夢していいですよ", language="ja", confidence=0.82)
    repaired, summary = pipeline_steps._repair_asr_chunks(
        [original],
        backend=SuspiciousLocalBackend(),
        project_dir=tmp_project_dir,
        repair_audio_path=repair_audio,
        audio_duration_sec=12.0,
        cfg=ProjectConfig(project_name=tmp_project_dir.name),
    )

    assert repaired[0].text == "悪夢し続けます"
    assert summary["repaired"] == 1
    assert summary["items"][0]["accepted"] is True
    assert summary["items"][0]["attempts"][0]["reason"] == "accepted"


def test_asr_review_rejects_selected_candidate_that_still_needs_review(tmp_project_dir: Path) -> None:
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_review_enabled=True,
        asr_review_backend="mock",
        asr_review_generate_candidates=False,
        asr_review_suspicious_text_patterns=["もちなとい", "悪夢し"],
        asr_review_candidate_replacements={"もちなとい": "悪夢し続けます"},
    )
    chunks = [
        ASRChunk(
            start=0.0,
            end=4.0,
            text="もちなとい",
            language="ja",
            confidence=0.96,
        )
    ]

    reviewed, summary = pipeline_steps._review_asr_chunks_with_model(
        chunks,
        backend=object(),
        project_dir=tmp_project_dir,
        review_audio_path=tmp_project_dir / "missing.wav",
        audio_duration_sec=5.0,
        cfg=cfg,
    )

    assert reviewed[0].text == "もちなとい"
    assert summary["replaced"] == 0
    assert summary["manual_review"] == 1
    assert summary["items"][0]["accepted"] is False
    assert summary["items"][0]["blocked_review_reasons"] == [
        "asr_suspicious_pattern:悪夢し"
    ]


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


def test_asr_repair_splits_very_long_sparse_chunk_with_short_text() -> None:
    chunk = ASRChunk(
        start=100.0,
        end=940.0,
        text="ほら見て ほら見てってば",
        language="ja",
        confidence=0.97,
    )

    repair_chunks = pipeline_steps._split_asr_chunks_for_repair(
        [chunk],
        audio_duration_sec=1000.0,
        max_chunk_sec=20.0,
    )

    assert len(repair_chunks) > 1
    assert repair_chunks[0].start == pytest.approx(100.0)
    assert repair_chunks[-1].end == pytest.approx(940.0)
    assert " ".join(part.text for part in repair_chunks).replace(" ", "") == chunk.text.replace(" ", "")
    assert max(part.end - part.start for part in repair_chunks) < chunk.end - chunk.start


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
        cfg=ProjectConfig(project_name=tmp_project_dir.name, asr_resegment_max_sec=20.0),
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
        cfg=ProjectConfig(project_name=tmp_project_dir.name, asr_resegment_max_sec=20.0),
    )

    assert summary["attempted"] == 1
    assert summary["repaired"] == 0
    assert summary["items"][0]["accepted"] is False
    assert summary["items"][0]["prompt_leaked"] is True
    assert repaired == [original]


def test_asr_repair_rejects_degenerate_repetition_candidate() -> None:
    cfg = ProjectConfig(project_name="test-project")
    original = ASRChunk(start=10.0, end=23.0, text="アクメの全", language="ja", confidence=0.86)
    candidate = ASRChunk(
        start=10.0,
        end=10.6,
        text="ビ" + "ー" * 120,
        language="ja",
        confidence=0.99,
    )

    accepted, _score, reason = pipeline_steps._asr_repair_candidate_score(
        original,
        [candidate],
        cfg=cfg,
        prompt_leaked=False,
    )

    assert accepted is False
    assert reason == "degenerate_repetition"


def test_asr_repair_rejects_numeric_runaway_candidate() -> None:
    cfg = ProjectConfig(project_name="test-project")
    original = ASRChunk(
        start=8249.82,
        end=8263.58,
        text="27、28、29、30、31、32、33、34、35、36、37、38、39、40、41、42、43、44、45、46、47、48、49、50",
        language="ja",
        confidence=0.90,
    )
    candidate = ASRChunk(
        start=8249.82,
        end=8263.58,
        text="15,14,13,14,15,16,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57,58,59,60,61,62,62,63,63,64,67,68,79,80,81,81,82,82",
        language="ja",
        confidence=0.96,
    )

    accepted, _score, reason = pipeline_steps._asr_repair_candidate_score(
        original,
        [candidate],
        cfg=cfg,
        prompt_leaked=False,
    )

    assert accepted is False
    assert reason == "numeric_runaway"


def test_asr_review_flags_degenerate_repetition_and_numeric_runaway() -> None:
    cfg = ProjectConfig(project_name="test-project")

    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="ビ" + "ー" * 120,
            language="ja",
            backend="faster_whisper",
            start=7866.22,
            end=7866.8,
            confidence=0.99,
        ),
        cfg,
    ) == ["asr_degenerate_repetition"]
    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="15,14,13,14,15,16,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,56,57,58,59,60,61,62,62,63,63,64,67,68,79,80,81,81,82,82",
            language="ja",
            backend="faster_whisper",
            start=8263.58,
            end=8277.34,
            confidence=0.92,
        ),
        cfg,
    ) == ["asr_numeric_runaway"]


def test_asr_review_flags_short_repeated_countdown_runaway() -> None:
    cfg = ProjectConfig(project_name="test-project")

    reasons = pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="自動連続悪夢まであと15 15 15 15 15 15",
            language="ja",
            backend="qwen_asr",
            start=120.0,
            end=126.0,
            confidence=0.92,
        ),
        cfg,
    )

    assert "asr_numeric_runaway" in reasons


def test_asr_review_keeps_interleaved_countdown_as_warning_not_manual_review() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="10 さあ 9 頑張りましょうね 8 体制を整えて 7 反応するんです 5 ほら 1 0 射精します",
        language="ja",
        backend="faster_whisper",
        start=8121.02,
        end=8137.96,
        confidence=0.92,
    )

    assert pipeline_steps._source_script_asr_review_reasons(source_script, cfg) == []
    assert (
        pipeline_steps._source_script_countdown_unverified_reason(source_script)
        == "asr_countdown_unverified"
    )


def test_asr_review_treats_dominant_moan_repetition_as_texture() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="イクイクイク" + "オ" * 80,
        language="ja",
        backend="faster_whisper",
        start=3045.23,
        end=3055.23,
        confidence=0.91,
    )

    assert (
        pipeline_steps._source_script_non_speech_texture_reason(source_script)
        == "asr_non_speech_texture"
    )
    assert pipeline_steps._source_script_asr_review_reasons(source_script, cfg) == []


def test_asr_review_downgrades_plausible_sparse_speech_to_warning() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="今の姿勢のまま",
        language="ja",
        backend="faster_whisper",
        start=9168.787,
        end=9183.54,
        confidence=0.94,
    )

    assert pipeline_steps._source_script_asr_review_reasons(source_script, cfg) == []
    assert (
        pipeline_steps._source_script_sparse_speech_unverified_reason(source_script, cfg)
        == "asr_sparse_speech_unverified"
    )


def test_asr_review_keeps_random_sparse_word_list_blocked() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="雨 桜 東 ドラム",
        language="ja",
        backend="faster_whisper",
        start=516.17,
        end=534.106,
        confidence=0.94,
    )

    assert pipeline_steps._source_script_asr_review_reasons(source_script, cfg) == [
        "asr_sparse_text_density"
    ]
    assert pipeline_steps._source_script_sparse_speech_unverified_reason(source_script, cfg) is None


def test_asr_review_downgrades_long_numeric_sequence_to_warning() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="296 295 294",
        language="ja",
        backend="faster_whisper",
        start=797.07,
        end=811.593,
        confidence=0.93,
    )

    assert pipeline_steps._source_script_asr_review_reasons(source_script, cfg) == []
    assert (
        pipeline_steps._source_script_numeric_sequence_unverified_reason(source_script)
        == "asr_numeric_sequence_unverified"
    )


def test_asr_repair_rejection_is_suppressed_for_warning_only_source() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="可愛らしい反応です",
        language="ja",
        backend="faster_whisper",
        start=1128.73,
        end=1144.615,
        confidence=0.95,
    )

    assert pipeline_steps._filter_asr_repair_review_reasons(
        source_script,
        cfg,
        review_reasons=[],
        repair_review_reasons=["asr_repair_rejected:prompt_or_hallucination_leak"],
    ) == ["asr_repair_rejected:prompt_or_hallucination_leak"]


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


def test_asr_text_replacements_include_observed_rj01410718_residuals() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=4773.383,
                end=4777.203,
                text="これが 深い女の悪夢",
                language="ja",
                confidence=0.98,
            ),
            ASRChunk(
                start=5251.733,
                end=5255.473,
                text="君が望んだ悪夢を受け入れていきたいよね",
                language="ja",
                confidence=0.96,
            ),
            ASRChunk(
                start=5473.833,
                end=5486.033,
                text="0 ほら りくんといく 私の手にもで 脈動が伝わってるよ",
                language="ja",
                confidence=0.93,
            ),
            ASRChunk(
                start=3576.265,
                end=3587.595,
                text="すぐやりたがる 陰難 陣法の苦戦女子なのに",
                language="ja",
                confidence=0.96,
            ),
            ASRChunk(
                start=1202.604,
                end=1207.064,
                text="紙をまとめるためのネットが",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=197.885,
                end=208.82,
                text="きやちょっと違うの",
                language="ja",
                confidence=0.95,
            ),
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 6
    assert [chunk.text for chunk in chunks] == [
        "これが 深い女のアクメ",
        "君が望んだアクメを受け入れていきたいよね",
        "0 ほら びくんとイく 私の手にも 脈動が伝わってるよ",
        "すぐやりたがる 淫乱 チンポの苦戦女子なのに",
        "髪をまとめるためのネットが",
        "いやちょっと違うの",
    ]


def test_observed_akume_residual_replacements_still_keep_true_nightmare_context() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="怖い女の悪夢を見て眠れない",
                language="ja",
                confidence=0.96,
            )
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 0
    assert chunks[0].text == "怖い女の悪夢を見て眠れない"


def test_asr_review_flags_observed_unresolved_rj01410718_fragments() -> None:
    cfg = ProjectConfig()

    reasons = pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="天群と心臓の鼓動に合わせて 吸って 吸うできる ごすーっとかぶさって読む",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=12.0,
            confidence=0.96,
        ),
        cfg,
    )

    assert reasons == ["asr_suspicious_pattern:天群,吸うできる,ごすーっとかぶさって読む"]


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


def test_asr_text_replacements_fix_observed_seikantai_mishear() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=3980.882,
                end=3987.782,
                text="全身をまさぐったり、静観体を刺激したり、足をもじもじしたりして、もだえることができます。",
                language="ja",
                confidence=0.98,
            )
        ],
        cfg.asr_text_replacements,
    )

    assert cfg.asr_text_replacements["静観体"] == "性感帯"
    assert summary["chunks_changed"] == 1
    assert summary["items"][0]["hits"] == [{"source": "静観体", "target": "性感帯", "count": 1}]
    assert (
        chunks[0].text
        == "全身をまさぐったり、性感帯を刺激したり、足をもじもじしたりして、もだえることができます。"
    )


def test_asr_text_replacements_include_observed_real_run_mishears() -> None:
    replacements = ProjectConfig().asr_text_replacements
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="メスイキ悪夢決めたい メスイキアカメが止まらない",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="これが私の助走 助走しながら 助走して また助走をしたくなったら",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=8.0,
                end=12.0,
                text="フェラチをして 気筒を舐め回すと 死を吹く",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=12.0,
                end=16.0,
                text="左の口岸が入ってきた スペンス入線 スペンス乳腺",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=16.0,
                end=20.0,
                text="処生できるよ 絶頂までのパウントダウン",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=20.0,
                end=24.0,
                text="幸せが足寄せて 成長を超える もう来る 成長が来る トロゲル",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=24.0,
                end=28.0,
                text="二なりの皆さん メス火薬で おっぱいも子宮も薄きっぱなし 陳腐を待っている",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=28.0,
                end=32.0,
                text="生きかけてる 息っぱなしや 息まくり",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=32.0,
                end=36.0,
                text="女体火薬で 全部くっぷくさせて 一年来る オナホシキューが負け息してる",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=36.0,
                end=40.0,
                text="こんなに窒内に出されたら 先ぽ ゴリゴリするの くりすこすこして 狩りの横の溝 私のフェラガを見て 愛部する",
                language="ja",
                confidence=0.92,
            ),
        ],
        replacements,
    )

    assert summary["chunks_changed"] == 10
    assert chunks[0].text == "メスイキアクメ決めたい メスイキアクメが止まらない"
    assert chunks[1].text == "これが私の女装 女装しながら 女装して また女装をしたくなったら"
    assert chunks[2].text == "フェラチオをして 亀頭を舐め回すと 潮を吹く"
    assert chunks[3].text == "左の睾丸が入ってきた スキーン腺 スキーン腺"
    assert chunks[4].text == "射精できるよ 絶頂までのカウントダウン"
    assert chunks[5].text == "幸せが押し寄せて 絶頂を超える もう来る 絶頂が来る とろける"
    assert chunks[6].text == "ふたなりの皆さん メス媚薬で おっぱいも子宮も疼きっぱなし チンポを待っている"
    assert chunks[7].text == "イキかけてる イキっぱなしや イキまくり"
    assert chunks[8].text == "女体化薬で 全部屈服させて 一気に来る オナホ子宮が負けイキしてる"
    assert chunks[9].text == "こんなに膣内に出されたら 先っぽ ゴリゴリするの クリをすこすこして カリの横の溝 私のフェラ顔を見て 愛撫する"


def test_contextual_asr_replacements_normalize_generic_akume_denma_and_countdown_residuals() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="私ももっと君の悪目顔見たいな",
                language="ja",
                confidence=0.93,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="すごい音でしょ電話の音 電話を挟むようにすると",
                language="ja",
                confidence=0.91,
            ),
            ASRChunk(
                start=8.0,
                end=10.0,
                text="ん…電話邪魔…",
                language="ja",
                confidence=0.82,
            ),
            ASRChunk(
                start=10.0,
                end=14.0,
                text="13 12 11 12",
                language="ja",
                confidence=0.9,
            ),
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 4
    assert summary["total_replacements"] == 5
    assert [chunk.text for chunk in chunks] == [
        "私ももっと君のアクメ顔見たいな",
        "すごい音でしょ電マの音 電マを挟むようにすると",
        "ん…電マ邪魔…",
        "13 12 11 10",
    ]


def test_countdown_replacements_repair_missing_number_and_duplicate_tail() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=8.0,
                text="10 9 8 7 6 4 3 2 1",
                language="ja",
                confidence=0.9,
            ),
            ASRChunk(
                start=8.0,
                end=16.0,
                text="10 9 8 7 6 5 4 3 2 1 1 1",
                language="ja",
                confidence=0.9,
            ),
            ASRChunk(
                start=16.0,
                end=20.0,
                text="1 0 1 0 0 0",
                language="ja",
                confidence=0.9,
            ),
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 2
    assert [chunk.text for chunk in chunks] == [
        "10 9 8 7 6 5 4 3 2 1",
        "10 9 8 7 6 5 4 3 2 1",
        "1 0 1 0 0 0",
    ]


def test_contextual_denma_replacement_keeps_true_phone_context() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="電話に出ると着信音がまだ鳴っていた",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="電話の電波が弱いみたい",
                language="ja",
                confidence=0.95,
            )
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 0
    assert chunks[0].text == "電話に出ると着信音がまだ鳴っていた"
    assert chunks[1].text == "電話の電波が弱いみたい"


def test_asr_review_patterns_do_not_flag_repaired_fellatio_term() -> None:
    cfg = ProjectConfig()

    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="フェラチオをして 亀頭を舐め回すと",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=3.0,
        ),
        cfg,
    ) == []
    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="フェラチをして 気筒を舐め回すと",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=3.0,
        ),
        cfg,
    ) == ["asr_suspicious_pattern:気筒,フェラチを"]


def test_asr_review_flags_observed_unresolved_monha_fragment() -> None:
    cfg = ProjectConfig()

    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="モンハの著しい乱れを 同時絶頂が近いものと推測されます",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=12.0,
        ),
        cfg,
    ) == ["asr_suspicious_pattern:モンハの著しい"]


def test_asr_review_flags_observed_terminal_clitoris_fragment() -> None:
    cfg = ProjectConfig()

    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="幸せを感じることだけができる世界 クリト",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=9.4,
        ),
        cfg,
    ) == ["asr_suspicious_pattern:クリト$"]
    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="幸せを感じることだけができる世界 クリトリス",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=9.4,
        ),
        cfg,
    ) == []


def test_asr_review_patterns_do_not_flag_train_strap_as_fishing() -> None:
    cfg = ProjectConfig()

    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="釣り革につかまると 背後に人の気配を感じた",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=3.0,
        ),
        cfg,
    ) == []
    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="あっという間に釣りが来ちゃう",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=3.0,
        ),
        cfg,
    ) == ["asr_suspicious_pattern:釣りが来"]


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


def test_contextual_asr_replacements_fix_observed_akume_mishears() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="本物の男性器ではありえない メスの快感を生み出し、激しく悪夢させるために作られた形",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="外観いいです いいですイーチ 悪夢していいですよ",
                language="ja",
                confidence=0.92,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 2
    assert summary["total_replacements"] == 2
    assert chunks[0].text == "本物の男性器ではありえない メスの快感を生み出し、激しくアクメさせるために作られた形"
    assert chunks[1].text == "外観いいです いいですイーチ アクメしていいですよ"


def test_contextual_asr_replacements_fix_machine_experiment_akume_mishears() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="悪夢の現象範囲を確認 自動連続悪夢まで",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="暴走した悪夢回路は淡々と あなたを悪夢させる 機械と融合しようとしている",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=8.0,
                end=12.0,
                text="ゼロ 悪夢します 減速悪夢計測 悪夢の前兆3分 自動連続悪夢になる",
                language="ja",
                confidence=0.92,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 3
    assert summary["total_replacements"] == 8
    assert chunks[0].text == "アクメの現象範囲を確認 自動連続アクメまで"
    assert chunks[1].text == "暴走したアクメ回路は淡々と あなたをアクメさせる 機械と融合しようとしている"
    assert chunks[2].text == "ゼロ アクメします 減速アクメ計測 アクメの前兆3分 自動連続アクメになる"


def test_contextual_asr_replacements_fix_broader_akume_homophones() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="波のように 強く 弱く 何度も打ち寄せる 体中が震えそうな悪夢",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="ああ、だらしない顔で悪目声漏らしながら体そらしちゃって",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=8.0,
                end=12.0,
                text="腰や背筋が悪夢の直前に来るゾワゾワした独特の感覚",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=12.0,
                end=16.0,
                text="悪夢寸前のあの感覚が一瞬で広がって",
                language="ja",
                confidence=0.92,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 4
    assert summary["total_replacements"] == 4
    assert chunks[0].text == "波のように 強く 弱く 何度も打ち寄せる 体中が震えそうなアクメ"
    assert chunks[1].text == "ああ、だらしない顔でアクメ声漏らしながら体そらしちゃって"
    assert chunks[2].text == "腰や背筋がアクメの直前に来るゾワゾワした独特の感覚"
    assert chunks[3].text == "アクメ寸前のあの感覚が一瞬で広がって"


def test_contextual_asr_replacements_fix_observed_sparse_akume_and_kaikan_residuals() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=5.0,
                text="そうならないようにいっぱい悪夢しようね",
                language="ja",
                confidence=0.97,
            ),
            ASRChunk(
                start=5.0,
                end=9.0,
                text="いつまで悪夢してるのよ 早く戻ってきてよね",
                language="ja",
                confidence=0.97,
            ),
            ASRChunk(
                start=9.0,
                end=14.0,
                text="開館で人の言葉も飛んじゃいましたね",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=14.0,
                end=19.0,
                text="怖い悪夢してるみたいで眠れない",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=19.0,
                end=24.0,
                text="市民会館で人の話を聞いた",
                language="ja",
                confidence=0.95,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 3
    assert [chunk.text for chunk in chunks] == [
        "そうならないようにいっぱいアクメしようね",
        "いつまでアクメしてるのよ 早く戻ってきてよね",
        "快感で人の言葉も飛んじゃいましたね",
        "怖い悪夢してるみたいで眠れない",
        "市民会館で人の話を聞いた",
    ]


def test_asr_text_replacements_fix_recent_kaikan_de_mishear() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="すごい絶頂の波 外観で真っ白な頭が電気のようにパチパチ",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="外観の電流に頭が白くなってスパークする",
                language="ja",
                confidence=0.91,
            ),
            ASRChunk(
                start=8.0,
                end=16.0,
                text="次の外観が待ちきれなくなる",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=16.0,
                end=22.0,
                text="外観という名の首輪がはめられている",
                language="ja",
                confidence=0.94,
            ),
            ASRChunk(
                start=22.0,
                end=28.0,
                text="あんたと比べるとまだまだ可愛い悪夢",
                language="ja",
                confidence=0.95,
            ),
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 5
    assert summary["total_replacements"] == 5
    assert chunks[0].text == "すごい絶頂の波 快感で真っ白な頭が電気のようにパチパチ"
    assert chunks[1].text == "快感の電流に頭が白くなってスパークする"
    assert chunks[2].text == "次の快感が待ちきれなくなる"
    assert chunks[3].text == "快感という名の首輪がはめられている"
    assert chunks[4].text == "あんたと比べるとまだまだ可愛いアクメ"
    assert cfg.asr_review_candidate_replacements["外観の電流"] == "快感の電流"
    assert cfg.asr_review_candidate_replacements["次の外観"] == "次の快感"
    assert cfg.asr_review_candidate_replacements["外観という名"] == "快感という名"
    assert cfg.asr_review_candidate_replacements["可愛い悪夢"] == "可愛いアクメ"


def test_contextual_asr_replacements_fix_generic_kaikan_and_keep_true_hall_context() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="こんな会館知りたくない うーんやめないで",
                language="ja",
                confidence=0.91,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="耳の奥 脳みそ 君が開館",
                language="ja",
                confidence=0.91,
            ),
            ASRChunk(
                start=8.0,
                end=12.0,
                text="市民会館のホールでイベントを見た",
                language="ja",
                confidence=0.95,
            ),
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 2
    assert [chunk.text for chunk in chunks] == [
        "こんな快感知りたくない うーんやめないで",
        "耳の奥 脳みそ 君が快感",
        "市民会館のホールでイベントを見た",
    ]
    assert pipeline_steps._source_script_asr_review_reasons(
        SourceScript(
            text="市民会館のホールでイベントを見た",
            language="ja",
            backend="faster_whisper",
            start=8.0,
            end=12.0,
            confidence=0.95,
        ),
        cfg,
    ) == []


def test_contextual_asr_replacements_fix_observed_akume_compounds() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=5.0,
                text="ほらほら出るよ ゼロ 行く 出てる 受精悪夢する",
                language="ja",
                confidence=0.91,
            ),
            ASRChunk(
                start=5.0,
                end=10.0,
                text="出た 行く 出産悪夢する 再現なく快感が湧き続ける",
                language="ja",
                confidence=0.91,
            ),
            ASRChunk(
                start=10.0,
                end=14.0,
                text="怖い夢を見たあと 強烈な悪夢する夜だった",
                language="ja",
                confidence=0.95,
            ),
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 2
    assert [chunk.text for chunk in chunks] == [
        "ほらほら出るよ ゼロ 行く 出てる 受精アクメする",
        "出た 行く 出産アクメする 再現なく快感が湧き続ける",
        "怖い夢を見たあと 強烈な悪夢する夜だった",
    ]


def test_contextual_asr_replacements_aggressively_normalize_observed_akume_residuals() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=5.0,
                text="あなたは今、悪夢スーツを着て、コックピットの柔らかいクッションの内側で横になっています。",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=5.0,
                end=10.0,
                text="そうそれそれこそがあの言葉あの言葉悪夢です",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=10.0,
                end=15.0,
                text="その悪夢はお姉ちゃんと同時にするんです",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=15.0,
                end=20.0,
                text="悪夢ノイドとなったあなたの成すべきこと",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=20.0,
                end=25.0,
                text="もうだめ 悪夢はもうだめ あなたはそう思ってしまいますが でも",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=25.0,
                end=30.0,
                text="来る また来る 悪夢の前兆 あなたはまた悪夢の予感に身構える そして",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=30.0,
                end=35.0,
                text="悪夢できなくなったら それで終わり",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=35.0,
                end=40.0,
                text="究極の悪夢を味わうか どっちにするか 決めなさい",
                language="ja",
                confidence=0.95,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 8
    assert [chunk.text for chunk in chunks] == [
        "あなたは今、アクメスーツを着て、コックピットの柔らかいクッションの内側で横になっています。",
        "そうそれそれこそがあの言葉あの言葉アクメです",
        "そのアクメはお姉ちゃんと同時にするんです",
        "アクメノイドとなったあなたの成すべきこと",
        "もうだめ アクメはもうだめ あなたはそう思ってしまいますが でも",
        "来る また来る アクメの前兆 あなたはまたアクメの予感に身構える そして",
        "アクメできなくなったら それで終わり",
        "究極のアクメを味わうか どっちにするか 決めなさい",
    ]


def test_contextual_asr_replacements_keep_true_nightmare_context() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="怖い悪夢の前兆で眠れない",
                language="ja",
                confidence=0.92,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 0
    assert summary["total_replacements"] == 0
    assert chunks[0].text == "怖い悪夢の前兆で眠れない"


def test_contextual_asr_replacements_keep_true_nightmare_context_with_machine_words() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="機械の音がする怖い夢を見たあと 悪夢まで眠れない",
                language="ja",
                confidence=0.92,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 0
    assert summary["total_replacements"] == 0
    assert chunks[0].text == "機械の音がする怖い夢を見たあと 悪夢まで眠れない"


def test_contextual_asr_replacements_keep_true_nightmare_context_with_generic_homophones() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="怖い夢を見たあと 悪目声と悪夢の直前のことを思い出して眠れない",
                language="ja",
                confidence=0.92,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 0
    assert summary["total_replacements"] == 0
    assert chunks[0].text == "怖い夢を見たあと 悪目声と悪夢の直前のことを思い出して眠れない"


def test_contextual_asr_replacements_keep_true_nightmare_face_context() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="怖い夢を見たあと 悪目顔のことを思い出して眠れない",
                language="ja",
                confidence=0.92,
            ),
        ],
        {},
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 0
    assert summary["total_replacements"] == 0
    assert chunks[0].text == "怖い夢を見たあと 悪目顔のことを思い出して眠れない"


def test_asr_review_item_keeps_true_nightmare_context_out_of_akume_candidates() -> None:
    cfg = ProjectConfig()
    chunks = [
        ASRChunk(
            start=0.0,
            end=4.0,
            text="怖い悪夢の前兆で眠れない",
            language="ja",
            confidence=0.92,
        )
    ]

    assert pipeline_steps._asr_review_item(chunks, 0, cfg=cfg) is None


def test_asr_review_item_still_suggests_akume_in_domain_context() -> None:
    cfg = ProjectConfig()
    chunks = [
        ASRChunk(
            start=0.0,
            end=4.0,
            text="快感で激しく悪夢させるために作られた形",
            language="ja",
            confidence=0.92,
        )
    ]

    item = pipeline_steps._asr_review_item(chunks, 0, cfg=cfg)

    assert item is not None
    assert "悪夢させ" in item["suspicious_patterns"]
    assert {"candidate_id": "domain_replacement", "text": "快感で激しくアクメさせるために作られた形"} in item["candidates"]


def test_asr_final_filter_drops_prompt_term_list_leaks() -> None:
    cfg = ProjectConfig()
    kept, dropped = pipeline_steps._filter_final_asr_chunks_for_hallucinations(
        [
            ASRChunk(
                start=45.11,
                end=64.46,
                text="おまんこ おちんぽ 女体化 採集マシーン エネルギー不足 睾丸 フェラチオ スキーン腺",
                language="ja",
                confidence=0.93,
            ),
            ASRChunk(
                start=65.0,
                end=68.0,
                text="少し近づきますね",
                language="ja",
                confidence=0.96,
            ),
        ],
        cfg=cfg,
    )

    assert [chunk.text for chunk in kept] == ["少し近づきますね"]
    assert len(dropped) == 1
    assert dropped[0]["reason"] == "prompt_term_list_leak"
    assert dropped[0]["text"] == "おまんこ おちんぽ 女体化 採集マシーン エネルギー不足 睾丸 フェラチオ スキーン腺"
    assert dropped[0]["duration"] == pytest.approx(19.35)


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
    assert diagnostics["source_separation_fallback"]["recommended"] is False
    assert diagnostics["source_separation_fallback"]["reason"] == "raw_asr_quality_within_threshold"
    assert diagnostics["source_separation_fallback"]["reasons"] == []
    assert diagnostics["source_separation_fallback"]["metrics"]["manual_review_threshold"] == 3
    assert summary["recommend_source_separation_fallback"] is False
    assert manifest.artifacts["asr_input_diagnostics"]


def test_asr_high_risk_report_classifies_segments_for_automation() -> None:
    cfg = ProjectConfig()
    auto_segment = sample_segment("seg_0001", start=0.0, end=2.0)
    auto_segment.status = "transcribed"
    auto_segment.source_script = SourceScript(
        text="少し近づきますね",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=2.0,
        confidence=0.96,
    )
    countdown_segment = sample_segment("seg_0002", start=2.0, end=7.0)
    countdown_segment.status = "transcribed"
    countdown_segment.source_script = SourceScript(
        text="4 3 2 1 0",
        language="ja",
        backend="faster_whisper",
        start=2.0,
        end=7.0,
        confidence=0.91,
    )
    countdown_segment.analysis["countdown_event"] = {
        "kind": "descending_countdown",
        "values": [4, 3, 2, 1, 0],
        "token_timeline": [{}, {}, {}, {}, {}],
    }
    texture_segment = sample_segment("seg_0003", start=7.0, end=10.0)
    texture_segment.status = "non_speech_texture"
    texture_segment.keep_original_texture = True
    texture_segment.errors = ["asr_non_speech_texture"]
    texture_segment.source_script = SourceScript(
        text="ん……",
        language="ja",
        backend="faster_whisper",
        start=7.0,
        end=10.0,
        confidence=0.8,
    )
    review_segment = sample_segment("seg_0004", start=10.0, end=14.0)
    review_segment.status = "needs_manual_review"
    review_segment.errors = ["asr_suspicious_pattern:(?:悪夢|悪目|明け目|アカメ)顔"]
    review_segment.source_script = SourceScript(
        text="悪目顔",
        language="ja",
        backend="faster_whisper",
        start=10.0,
        end=14.0,
        confidence=0.93,
    )
    manifest = PipelineManifest(
        project_config=cfg,
        segments=[auto_segment, countdown_segment, texture_segment, review_segment],
    )

    report = pipeline_steps._build_asr_high_risk_report(
        manifest,
        cfg=cfg,
        replacements_summary={
            "items": [
                {
                    "chunk_id": "chunk_0004",
                    "start": 10.0,
                    "end": 14.0,
                    "original_text": "悪目顔",
                    "replaced_text": "アクメ顔",
                    "hits": [{"source": "悪目", "target": "アクメ", "count": 1}],
                }
            ]
        },
        repair_summary={"attempted": 1, "repaired": 0},
        asr_review_summary={"attempted": 1, "manual_review": 1},
        filtered_summary=[],
    )

    summary = report["summary"]
    assert summary["segment_count"] == 4
    assert summary["auto_accept"] == 1
    assert summary["countdown_verified"] == 1
    assert summary["texture"] == 1
    assert summary["needs_review"] == 1
    assert summary["severe"] == 1
    assert summary["automated_dubbing_ready"] is False
    assert summary["blocking_reasons"] == ["needs_manual_review"]
    decisions = {item["segment_id"]: item["decision"] for item in report["items"]}
    assert decisions == {
        "seg_0002": "countdown_verified",
        "seg_0003": "texture",
        "seg_0004": "needs_review",
    }
    review_item = next(item for item in report["items"] if item["segment_id"] == "seg_0004")
    assert review_item["replacement_hits"][0]["source"] == "悪目"


def test_asr_high_risk_report_warns_for_unverified_sparse_and_numeric_sequences() -> None:
    cfg = ProjectConfig()
    sparse_segment = sample_segment("seg_0001", start=0.0, end=15.0)
    sparse_segment.status = "transcribed"
    sparse_segment.analysis["asr_sparse_speech_unverified"] = {
        "reason": "asr_sparse_speech_unverified",
    }
    sparse_segment.source_script = SourceScript(
        text="今の姿勢のまま",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=15.0,
        confidence=0.94,
    )
    numeric_segment = sample_segment("seg_0002", start=15.0, end=30.0)
    numeric_segment.status = "transcribed"
    numeric_segment.analysis["asr_numeric_sequence_unverified"] = {
        "reason": "asr_numeric_sequence_unverified",
    }
    numeric_segment.source_script = SourceScript(
        text="296 295 294",
        language="ja",
        backend="faster_whisper",
        start=15.0,
        end=30.0,
        confidence=0.93,
    )
    manifest = PipelineManifest(project_config=cfg, segments=[sparse_segment, numeric_segment])

    report = pipeline_steps._build_asr_high_risk_report(
        manifest,
        cfg=cfg,
        replacements_summary={"items": []},
        repair_summary={},
        asr_review_summary={},
        filtered_summary=[],
    )

    summary = report["summary"]
    assert summary["severe"] == 0
    assert summary["warning"] == 2
    assert summary["automated_dubbing_ready"] is True
    assert summary["blocking_reasons"] == []
    decisions = {item["segment_id"]: item["decision"] for item in report["items"]}
    assert decisions == {
        "seg_0001": "sparse_speech_unverified",
        "seg_0002": "numeric_sequence_unverified",
    }


def test_transcribe_writes_asr_high_risk_report(
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
                    end=2.0,
                    text="悪目顔",
                    language="ja",
                    confidence=0.72,
                )
            ]

    monkeypatch.setattr(pipeline_steps, "create_asr_backend", lambda *_args, **_kwargs: FakeASRBackend())

    manifest = transcribe_step(tmp_project_dir, asr_backend="faster_whisper", confirm_rights=True)

    report_path = Path(manifest.artifacts["asr_high_risk_report"])
    report = json.loads(report_path.read_text("utf-8"))
    assert report["summary"]["automated_dubbing_ready"] is False
    assert report["summary"]["needs_review"] == 1
    assert report["items"][0]["decision"] == "needs_review"
    assert report["items"][0]["severity"] == "severe"
    postprocess_path = Path(manifest.artifacts["asr_postprocess_review"])
    postprocess = json.loads(postprocess_path.read_text("utf-8"))
    assert postprocess["summary"]["segment_count"] == 1
    assert postprocess["summary"]["item_count"] == 1
    assert postprocess["summary"]["manual_review"] == 1
    assert postprocess["items"][0]["action"] == "manual_review"
    assert manifest.stage_state["transcribe"]["asr_auto_dub_ready"] is False
    assert manifest.stage_state["transcribe"]["asr_high_risk_severe"] == 1
    assert manifest.stage_state["transcribe"]["asr_postprocess_review_items"] == 1
    assert manifest.stage_state["transcribe"]["asr_postprocess_manual_review"] == 1


def test_asr_source_separation_fallback_recommends_demucs_for_poor_raw_signal() -> None:
    cfg = ProjectConfig(source_separation_backend="none")
    segments: list[Segment] = []
    for index in range(100):
        segment = sample_segment(
            f"seg_{index:04d}",
            start=float(index),
            end=float(index + 1),
        )
        segment.status = "transcribed"
        segments.append(segment)
    for segment in segments[:3]:
        segment.status = "needs_manual_review"
        segment.errors = ["asr_sparse_text_density"]
    segments[0].errors.append("asr_repair_rejected:prompt_or_hallucination_leak")
    manifest = PipelineManifest(project_config=cfg, segments=segments)

    recommendation = pipeline_steps._source_separation_fallback_recommendation(
        manifest,
        cfg=cfg,
        input_diagnostics={"selected": {"source": "gemma_mono_16k"}},
        repair_summary={"attempted": 24, "repaired": 4},
        asr_review_summary={"attempted": 16, "failed": 1},
    )

    assert recommendation["recommended"] is True
    assert recommendation["recommended_backend"] == "demucs"
    assert recommendation["reasons"] == ["manual_review_rate"]
    assert recommendation["metrics"]["needs_manual_review"] == 3
    assert recommendation["metrics"]["manual_review_rate"] == pytest.approx(0.03)


def test_asr_source_separation_fallback_ignores_noisy_low_rate_raw_signal() -> None:
    cfg = ProjectConfig(source_separation_backend="none")
    segments: list[Segment] = []
    for index in range(1265):
        segment = sample_segment(
            f"seg_{index:04d}",
            start=float(index),
            end=float(index + 1),
        )
        segment.status = "transcribed"
        segments.append(segment)
    for segment in segments[:25]:
        segment.status = "needs_manual_review"
        segment.errors = ["asr_suspicious_pattern:悪夢まで"]
    manifest = PipelineManifest(project_config=cfg, segments=segments)

    recommendation = pipeline_steps._source_separation_fallback_recommendation(
        manifest,
        cfg=cfg,
        input_diagnostics={"selected": {"source": "gemma_mono_16k"}},
        repair_summary={"attempted": 41, "repaired": 41},
        asr_review_summary={"attempted": 31, "failed": 0},
    )

    assert recommendation["recommended"] is False
    assert recommendation["recommended_backend"] is None
    assert recommendation["reason"] == "raw_asr_quality_within_threshold"
    assert recommendation["reasons"] == []
    assert recommendation["metrics"]["manual_review_rate"] == pytest.approx(0.019763)


def test_asr_source_separation_fallback_uses_per_file_manual_review_threshold() -> None:
    cfg = ProjectConfig(source_separation_backend="none")
    segments: list[Segment] = []
    for index in range(500):
        segment = sample_segment(
            f"seg_{index:04d}",
            start=float(index),
            end=float(index + 1),
        )
        segment.status = "transcribed"
        segments.append(segment)
    for segment in segments[400:403]:
        segment.status = "needs_manual_review"
        segment.errors = ["asr_sparse_text_density"]
    manifest = PipelineManifest(
        project_config=cfg,
        source_info=SourceInfo(
            path="folder",
            duration_sec=500.0,
            raw={
                "folder_input": {
                    "asr_parts": [
                        {
                            "part_index": 1,
                            "stem": "clean",
                            "path": "clean.wav",
                            "start_sec": 0.0,
                            "end_sec": 400.0,
                        },
                        {
                            "part_index": 2,
                            "stem": "bad",
                            "path": "bad.wav",
                            "start_sec": 400.0,
                            "end_sec": 500.0,
                        },
                    ]
                }
            },
        ),
        segments=segments,
    )

    recommendation = pipeline_steps._source_separation_fallback_recommendation(
        manifest,
        cfg=cfg,
        input_diagnostics={"selected": {"source": "gemma_mono_16k"}},
        repair_summary={},
        asr_review_summary={},
    )

    assert recommendation["recommended"] is True
    assert recommendation["recommended_backend"] == "demucs"
    assert recommendation["reasons"] == ["manual_review_rate"]
    assert recommendation["metrics"]["manual_review_rate"] == pytest.approx(0.006)
    assert recommendation["metrics"]["manual_review_threshold"] == 10
    assert recommendation["metrics"]["manual_review_file_trigger_count"] == 1
    file_metrics = recommendation["metrics"]["manual_review_file_metrics"]
    assert file_metrics[0]["recommended"] is False
    assert file_metrics[0]["manual_review_threshold"] == 8
    assert file_metrics[1]["recommended"] is True
    assert file_metrics[1]["part_index"] == 2
    assert file_metrics[1]["segment_count"] == 100
    assert file_metrics[1]["needs_manual_review"] == 3
    assert file_metrics[1]["manual_review_threshold"] == 3


def test_asr_source_separation_fallback_requires_minimum_manual_review_count_per_file() -> None:
    cfg = ProjectConfig(source_separation_backend="none")
    segments: list[Segment] = []
    for index in range(100):
        segment = sample_segment(
            f"seg_{index:04d}",
            start=float(index),
            end=float(index + 1),
        )
        segment.status = "transcribed"
        segments.append(segment)
    for segment in segments[:2]:
        segment.status = "needs_manual_review"
        segment.errors = ["asr_sparse_text_density"]
    manifest = PipelineManifest(
        project_config=cfg,
        source_info=SourceInfo(
            path="folder",
            duration_sec=100.0,
            raw={
                "folder_input": {
                    "asr_parts": [
                        {
                            "part_index": 1,
                            "stem": "borderline",
                            "path": "borderline.wav",
                            "start_sec": 0.0,
                            "end_sec": 100.0,
                        }
                    ]
                }
            },
        ),
        segments=segments,
    )

    recommendation = pipeline_steps._source_separation_fallback_recommendation(
        manifest,
        cfg=cfg,
        input_diagnostics={"selected": {"source": "gemma_mono_16k"}},
        repair_summary={},
        asr_review_summary={},
    )

    assert recommendation["recommended"] is False
    assert recommendation["recommended_backend"] is None
    assert recommendation["reason"] == "raw_asr_quality_within_threshold"
    assert recommendation["metrics"]["manual_review_rate"] == pytest.approx(0.02)
    assert recommendation["metrics"]["manual_review_threshold"] == 3
    assert recommendation["metrics"]["manual_review_file_metrics"][0]["manual_review_threshold"] == 3
    assert recommendation["metrics"]["manual_review_file_metrics"][0]["recommended"] is False


def test_asr_source_separation_fallback_skips_after_demucs_input() -> None:
    cfg = ProjectConfig(source_separation_backend="demucs")
    segment = sample_segment("seg_0001")
    segment.status = "needs_manual_review"
    segment.errors = ["asr_sparse_text_density"]
    manifest = PipelineManifest(project_config=cfg, segments=[segment])

    recommendation = pipeline_steps._source_separation_fallback_recommendation(
        manifest,
        cfg=cfg,
        input_diagnostics={"selected": {"source": "source_vocals_mono_16k"}},
        repair_summary={"attempted": 40, "repaired": 0},
        asr_review_summary={},
    )

    assert recommendation["recommended"] is False
    assert recommendation["reason"] == "source_separation_already_used"
    assert recommendation["recommended_backend"] is None


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
    assert ProjectConfig().gemma_llama_cpp_ctx_size == 16384
    assert ProjectConfig().gemma_text_context_radius == 10
    assert ProjectConfig().gemma_text_span_size == 10
    assert ProjectConfig().gemma_text_span_max_sec == pytest.approx(75.0)
    assert ProjectConfig().gemma_text_span_max_gap_sec == pytest.approx(5.0)
    assert ProjectConfig().gemma_text_n_predict == 1536
    assert ProjectConfig().gemma_text_two_pass is True
    assert ProjectConfig().gemma_text_concurrency == 1
    assert ProjectConfig().gemma_audio_style_concurrency == 2
    assert ProjectConfig().gsv_concurrency == 4
    assert ProjectConfig().asr_device == "cuda"
    assert ProjectConfig().source_language == "ja"
    assert ProjectConfig().target_language == "ko"
    assert ProjectConfig(target_language="kr").target_language == "ko"
    assert ProjectConfig().asr_resegment_from_chunks is True
    assert ProjectConfig().asr_resegment_min_sec == pytest.approx(3.0)
    assert ProjectConfig().asr_resegment_max_sec == pytest.approx(10.0)
    assert ProjectConfig().asr_resegment_merge_gap_sec == pytest.approx(1.0)
    assert ProjectConfig().asr_countdown_merge_max_span_sec == pytest.approx(60.0)
    assert ProjectConfig().asr_word_timestamps is False
    assert ProjectConfig().asr_hallucination_silence_threshold is None
    assert ProjectConfig().asr_sparse_chunk_max_sec == pytest.approx(30.0)
    assert ProjectConfig().asr_sparse_chunk_min_chars_per_sec == pytest.approx(0.5)
    assert ProjectConfig().asr_repair_enabled is True
    assert ProjectConfig().asr_repair_confidence_threshold == pytest.approx(0.90)
    assert ProjectConfig().asr_repair_sparse_min_sec == pytest.approx(12.0)
    assert ProjectConfig().asr_repair_sparse_min_chars_per_sec == pytest.approx(1.0)
    assert ProjectConfig().asr_repair_padding_sec == pytest.approx(1.0)
    assert ProjectConfig().asr_repair_max_chunks == 160
    assert "もちなとい" in ProjectConfig().asr_repair_suspicious_text_patterns
    assert "ご処生" in ProjectConfig().asr_repair_suspicious_text_patterns
    assert ProjectConfig(asr_review_backend="llama_server_audio").asr_review_backend == "llama_server_audio"
    assert ProjectConfig().asr_review_audio_padding_sec == pytest.approx(0.4)
    assert "gemma-4-E4B-it-OBLITERATED-Q8_0.gguf" in ProjectConfig().gemma_llama_cpp_audio_model_path
    assert "gemma-4-E4B-it-OBLITERATED-mmproj-f16.gguf" in ProjectConfig().gemma_llama_cpp_audio_mmproj_path
    assert "女体化" in ProjectConfig().asr_hotwords
    assert "採集マシーン" in ProjectConfig().asr_hotwords
    assert "電マ" in ProjectConfig().asr_hotwords
    assert "亀頭" in ProjectConfig().asr_hotwords
    assert "スキーン腺" in ProjectConfig().asr_hotwords
    assert "陰核基部" in ProjectConfig().asr_hotwords
    assert "ポルチオ" in ProjectConfig().asr_hotwords
    assert "オナニー" not in ProjectConfig().asr_hotwords
    assert "ゆっくり" not in ProjectConfig().asr_hotwords
    assert "気持ちいい" not in ProjectConfig().asr_hotwords
    assert "快感" not in ProjectConfig().asr_hotwords
    assert ProjectConfig().asr_text_replacements["釣りが来ちゃう"] == "絶頂が来ちゃう"
    assert ProjectConfig().asr_text_replacements["女体科"] == "女体化"
    assert ProjectConfig().asr_text_replacements["生還体"] == "性感帯"
    assert ProjectConfig().asr_text_replacements["薄いて"] == "疼いて"
    assert ProjectConfig().asr_text_replacements["尿位"] == "尿意"
    assert ProjectConfig().asr_text_replacements["中八菌催眠音声"] == "18禁催眠音声"
    assert ProjectConfig().asr_text_replacements["手帳が来る"] == "絶頂が来る"
    assert ProjectConfig().asr_text_replacements["ピスタン"] == "ピストン"
    assert ProjectConfig().asr_text_replacements["ウニアクナで触手"] == "耳奥まで触手"
    assert ProjectConfig().source_separation_backend == "auto"
    assert ProjectConfig().source_separation_model == "htdemucs"
    assert cli_module.FULL_REAL_QUALITY_PRESET["source_separation_backend"] == "auto"
    assert ProjectConfig().gsv_trim_edge_silence is True
    assert ProjectConfig().gsv_ref_min_sec == pytest.approx(3.0)
    assert ProjectConfig().gsv_ref_max_sec == pytest.approx(10.0)
    assert ProjectConfig().gsv_tts_min_speed_factor == pytest.approx(0.85)
    assert ProjectConfig().gsv_tts_max_speed_factor == pytest.approx(1.12)
    assert ProjectConfig().gsv_countdown_renderer == "numeric_phrase"
    assert ProjectConfig().gsv_countdown_carrier_templates == [
        "이번 숫자는 {token}. 입니다.",
        "숫자만 조용히 말해요. {token}. 다시.",
    ]
    assert ProjectConfig().gsv_countdown_carrier_numeric_unit_enabled is True
    assert ProjectConfig().gsv_countdown_carrier_numeric_unit_templates == [
        "{token} 번만요.",
        "{token} 입니다.",
        "{token} 하고요.",
        "{token} 초만요.",
    ]
    assert ProjectConfig().gsv_countdown_carrier_token_templates["구"] == [
        "구팔칠.",
        "구, 팔, 칠.",
        "{token}, 살짝만.",
    ]
    assert ProjectConfig().gsv_countdown_carrier_token_templates["사"] == [
        "사아 번만요.",
    ]
    assert ProjectConfig().gsv_countdown_carrier_token_templates["일"] == [
        "1.",
        "일, 천천히요.",
        "일 번만요.",
        "일, 일.",
    ]
    assert ProjectConfig().gsv_countdown_carrier_numeric_unit_onset_window_sec == pytest.approx(
        [0.18, 0.24, 0.30, 0.36]
    )
    assert ProjectConfig().gsv_countdown_carrier_numeric_unit_tail_pad_sec == pytest.approx(0.04)
    assert ProjectConfig().gsv_countdown_carrier_energy_extend_enabled is True
    assert ProjectConfig().gsv_countdown_carrier_energy_extend_max_sec == pytest.approx(0.10)
    assert ProjectConfig().gsv_countdown_carrier_energy_extend_coda_max_sec == pytest.approx(0.20)
    assert ProjectConfig().gsv_countdown_carrier_energy_extend_edge_threshold_ratio == pytest.approx(0.12)
    assert ProjectConfig().gsv_countdown_carrier_energy_extend_quiet_threshold_ratio == pytest.approx(0.08)
    assert ProjectConfig().gsv_countdown_carrier_candidate_count == 2
    assert ProjectConfig().gsv_countdown_carrier_slice_search_enabled is True
    assert ProjectConfig().gsv_countdown_carrier_slice_window_sec == pytest.approx([0.30, 0.42, 0.55])
    assert ProjectConfig().gsv_countdown_carrier_slice_window_offset_sec == pytest.approx([-0.06, 0.0, 0.06])
    assert ProjectConfig().gsv_countdown_carrier_max_slice_windows_per_candidate == 5
    assert ProjectConfig().gsv_countdown_carrier_full_sentence_prefilter_enabled is True
    assert ProjectConfig().gsv_countdown_carrier_full_sentence_prefilter_min_coverage == pytest.approx(1.0)
    assert ProjectConfig().gsv_countdown_carrier_quality_retry_enabled is True
    assert ProjectConfig().gsv_countdown_carrier_quality_retry_max_rounds == 3
    assert ProjectConfig().gsv_countdown_carrier_quality_retry_target_tier == "A"
    assert ProjectConfig().gsv_countdown_token_bank_enabled is True
    assert ProjectConfig().gsv_countdown_token_bank_warmup_enabled is True
    assert ProjectConfig().gsv_countdown_token_bank_pack_warmup_enabled is True
    assert ProjectConfig().gsv_countdown_token_bank_pack_templates[0] == "{token}, {token}, {token}"
    assert ProjectConfig().gsv_countdown_token_bank_max_ref_count == 4
    assert ProjectConfig().gsv_countdown_token_bank_beam_width == 8
    assert ProjectConfig().gsv_countdown_carrier_stop_window_search_after_pronunciation_pass is True
    assert ProjectConfig().gsv_countdown_carrier_target_pronunciation_passes == 2
    assert ProjectConfig().gsv_countdown_chunk_candidate_count == 10
    assert ProjectConfig().gsv_countdown_chunk_max_size == 10
    assert ProjectConfig().gsv_countdown_token_speed_factor == pytest.approx(1.0)
    assert ProjectConfig().gsv_countdown_fallback_renderer == "manual_review"
    assert ProjectConfig().gsv_countdown_timing_mode == "source_smoothed"
    assert ProjectConfig().gsv_countdown_strict_token_pronunciation is True
    assert ProjectConfig().gsv_countdown_pack_min_span_occupancy == pytest.approx(0.55)
    assert ProjectConfig().gsv_countdown_phrase_slice_edge_pad_sec == pytest.approx(0.04)
    assert ProjectConfig().gsv_countdown_token_single_syllable_max_sec == pytest.approx(0.55)
    assert ProjectConfig().gsv_countdown_token_double_syllable_max_sec == pytest.approx(0.75)
    assert ProjectConfig().gsv_countdown_token_multi_syllable_max_sec == pytest.approx(0.95)
    assert ProjectConfig().gsv_countdown_max_tempo == pytest.approx(1.0)
    assert ProjectConfig().gsv_countdown_prosody_qc_enabled is True
    assert ProjectConfig().gsv_countdown_prosody_max_median_semitone_error == pytest.approx(12.0)
    assert ProjectConfig().gsv_countdown_prosody_min_pass_score == pytest.approx(0.74)
    assert ProjectConfig().gsv_countdown_prosody_min_warn_score == pytest.approx(0.45)
    assert ProjectConfig().gsv_countdown_prosody_failure_blocks_mix is True
    assert ProjectConfig().gsv_pronunciation_qc_max_observed_unit_ratio == pytest.approx(1.8)
    assert ProjectConfig().gsv_pronunciation_qc_max_extra_units == 1
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


def test_asr_audio_review_server_uses_audio_llama_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ProjectConfig(
        gemma_llama_cpp_model_path="text-model.gguf",
        gemma_llama_cpp_mmproj_path="text-mmproj.gguf",
        gemma_llama_cpp_audio_model_path="audio-model.gguf",
        gemma_llama_cpp_audio_mmproj_path="audio-mmproj.gguf",
    )
    captured: dict[str, object] = {}

    def fake_default_llama_server_command(**kwargs: object) -> list[str]:
        captured.clear()
        captured.update(kwargs)
        return ["llama-server"]

    monkeypatch.setattr(
        pipeline_steps._common_stage,
        "default_llama_server_command",
        fake_default_llama_server_command,
    )

    assert pipeline_steps._gemma_text_server_command(cfg, include_mmproj=True) == ["llama-server"]
    assert captured["model_path"] == "audio-model.gguf"
    assert captured["mmproj_path"] == "audio-mmproj.gguf"

    assert pipeline_steps._gemma_text_server_command(cfg, include_mmproj=False) == ["llama-server"]
    assert captured["model_path"] == "text-model.gguf"
    assert captured["mmproj_path"] is None


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


def test_source_voice_ref_selection_rejects_stale_audio_duration_mismatch(
    tmp_project_dir: Path,
) -> None:
    stale_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_stale_mix.wav"
    valid_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_valid_mix.wav"
    write_audio(stale_audio, np.full((int(48_000 * 2.8), 2), 0.05, dtype=np.float32), 48_000)
    write_audio(valid_audio, np.full((int(48_000 * 4.0), 2), 0.05, dtype=np.float32), 48_000)
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gsv_ref_min_quality_score=0.0,
        ),
        segments=[
            Segment(
                id="seg_stale",
                start=0.0,
                end=5.0,
                duration=5.0,
                audio_for_gemma=str(stale_audio.relative_to(tmp_project_dir)),
                audio_for_mix=str(stale_audio.relative_to(tmp_project_dir)),
                analysis={"speaker_count": 1},
                source_script=SourceScript(
                    text="これは古い音声長の参照候補です。",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=5.0,
                ),
            ),
            Segment(
                id="seg_valid",
                start=6.0,
                end=10.0,
                duration=4.0,
                audio_for_gemma=str(valid_audio.relative_to(tmp_project_dir)),
                audio_for_mix=str(valid_audio.relative_to(tmp_project_dir)),
                analysis={"speaker_count": 1},
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=6.0,
                    end=10.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_valid"]]


def test_source_voice_ref_selection_prefers_plain_prompt_over_intense_longer_prompt(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_intense",
                start=0.0,
                end=5.5,
                duration=5.5,
                audio_for_gemma="work/segments/audio/seg_intense_gemma.wav",
                audio_for_mix="work/segments/audio/seg_intense_mix.wav",
                source_script=SourceScript(
                    text="ところでピチッと電気が走るような感触も足に感じる タイツ触手が動き始めたようだね。",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=5.5,
                ),
            ),
            Segment(
                id="seg_plain",
                start=6.0,
                end=10.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=6.0,
                    end=10.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_avoids_short_explicit_prompt_when_plain_prompt_exists(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_explicit",
                start=0.0,
                end=5.0,
                duration=5.0,
                audio_for_gemma="work/segments/audio/seg_explicit_gemma.wav",
                audio_for_mix="work/segments/audio/seg_explicit_mix.wav",
                source_script=SourceScript(
                    text="もう、ほら、ビンビンだよ。",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=5.0,
                ),
            ),
            Segment(
                id="seg_plain",
                start=6.0,
                end=10.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=6.0,
                    end=10.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_avoids_twitch_effect_prompt_when_plain_prompt_exists(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_effect",
                start=0.0,
                end=5.0,
                duration=5.0,
                audio_for_gemma="work/segments/audio/seg_effect_gemma.wav",
                audio_for_mix="work/segments/audio/seg_effect_mix.wav",
                source_script=SourceScript(
                    text="あらピクピクケアレンしちゃった",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=5.0,
                ),
            ),
            Segment(
                id="seg_plain",
                start=6.0,
                end=10.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="ありがとうございます。じゃああちらでお会計お願いします。",
                    language="ja",
                    backend="mock",
                    start=6.0,
                    end=10.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_rejects_prompt_split_inside_japanese_word(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_bad_left",
                start=0.0,
                end=1.9,
                duration=1.9,
                audio_for_gemma="work/segments/audio/seg_bad_left_gemma.wav",
                audio_for_mix="work/segments/audio/seg_bad_left_mix.wav",
                source_script=SourceScript(
                    text="ねえ、聞",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=1.9,
                ),
            ),
            Segment(
                id="seg_bad_right",
                start=1.9,
                end=7.2,
                duration=5.3,
                audio_for_gemma="work/segments/audio/seg_bad_right_gemma.wav",
                audio_for_mix="work/segments/audio/seg_bad_right_mix.wav",
                source_script=SourceScript(
                    text="こえる?お姉さんの声が近くで聞こえるでしょう。",
                    language="ja",
                    backend="mock",
                    start=1.9,
                    end=7.2,
                ),
            ),
            Segment(
                id="seg_plain",
                start=8.0,
                end=12.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=8.0,
                    end=12.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_rejects_unclean_voice_training_segment(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_unclean",
                start=0.0,
                end=5.2,
                duration=5.2,
                audio_for_gemma="work/segments/audio/seg_unclean_gemma.wav",
                audio_for_mix="work/segments/audio/seg_unclean_mix.wav",
                analysis={
                    "voice_training": {
                        "clean_voice": False,
                        "eligible": False,
                        "reason": "head_bang",
                        "effect_tags": ["none"],
                    }
                },
                source_script=SourceScript(
                    text="今日はとても静かに耳元で話していますね。",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=5.2,
                ),
            ),
            Segment(
                id="seg_plain",
                start=6.0,
                end=10.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=6.0,
                    end=10.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_rejects_repetitive_prompt_when_plain_preference_is_disabled(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gsv_few_shot_prefer_plain_text=False,
        ),
        segments=[
            Segment(
                id="seg_repeat",
                start=0.0,
                end=5.0,
                duration=5.0,
                audio_for_gemma="work/segments/audio/seg_repeat_gemma.wav",
                audio_for_mix="work/segments/audio/seg_repeat_mix.wav",
                source_script=SourceScript(
                    text="いく いく いく",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=5.0,
                ),
            ),
            Segment(
                id="seg_plain",
                start=6.0,
                end=10.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=6.0,
                    end=10.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_rejects_partial_repetition_prompt(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_partial_repeat",
                start=0.0,
                end=4.5,
                duration=4.5,
                audio_for_gemma="work/segments/audio/seg_partial_repeat_gemma.wav",
                audio_for_mix="work/segments/audio/seg_partial_repeat_mix.wav",
                source_script=SourceScript(
                    text="強くなる強く",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=4.5,
                ),
            ),
            Segment(
                id="seg_plain",
                start=5.5,
                end=9.5,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=5.5,
                    end=9.5,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_rejects_sparse_long_prompt(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_sparse",
                start=0.0,
                end=8.0,
                duration=8.0,
                audio_for_gemma="work/segments/audio/seg_sparse_gemma.wav",
                audio_for_mix="work/segments/audio/seg_sparse_mix.wav",
                source_script=SourceScript(
                    text="なる",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=8.0,
                ),
            ),
            Segment(
                id="seg_plain",
                start=9.0,
                end=13.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="今日は少しだけ静かに話しますね。",
                    language="ja",
                    backend="mock",
                    start=9.0,
                    end=13.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_rejects_low_diversity_repetition_prompt(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gsv_few_shot_prefer_plain_text=False,
        ),
        segments=[
            Segment(
                id="seg_low_diversity",
                start=0.0,
                end=7.58,
                duration=7.58,
                audio_for_gemma="work/segments/audio/seg_low_diversity_gemma.wav",
                audio_for_mix="work/segments/audio/seg_low_diversity_mix.wav",
                source_script=SourceScript(
                    text="くなる強くくく強強く強く強くなる強くなる強なる",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=7.58,
                ),
            ),
            Segment(
                id="seg_plain",
                start=8.5,
                end=14.75,
                duration=6.25,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="お兄さんの意識に合わせて、ゆらゆらと形を変えていくよ。",
                    language="ja",
                    backend="mock",
                    start=8.5,
                    end=14.75,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_avoids_repeated_effect_prompt_when_plain_prompt_exists(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gsv_few_shot_prefer_plain_text=False,
        ),
        segments=[
            Segment(
                id="seg_effect_repeat",
                start=0.0,
                end=7.01,
                duration=7.01,
                audio_for_gemma="work/segments/audio/seg_effect_repeat_gemma.wav",
                audio_for_mix="work/segments/audio/seg_effect_repeat_mix.wav",
                source_script=SourceScript(
                    text="スキーンフェラチオスキーン",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=7.01,
                ),
            ),
            Segment(
                id="seg_plain",
                start=8.0,
                end=14.25,
                duration=6.25,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="お兄さんの意識に合わせて、ゆらゆらと形を変えていくよ。",
                    language="ja",
                    backend="mock",
                    start=8.0,
                    end=14.25,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


def test_source_voice_ref_selection_avoids_explicit_jargon_prompt_when_plain_prompt_exists(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gsv_few_shot_prefer_plain_text=False,
        ),
        segments=[
            Segment(
                id="seg_jargon",
                start=0.0,
                end=7.01,
                duration=7.01,
                audio_for_gemma="work/segments/audio/seg_jargon_gemma.wav",
                audio_for_mix="work/segments/audio/seg_jargon_mix.wav",
                source_script=SourceScript(
                    text="エネルギー不足睾丸フチオ",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=7.01,
                ),
            ),
            Segment(
                id="seg_plain",
                start=8.0,
                end=14.25,
                duration=6.25,
                audio_for_gemma="work/segments/audio/seg_plain_gemma.wav",
                audio_for_mix="work/segments/audio/seg_plain_mix.wav",
                source_script=SourceScript(
                    text="お兄さんの意識に合わせて、ゆらゆらと形を変えていくよ。",
                    language="ja",
                    backend="mock",
                    start=8.0,
                    end=14.25,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(
        tmp_project_dir,
        manifest,
        manifest.project_config,
        max_refs=1,
    )

    assert [[segment.id for segment in span.segments] for span in selected] == [["seg_plain"]]


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
    assert refs["whisper_close"]["prompt_text"] == "みじかいです。つずきです。"
    assert refs["whisper_close"]["prompt_text_original"] == "短いです。 続きです。"
    ref_qc = json.loads(Path(manifest.artifacts["source_voice_ref_qc"]).read_text("utf-8"))
    assert ref_qc["refs"][0]["selected_segment_ids"] == ["seg_short", "seg_next"]


def test_prepare_source_voice_refs_records_rejected_stale_audio_duration(
    tmp_project_dir: Path,
) -> None:
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            gsv_ref_min_quality_score=0.0,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    stale_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_stale_mix.wav"
    valid_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_valid_mix.wav"
    write_audio(stale_audio, np.full((int(48_000 * 2.8), 2), 0.05, dtype=np.float32), 48_000)
    write_audio(valid_audio, np.full((int(48_000 * 4.0), 2), 0.05, dtype=np.float32), 48_000)
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            project_config=ProjectConfig(
                project_name=tmp_project_dir.name,
                gsv_ref_min_quality_score=0.0,
            ),
            segments=[
                Segment(
                    id="seg_stale",
                    start=0.0,
                    end=5.0,
                    duration=5.0,
                    audio_for_gemma=str(stale_audio.relative_to(tmp_project_dir)),
                    audio_for_mix=str(stale_audio.relative_to(tmp_project_dir)),
                    analysis={"speaker_count": 1},
                    source_script=SourceScript(
                        text="これは古い音声長の参照候補です。",
                        language="ja",
                        backend="mock",
                        start=0.0,
                        end=5.0,
                    ),
                ),
                Segment(
                    id="seg_valid",
                    start=6.0,
                    end=10.0,
                    duration=4.0,
                    audio_for_gemma=str(valid_audio.relative_to(tmp_project_dir)),
                    audio_for_mix=str(valid_audio.relative_to(tmp_project_dir)),
                    analysis={"speaker_count": 1},
                    source_script=SourceScript(
                        text="今日は少しだけ静かに話しますね。",
                        language="ja",
                        backend="mock",
                        start=6.0,
                        end=10.0,
                    ),
                ),
            ],
        ),
    )

    manifest = prepare_source_voice_refs_step(tmp_project_dir, confirm_rights=True)

    refs = json.loads((tmp_project_dir / "refs" / "refs.json").read_text("utf-8"))
    assert duration_sec(tmp_project_dir / refs["whisper_close"]["ref_audio_path"]) == pytest.approx(4.0)
    ref_qc = json.loads(Path(manifest.artifacts["source_voice_ref_qc"]).read_text("utf-8"))
    assert ref_qc["refs"][0]["selected_segment_ids"] == ["seg_valid"]
    assert ref_qc["refs"][0]["selected_actual_duration_sec"] == pytest.approx(4.0)
    rejected = ref_qc["rejected_spans"]
    assert any(
        row["segment_ids"] == ["seg_stale"]
        and any(reason.startswith("audio_duration_mismatch") for reason in row["reject_reasons"])
        for row in rejected
    )


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
            labels = ["첫번째", "두번째", "세번째"]
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {labels[index]}",
                    ko_natural=f"자연 {labels[index]}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for index, segment in enumerate(segments)
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
            labels = ["첫번째", "두번째", "세번째"]
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {labels[index]}",
                    ko_natural=f"자연 {labels[index]}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for index, segment in enumerate(segments)
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


def test_translate_ko_stores_repaired_numeric_counting_translation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    segment = sample_segment("seg_0001", start=0.0, end=5.0)
    segment.source_script = SourceScript(
        text="21 22 23 24 25",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=0.0,
        end=5.0,
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["transcribe"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    class TruncatedCountingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            return {
                item.id: KoreanTranslation(
                    ko_literal="21 22 23 24 25",
                    ko_natural="스물하나 스물둘 스물셋",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for item in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", TruncatedCountingClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)

    expected = "스물하나, 스물둘, 스물셋, 스물넷, 스물다섯"
    rows = [
        json.loads(line)
        for line in Path(tmp_project_dir / "work" / "translate_ko" / "translation_bundles.jsonl").read_text().splitlines()
    ]
    assert rows[0]["translation_ko"]["ko_natural"] == expected


def test_korean_ordinal_postprocess_repairs_second_ordinal_mistranslation() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="それはまるで第二の皮膚のように全身に貼り付いています",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="그것은 마치 제이의 피부처럼 온몸에 붙어 있습니다.",
        ko_natural="마치 제이의 피부처럼 온몸에 붙어 있어요.",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )
    diagnostics: list[dict[str, object]] = []
    quality_counters: Counter[str] = Counter()

    rewritten = pipeline_steps._apply_korean_ordinal_postprocess(
        [segment],
        diagnostics,
        quality_counters,
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "마치 두 번째 피부처럼 온몸에 붙어 있어요."
    assert "korean_ordinal_postprocess" in segment.translation_ko.notes
    assert quality_counters["ordinal_mistranslation_repaired"] == 1
    assert diagnostics[0]["repair_reasons"] == ["ordinal_mistranslation"]


def test_korean_asr_homophone_postprocess_repairs_akume_mistranslation() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="メスイキ悪夢が止まらない",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="암컷 절정 악몽이 멈추지 않습니다.",
        ko_natural="암컷 절정 악몽이 멈추지 않아요.",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )
    diagnostics: list[dict[str, object]] = []
    quality_counters: Counter[str] = Counter()

    rewritten = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [segment],
        diagnostics,
        quality_counters,
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "암컷 절정이 멈추지 않아요."
    assert "korean_asr_homophone_postprocess" in segment.translation_ko.notes
    assert quality_counters["asr_homophone_repaired"] == 1
    assert diagnostics[0]["repair_reasons"] == ["asr_homophone_akume"]


def test_korean_asr_homophone_postprocess_repairs_akume_ochi_mistranslation() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="完全敗北 悪夢落ちへの道を歩み始める 快感が込み上げ 行く",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="완전 패배, 악몽으로 떨어지는 길을 걷기 시작합니다.",
        ko_natural="완전 패배... 악몽으로 떨어지는 길을 걷기 시작해요.",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )

    rewritten = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [segment],
        [],
        Counter(),
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "완전 패배... 절정에 빠지는 길을 걷기 시작해요."


def test_korean_asr_homophone_postprocess_repairs_akeme_ochi_mistranslation() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="完全敗北 明け目落ちへの道を歩み始める 快感が込み上げ イク",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="완전 패배, 눈이 뒤집히는 타락의 길로 들어섭니다.",
        ko_natural="완전 패배... 눈이 뒤집히는 타락의 길로 들어서고 있어요.",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )

    rewritten = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [segment],
        [],
        Counter(),
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "완전 패배... 절정에 빠지는 길로 들어서고 있어요."


def test_korean_asr_homophone_postprocess_repairs_remaining_akume_after_prior_note() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="メスイキ悪夢決めたい 快感が止まらない",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="메스이키 악몽을 결정하고 싶어.",
        ko_natural="암컷으로서 가는 악몽을 꾸고 싶어.",
        notes=["korean_asr_homophone_postprocess"],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )

    rewritten = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [segment],
        [],
        Counter(),
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert "악몽" not in segment.translation_ko.ko_natural
    assert segment.translation_ko.notes.count("korean_asr_homophone_postprocess") == 1


def test_korean_asr_homophone_postprocess_repairs_akame_asr_variant() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="メスイキアカメが止まらない",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="암컷 절정의 악몽이 멈추지 않습니다.",
        ko_natural="암컷 절정의 악몽이 멈추지 않아요.",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )

    rewritten = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [segment],
        [],
        Counter(),
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "암컷 절정이 멈추지 않아요."


def test_korean_asr_homophone_postprocess_repairs_josou_mistranslation() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="これが私の助走",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="이것이 저의 도움닫기입니다.",
        ko_natural="이게 저의 도움닫기예요.",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )
    diagnostics: list[dict[str, object]] = []
    quality_counters: Counter[str] = Counter()

    rewritten = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [segment],
        diagnostics,
        quality_counters,
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "이게 저의 여장이에요."
    assert "korean_asr_homophone_postprocess" in segment.translation_ko.notes
    assert quality_counters["asr_homophone_repaired"] == 1
    assert diagnostics[0]["repair_reasons"] == ["asr_homophone_josou"]


def test_korean_asr_homophone_postprocess_repairs_josou_verb_mistranslation() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="また助走をしたくなったら聞きに来てね",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="다시 달려들고 싶어지면 들으러 오세요.",
        ko_natural="다시 달려들고 싶어지면 들으러 오세요.",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )

    rewritten = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [segment],
        [],
        Counter(),
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "다시 여장하고 싶어지면 들으러 오세요."


def test_korean_onomatopoeia_postprocess_repairs_guriguri_transliteration() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="グリ 腰をもじもじ動かして 気持ちいいのね いいよ",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="그리 그리, 허리를 꼼지락꼼지락 움직여서 기분 좋은 거구나, 좋아",
        ko_natural="그리 그리, 허리를 꼼지락거리는 게 기분 좋은가 보네, 좋아...",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )
    diagnostics: list[dict[str, object]] = []
    quality_counters: Counter[str] = Counter()

    rewritten = pipeline_steps._apply_korean_onomatopoeia_postprocess(
        [segment],
        diagnostics,
        quality_counters,
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "문질문질, 허리를 꼼지락거리는 게 기분 좋은가 보네, 좋아..."
    assert "korean_onomatopoeia_postprocess" in segment.translation_ko.notes
    assert quality_counters["onomatopoeia_transliteration_repaired"] == 1
    assert diagnostics[0]["repair_reasons"] == ["onomatopoeia_guriguri"]


def test_korean_onomatopoeia_postprocess_repairs_compact_guriguri_transliteration() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="やわらかくねじるように グリグリしてあげる",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="부드럽게 비틀듯이 그리그리 해줄게",
        ko_natural="부드럽게 비틀듯이, 그리그리 해줄게",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )

    rewritten = pipeline_steps._apply_korean_onomatopoeia_postprocess(
        [segment],
        [],
        Counter(),
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_literal == "부드럽게 비틀듯이 문질문질 해줄게"
    assert segment.translation_ko.ko_natural == "부드럽게 비틀듯이, 문질문질 해줄게"


def test_korean_fluency_postprocess_repairs_observed_broken_ending_translation() -> None:
    segment = sample_segment("seg_0001")
    segment.source_script = SourceScript(
        text="終わっていく日々に絶望を感じて",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="끝져가는 나날들에 절망을 느끼며",
        ko_natural="끝져가는 나날들에 절망을 느끼며",
        notes=[],
        confidence=0.9,
        model="fake",
        batch_id="batch_0001",
    )
    diagnostics: list[dict[str, object]] = []
    quality_counters: Counter[str] = Counter()

    rewritten = pipeline_steps._apply_korean_fluency_postprocess(
        [segment],
        diagnostics,
        quality_counters,
    )

    assert rewritten == 1
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "끝나가는 나날들에 절망을 느끼며"
    assert "korean_fluency_postprocess" in segment.translation_ko.notes
    assert quality_counters["fluency_repaired"] == 1
    assert diagnostics[0]["repair_reasons"] == ["broken_korean_ending"]


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
            labels = ["첫번째", "두번째", "세번째"]
            label = labels[len(calls) - 1]
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {label}",
                    ko_natural=f"자연 {label}",
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
    assert first_translation.ko_natural == "자연 첫번째"

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")
    assert calls == ["batch_0001"]

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", force_retranslate=True)

    manifest = load_manifest(tmp_project_dir)
    translation = manifest.segments[0].translation_ko
    assert translation is not None
    assert translation.ko_natural == "자연 두번째"
    assert manifest.stage_state["translate-ko"]["force_retranslate"] is True


def test_translate_ko_force_retranslate_resets_downstream_failed_segments(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    segment.status = "failed"
    segment.errors = ["All TTS candidates failed."]
    segment.translation_ko = KoreanTranslation(
        ko_literal="오래된 직역",
        ko_natural="오래된 번역",
        model="old",
        batch_id="old_batch",
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
            calls.extend(segment.id for segment in segments)
            return {
                segment.id: KoreanTranslation(
                    ko_literal="새 직역",
                    ko_natural="새 번역",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", force_retranslate=True)

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert calls == ["seg_0001"]
    assert segment.status == "transcribed"
    assert segment.errors == []
    assert segment.translation_ko is not None
    assert segment.translation_ko.ko_natural == "새 번역"


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


def test_mix_includes_qc_passed_rvc_output_even_if_tts_candidate_gate_failed() -> None:
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
                duration_sec=1.3,
                selected=True,
                duration_ratio=1.3,
                duration_gate="too_long",
                acceptable_for_mix=False,
            )
        ],
    )
    segment.rvc = RVCMetadata(
        backend="command",
        input_path="work/tts/seg_0001_final.wav",
        output_path="work/rvc/seg_0001_final.wav",
        accepted=True,
    )
    segment.qc = QCMetadata(recommendation="pass", status="ok")

    assert pipeline_steps._include_segment_in_mix(segment, allow_korean_timing_draft=False)


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


def test_llama_server_translation_client_salvages_over_budget_text() -> None:
    requests: list[dict[str, object]] = []
    response = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "좋아요 반드시 그렇게 돼요",
            "ko_natural": "좋아요 반드시 그렇게 돼요",
            "notes": [],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]},
        )

    segment = sample_segment(start=0.0, end=2.98)
    segment.source_script = SourceScript(
        text="いいね 必ずそうなる",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=2.98,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
        two_pass=False,
    )

    translations = client.translate_batch([segment], "batch_0001")

    translation = translations["seg_0001"]
    assert korean_tts_speech_char_count(translation.ko_natural) <= 10
    assert "korean_tts_budget_fit" in translation.notes
    assert len(requests) == 1


def test_llama_server_translation_client_sanitizes_latin_and_pronunciation_symbols() -> None:
    response = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "ASMR 5/OK—TTS",
            "ko_natural": "ASMR 5/OK—TTS",
            "notes": [],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}]},
        )

    segment = sample_segment(start=0.0, end=8.0)
    segment.source_script = SourceScript(
        text="エーエスエムアール オーケー ティーティーエス",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=8.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    translation = translations["seg_0001"]
    assert translation.ko_natural == "에이에스엠알 오 오케이 티티에스"
    assert "korean_tts_sanitized" in translation.notes


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


def test_llama_server_translation_client_analyzes_audio_style(tmp_path: Path) -> None:
    requests: list[dict[str, object]] = []
    audio_path = tmp_path / "seg_0001.wav"
    audio_path.write_bytes(b"RIFFmock-wav-data")
    segment = sample_segment("seg_0001")
    response = {
        "nonverbal_cues": [],
        "spatial_style": "center",
        "style_tags": ["echo"],
        "estimated_pan": 0.0,
        "keep_original_texture": True,
        "risk_flags": [],
        "confidence": 0.92,
        "voice_training": {
            "clean_voice": False,
            "eligible": False,
            "reason": "echo on same speaker",
            "effect_tags": ["echo"],
            "same_speaker_under_effect": True,
        },
    }

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
        n_predict=512,
    )

    result = client.analyze_audio_style(audio_path, segment)

    assert result["voice_training"]["effect_tags"] == ["echo"]
    assert requests[0]["max_tokens"] == 384
    content = requests[0]["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Analyze only the audible style" in content[0]["text"]
    assert "Context:" not in content[0]["text"]
    assert content[1]["type"] == "input_audio"
    assert base64.b64decode(content[1]["input_audio"]["data"]) == audio_path.read_bytes()


def test_llama_server_translation_client_normalizes_audio_style_without_repair(
    tmp_path: Path,
) -> None:
    requests: list[dict[str, object]] = []
    audio_path = tmp_path / "seg_0001.wav"
    audio_path.write_bytes(b"RIFFmock-wav-data")
    segment = sample_segment("seg_0001", start=0.0, end=1.0)
    response = {
        "spatial_style": "rich",
        "estimated_pan": "0.2",
        "keep_original_texture": "true",
        "confidence": "high",
        "voice_training": {"effect_tags": ["echo"]},
    }

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
        retries=1,
        n_predict=384,
    )

    result = client.analyze_audio_style(audio_path, segment)

    assert len(requests) == 1
    assert result["spatial_style"] == "center"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["voice_training"]["effect_tags"] == ["echo"]
    assert result["estimated_pan"] == pytest.approx(0.2)
    assert result["keep_original_texture"] is True
    content = requests[0]["messages"][0]["content"]
    assert isinstance(content, list)
    assert "JSON shape:" in content[0]["text"]
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
    assert "Translate as literally as possible" in first_prompt
    assert "Second pass" in second_prompt
    assert "ko_literal" in second_prompt
    assert "staying as literal to ko_literal/source_text" in second_prompt
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


def test_llama_server_translation_client_warns_when_retrying_truncated_response(caplog) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "content": '[{"segment_id":"seg_0001","ko_natural":"잘린 응답"',
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                [{"segment_id": "seg_0001", "ko_natural": "안녕하세요."}],
                                ensure_ascii=False,
                            )
                        },
                    }
                ]
            },
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="こんばんは。",
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
        two_pass=False,
    )

    with caplog.at_level(py_logging.WARNING, logger="asmr_dub_pipeline.gemma.text_translate"):
        translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "안녕하세요."
    assert len(requests) == 2
    assert any(
        "Gemma text translation retry after possible context overflow/truncation" in record.message
        for record in caplog.records
    )


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


def test_llama_server_translation_client_repairs_foreign_pronunciation_symbol() -> None:
    requests: list[dict[str, object]] = []
    responses = [
        [{"segment_id": "seg_0001", "ko_literal": "마치 말미ز처럼 생긴 여러 개의 가는 튜브가"}],
        [{"segment_id": "seg_0001", "ko_natural": "마치 말미ز처럼 생긴 여러 개의 가는 튜브가"}],
        [{"segment_id": "seg_0001", "ko_natural": "마치 말미잘처럼 생긴 여러 개의 가는 튜브가"}],
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

    segment = sample_segment(start=0.0, end=4.96)
    segment.source_script = SourceScript(
        text="まるでイソギンチャクのような何本もの細いチューブが",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=4.96,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "마치 말미잘처럼 생긴 여러 개의 가는 튜브가"
    assert len(requests) == 3
    assert "contains non-Korean pronunciation symbols" in requests[2]["messages"][0]["content"]


def test_translate_ko_prompt_includes_korean_tts_timing_budget() -> None:
    segment = sample_segment(end=4.04)
    segment.source_script = SourceScript(
        text="ここは今の世界とは違う、平行世界です。",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=4.04,
    )

    prompt = build_translate_ko_prompt([segment], "batch_0001")
    budget = korean_tts_timing_budget(segment.duration, segment.source_script.text)

    assert "korean_tts_timing" in prompt
    assert "Translate as literally as possible" in prompt
    assert "semantically compress" in prompt
    assert japanese_pronunciation_count(segment.source_script.text) == 25
    assert budget["budget_basis"] == "source_japanese_pronunciation"
    assert budget["source_japanese_pronunciation_count"] == 25
    assert f'"target_speech_chars": {budget["target_speech_chars"]}' in prompt
    assert f'"max_speech_chars": {budget["max_speech_chars"]}' in prompt


def test_translate_ko_prompt_omits_timing_budget_from_neighbor_context() -> None:
    target = sample_segment("seg_0001", start=0.0, end=4.04)
    neighbor = sample_segment("seg_0002", start=4.04, end=8.0)
    target.source_script = SourceScript(
        text="ここは今の世界とは違う、平行世界です。",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=target.start,
        end=target.end,
    )
    neighbor.source_script = SourceScript(
        text="隣の文脈です。",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=neighbor.start,
        end=neighbor.end,
    )

    prompt = build_translate_ko_prompt([target], "batch_0001", [target, neighbor])
    payload = json.loads(prompt.split("Input:\n", 1)[1])

    assert "korean_tts_timing" in payload["segments"][0]
    context_by_id = {item["segment_id"]: item for item in payload["context"]}
    assert "korean_tts_timing" not in context_by_id["seg_0001"]
    assert "korean_tts_timing" not in context_by_id["seg_0002"]


def test_llama_server_translation_client_repairs_over_timing_budget_korean() -> None:
    requests: list[dict[str, object]] = []
    responses = [
        [{"segment_id": "seg_0001", "ko_literal": "여기예요."}],
        [{"segment_id": "seg_0001", "ko_natural": "여기는 지금의 세계와는 다른, 평행세계예요."}],
        [{"segment_id": "seg_0001", "ko_natural": "여기예요."}],
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

    segment = sample_segment(end=4.04)
    segment.source_script = SourceScript(
        text="ここです。",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=4.04,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "여기예요."
    assert len(requests) == 3
    repair_prompt = requests[2]["messages"][0]["content"]
    assert "korean_tts_timing" in repair_prompt
    assert "exceeds Korean TTS max speech chars" in repair_prompt


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
    assert refs["whisper_close"]["prompt_text"].startswith("エムオーシーケー エスオーユーアールシーイー")
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


def test_translate_ko_deterministically_repairs_manual_review_countdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    segment = sample_segment(start=0.0, end=8.0)
    segment.status = "needs_manual_review"
    segment.errors.append("previous countdown translation failed")
    segment.source_script = SourceScript(
        text="10 9 8 7 6 5 4 3 2 1 0",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=8.0,
        confidence=0.95,
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["transcribe"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    class FailingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            raise AssertionError(f"countdown should not reach model batch {batch_id}")

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FailingClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    translated = manifest.segments[0]
    assert translated.status == "transcribed"
    assert translated.errors == []
    assert translated.translation_ko is not None
    assert translated.translation_ko.ko_natural == "십, 구, 팔, 칠, 육, 오, 사, 삼, 이, 일, 영"
    assert translated.translation_ko.notes == ["deterministic_countdown_event"]
    assert translated.analysis["countdown_event"]["values"] == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    assert manifest.stage_state["translate-ko"]["translated"] == 1
    assert manifest.stage_state["translate-ko"]["span_count"] == 0


def test_translate_ko_deterministically_repairs_manual_review_counting_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    segment = sample_segment(start=0.0, end=8.0)
    segment.status = "needs_manual_review"
    segment.errors.append("previous numeric translation failed")
    segment.source_script = SourceScript(
        text="1、2、3、4、5、",
        language="ja",
        backend="faster_whisper",
        start=0.0,
        end=8.0,
        confidence=0.95,
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["transcribe"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    class FailingClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            raise AssertionError(f"numeric counting should not reach model batch {batch_id}")

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FailingClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    translated = manifest.segments[0]
    assert translated.status == "transcribed"
    assert translated.errors == []
    assert translated.translation_ko is not None
    assert translated.translation_ko.ko_natural == "하나, 둘, 셋, 넷, 다섯"
    assert translated.translation_ko.notes == ["numeric_counting_postprocess"]
    assert manifest.stage_state["translate-ko"]["translated"] == 1
    assert manifest.stage_state["translate-ko"]["numeric_counting_postprocessed"] == 1
    assert manifest.stage_state["translate-ko"]["span_count"] == 0


def test_translate_ko_repairs_raw_digits_and_records_diagnostics(
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
        text="快感レベル100",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    save_manifest(tmp_project_dir, manifest)

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
                    ko_literal="쾌감 레벨 100",
                    ko_natural="쾌감 레벨 100이에요.",
                    notes=[],
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
    assert translation.ko_natural == "쾌감 레벨 백이에요."
    assert "korean_digit_pronunciation_postprocess" in translation.notes
    assert manifest.segments[0].status != "needs_manual_review"
    diagnostics_path = Path(manifest.artifacts["translation_diagnostics"])
    diagnostics = json.loads(diagnostics_path.read_text("utf-8"))
    assert diagnostics["quality_counters"]["raw_digit"] == 1
    assert diagnostics["repaired_translation_bundles"][0]["repair_reasons"] == ["raw_digit"]
    assert diagnostics["final_translation_bundles"][0]["status"] == "translated"


def test_translate_ko_severe_domain_smell_does_not_reach_korean_script(
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
        text="ザーメン媚薬を飲ませます",
        language="ja",
        confidence=0.99,
        backend="mock",
        start=segment.start,
        end=segment.end,
    )
    save_manifest(tmp_project_dir, manifest)

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
                    ko_literal="정액 변비약을 먹일게요.",
                    ko_natural="정액 변비약을 먹일게요.",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")
    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "needs_manual_review"
    assert segment.script is None
    assert any("severe_translation_smell" in error for error in segment.errors)
    diagnostics = json.loads(Path(manifest.artifacts["translation_diagnostics"]).read_text("utf-8"))
    assert diagnostics["quality_counters"]["domain_mistranslation"] == 1
    assert diagnostics["final_translation_bundles"][0]["status"] == "needs_manual_review"


def test_korean_script_records_duration_budget_pressure(
    tmp_project_dir: Path,
) -> None:
    text = "여기는 지금의 세계와는 다른, 평행세계예요."
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.04,
        duration=4.04,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        status="transcribed",
        source_script=SourceScript(
            text="ここは今の世界とは違う、平行世界です。",
            language="ja",
            backend="mock",
            start=0.0,
            end=4.04,
        ),
        translation_ko=KoreanTranslation(
            ko_literal=text,
            ko_natural=text,
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    scripted = manifest.segments[0]
    assert scripted.script is not None
    timing = scripted.analysis["korean_tts_timing"]
    budget = korean_tts_timing_budget(segment.duration, segment.source_script.text)
    assert timing["source_japanese_pronunciation_count"] == 25
    assert timing["speech_chars"] == korean_tts_speech_char_count(text)
    assert timing["max_speech_chars"] == budget["max_speech_chars"]
    assert timing["over_budget"] is False
    assert scripted.script.expected_tts_duration_sec == pytest.approx(
        estimate_tts_duration(text, "ko")
    )
    assert "korean_tts_timing_over_budget" not in scripted.script.risk_flags


def test_korean_script_adapts_stiff_translation_into_spoken_tts(
    tmp_project_dir: Path,
) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        status="transcribed",
        source_script=SourceScript(
            text="私は大丈夫です。これは良いものです。",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="저는 괜찮습니다. 이것은 좋은 것입니다.",
            ko_natural="저는 괜찮습니다. 이것은 좋은 것입니다.",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    scripted = manifest.segments[0]
    assert scripted.script is not None
    assert scripted.script.tts_text == "전 괜찮아요. 이건 좋은 거예요."
    assert scripted.analysis["korean_tts_adaptation"]["tts_text_before"] == (
        "저는 괜찮습니다. 이것은 좋은 것입니다."
    )
    assert scripted.analysis["korean_tts_adaptation"]["reasons"] == [
        "korean_colloquial_postprocess"
    ]


def test_korean_script_defers_dense_micro_segment_to_structure_policy(
    tmp_project_dir: Path,
) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=0.25,
        duration=0.25,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        status="transcribed",
        source_script=SourceScript(
            text="でも、こんなに硬くなってしまっているし",
            language="ja",
            backend="mock",
            start=0.0,
            end=0.25,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="게다가, 이렇게 딱딱해져 버렸고 말이에요.",
            ko_natural="게다가, 이렇게 딱딱해져 버렸고 말이에요.",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    deferred = manifest.segments[0]
    assert deferred.status == "needs_manual_review"
    assert deferred.script is None
    assert deferred.analysis["korean_tts_micro_segment_policy"]["action"] == (
        "merge_or_absorb_required"
    )
    assert any("Micro segment too dense for standalone Korean TTS" in error for error in deferred.errors)


def test_korean_script_absorbs_dense_micro_segment_into_previous_neighbor(
    tmp_project_dir: Path,
) -> None:
    previous = sample_segment("seg_0001", start=0.0, end=2.0)
    previous.status = "transcribed"
    previous.source_script = SourceScript(
        text="ゆっくり聞いてください",
        language="ja",
        backend="mock",
        start=0.0,
        end=2.0,
    )
    previous.translation_ko = KoreanTranslation(
        ko_literal="천천히 들어주세요.",
        ko_natural="천천히 들어주세요.",
        model="mock",
        batch_id="batch_0001",
    )
    micro = sample_segment("seg_0002", start=4.18, end=4.52)
    micro.status = "transcribed"
    micro.source_script = SourceScript(
        text="そう。",
        language="ja",
        backend="mock",
        start=4.18,
        end=4.52,
    )
    micro.translation_ko = KoreanTranslation(
        ko_literal="그래요.",
        ko_natural="그래요.",
        model="mock",
        batch_id="batch_0001",
    )
    following = sample_segment("seg_0003", start=7.0, end=9.0)
    following.status = "transcribed"
    following.source_script = SourceScript(
        text="息をして",
        language="ja",
        backend="mock",
        start=7.0,
        end=9.0,
    )
    following.translation_ko = KoreanTranslation(
        ko_literal="숨을 쉬세요.",
        ko_natural="숨을 쉬세요.",
        model="mock",
        batch_id="batch_0001",
    )
    manifest = PipelineManifest(segments=[previous, micro, following])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    previous, micro, following = manifest.segments
    assert manifest.stage_state["korean-script"]["needs_manual_review"] == 0
    assert manifest.stage_state["korean-script"]["absorbed_micro_segments"] == 1
    assert previous.status == "scripted"
    assert previous.end == pytest.approx(4.52)
    assert previous.script is not None
    assert previous.script.tts_text == "천천히 들어주세요. 그래요."
    assert micro.status == "absorbed"
    assert micro.script is None
    assert micro.analysis["korean_tts_absorption"]["absorbed_into_segment_id"] == "seg_0001"
    assert following.status == "scripted"


def test_korean_script_retries_existing_dense_micro_manual_review_for_absorption(
    tmp_project_dir: Path,
) -> None:
    previous = sample_segment("seg_0001", start=0.0, end=2.0)
    previous.status = "scripted"
    previous.source_script = SourceScript(
        text="ゆっくり聞いてください",
        language="ja",
        backend="mock",
        start=0.0,
        end=2.0,
    )
    previous.translation_ko = KoreanTranslation(
        ko_literal="천천히 들어주세요.",
        ko_natural="천천히 들어주세요.",
        model="mock",
        batch_id="batch_0001",
    )
    previous.script = JapaneseScript(
        literal_ja="ゆっくり聞いてください",
        ja_text="ゆっくり聞いてください",
        tts_text="천천히 들어주세요.",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    micro = sample_segment("seg_0002", start=2.08, end=2.42)
    micro.status = "needs_manual_review"
    micro.errors = [
        "Micro segment too dense for standalone Korean TTS; merge or absorb required.",
        "korean-script skipped segment status needs_manual_review.",
    ]
    micro.source_script = SourceScript(
        text="そう。",
        language="ja",
        backend="mock",
        start=2.08,
        end=2.42,
    )
    micro.translation_ko = KoreanTranslation(
        ko_literal="그래요.",
        ko_natural="그래요.",
        model="mock",
        batch_id="batch_0001",
    )
    manifest = PipelineManifest(segments=[previous, micro])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    previous, micro = manifest.segments
    assert manifest.stage_state["korean-script"]["needs_manual_review"] == 0
    assert manifest.stage_state["korean-script"]["absorbed_micro_segments"] == 1
    assert previous.script is not None
    assert previous.script.tts_text == "천천히 들어주세요. 그래요."
    assert micro.status == "absorbed"
    assert micro.errors == []


def test_korean_script_absorbs_dense_micro_segment_into_following_neighbor_when_closer(
    tmp_project_dir: Path,
) -> None:
    previous = sample_segment("seg_0001", start=0.0, end=1.0)
    previous.status = "transcribed"
    previous.source_script = SourceScript(
        text="ゆっくり",
        language="ja",
        backend="mock",
        start=0.0,
        end=1.0,
    )
    previous.translation_ko = KoreanTranslation(
        ko_literal="천천히요.",
        ko_natural="천천히요.",
        model="mock",
        batch_id="batch_0001",
    )
    micro = sample_segment("seg_0002", start=2.4, end=2.74)
    micro.status = "transcribed"
    micro.source_script = SourceScript(
        text="そう。",
        language="ja",
        backend="mock",
        start=2.4,
        end=2.74,
    )
    micro.translation_ko = KoreanTranslation(
        ko_literal="그래요.",
        ko_natural="그래요.",
        model="mock",
        batch_id="batch_0001",
    )
    following = sample_segment("seg_0003", start=2.82, end=5.0)
    following.status = "transcribed"
    following.source_script = SourceScript(
        text="息をして",
        language="ja",
        backend="mock",
        start=2.82,
        end=5.0,
    )
    following.translation_ko = KoreanTranslation(
        ko_literal="숨을 쉬세요.",
        ko_natural="숨을 쉬세요.",
        model="mock",
        batch_id="batch_0001",
    )
    manifest = PipelineManifest(segments=[previous, micro, following])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    previous, micro, following = manifest.segments
    assert manifest.stage_state["korean-script"]["needs_manual_review"] == 0
    assert manifest.stage_state["korean-script"]["absorbed_micro_segments"] == 1
    assert previous.status == "scripted"
    assert micro.status == "absorbed"
    assert micro.script is None
    assert micro.analysis["korean_tts_absorption"]["absorbed_into_segment_id"] == "seg_0003"
    assert following.status == "scripted"
    assert following.start == pytest.approx(2.4)
    assert following.script is not None
    assert following.script.tts_text == "그래요. 숨을 쉬세요."


def test_korean_script_merges_short_same_speaker_segment_under_one_point_five_seconds(
    tmp_project_dir: Path,
) -> None:
    previous = sample_segment("seg_0001", start=0.0, end=2.0)
    previous.speaker_id = "speaker_0001"
    previous.status = "transcribed"
    previous.source_script = SourceScript(
        text="ゆっくり聞いてください",
        language="ja",
        backend="mock",
        start=0.0,
        end=2.0,
    )
    previous.translation_ko = KoreanTranslation(
        ko_literal="천천히 들어주세요.",
        ko_natural="천천히 들어주세요.",
        model="mock",
        batch_id="batch_0001",
    )
    short = sample_segment("seg_0002", start=2.1, end=3.3)
    short.speaker_id = "speaker_0001"
    short.status = "transcribed"
    short.source_script = SourceScript(
        text="そう。",
        language="ja",
        backend="mock",
        start=2.1,
        end=3.3,
    )
    short.translation_ko = KoreanTranslation(
        ko_literal="그래요.",
        ko_natural="그래요.",
        model="mock",
        batch_id="batch_0001",
    )
    following = sample_segment("seg_0003", start=4.0, end=6.0)
    following.speaker_id = "speaker_0002"
    following.status = "transcribed"
    following.source_script = SourceScript(
        text="息をして",
        language="ja",
        backend="mock",
        start=4.0,
        end=6.0,
    )
    following.translation_ko = KoreanTranslation(
        ko_literal="숨을 쉬세요.",
        ko_natural="숨을 쉬세요.",
        model="mock",
        batch_id="batch_0001",
    )
    manifest = PipelineManifest(segments=[previous, short, following])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    previous, short, following = manifest.segments
    assert manifest.stage_state["korean-script"]["needs_manual_review"] == 0
    assert manifest.stage_state["korean-script"]["absorbed_micro_segments"] == 1
    assert previous.status == "scripted"
    assert previous.end == pytest.approx(3.3)
    assert previous.script is not None
    assert previous.script.tts_text == "천천히 들어주세요. 그래요."
    assert short.status == "absorbed"
    assert short.analysis["korean_tts_absorption"]["absorbed_into_segment_id"] == "seg_0001"
    assert following.status == "scripted"


def test_translate_ko_diagnostics_record_split_and_single_retry_failure(
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
            if len(segments) > 1:
                raise RuntimeError("Could not parse translation JSON array")
            if segments[0].id == "seg_0002":
                raise RuntimeError("single JSON parse failed")
            return {
                segments[0].id: KoreanTranslation(
                    ko_literal="좋아요.",
                    ko_natural="좋아요.",
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
    assert manifest.segments[0].translation_ko is not None
    assert manifest.segments[1].status == "needs_manual_review"
    diagnostics = json.loads(Path(manifest.artifacts["translation_diagnostics"]).read_text("utf-8"))
    assert any(
        attempt["accepted"] is False and attempt["attempt_type"] == "batch"
        for attempt in diagnostics["retry_attempts"]
    )
    assert any(
        attempt["accepted"] is False
        and attempt["attempt_type"] == "single"
        and attempt["segment_ids"] == ["seg_0002"]
        for attempt in diagnostics["retry_attempts"]
    )
    final_by_id = {row["segment_id"]: row for row in diagnostics["final_translation_bundles"]}
    assert final_by_id["seg_0002"]["status"] == "needs_manual_review"
    assert "single JSON parse failed" in final_by_id["seg_0002"]["rejected_reasons"][0]


def test_korean_script_skips_existing_manual_review_translation(tmp_project_dir: Path) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        status="needs_manual_review",
        source_script=SourceScript(
            text="ザーメン媚薬を飲ませます",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="정액 변비약을 먹일게요.",
            ko_natural="정액 변비약을 먹일게요.",
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
    assert manifest.segments[0].script is None


def test_audio_style_effect_tags_flow_into_korean_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    gemma_audio = audio_dir / "seg_0001_gemma.wav"
    mix_audio = audio_dir / "seg_0001_mix.wav"
    write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
    write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(gemma_audio),
        audio_for_mix=str(mix_audio),
        status="transcribed",
        source_script=SourceScript(
            text="声に反響と機械的な加工があります",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="목소리에 반향과 기계적인 처리가 있어요.",
            ko_natural="목소리에 반향과 기계적인 처리가 있어요.",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["segment"] = {"status": "completed"}
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    class FakeGemmaBackend:
        def analyze_audio_style(self, audio_path: Path, segment: Segment, _context: dict[str, object]) -> dict[str, object]:
            assert audio_path == gemma_audio
            return {
                "nonverbal_cues": [],
                "spatial_style": "left_close",
                "style_tags": ["sleepy", "robot", "echo"],
                "emotion": "sleepy",
                "pace": "very_slow",
                "volume": "whisper",
                "estimated_pan": 0.0,
                "keep_original_texture": True,
                "risk_flags": [],
                "confidence": 0.91,
                "voice_training": {
                    "clean_voice": False,
                    "eligible": False,
                    "reason": "same speaker has robotic echo processing",
                    "effect_tags": ["robot", "echo"],
                    "same_speaker_under_effect": True,
                },
            }

    monkeypatch.setattr(pipeline_steps, "create_gemma_backend", lambda *_args, **_kwargs: FakeGemmaBackend())

    audio_style_step(tmp_project_dir, "mock", confirm_rights=True)
    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    scripted = manifest.segments[0]
    assert manifest.stage_state["audio-style"]["tagged_segments"] == 1
    assert scripted.analysis["voice_training"]["effect_tags"] == ["robot", "echo"]
    assert scripted.script is not None
    assert scripted.script.emotion == "sleepy"
    assert scripted.script.pace == "very_slow"
    assert scripted.script.volume == "whisper"
    assert scripted.script.spatial_style == "left_close"
    assert scripted.script.ref_style == "sleepy"
    assert "sleepy" in scripted.script.style_tags
    assert "robot" in scripted.script.style_tags
    assert "echo" in scripted.script.style_tags


def test_audio_style_none_tag_marks_segments_without_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    gemma_audio = audio_dir / "seg_0001_gemma.wav"
    mix_audio = audio_dir / "seg_0001_mix.wav"
    write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
    write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(gemma_audio),
        audio_for_mix=str(mix_audio),
        status="transcribed",
        source_script=SourceScript(
            text="普通の囁きです",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="평범한 속삭임이에요.",
            ko_natural="평범한 속삭임이에요.",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["segment"] = {"status": "completed"}
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    class FakeGemmaBackend:
        def analyze_audio_style(self, audio_path: Path, segment: Segment, _context: dict[str, object]) -> dict[str, object]:
            assert audio_path == gemma_audio
            return {
                "nonverbal_cues": [],
                "spatial_style": "center",
                "style_tags": [],
                "estimated_pan": 0.0,
                "keep_original_texture": True,
                "risk_flags": [],
                "confidence": 0.86,
                "effect_events": [],
                "voice_training": {
                    "clean_voice": True,
                    "eligible": True,
                    "reason": "no audible voice effect",
                    "effect_tags": [],
                    "same_speaker_under_effect": False,
                },
            }

    monkeypatch.setattr(pipeline_steps, "create_gemma_backend", lambda *_args, **_kwargs: FakeGemmaBackend())

    audio_style_step(tmp_project_dir, "mock", confirm_rights=True)
    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    scripted = manifest.segments[0]
    assert manifest.stage_state["audio-style"]["tagged_segments"] == 0
    assert scripted.analysis["voice_training"]["effect_tags"] == ["none"]
    assert scripted.analysis["audio_style"]["effect_tags"] == ["none"]
    assert scripted.analysis["audio_style"]["effect_events"] == []
    assert scripted.script is not None
    assert "none" in scripted.script.style_tags


def test_audio_style_speaker_suspicious_scope_targets_minor_merge_and_low_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"

    def segment(
        segment_id: str,
        *,
        start: float,
        end: float,
        speaker_id: str | None,
        analysis: dict[str, object],
    ) -> Segment:
        gemma_audio = audio_dir / f"{segment_id}_gemma.wav"
        mix_audio = audio_dir / f"{segment_id}_mix.wav"
        write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
        write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
        return Segment(
            id=segment_id,
            start=start,
            end=end,
            duration=end - start,
            audio_for_gemma=str(gemma_audio),
            audio_for_mix=str(mix_audio),
            status="transcribed",
            speaker_id=speaker_id,
            analysis=analysis,
        )

    normal = segment(
        "seg_0001",
        start=0.0,
        end=2.0,
        speaker_id="speaker_0001",
        analysis={
            "source_speaker_assignment": {
                "speaker_id": "speaker_0001",
                "speaker_count": 1,
                "dominant_overlap_ratio": 0.92,
                "overlaps": {"speaker_0001": 1.84},
            }
        },
    )
    minor_merge = segment(
        "seg_0002",
        start=2.0,
        end=5.0,
        speaker_id="speaker_0001",
        analysis={
            "source_speaker_assignment": {
                "speaker_id": "speaker_0004",
                "speaker_count": 1,
                "dominant_overlap_ratio": 0.82,
                "overlaps": {"speaker_0004": 2.46},
            },
            "source_speaker_bucket_normalization": {
                "original_speaker_id": "speaker_0004",
                "merged_into_speaker_id": "speaker_0001",
                "merge_confidence": "low",
                "centroid_similarity": 0.51,
                "clean_training_duration_sec": 7.14,
                "reason": "minor_bucket_auto_merged",
            },
        },
    )
    low_overlap = segment(
        "seg_0003",
        start=5.0,
        end=8.0,
        speaker_id=None,
        analysis={
            "source_speaker_assignment": {
                "speaker_id": None,
                "speaker_count": 1,
                "dominant_overlap_ratio": 0.45,
                "overlaps": {"speaker_0002": 1.35},
            }
        },
    )
    manifest = PipelineManifest(segments=[normal, minor_merge, low_overlap])
    manifest.stage_state["segment"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    calls: list[str] = []

    class FakeGemmaBackend:
        def analyze_audio_style(self, _audio_path: Path, segment: Segment, _context: dict[str, object]) -> dict[str, object]:
            calls.append(segment.id)
            return {
                "nonverbal_cues": [],
                "spatial_style": "center",
                "style_tags": [],
                "estimated_pan": 0.0,
                "keep_original_texture": True,
                "risk_flags": [],
                "confidence": 0.9,
                "effect_events": [],
                "voice_training": {
                    "clean_voice": True,
                    "eligible": True,
                    "reason": "no audible voice effect",
                    "effect_tags": [],
                    "same_speaker_under_effect": False,
                },
            }

    monkeypatch.setattr(pipeline_steps, "create_gemma_backend", lambda *_args, **_kwargs: FakeGemmaBackend())

    audio_style_step(tmp_project_dir, "mock", confirm_rights=True, scope="speaker-suspicious")

    manifest = load_manifest(tmp_project_dir)
    assert calls == ["seg_0002", "seg_0003"]
    assert manifest.stage_state["audio-style"]["scope"] == "speaker_suspicious"
    assert manifest.stage_state["audio-style"]["selected_segments"] == ["seg_0002", "seg_0003"]
    assert manifest.stage_state["audio-style"]["skipped_scope"] == 1
    assert "audio_style" not in manifest.segments[0].analysis
    assert manifest.segments[1].analysis["audio_style"]["effect_tags"] == ["none"]
    assert manifest.segments[2].analysis["audio_style"]["effect_tags"] == ["none"]


def test_audio_style_llama_server_audio_reuses_managed_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    gemma_audio = audio_dir / "seg_0001_gemma.wav"
    mix_audio = audio_dir / "seg_0001_mix.wav"
    write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
    write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(gemma_audio),
        audio_for_mix=str(mix_audio),
        status="transcribed",
        source_script=SourceScript(
            text="声に反響があります",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
    )
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_server_auto_start=True,
            gemma_text_server_url="http://127.0.0.1:18080",
        ),
        segments=[segment],
    )
    manifest.stage_state["segment"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)
    save_project_config(manifest.project_config, tmp_project_dir / "pipeline.yaml")
    events: list[tuple[str, object]] = []

    class FakeServer:
        def __init__(self, **kwargs: object) -> None:
            events.append(("server_init", kwargs))
            self.started = False
            self.reused_existing = False
            self.base_url = str(kwargs["base_url"])

        def start(self) -> None:
            events.append(("server_start", self.base_url))
            self.started = True

        def stop(self) -> None:
            events.append(("server_stop", self.base_url))

    class FakeClient:
        def __init__(self, base_url: str, **kwargs: object) -> None:
            events.append(("client_init", {"base_url": base_url, **kwargs}))

        def analyze_audio_style(self, audio_path: Path, segment: Segment) -> dict[str, object]:
            events.append(("analyze_audio_style", {"audio_path": audio_path, "segment_id": segment.id}))
            return {
                "nonverbal_cues": [],
                "spatial_style": "center",
                "style_tags": ["echo"],
                "estimated_pan": 0.0,
                "keep_original_texture": True,
                "risk_flags": [],
                "confidence": 0.9,
                "voice_training": {
                    "clean_voice": False,
                    "eligible": False,
                    "reason": "echo on same speaker",
                    "effect_tags": ["echo"],
                    "same_speaker_under_effect": True,
                },
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", FakeServer)
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)
    monkeypatch.setattr(pipeline_steps, "_gemma_text_server_command", lambda *args, **kwargs: ["llama-server"])
    monkeypatch.setattr(
        pipeline_steps,
        "create_gemma_backend",
        lambda *_args, **_kwargs: pytest.fail("llama_server_audio should not use per-segment CLI backend"),
    )

    audio_style_step(tmp_project_dir, "llama_server_audio", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["audio-style"]["server"]["base_url"] == "http://127.0.0.1:18080"
    assert manifest.stage_state["audio-style"]["server"]["server_count"] == 1
    assert manifest.segments[0].analysis["voice_training"]["effect_tags"] == ["echo"]
    assert ("server_start", "http://127.0.0.1:18080") in events
    assert ("server_stop", "http://127.0.0.1:18080") in events
    assert any(event[0] == "analyze_audio_style" for event in events)


def test_audio_style_llama_server_audio_keeps_adjacent_segments_separate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    for index, (start, end) in enumerate([(0.0, 1.0), (1.1, 2.1)], start=1):
        segment_id = f"seg_{index:04d}"
        gemma_audio = audio_dir / f"{segment_id}_gemma.wav"
        mix_audio = audio_dir / f"{segment_id}_mix.wav"
        write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
        write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
        segments.append(
            Segment(
                id=segment_id,
                start=start,
                end=end,
                duration=end - start,
                audio_for_gemma=str(gemma_audio),
                audio_for_mix=str(mix_audio),
                status="transcribed",
                source_script=SourceScript(
                    text="普通の囁きです",
                    language="ja",
                    backend="mock",
                    start=start,
                    end=end,
                ),
            )
        )
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_server_auto_start=True,
            gemma_text_server_url="http://127.0.0.1:18080",
            gemma_audio_style_concurrency=2,
            gemma_text_span_size=4,
            gemma_text_span_max_sec=30.0,
            gemma_text_span_max_gap_sec=1.0,
        ),
        segments=segments,
    )
    manifest.stage_state["segment"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)
    save_project_config(manifest.project_config, tmp_project_dir / "pipeline.yaml")
    events: list[tuple[str, object]] = []
    started_segment_ids: list[str] = []
    started_lock = Lock()
    both_requests_started = Event()

    class FakeServer:
        def __init__(self, **kwargs: object) -> None:
            self.started = False
            self.reused_existing = False
            self.base_url = str(kwargs["base_url"])

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            events.append(("server_stop", self.base_url))

    class FakeClient:
        def __init__(self, base_url: str, **kwargs: object) -> None:
            events.append(("client_init", {"base_url": base_url, **kwargs}))

        def analyze_audio_style(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            audio_path, segment = _args
            with started_lock:
                started_segment_ids.append(segment.id)
                if len(started_segment_ids) == 2:
                    both_requests_started.set()
            assert both_requests_started.wait(1.0), "audio-style requests should run concurrently"
            events.append(("analyze_audio_style", {"audio_path": audio_path, "segment_id": segment.id}))
            return {
                "nonverbal_cues": [],
                "spatial_style": "center",
                "style_tags": [],
                "estimated_pan": segment.estimated_pan,
                "keep_original_texture": True,
                "risk_flags": [],
                "confidence": 0.9,
                "effect_events": [],
                "voice_training": {
                    "clean_voice": True,
                    "eligible": True,
                    "reason": "plain segment voice",
                    "effect_tags": [],
                    "same_speaker_under_effect": False,
                },
            }

        def analyze_audio_style_batch(
            self,
            audio_path: Path,
            batch_segments: list[Segment],
            *,
            clip_segments: list[dict[str, object]],
        ) -> dict[str, dict[str, object]]:
            pytest.fail("audio-style should not batch adjacent segments because effects can differ")

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", FakeServer)
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    def fake_server_command(*args: object, **kwargs: object) -> list[str]:
        events.append(("server_command", kwargs))
        return ["llama-server"]

    monkeypatch.setattr(pipeline_steps, "_gemma_text_server_command", fake_server_command)

    audio_style_step(tmp_project_dir, "llama_server_audio", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    style_events = [event for event in events if event[0] == "analyze_audio_style"]
    assert sorted(event[1]["segment_id"] for event in style_events) == ["seg_0001", "seg_0002"]
    server_command_events = [event for event in events if event[0] == "server_command"]
    assert server_command_events[0][1]["parallel_slots"] == 2
    assert manifest.stage_state["audio-style"]["styled"] == 2
    assert manifest.stage_state["audio-style"]["concurrency"] == 2
    assert manifest.stage_state["audio-style"]["server"]["parallel_slots"] == 2
    assert "batch_requests" not in manifest.stage_state["audio-style"]
    assert manifest.segments[0].analysis["audio_style"]["effect_tags"] == ["none"]
    assert manifest.segments[1].analysis["audio_style"]["effect_tags"] == ["none"]


def test_audio_style_skips_existing_analysis_without_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    gemma_audio = audio_dir / "seg_0001_gemma.wav"
    mix_audio = audio_dir / "seg_0001_mix.wav"
    write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
    write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(gemma_audio),
        audio_for_mix=str(mix_audio),
        status="transcribed",
        analysis={
            "audio_style": {
                "backend_task": "audio_style",
                "effect_tags": ["echo"],
                "effect_events": [],
                "confidence": 0.9,
            },
            "voice_training": {
                "clean_voice": False,
                "eligible": False,
                "reason": "existing echo analysis",
                "effect_tags": ["echo"],
                "same_speaker_under_effect": True,
            },
        },
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["segment"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    class FailingGemmaBackend:
        def analyze_audio_style(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            pytest.fail("audio-style should reuse existing segment analysis unless force=True")

    monkeypatch.setattr(pipeline_steps, "create_gemma_backend", lambda *_args, **_kwargs: FailingGemmaBackend())

    audio_style_step(tmp_project_dir, "mock", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["audio-style"]["styled"] == 0
    assert manifest.stage_state["audio-style"]["skipped_existing"] == 1
    assert manifest.stage_state["audio-style"]["tagged_segments"] == 1
    assert manifest.segments[0].analysis["audio_style"]["effect_tags"] == ["echo"]


def test_audio_style_skip_existing_analysis_does_not_start_llama_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    gemma_audio = audio_dir / "seg_0001_gemma.wav"
    mix_audio = audio_dir / "seg_0001_mix.wav"
    write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
    write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(gemma_audio),
        audio_for_mix=str(mix_audio),
        status="transcribed",
        analysis={
            "audio_style": {
                "backend_task": "audio_style",
                "effect_tags": ["none"],
                "effect_events": [],
                "confidence": 0.9,
            },
            "voice_training": {
                "clean_voice": True,
                "eligible": True,
                "reason": "existing plain analysis",
                "effect_tags": ["none"],
                "same_speaker_under_effect": False,
            },
        },
    )
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_server_auto_start=True,
            gemma_text_server_url="http://127.0.0.1:18080",
        ),
        segments=[segment],
    )
    manifest.stage_state["segment"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)
    save_project_config(manifest.project_config, tmp_project_dir / "pipeline.yaml")

    class FailingServer:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("audio-style should not start llama-server when every segment is reused")

    class FailingClient:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pytest.fail("audio-style should not create a llama-server client when every segment is reused")

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", FailingServer)
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FailingClient)

    audio_style_step(tmp_project_dir, "llama_server_audio", confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["audio-style"]["styled"] == 0
    assert manifest.stage_state["audio-style"]["skipped_existing"] == 1
    assert manifest.stage_state["audio-style"]["server"] is None


def test_audio_style_force_reanalyzes_existing_analysis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    gemma_audio = audio_dir / "seg_0001_gemma.wav"
    mix_audio = audio_dir / "seg_0001_mix.wav"
    write_audio(gemma_audio, np.zeros((16_000, 1), dtype=np.float32), 16_000)
    write_audio(mix_audio, np.zeros((48_000, 2), dtype=np.float32), 48_000)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(gemma_audio),
        audio_for_mix=str(mix_audio),
        status="transcribed",
        analysis={
            "audio_style": {
                "backend_task": "audio_style",
                "effect_tags": ["none"],
                "effect_events": [],
                "confidence": 0.8,
            },
            "voice_training": {
                "clean_voice": True,
                "eligible": True,
                "reason": "existing plain analysis",
                "effect_tags": ["none"],
                "same_speaker_under_effect": False,
            },
        },
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["segment"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)
    calls: list[str] = []

    class FakeGemmaBackend:
        def analyze_audio_style(self, audio_path: Path, segment: Segment, _context: dict[str, object]) -> dict[str, object]:
            calls.append(segment.id)
            assert audio_path == gemma_audio
            return {
                "nonverbal_cues": [],
                "spatial_style": "center",
                "style_tags": ["robot"],
                "estimated_pan": 0.0,
                "keep_original_texture": True,
                "risk_flags": [],
                "confidence": 0.93,
                "voice_training": {
                    "clean_voice": False,
                    "eligible": False,
                    "reason": "forced reanalysis found robot processing",
                    "effect_tags": ["robot"],
                    "same_speaker_under_effect": True,
                },
            }

    monkeypatch.setattr(pipeline_steps, "create_gemma_backend", lambda *_args, **_kwargs: FakeGemmaBackend())

    audio_style_step(tmp_project_dir, "mock", confirm_rights=True, force=True)

    manifest = load_manifest(tmp_project_dir)
    assert calls == ["seg_0001"]
    assert manifest.stage_state["audio-style"]["styled"] == 1
    assert manifest.stage_state["audio-style"]["skipped_existing"] == 0
    assert manifest.segments[0].analysis["audio_style"]["effect_tags"] == ["robot"]


@pytest.mark.parametrize(
    ("text", "issue"),
    [("목소에 계속 따만 ة", "korean_tts_contains_pronunciation_symbol")],
)
def test_korean_script_blocks_foreign_symbols(
    tmp_project_dir: Path,
    text: str,
    issue: str,
) -> None:
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
            ko_literal=text,
            ko_natural=text,
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
    assert issue in manifest.segments[0].errors[-1]


@pytest.mark.parametrize(
    "text",
    [
        "그럼 다음 푸",
        "초상 말고, 사람",
    ],
)
def test_korean_script_repairs_truncated_fragments_before_manual_review(
    tmp_project_dir: Path,
    text: str,
) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.5,
        duration=1.5,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        source_script=SourceScript(
            text="少し近づきますね",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.5,
        ),
        translation_ko=KoreanTranslation(
            ko_literal=text,
            ko_natural=text,
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "scripted"
    assert segment.script is not None
    assert segment.script.tts_text == f"{text}..."
    assert segment.analysis["pre_synth_text_qc_recovery"] == "repaired_truncated_sentence"


def test_korean_script_softens_safe_truncated_connector_for_tts(tmp_project_dir: Path) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        source_script=SourceScript(
            text="でも",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="하지만,",
            ko_natural="하지만,",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].status == "scripted"
    assert manifest.segments[0].script is not None
    assert manifest.segments[0].script.tts_text == "하지만..."
    assert (
        manifest.segments[0].analysis["pre_synth_text_qc_recovery"]
        == "softened_truncated_sentence"
    )


def test_korean_script_softens_safe_truncated_malgo_connector(tmp_project_dir: Path) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        source_script=SourceScript(
            text="他のことは考えないで",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="다른 생각은 하지 말고",
            ko_natural="다른 생각은 하지 말고",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].status == "scripted"
    assert manifest.segments[0].script is not None
    assert manifest.segments[0].script.tts_text == "다른 생각은 하지 말고..."


def test_korean_script_recovers_previous_truncated_preflight_manual_review(
    tmp_project_dir: Path,
) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        status="needs_manual_review",
        errors=[
            "Korean TTS preflight blocked synthesis: korean_tts_suspicious_truncated_sentence",
            "korean-script skipped segment status needs_manual_review.",
        ],
        source_script=SourceScript(
            text="でも",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="하지만,",
            ko_natural="하지만,",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].status == "scripted"
    assert manifest.segments[0].errors == []
    assert manifest.segments[0].script is not None
    assert manifest.segments[0].script.tts_text == "하지만..."


def test_translate_ko_cli_accepts_repair_retry_options(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_translate_ko_step(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest()

    monkeypatch.setattr(cli_module, "translate_ko_step", fake_translate_ko_step)

    result = cli_runner.invoke(
        app,
        [
            "translate-ko",
            "-p",
            str(tmp_project_dir),
            "--gemma-text-backend",
            "mock",
            "--retry-failed",
            "--repair-only",
            "--force-retranslate-failed",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["retry_failed"] is True
    assert kwargs["repair_only"] is True
    assert kwargs["force_retranslate_failed"] is True


def test_synth_cli_accepts_only_segments(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_synth_step(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest()

    monkeypatch.setattr(cli_module, "synth_step", fake_synth_step)

    result = cli_runner.invoke(
        app,
        [
            "synth",
            "-p",
            str(tmp_project_dir),
            "--mock",
            "--confirm-rights",
            "--only-segments",
            "seg_0001, seg_0020,,seg_0001",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["only_segment_ids"] == {"seg_0001", "seg_0020"}
    assert kwargs["render_countdowns"] is False


def test_countdown_synth_cli_accepts_only_segments(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_countdown_synth_step(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest()

    monkeypatch.setattr(cli_module, "countdown_synth_step", fake_countdown_synth_step)

    result = cli_runner.invoke(
        app,
        [
            "countdown-synth",
            "-p",
            str(tmp_project_dir),
            "--mock",
            "--confirm-rights",
            "--only-segments",
            "seg_0001 seg_0020",
        ],
    )

    assert result.exit_code == 0, result.output
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["only_segment_ids"] == {"seg_0001", "seg_0020"}


def test_korean_script_cli_runs_stage(
    cli_runner,
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_korean_script_step(*args: object, **kwargs: object) -> PipelineManifest:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return PipelineManifest()

    monkeypatch.setattr(cli_module, "korean_script_step", fake_korean_script_step)

    result = cli_runner.invoke(
        app,
        [
            "korean-script",
            "-p",
            str(tmp_project_dir),
            "--confirm-rights",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"] == (tmp_project_dir.resolve(),)
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["confirm_rights"] is True


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
            "--no-asr-repair",
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
    assert kwargs["asr_repair_enabled"] is False


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
        def __init__(self, **kwargs: object) -> None:
            self.base_url = str(kwargs["base_url"])
            self.command = [str(part) for part in kwargs.get("command", [])]
            self.log_path = kwargs.get("log_path")
            self.started = False
            self.reused_existing = False

        def start(self) -> None:
            self.started = True
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

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", FakeServer)
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    with pytest.raises(RuntimeError, match="Connection refused"):
        translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].translation_ko is None
    assert state["stopped"] is True
    assert manifest.stage_state["translate-ko"]["status"] == "failed"
    assert "Connection refused" in manifest.stage_state["translate-ko"]["error"]

    summary = json.loads((tmp_project_dir / "work" / "translate_ko" / "summary.json").read_text())
    diagnostics = json.loads(
        (tmp_project_dir / "work" / "translate_ko" / "diagnostics.json").read_text()
    )
    assert summary["partial"] is True
    assert summary["server"]["server_count"] == 1
    assert summary["server"]["instances"][0]["base_url"] == "http://127.0.0.1:8080"
    assert summary["server"]["instances"][0]["started"] is True
    assert summary["server"]["instances"][0]["reused_existing"] is False
    assert summary["server"]["instances"][0]["log_path"].endswith("work/translate_ko/llama_server.log")
    assert "llama-server" in summary["server"]["instances"][0]["command_preview"]
    assert diagnostics["server"] == summary["server"]


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
    monkeypatch.setattr(orchestrator, "audio_style_step", step("audio-style"))
    monkeypatch.setattr(orchestrator, "script_step", step("script"))
    monkeypatch.setattr(orchestrator, "transcribe_step", step("transcribe"))
    monkeypatch.setattr(orchestrator, "translate_ko_step", step("translate-ko"))
    monkeypatch.setattr(orchestrator, "korean_script_step", step("korean-script"))
    monkeypatch.setattr(orchestrator, "source_speakers_step", step("source-speakers"))
    monkeypatch.setattr(orchestrator, "prepare_source_voice_refs_step", step("prepare-refs"))
    monkeypatch.setattr(orchestrator, "gsv_few_shot_step", step("gsv-few-shot"))
    monkeypatch.setattr(orchestrator, "synth_step", step("synth"))
    monkeypatch.setattr(orchestrator, "countdown_synth_step", step("countdown-synth"))
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
        "source-speakers",
        "audio-style",
        "prepare-refs",
        "translate-ko",
        "korean-script",
        "gsv-few-shot",
        "synth",
        "countdown-synth",
        "train-rvc",
        "rvc",
        "qc",
        "mix",
        "export",
    ]
    assert calls[3][2]["asr_backend"] == "qwen_asr"
    assert calls[6][1][1] == "hf"
    assert calls[8][1][1] == "llama_server"
    assert calls[11][2]["mock"] is False
    assert calls[11][2]["render_countdowns"] is False
    assert calls[12][2]["mock"] is False
    assert calls[13][2]["mock"] is False
    assert calls[14][2]["mock"] is False
    assert calls[15][1][1] == "mock"

    calls.clear()
    orchestrator.run_pipeline(
        tmp_path / "input.wav",
        tmp_path / "project_llama_cpp",
        confirm_rights=True,
        mock=False,
        gemma_backend="llama_cpp",
        target_language="kr",
        asr_backend="qwen_asr",
    )
    assert calls[6][1][1] == "llama_server_audio"

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


def test_run_pipeline_passes_force_to_train_rvc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(orchestrator, "audio_style_step", step("audio-style"))
    monkeypatch.setattr(orchestrator, "script_step", step("script"))
    monkeypatch.setattr(orchestrator, "transcribe_step", step("transcribe"))
    monkeypatch.setattr(orchestrator, "translate_ko_step", step("translate-ko"))
    monkeypatch.setattr(orchestrator, "korean_script_step", step("korean-script"))
    monkeypatch.setattr(orchestrator, "source_speakers_step", step("source-speakers"))
    monkeypatch.setattr(orchestrator, "prepare_source_voice_refs_step", step("prepare-refs"))
    monkeypatch.setattr(orchestrator, "gsv_few_shot_step", step("gsv-few-shot"))
    monkeypatch.setattr(orchestrator, "synth_step", step("synth"))
    monkeypatch.setattr(orchestrator, "countdown_synth_step", step("countdown-synth"))
    monkeypatch.setattr(orchestrator, "rvc_train_step", step("train-rvc"))
    monkeypatch.setattr(orchestrator, "rvc_step", step("rvc"))
    monkeypatch.setattr(orchestrator, "qc_step", step("qc"))
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
        rvc_train_force=True,
    )

    train_calls = [call for call in calls if call[0] == "train-rvc"]
    assert train_calls
    assert train_calls[0][2]["force"] is True


def test_target_language_ko_auto_demucs_fallback_reruns_asr_after_poor_raw_signal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    project_dir = tmp_path / "project"
    transcribe_call_count = 0

    def step(name: str):
        def inner(*args: object, **kwargs: object) -> PipelineManifest:
            calls.append((name, args, kwargs))
            return PipelineManifest()

        return inner

    def fake_init(path: Path) -> PipelineManifest:
        Path(path).mkdir(parents=True, exist_ok=True)
        calls.append(("init", (path,), {}))
        return PipelineManifest()

    def fake_transcribe(*args: object, **kwargs: object) -> PipelineManifest:
        nonlocal transcribe_call_count
        transcribe_call_count += 1
        calls.append(("transcribe", args, kwargs))
        summary_path = project_dir / "work/transcribe/asr_diagnostics_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if transcribe_call_count == 1:
            summary = {
                "source_separation_fallback": {
                    "recommended": True,
                    "recommended_backend": "demucs",
                    "reasons": ["manual_review_rate"],
                    "metrics": {"manual_review_rate": 0.03},
                },
                "recommend_source_separation_fallback": True,
            }
        else:
            summary = {
                "source_separation_fallback": {
                    "recommended": False,
                    "reason": "source_separation_already_used",
                },
                "recommend_source_separation_fallback": False,
            }
        summary_path.write_text(json.dumps(summary), "utf-8")
        return PipelineManifest(artifacts={"asr_diagnostics_summary": str(summary_path)})

    monkeypatch.setattr(orchestrator, "init_project", fake_init)
    monkeypatch.setattr(orchestrator, "extract_step", step("extract"))
    monkeypatch.setattr(orchestrator, "source_separation_step", step("source-separation"))
    monkeypatch.setattr(orchestrator, "segment_step", step("segment"))
    monkeypatch.setattr(orchestrator, "analyze_step", step("analyze"))
    monkeypatch.setattr(orchestrator, "audio_style_step", step("audio-style"))
    monkeypatch.setattr(orchestrator, "script_step", step("script"))
    monkeypatch.setattr(orchestrator, "transcribe_step", fake_transcribe)
    monkeypatch.setattr(orchestrator, "translate_ko_step", step("translate-ko"))
    monkeypatch.setattr(orchestrator, "korean_script_step", step("korean-script"))
    monkeypatch.setattr(orchestrator, "source_speakers_step", step("source-speakers"))
    monkeypatch.setattr(orchestrator, "prepare_source_voice_refs_step", step("prepare-refs"))
    monkeypatch.setattr(orchestrator, "gsv_few_shot_step", step("gsv-few-shot"))
    monkeypatch.setattr(orchestrator, "synth_step", step("synth"))
    monkeypatch.setattr(orchestrator, "countdown_synth_step", step("countdown-synth"))
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
        project_dir,
        confirm_rights=True,
        mock=False,
        gemma_backend="hf",
        target_language="ko",
        asr_backend="faster_whisper",
        asr_preset="whisper",
        asr_batched_inference=True,
        asr_batch_size=16,
    )

    assert [name for name, _, _ in calls[:6]] == [
        "init",
        "extract",
        "source-separation",
        "transcribe",
        "source-separation",
        "transcribe",
    ]
    source_separation_calls = [call for call in calls if call[0] == "source-separation"]
    assert len(source_separation_calls) == 2
    assert source_separation_calls[1][2] == {"confirm_rights": True, "force": True}
    transcribe_calls = [call for call in calls if call[0] == "transcribe"]
    assert len(transcribe_calls) == 2
    assert transcribe_calls[1][2]["asr_backend"] == "faster_whisper"
    assert transcribe_calls[1][2]["asr_preset"] == "whisper"
    assert transcribe_calls[1][2]["asr_batched_inference"] is True
    assert transcribe_calls[1][2]["asr_batch_size"] == 16
    assert orchestrator.load_project_config(project_dir).source_separation_backend == "demucs"
