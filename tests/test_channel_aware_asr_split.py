from __future__ import annotations

from pathlib import Path

import numpy as np

from asmr_dub_pipeline.asr.base import ASRChunk, ASRWord
from asmr_dub_pipeline.audio.features import write_audio
from asmr_dub_pipeline.pipeline.stages import common as pipeline_common
from asmr_dub_pipeline.pipeline.stages import transcribe as transcribe_stage
from asmr_dub_pipeline.schemas import (
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceLaneTranscript,
    SourceScript,
)


def _word(start: float, end: float, text: str, confidence: float = 0.95) -> ASRWord:
    return ASRWord(start=start, end=end, text=text, confidence=confidence)


def _segment(segment_id: str, audio_path: Path, *, text: str = "3 2 1 10 0 9 心地よさ") -> Segment:
    return Segment(
        id=segment_id,
        start=100.0,
        end=104.0,
        duration=4.0,
        audio_for_gemma=str(audio_path),
        audio_for_mix=str(audio_path),
        status="needs_manual_review",
        errors=["asr_repair_rejected"],
        source_script=SourceScript(
            text=text,
            language="ja",
            confidence=0.82,
            backend="faster_whisper:segment_retry:no_vad_clean",
            start=100.0,
            end=104.0,
        ),
    )


def test_padded_asr_boundary_clip_rebuilds_text_from_words() -> None:
    chunks = [
        ASRChunk(
            start=0.0,
            end=5.0,
            text="5 4 3 2 1 0",
            language="ja",
            confidence=0.96,
            words=[
                _word(0.1, 0.2, "5"),
                _word(1.1, 1.2, "4"),
                _word(2.1, 2.2, "3"),
                _word(3.1, 3.2, "2"),
                _word(3.4, 3.5, "1"),
                _word(3.7, 3.8, "0"),
            ],
        )
    ]

    result = pipeline_common._clip_asr_chunks_to_window(
        chunks,
        clip_start=98.0,
        clip_end=103.0,
        window_start=100.0,
        window_end=104.0,
        require_word_timestamps_for_boundary=True,
    )

    assert result.reject_reason is None
    assert result.boundary_clipped is True
    assert [chunk.text for chunk in result.chunks] == ["3 2 1 0"]
    assert result.chunks[0].start >= 100.0
    assert all(word.start >= 100.0 for word in result.chunks[0].words)


def test_padded_asr_boundary_clip_rejects_boundary_crossing_text_without_words() -> None:
    chunks = [
        ASRChunk(
            start=0.0,
            end=5.0,
            text="5 4 3 2 1 0",
            language="ja",
            confidence=0.96,
        )
    ]

    result = pipeline_common._clip_asr_chunks_to_window(
        chunks,
        clip_start=98.0,
        clip_end=103.0,
        window_start=100.0,
        window_end=104.0,
        require_word_timestamps_for_boundary=True,
    )

    assert result.chunks == []
    assert result.reject_reason == "boundary_clipping_requires_word_timestamps"


def test_source_lane_transcript_schema_and_config_defaults() -> None:
    cfg = ProjectConfig(project_name="project")
    lane = SourceLaneTranscript(
        lane_id="seg_0068_L",
        channel="left",
        text="2 1 0 心地よさ",
        language="ja",
        confidence=0.94,
        start=506.03,
        end=515.43,
        pan=-0.72,
        spatial_style="left_close",
        backend="faster_whisper:channel_split:left",
        candidate_id="padded",
        clip_start=504.63,
        clip_end=516.83,
        words=[],
        boundary_clipped=True,
        review_reasons=[],
    )
    segment = Segment(
        id="seg_0068_L",
        parent_segment_id="seg_0068",
        source_lane=lane,
        source_lanes=[],
        start=506.03,
        end=515.43,
        duration=9.4,
        audio_for_gemma="seg_0068_L_gemma.wav",
        audio_for_mix="seg_0068_mix.wav",
    )

    assert cfg.asr_channel_split_enabled is True
    assert cfg.asr_channel_split_padding_sec == 1.4
    assert cfg.asr_channel_split_wide_padding_sec == 3.0
    assert cfg.asr_channel_split_max_segments == 80
    assert segment.source_lane == lane
    assert segment.parent_segment_id == "seg_0068"


def test_channel_aware_asr_split_creates_left_and_right_lane_segments(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "segments" / "audio" / "seg_0068_mix.wav"
    samples = np.zeros((48_000, 2), dtype=np.float32)
    write_audio(audio_path, samples, 48_000)
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            asr_channel_split_enabled=True,
            asr_channel_split_max_segments=10,
        ),
        segments=[_segment("seg_0068", audio_path)],
    )
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(transcribe_stage, "duration_sec", lambda _path: 120.0)

    def fake_slice_audio_channel(
        input_path: Path,
        start_sec: float,
        end_sec: float,
        output_path: Path,
        *,
        channel: str,
        sample_rate: int | None = None,
    ) -> Path:
        assert input_path == audio_path
        assert start_sec <= 100.0
        assert end_sec >= 104.0
        assert sample_rate == 16_000
        calls.append((channel, output_path))
        write_audio(output_path, np.zeros((16_000, 1), dtype=np.float32), 16_000)
        return output_path

    class FakeBackend:
        name = "faster_whisper"

        def transcribe_with_options(
            self,
            audio_path: Path,
            _segments: list[Segment],
            **overrides: object,
        ) -> list[ASRChunk]:
            assert overrides["word_timestamps"] is True
            channel = audio_path.name.split("_")[2]
            if channel == "left":
                return [
                    ASRChunk(
                        start=0.0,
                        end=4.0,
                        text="2 1 0 心地よさ",
                        language="ja",
                        confidence=0.94,
                        words=[
                            _word(0.1, 0.2, "2"),
                            _word(0.4, 0.5, "1"),
                            _word(0.7, 0.8, "0"),
                            _word(1.0, 1.4, "心地よさ"),
                        ],
                    )
                ]
            if channel == "right":
                return [
                    ASRChunk(
                        start=0.0,
                        end=4.0,
                        text="10 9 8 7",
                        language="ja",
                        confidence=0.93,
                        words=[
                            _word(0.1, 0.2, "10"),
                            _word(0.4, 0.5, "9"),
                            _word(0.7, 0.8, "8"),
                            _word(1.0, 1.1, "7"),
                        ],
                    )
                ]
            return []

    monkeypatch.setattr(transcribe_stage.ffmpeg, "slice_audio_channel", fake_slice_audio_channel)

    summary = transcribe_stage._apply_channel_aware_asr_split(
        manifest,
        backend=FakeBackend(),
        project_dir=tmp_project_dir,
        audio_duration_sec=120.0,
        cfg=manifest.project_config,
    )

    by_id = {segment.id: segment for segment in manifest.segments}
    assert summary["split"] == 1
    assert by_id["seg_0068"].status == "absorbed"
    assert by_id["seg_0068"].source_lanes
    assert by_id["seg_0068_L"].parent_segment_id == "seg_0068"
    assert by_id["seg_0068_L"].source_script.text == "2 1 0 心地よさ"
    assert by_id["seg_0068_L"].source_lane.channel == "left"
    assert by_id["seg_0068_L"].source_lane.spatial_style == "left_close"
    assert by_id["seg_0068_L"].keep_original_texture is False
    assert by_id["seg_0068_R"].source_script.text == "10 9 8 7"
    assert by_id["seg_0068_R"].source_lane.channel == "right"
    assert by_id["seg_0068_R"].source_lane.spatial_style == "right_close"
    assert sorted({channel for channel, _path in calls}) == ["left", "mid", "right", "side"]


def test_channel_aware_asr_split_uses_local_offsets_for_segment_mix_clips(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "segments" / "audio" / "seg_0177_mix.wav"
    samples = np.zeros((48_000 * 4, 2), dtype=np.float32)
    write_audio(audio_path, samples, 48_000)
    manifest = PipelineManifest(
        project_config=ProjectConfig(
            project_name=tmp_project_dir.name,
            asr_channel_split_enabled=True,
            asr_channel_split_max_segments=10,
        ),
        segments=[_segment("seg_0177", audio_path, text="れ")],
    )
    calls: list[tuple[str, float, float]] = []

    def fake_slice_audio_channel(
        input_path: Path,
        start_sec: float,
        end_sec: float,
        output_path: Path,
        *,
        channel: str,
        sample_rate: int | None = None,
    ) -> Path:
        assert input_path == audio_path
        assert start_sec == 0.0
        assert end_sec == 4.0
        calls.append((channel, start_sec, end_sec))
        write_audio(output_path, np.zeros((48_000 * 4, 1), dtype=np.float32), sample_rate or 16_000)
        return output_path

    class FakeBackend:
        name = "faster_whisper"

        def transcribe_with_options(
            self,
            audio_path: Path,
            _segments: list[Segment],
            **_overrides: object,
        ) -> list[ASRChunk]:
            channel = audio_path.name.split("_")[2]
            text = "左の声" if channel == "left" else "右の声" if channel == "right" else ""
            if not text:
                return []
            return [
                ASRChunk(
                    start=0.0,
                    end=4.0,
                    text=text,
                    language="ja",
                    confidence=0.94,
                    words=[_word(0.5, 1.0, text)],
                )
            ]

    monkeypatch.setattr(transcribe_stage.ffmpeg, "slice_audio_channel", fake_slice_audio_channel)

    summary = transcribe_stage._apply_channel_aware_asr_split(
        manifest,
        backend=FakeBackend(),
        project_dir=tmp_project_dir,
        audio_duration_sec=120.0,
        cfg=manifest.project_config,
    )

    by_id = {segment.id: segment for segment in manifest.segments}
    assert summary["split"] == 1
    assert sorted({channel for channel, _start, _end in calls}) == ["left", "mid", "right", "side"]
    assert by_id["seg_0177_L"].source_lane.words[0].start == 100.5
    assert by_id["seg_0177_R"].source_lane.words[0].end == 101.0


def test_channel_aware_asr_split_skips_mono_audio(tmp_project_dir: Path) -> None:
    audio_path = tmp_project_dir / "work" / "segments" / "audio" / "seg_0068_mix.wav"
    write_audio(audio_path, np.zeros((48_000, 1), dtype=np.float32), 48_000)
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[_segment("seg_0068", audio_path)],
    )

    class UnusedBackend:
        name = "faster_whisper"

    summary = transcribe_stage._apply_channel_aware_asr_split(
        manifest,
        backend=UnusedBackend(),
        project_dir=tmp_project_dir,
        audio_duration_sec=120.0,
        cfg=manifest.project_config,
    )

    assert summary["split"] == 0
    assert summary["items"][0]["reject_reason"] == "audio_not_stereo"
    assert manifest.segments[0].status == "needs_manual_review"


def test_channel_aware_asr_split_demotes_short_empty_repair_reject_to_texture(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    audio_path = tmp_project_dir / "work" / "segments" / "audio" / "seg_0992_mix.wav"
    write_audio(audio_path, np.zeros((48_000 * 4, 2), dtype=np.float32), 48_000)
    segment = Segment(
        id="seg_0992",
        start=1.0,
        end=1.56,
        duration=0.56,
        audio_for_gemma=str(audio_path),
        audio_for_mix=str(audio_path),
        status="needs_manual_review",
        errors=["asr_repair_rejected"],
        keep_original_texture=True,
        analysis={
            "asr_quality_gate": {
                "decision": "block_tts",
                "reasons": ["asr_repair_rejected"],
                "tts_blocked": True,
            }
        },
        source_script=SourceScript(
            text="これ",
            language="ja",
            confidence=0.897,
            backend="faster_whisper",
            start=1.0,
            end=1.56,
        ),
    )
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[segment],
    )

    def fake_slice_audio_channel(
        _input_path: Path,
        _start_sec: float,
        _end_sec: float,
        output_path: Path,
        *,
        channel: str,
        sample_rate: int | None = None,
    ) -> Path:
        write_audio(output_path, np.zeros((16_000, 1), dtype=np.float32), sample_rate or 16_000)
        return output_path

    class EmptyBackend:
        name = "faster_whisper"

        def transcribe_with_options(
            self,
            _audio_path: Path,
            _segments: list[Segment],
            **_overrides: object,
        ) -> list[ASRChunk]:
            return []

    monkeypatch.setattr(transcribe_stage.ffmpeg, "slice_audio_channel", fake_slice_audio_channel)

    summary = transcribe_stage._apply_channel_aware_asr_split(
        manifest,
        backend=EmptyBackend(),
        project_dir=tmp_project_dir,
        audio_duration_sec=4.0,
        cfg=manifest.project_config,
    )

    assert summary["no_dialog_texture"] == 1
    assert summary["items"][0]["reject_reason"] == "no_dialog_texture"
    assert manifest.segments[0].status == "non_speech_texture"
    assert manifest.segments[0].keep_original_texture is True
    assert manifest.segments[0].errors == ["asr_non_speech_texture"]
    assert manifest.segments[0].analysis["asr_quality_gate"] == {
        "decision": "texture",
        "reasons": ["asr_non_speech_texture"],
        "tts_blocked": True,
    }
