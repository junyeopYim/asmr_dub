from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from asmr_dub_pipeline.audio.mixing import (
    build_dialogue_stem,
    build_segment_mix_plan,
    build_source_suppressed_background,
    db_to_gain,
    mix_with_background,
    overlay,
    overlay_segment,
    reduce_center_speech_bleed,
    suppress_timeline_intervals,
)
from asmr_dub_pipeline.qc.audio_qc import measure_audio_qc
from asmr_dub_pipeline.qc.scoring import score_qc
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    NonverbalCue,
    QCMetadata,
    Segment,
    TTSMetadata,
)

SAMPLE_RATE = 48_000


def _frame(sec: float, sample_rate: int = SAMPLE_RATE) -> int:
    return int(round(sec * sample_rate))


def _sine(
    duration_sec: float,
    *,
    frequency: float = 220.0,
    amplitude: float = 0.08,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 2,
) -> np.ndarray:
    t = np.arange(_frame(duration_sec, sample_rate), dtype=np.float32) / sample_rate
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    if channels == 1:
        return tone[:, None].astype(np.float32)
    return np.repeat(tone[:, None], channels, axis=1).astype(np.float32)


def _room_tone(
    duration_sec: float,
    *,
    amplitude: float = 0.015,
    seed: int = 7,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frames = _frame(duration_sec, sample_rate)
    left = rng.normal(0.0, amplitude, frames)
    right = rng.normal(0.0, amplitude * 0.85, frames)
    return np.stack([left, right], axis=1).astype(np.float32)


def _write_wav(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(data, dtype=np.float32), sample_rate, subtype="FLOAT")
    return path


def _read_wav(path: Path) -> np.ndarray:
    data, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    assert sample_rate == SAMPLE_RATE
    return data


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data))))


def _passing_segment(segment_id: str, start: float, duration: float, pan: float, tts_path: Path) -> Segment:
    return Segment(
        id=segment_id,
        start=start,
        end=start + duration,
        duration=duration,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        estimated_pan=pan,
        status="ok",
        tts=TTSMetadata(selected_candidate_path=str(tts_path)),
        qc=QCMetadata(recommendation="pass", status="ok"),
    )


def test_overlay_uses_deterministic_pan_gain_and_fade_profile() -> None:
    base = np.zeros((_frame(0.22), 2), dtype=np.float32)
    clip = np.ones((_frame(0.12), 1), dtype=np.float32) * 0.25
    pan = -0.5
    gain_db = -6.0
    fade_ms = 10.0
    start_sec = 0.05

    first = overlay(base, clip, SAMPLE_RATE, start_sec, gain_db=gain_db, pan=pan, fade_ms=fade_ms)
    second = overlay(base, clip, SAMPLE_RATE, start_sec, gain_db=gain_db, pan=pan, fade_ms=fade_ms)

    np.testing.assert_array_equal(first, second)
    start = _frame(start_sec)
    end = start + len(clip)
    fade_frames = _frame(fade_ms / 1000.0)
    angle = (pan + 1.0) * np.pi / 4.0
    expected_mid = np.array(
        [
            0.25 * np.cos(angle) * db_to_gain(gain_db),
            0.25 * np.sin(angle) * db_to_gain(gain_db),
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(first[:start], 0.0)
    np.testing.assert_array_equal(first[end:], 0.0)
    np.testing.assert_allclose(first[start], [0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(first[start + fade_frames + 10], expected_mid, rtol=1e-6, atol=1e-7)
    assert 0.45 < first[start + fade_frames // 2, 0] / expected_mid[0] < 0.55
    np.testing.assert_allclose(first[end - 1], [0.0, 0.0], atol=1e-7)


def test_dialogue_stem_pads_timeline_and_preserves_natural_pauses(tmp_path: Path) -> None:
    left_tts = _write_wav(tmp_path / "tts_left.wav", _sine(0.16, frequency=180.0, channels=1))
    right_tts = _write_wav(tmp_path / "tts_right.wav", _sine(0.16, frequency=300.0, channels=1))
    output = tmp_path / "dialogue_stem.wav"
    segments = [
        _passing_segment("seg_left", 0.10, 0.16, -0.75, left_tts),
        _passing_segment("seg_right", 0.55, 0.16, 0.75, right_tts),
    ]

    build_dialogue_stem(segments, output, target_duration_sec=1.0, sample_rate=SAMPLE_RATE)

    stem = _read_wav(output)
    assert stem.shape == (_frame(1.0), 2)
    assert np.max(np.abs(stem[: _frame(0.08)])) == 0.0
    assert np.max(np.abs(stem[_frame(0.30) : _frame(0.50)])) == 0.0
    assert np.max(np.abs(stem[_frame(0.78) :])) == 0.0
    assert _rms(stem[_frame(0.12) : _frame(0.22), 0]) > _rms(stem[_frame(0.12) : _frame(0.22), 1])
    assert _rms(stem[_frame(0.57) : _frame(0.67), 1]) > _rms(stem[_frame(0.57) : _frame(0.67), 0])


def test_dialogue_stem_can_include_segment_with_predicate(tmp_path: Path) -> None:
    tts = _write_wav(tmp_path / "tts.wav", _sine(0.16, frequency=180.0, channels=1))
    segment = _passing_segment("seg_draft", 0.10, 0.16, 0.0, tts)
    segment.status = "needs_regeneration"
    segment.qc = QCMetadata(
        recommendation="regenerate",
        status="needs_regeneration",
        issues=["duration_ratio_out_of_range"],
    )
    default_output = tmp_path / "default.wav"
    draft_output = tmp_path / "draft.wav"

    build_dialogue_stem([segment], default_output, target_duration_sec=0.5, sample_rate=SAMPLE_RATE)
    build_dialogue_stem(
        [segment],
        draft_output,
        target_duration_sec=0.5,
        sample_rate=SAMPLE_RATE,
        include_segment=lambda _: True,
    )

    assert np.max(np.abs(_read_wav(default_output))) == 0.0
    assert np.max(np.abs(_read_wav(draft_output))) > 0.0


def test_overlay_segment_reuses_base_buffer(tmp_path: Path) -> None:
    base = np.zeros((_frame(0.35), 2), dtype=np.float32)
    clip = np.ones((_frame(0.12), 1), dtype=np.float32) * 0.08
    segment = _passing_segment("seg_in_place", 0.10, 0.12, 0.0, tmp_path / "unused.wav")
    plan = build_segment_mix_plan(segment)

    mixed, updated_plan = overlay_segment(base, clip, SAMPLE_RATE, plan)

    assert mixed is base
    assert updated_plan.room_tone_used is False
    assert np.max(np.abs(base)) > 0.0


def test_spatial_style_plan_uses_profile_fade_and_cue_pause_sec(tmp_path: Path) -> None:
    tts_path = _write_wav(tmp_path / "sleepy_tts.wav", _sine(0.18, frequency=240.0, channels=1))
    room_path = _write_wav(tmp_path / "source_room.wav", _room_tone(0.70, amplitude=0.01, seed=13))
    output = tmp_path / "sleepy_dialogue.wav"
    segment = Segment(
        id="seg_sleepy",
        start=0.30,
        end=0.48,
        duration=0.18,
        audio_for_gemma=str(room_path),
        audio_for_mix=str(room_path),
        status="ok",
        script=JapaneseScript(
            ja_text="ゆっくり待ってね。",
            tts_text="ゆっくり待ってね。",
            spatial_style="sleepy_center",
            nonverbal_cues=[NonverbalCue(kind="pause", position=99, pause_sec=0.5)],
        ),
        tts=TTSMetadata(selected_candidate_path=str(tts_path)),
        qc=QCMetadata(recommendation="pass", status="ok"),
    )

    plan = build_segment_mix_plan(segment)
    assert plan.style == "sleepy_center"
    assert plan.fade_in_ms == 72.0
    assert plan.fade_out_ms == 120.0
    assert plan.trailing_pause_sec == 0.5

    build_dialogue_stem([segment], output, target_duration_sec=0.90, sample_rate=SAMPLE_RATE)

    assert segment.mix["fade_in_ms"] == 72.0
    assert segment.mix["fade_out_ms"] == 120.0
    assert segment.mix["trailing_pause_sec"] == 0.5
    assert segment.mix["room_tone_used"] is True


def test_background_bed_is_preserved_without_aggressive_normalization(tmp_path: Path) -> None:
    duration_sec = 0.60
    dialogue = np.zeros((_frame(duration_sec), 2), dtype=np.float32)
    dialogue[_frame(0.10) : _frame(0.24)] = _sine(0.14, amplitude=0.07)
    room_tone = _room_tone(duration_sec, amplitude=0.012, seed=11)
    dialogue_path = _write_wav(tmp_path / "dialogue.wav", dialogue)
    room_tone_path = _write_wav(tmp_path / "room_tone.wav", room_tone)
    output = tmp_path / "final_audio.wav"
    background_gain_db = -12.0

    mix_with_background(
        dialogue_path,
        output,
        room_tone_path,
        background_gain_db=background_gain_db,
        sample_rate=SAMPLE_RATE,
        suppress_background_speech=False,
    )

    mixed = _read_wav(output)
    expected = dialogue + room_tone * db_to_gain(background_gain_db)
    np.testing.assert_allclose(mixed, expected, atol=8e-5)
    pause = slice(_frame(0.32), _frame(0.48))
    assert _rms(mixed[pause]) > 0.001
    assert np.max(np.abs(mixed)) < 0.10


def test_background_speech_bleed_reduction_attenuates_center_voice_band() -> None:
    duration_sec = 0.75
    center_speech = _sine(duration_sec, frequency=1_000.0, amplitude=0.08)
    side_texture = _sine(duration_sec, frequency=300.0, amplitude=0.035)
    side_texture[:, 1] *= -1.0
    source = center_speech + side_texture

    reduced = reduce_center_speech_bleed(source, SAMPLE_RATE)

    center_before = _rms((source[:, 0] + source[:, 1]) * 0.5)
    center_after = _rms((reduced[:, 0] + reduced[:, 1]) * 0.5)
    side_before = _rms((source[:, 0] - source[:, 1]) * 0.5)
    side_after = _rms((reduced[:, 0] - reduced[:, 1]) * 0.5)
    assert center_after < center_before * 0.25
    assert side_after > side_before * 0.80


def test_mix_with_background_reduces_center_speech_by_default(tmp_path: Path) -> None:
    duration_sec = 0.75
    dialogue = np.zeros((_frame(duration_sec), 2), dtype=np.float32)
    center_speech = _sine(duration_sec, frequency=1_000.0, amplitude=0.08)
    background_path = _write_wav(tmp_path / "background.wav", center_speech)
    dialogue_path = _write_wav(tmp_path / "dialogue.wav", dialogue)
    suppressed_output = tmp_path / "suppressed.wav"
    unsuppressed_output = tmp_path / "unsuppressed.wav"

    mix_with_background(
        dialogue_path,
        suppressed_output,
        background_path,
        background_gain_db=0.0,
        sample_rate=SAMPLE_RATE,
    )
    mix_with_background(
        dialogue_path,
        unsuppressed_output,
        background_path,
        background_gain_db=0.0,
        sample_rate=SAMPLE_RATE,
        suppress_background_speech=False,
    )

    assert _rms(_read_wav(suppressed_output)) < _rms(_read_wav(unsuppressed_output)) * 0.25


def test_timeline_suppression_targets_source_speech_intervals() -> None:
    duration_sec = 1.0
    source = _sine(duration_sec, frequency=750.0, amplitude=0.08)
    suppressed = suppress_timeline_intervals(
        source,
        SAMPLE_RATE,
        [(0.25, 0.55)],
        attenuation_db=-36.0,
        pad_sec=0.0,
        fade_ms=5.0,
    )

    before = slice(_frame(0.05), _frame(0.18))
    during = slice(_frame(0.32), _frame(0.48))
    after = slice(_frame(0.72), _frame(0.90))
    assert _rms(suppressed[during]) < _rms(source[during]) * 0.05
    assert _rms(suppressed[before]) > _rms(source[before]) * 0.90
    assert _rms(suppressed[after]) > _rms(source[after]) * 0.90


def test_source_suppressed_background_uses_segment_timeline(tmp_path: Path) -> None:
    duration_sec = 1.0
    source = _sine(duration_sec, frequency=900.0, amplitude=0.08)
    source_path = _write_wav(tmp_path / "source.wav", source)
    output = tmp_path / "suppressed_background.wav"
    segment = Segment(
        id="seg_0001",
        start=0.20,
        end=0.60,
        duration=0.40,
        audio_for_gemma=str(source_path),
        audio_for_mix=str(source_path),
        source_script={
            "text": "source speech",
            "language": "ja",
            "backend": "mock",
            "start": 0.20,
            "end": 0.60,
        },
    )

    build_source_suppressed_background(
        source_path,
        output,
        [segment],
        sample_rate=SAMPLE_RATE,
        attenuation_db=-36.0,
        pad_sec=0.0,
        fade_ms=5.0,
        reduce_center_bleed=False,
    )

    suppressed = _read_wav(output)
    during = slice(_frame(0.30), _frame(0.50))
    outside = slice(_frame(0.72), _frame(0.90))
    assert _rms(suppressed[during]) < _rms(source[during]) * 0.05
    assert _rms(suppressed[outside]) > _rms(source[outside]) * 0.90


def test_score_qc_warns_for_clipping_and_tts_duration_mismatch(tmp_path: Path) -> None:
    clipped_path = _write_wav(tmp_path / "clipped.wav", np.ones((_frame(0.50), 2), dtype=np.float32))
    clipped_metrics = measure_audio_qc(clipped_path, target_duration_sec=0.50)
    clipped_qc = score_qc(clipped_metrics, {"recommendation": "pass"})
    assert "clipping_detected" in clipped_qc.issues
    assert clipped_qc.status == "needs_regeneration"

    too_long_path = _write_wav(tmp_path / "too_long.wav", _sine(1.40))
    too_long_qc = score_qc(
        measure_audio_qc(too_long_path, target_duration_sec=1.0),
        {"recommendation": "pass"},
    )
    assert too_long_qc.duration_ratio and too_long_qc.duration_ratio > 1.35
    assert "duration_ratio_out_of_range" in too_long_qc.issues

    too_short_path = _write_wav(tmp_path / "too_short.wav", _sine(0.50))
    too_short_qc = score_qc(
        measure_audio_qc(too_short_path, target_duration_sec=1.0),
        {"recommendation": "pass"},
    )
    assert too_short_qc.duration_ratio and too_short_qc.duration_ratio < 0.75
    assert "duration_ratio_out_of_range" in too_short_qc.issues


def test_score_qc_warns_when_audio_is_too_loud_for_asmr(tmp_path: Path) -> None:
    loud_path = _write_wav(tmp_path / "loud_but_not_clipped.wav", _sine(0.75, amplitude=0.50))
    metrics = measure_audio_qc(loud_path, target_duration_sec=0.75)

    qc = score_qc(metrics, {"recommendation": "pass"})

    assert metrics["clipping_ratio"] == 0.0
    assert metrics["rms_dbfs"] > -18.0
    assert "too_loud_for_asmr" in qc.issues
    assert qc.status == "needs_regeneration"


def test_score_qc_warns_for_too_much_leading_or_trailing_silence(tmp_path: Path) -> None:
    data = np.concatenate(
        [
            np.zeros((_frame(0.45), 2), dtype=np.float32),
            _sine(0.20, amplitude=0.06),
            np.zeros((_frame(0.55), 2), dtype=np.float32),
        ],
        axis=0,
    )
    silence_path = _write_wav(tmp_path / "excessive_silence.wav", data)
    metrics = measure_audio_qc(silence_path, target_duration_sec=1.20)

    qc = score_qc(metrics, {"recommendation": "pass"})

    assert metrics["leading_silence_sec"] >= 0.44
    assert metrics["trailing_silence_sec"] >= 0.54
    assert "too_much_silence" in qc.issues
    assert qc.status == "needs_regeneration"


def test_score_qc_allows_short_asmr_edge_pause_on_long_segment(tmp_path: Path) -> None:
    data = np.concatenate(
        [
            np.zeros((_frame(0.08), 2), dtype=np.float32),
            _sine(8.80, amplitude=0.06),
            np.zeros((_frame(0.64), 2), dtype=np.float32),
        ],
        axis=0,
    )
    pause_path = _write_wav(tmp_path / "soft_tail_pause.wav", data)
    metrics = measure_audio_qc(pause_path, target_duration_sec=9.52)

    qc = score_qc(metrics, {"recommendation": "pass"})

    assert metrics["trailing_silence_sec"] >= 0.63
    assert "too_much_silence" not in qc.issues
    assert qc.status == "ok"
