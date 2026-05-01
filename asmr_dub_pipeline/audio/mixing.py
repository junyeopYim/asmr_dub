from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asmr_dub_pipeline.schemas import NonverbalCue, Segment

from .features import ensure_stereo, load_audio, resample_linear, write_audio


@dataclass(frozen=True)
class SpatialMixProfile:
    """Deterministic ASMR placement and envelope defaults for one spatial style."""

    style: str
    pan: float
    gain_db: float
    fade_in_ms: float
    fade_out_ms: float
    room_pre_pad_sec: float
    room_post_pad_sec: float
    room_gain_db: float
    pause_scale: float
    max_cue_pause_sec: float
    use_estimated_pan: bool = False


@dataclass(frozen=True)
class SegmentMixPlan:
    style: str
    pan: float
    gain_db: float
    fade_in_ms: float
    fade_out_ms: float
    room_pre_pad_sec: float
    room_post_pad_sec: float
    room_gain_db: float
    leading_pause_sec: float
    trailing_pause_sec: float
    overlay_start_sec: float
    voice_start_sec: float
    room_tone_path: str | None
    room_tone_used: bool = False

    def as_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 6)
        return data


SPATIAL_MIX_PROFILES: dict[str, SpatialMixProfile] = {
    "center": SpatialMixProfile(
        style="center",
        pan=0.0,
        gain_db=-2.0,
        fade_in_ms=24.0,
        fade_out_ms=36.0,
        room_pre_pad_sec=0.06,
        room_post_pad_sec=0.12,
        room_gain_db=-30.0,
        pause_scale=0.12,
        max_cue_pause_sec=0.30,
        use_estimated_pan=True,
    ),
    "left_close": SpatialMixProfile(
        style="left_close",
        pan=-0.72,
        gain_db=-1.0,
        fade_in_ms=18.0,
        fade_out_ms=34.0,
        room_pre_pad_sec=0.08,
        room_post_pad_sec=0.16,
        room_gain_db=-29.0,
        pause_scale=0.14,
        max_cue_pause_sec=0.36,
    ),
    "right_close": SpatialMixProfile(
        style="right_close",
        pan=0.72,
        gain_db=-1.0,
        fade_in_ms=18.0,
        fade_out_ms=34.0,
        room_pre_pad_sec=0.08,
        room_post_pad_sec=0.16,
        room_gain_db=-29.0,
        pause_scale=0.14,
        max_cue_pause_sec=0.36,
    ),
    "center_close": SpatialMixProfile(
        style="center_close",
        pan=0.0,
        gain_db=-1.6,
        fade_in_ms=22.0,
        fade_out_ms=40.0,
        room_pre_pad_sec=0.08,
        room_post_pad_sec=0.18,
        room_gain_db=-30.0,
        pause_scale=0.14,
        max_cue_pause_sec=0.36,
    ),
    "center_far": SpatialMixProfile(
        style="center_far",
        pan=0.0,
        gain_db=-5.0,
        fade_in_ms=46.0,
        fade_out_ms=82.0,
        room_pre_pad_sec=0.18,
        room_post_pad_sec=0.28,
        room_gain_db=-27.0,
        pause_scale=0.16,
        max_cue_pause_sec=0.42,
    ),
    "sleepy_center": SpatialMixProfile(
        style="sleepy_center",
        pan=0.0,
        gain_db=-3.8,
        fade_in_ms=72.0,
        fade_out_ms=120.0,
        room_pre_pad_sec=0.22,
        room_post_pad_sec=0.42,
        room_gain_db=-28.0,
        pause_scale=0.22,
        max_cue_pause_sec=0.60,
    ),
    "ambient": SpatialMixProfile(
        style="ambient",
        pan=0.0,
        gain_db=-6.0,
        fade_in_ms=64.0,
        fade_out_ms=96.0,
        room_pre_pad_sec=0.20,
        room_post_pad_sec=0.34,
        room_gain_db=-26.0,
        pause_scale=0.18,
        max_cue_pause_sec=0.48,
    ),
    "binaural_sweep": SpatialMixProfile(
        style="binaural_sweep",
        pan=0.0,
        gain_db=-2.5,
        fade_in_ms=26.0,
        fade_out_ms=48.0,
        room_pre_pad_sec=0.10,
        room_post_pad_sec=0.22,
        room_gain_db=-29.0,
        pause_scale=0.16,
        max_cue_pause_sec=0.42,
        use_estimated_pan=True,
    ),
}


def db_to_gain(db: float) -> float:
    return float(10 ** (db / 20.0))


def _peak_guard(data: np.ndarray, peak_limit_dbfs: float | None) -> np.ndarray:
    if peak_limit_dbfs is None:
        return data
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    peak_limit = db_to_gain(peak_limit_dbfs)
    if peak > peak_limit > 0:
        return data * (peak_limit / peak)
    return data


def reduce_center_speech_bleed(
    data: np.ndarray,
    sample_rate: int,
    attenuation_db: float = -18.0,
    voice_band_hz: tuple[float, float] = (120.0, 6_000.0),
    dominance_db: float = 2.5,
) -> np.ndarray:
    """Attenuate mid-channel speech-band energy while preserving stereo-side texture.

    This is a deterministic local fallback, not a neural source separator. It is
    intended for ASMR beds where original dialogue is often near-center and the
    useful room/music texture has more side-channel energy.
    """

    stereo = ensure_stereo(data).astype(np.float32, copy=False)
    if stereo.size == 0:
        return stereo.copy()

    mid = ((stereo[:, 0] + stereo[:, 1]) * 0.5).astype(np.float32)
    side = ((stereo[:, 0] - stereo[:, 1]) * 0.5).astype(np.float32)
    n_fft = min(4096, max(512, 2 ** int(np.ceil(np.log2(max(1, min(len(mid), 4096)))))))
    hop = max(1, n_fft // 4)
    window = np.hanning(n_fft).astype(np.float32)
    if not np.any(window):
        return stereo.copy()

    pad = n_fft
    padded_mid = np.pad(mid, (pad, pad), mode="constant")
    padded_side = np.pad(side, (pad, pad), mode="constant")
    out = np.zeros_like(padded_mid, dtype=np.float32)
    norm = np.zeros_like(padded_mid, dtype=np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    low, high = voice_band_hz
    band = (freqs >= low) & (freqs <= high)
    attenuation = db_to_gain(attenuation_db)
    dominance = db_to_gain(dominance_db)

    for start in range(0, len(padded_mid) - n_fft + 1, hop):
        mid_frame = padded_mid[start : start + n_fft] * window
        side_frame = padded_side[start : start + n_fft] * window
        mid_spec = np.fft.rfft(mid_frame)
        side_spec = np.fft.rfft(side_frame)
        mid_mag = np.abs(mid_spec)
        side_mag = np.abs(side_spec)
        speech_like = band & (mid_mag > (side_mag * dominance + 1e-9))
        if np.any(speech_like):
            mid_spec[speech_like] *= attenuation
        reduced = np.fft.irfft(mid_spec, n_fft).astype(np.float32)
        out[start : start + n_fft] += reduced * window
        norm[start : start + n_fft] += window * window

    valid = norm > 1e-8
    out[valid] /= norm[valid]
    reduced_mid = out[pad : pad + len(mid)]
    reduced = np.stack([reduced_mid + side, reduced_mid - side], axis=1)
    return np.clip(reduced, -1.0, 1.0).astype(np.float32)


def suppress_timeline_intervals(
    data: np.ndarray,
    sample_rate: int,
    intervals: list[tuple[float, float]],
    attenuation_db: float = -42.0,
    pad_sec: float = 0.06,
    fade_ms: float = 30.0,
) -> np.ndarray:
    stereo = ensure_stereo(data).astype(np.float32, copy=False)
    if stereo.size == 0 or not intervals:
        return stereo.copy()

    envelope = np.ones(len(stereo), dtype=np.float32)
    low = db_to_gain(attenuation_db)
    pad_frames = int(round(max(0.0, pad_sec) * sample_rate))
    fade_frames = int(round(max(0.0, fade_ms) * sample_rate / 1000.0))
    for start_sec, end_sec in intervals:
        if end_sec <= start_sec:
            continue
        start = max(0, int(round(start_sec * sample_rate)) - pad_frames)
        end = min(len(envelope), int(round(end_sec * sample_rate)) + pad_frames)
        if end <= start:
            continue
        fade_in_start = max(0, start - fade_frames)
        fade_out_end = min(len(envelope), end + fade_frames)
        if start > fade_in_start:
            ramp = np.linspace(1.0, low, start - fade_in_start, endpoint=False, dtype=np.float32)
            envelope[fade_in_start:start] = np.minimum(envelope[fade_in_start:start], ramp)
        envelope[start:end] = np.minimum(envelope[start:end], low)
        if fade_out_end > end:
            ramp = np.linspace(low, 1.0, fade_out_end - end, endpoint=False, dtype=np.float32)
            envelope[end:fade_out_end] = np.minimum(envelope[end:fade_out_end], ramp)
    return (stereo * envelope[:, None]).astype(np.float32)


def build_source_suppressed_background(
    source_path: Path,
    output_path: Path,
    segments: list[Segment],
    sample_rate: int = 48_000,
    attenuation_db: float = -42.0,
    pad_sec: float = 0.06,
    fade_ms: float = 30.0,
    reduce_center_bleed: bool = True,
    peak_limit_dbfs: float | None = -1.0,
) -> Path:
    source = _load_for_mix(source_path, sample_rate)
    intervals = [
        (segment.start, segment.end)
        for segment in segments
        if segment.source_script and segment.source_script.text.strip()
    ]
    suppressed = suppress_timeline_intervals(
        source,
        sample_rate,
        intervals,
        attenuation_db=attenuation_db,
        pad_sec=pad_sec,
        fade_ms=fade_ms,
    )
    if reduce_center_bleed:
        suppressed = reduce_center_speech_bleed(suppressed, sample_rate)
    suppressed = _peak_guard(suppressed, peak_limit_dbfs)
    write_audio(output_path, suppressed, sample_rate)
    return output_path


def apply_fade(data: np.ndarray, sample_rate: int, fade_in_ms: float = 8.0, fade_out_ms: float = 8.0) -> np.ndarray:
    out = np.array(data, dtype=np.float32, copy=True)
    if out.ndim == 1:
        out = out[:, None]
    fade_in = min(len(out), int(sample_rate * fade_in_ms / 1000.0))
    fade_out = min(len(out), int(sample_rate * fade_out_ms / 1000.0))
    if fade_in > 1:
        out[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[:, None]
    if fade_out > 1:
        out[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)[:, None]
    return out


def pan_stereo(data: np.ndarray, pan: float) -> np.ndarray:
    pan = max(-1.0, min(1.0, pan))
    if data.ndim == 2 and data.shape[1] >= 2 and abs(pan) < 1e-6:
        return ensure_stereo(data).astype(np.float32)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    angle = (pan + 1.0) * np.pi / 4.0
    left = np.cos(angle) * mono
    right = np.sin(angle) * mono
    return np.stack([left, right], axis=1).astype(np.float32)


def overlay(
    base: np.ndarray,
    clip: np.ndarray,
    sample_rate: int,
    start_sec: float,
    gain_db: float = 0.0,
    pan: float = 0.0,
    fade_ms: float = 8.0,
) -> np.ndarray:
    out = np.array(base, dtype=np.float32, copy=True)
    stereo_clip = pan_stereo(clip, pan)
    stereo_clip = apply_fade(stereo_clip, sample_rate, fade_ms, fade_ms) * db_to_gain(gain_db)
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(len(out), start + len(stereo_clip))
    if end > start:
        out[start:end] += stereo_clip[: end - start]
    return out


def _load_for_mix(path: Path, target_sr: int) -> np.ndarray:
    data, sr = load_audio(path)
    if sr != target_sr:
        data = resample_linear(data, sr, target_sr)
    return ensure_stereo(data)


def _clip_pan(pan: float) -> float:
    return max(-1.0, min(1.0, float(pan)))


def _script_spatial_style(segment: Segment) -> str:
    if segment.script and segment.script.spatial_style:
        return segment.script.spatial_style
    style = segment.analysis.get("spatial_style")
    return str(style) if style else "center"


def _script_text_len(segment: Segment) -> int:
    if not segment.script:
        return 0
    return max(len(segment.script.tts_text), len(segment.script.ja_text), 1)


def _cue_pause_sec(cue: NonverbalCue, profile: SpatialMixProfile) -> float:
    if cue.pause_sec is not None:
        return max(0.0, min(profile.max_cue_pause_sec, cue.pause_sec))
    text = " ".join([cue.kind, cue.source_text, cue.normalized_text, cue.notes]).lower()
    intensity = max(0.0, min(1.0, cue.intensity))
    if any(token in text for token in ("pause", "silence", "間", "沈黙", "余韻")):
        return (0.18 + 0.32 * intensity) * profile.pause_scale
    if any(
        token in text
        for token in ("breath", "breathe", "sigh", "laugh", "mouth", "息", "吐息", "笑", "リップ")
    ):
        return (0.07 + 0.18 * intensity) * profile.pause_scale
    return 0.0


def _cue_pauses(
    cues: list[NonverbalCue],
    profile: SpatialMixProfile,
    text_len: int,
) -> tuple[float, float]:
    leading = 0.0
    trailing = 0.0
    for cue in cues:
        pause = _cue_pause_sec(cue, profile)
        if pause <= 0:
            continue
        if text_len > 0:
            position = max(0.0, min(1.0, cue.position / text_len))
        else:
            position = 1.0 if cue.position else 0.0
        if position <= 0.25:
            leading += pause
        else:
            trailing += pause
    return (
        round(min(profile.max_cue_pause_sec, leading), 6),
        round(min(profile.max_cue_pause_sec, trailing), 6),
    )


def build_segment_mix_plan(segment: Segment, gain_offset_db: float = 0.0) -> SegmentMixPlan:
    style = _script_spatial_style(segment)
    profile = SPATIAL_MIX_PROFILES.get(style, SPATIAL_MIX_PROFILES["center"])
    pan = segment.estimated_pan if profile.use_estimated_pan else profile.pan
    leading_pause, trailing_pause = _cue_pauses(
        segment.script.nonverbal_cues if segment.script else [],
        profile,
        _script_text_len(segment),
    )
    requested_pre_pad = profile.room_pre_pad_sec + leading_pause
    overlay_start = max(0.0, segment.start - requested_pre_pad)
    effective_pre_pad = max(0.0, segment.start - overlay_start)
    room_pre_pad = min(profile.room_pre_pad_sec, effective_pre_pad)
    leading_pause = max(0.0, effective_pre_pad - room_pre_pad)
    return SegmentMixPlan(
        style=profile.style,
        pan=_clip_pan(pan),
        gain_db=profile.gain_db + gain_offset_db,
        fade_in_ms=profile.fade_in_ms,
        fade_out_ms=profile.fade_out_ms,
        room_pre_pad_sec=room_pre_pad,
        room_post_pad_sec=profile.room_post_pad_sec,
        room_gain_db=profile.room_gain_db,
        leading_pause_sec=leading_pause,
        trailing_pause_sec=trailing_pause,
        overlay_start_sec=overlay_start,
        voice_start_sec=overlay_start + effective_pre_pad,
        room_tone_path=segment.audio_for_mix if segment.keep_original_texture else None,
    )


def _quietest_slice(data: np.ndarray, frames: int, sample_rate: int) -> np.ndarray:
    if frames <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    data = ensure_stereo(data).astype(np.float32, copy=False)
    if len(data) == 0:
        return np.zeros((frames, 2), dtype=np.float32)
    window = min(len(data), max(1, int(round(0.05 * sample_rate))))
    hop = max(1, window // 2)
    best_start = 0
    best_rms = float("inf")
    mono = data.mean(axis=1)
    for start in range(0, max(1, len(mono) - window + 1), hop):
        chunk = mono[start : start + window]
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0
        if rms < best_rms:
            best_start = start
            best_rms = rms
    texture = data[best_start : best_start + window]
    if len(texture) >= frames:
        return texture[:frames].copy()
    repeats = int(np.ceil(frames / max(1, len(texture))))
    return np.tile(texture, (repeats, 1))[:frames].astype(np.float32)


def _room_tone(path: str | None, sample_rate: int, frames: int, gain_db: float) -> np.ndarray:
    if frames <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if not path:
        return np.zeros((frames, 2), dtype=np.float32)
    source_path = Path(path)
    if not source_path.exists():
        return np.zeros((frames, 2), dtype=np.float32)
    try:
        source = _load_for_mix(source_path, sample_rate)
    except Exception:
        return np.zeros((frames, 2), dtype=np.float32)
    tone = _quietest_slice(source, frames, sample_rate) * db_to_gain(gain_db)
    return apply_fade(tone, sample_rate, 12.0, 24.0)


def _prepare_segment_clip(
    clip: np.ndarray,
    sample_rate: int,
    plan: SegmentMixPlan,
) -> tuple[np.ndarray, bool]:
    clip = pan_stereo(clip, plan.pan)
    clip = apply_fade(clip, sample_rate, plan.fade_in_ms, plan.fade_out_ms)
    clip = clip * db_to_gain(plan.gain_db)
    pre_frames = int(round((plan.room_pre_pad_sec + plan.leading_pause_sec) * sample_rate))
    post_frames = int(round((plan.room_post_pad_sec + plan.trailing_pause_sec) * sample_rate))
    pre = _room_tone(plan.room_tone_path, sample_rate, pre_frames, plan.room_gain_db)
    post = _room_tone(plan.room_tone_path, sample_rate, post_frames, plan.room_gain_db)
    room_used = bool((pre.size and np.any(pre)) or (post.size and np.any(post)))
    return np.concatenate([pre, clip, post], axis=0), room_used


def overlay_segment(
    base: np.ndarray,
    clip: np.ndarray,
    sample_rate: int,
    plan: SegmentMixPlan,
) -> tuple[np.ndarray, SegmentMixPlan]:
    prepared, room_used = _prepare_segment_clip(clip, sample_rate, plan)
    updated_plan = SegmentMixPlan(**{**asdict(plan), "room_tone_used": room_used})
    start = max(0, int(round(updated_plan.overlay_start_sec * sample_rate)))
    end = min(len(base), start + len(prepared))
    if end > start:
        base[start:end] += prepared[: end - start]
    return base, updated_plan


def build_dialogue_stem(
    segments: list[Segment],
    output_path: Path,
    target_duration_sec: float,
    sample_rate: int = 48_000,
    dialogue_gain_db: float = 0.0,
    dialogue_fade_ms: float | None = None,
    peak_limit_dbfs: float | None = -1.0,
    progress_callback: Callable[[int, int, Segment], None] | None = None,
    include_segment: Callable[[Segment], bool] | None = None,
) -> Path:
    frames = max(1, int(round(target_duration_sec * sample_rate)))
    base = np.zeros((frames, 2), dtype=np.float32)
    total = len(segments)
    for index, segment in enumerate(segments, start=1):
        included = (
            include_segment(segment)
            if include_segment is not None
            else segment.status == "ok" and segment.qc is not None and segment.qc.recommendation == "pass"
        )
        if not included:
            if progress_callback:
                progress_callback(index, total, segment)
            continue
        selected = segment.rvc.output_path if segment.rvc and segment.rvc.output_path else None
        if selected is None:
            selected = segment.tts.selected_candidate_path if segment.tts else None
        if not selected:
            if progress_callback:
                progress_callback(index, total, segment)
            continue
        clip = _load_for_mix(Path(selected), sample_rate)
        plan = build_segment_mix_plan(segment, gain_offset_db=dialogue_gain_db)
        if dialogue_fade_ms is not None:
            plan = SegmentMixPlan(
                **{
                    **asdict(plan),
                    "fade_in_ms": dialogue_fade_ms,
                    "fade_out_ms": dialogue_fade_ms,
                }
            )
        base, plan = overlay_segment(base, clip, sample_rate, plan)
        segment.mix.update(plan.as_manifest())
        if progress_callback:
            progress_callback(index, total, segment)
    base = _peak_guard(base, peak_limit_dbfs)
    write_audio(output_path, base, sample_rate)
    return output_path


def mix_with_background(
    dialogue_path: Path,
    output_path: Path,
    background_path: Path | None = None,
    background_gain_db: float = -18.0,
    sample_rate: int = 48_000,
    peak_limit_dbfs: float | None = -1.0,
    suppress_background_speech: bool = True,
) -> Path:
    dialogue = _load_for_mix(dialogue_path, sample_rate)
    mix = np.array(dialogue, copy=True)
    if background_path and background_path.exists():
        background = _load_for_mix(background_path, sample_rate)
        if suppress_background_speech:
            background = reduce_center_speech_bleed(background, sample_rate)
        if len(background) < len(mix):
            padded = np.zeros_like(mix)
            padded[: len(background)] = background
            background = padded
        else:
            background = background[: len(mix)]
        mix += background * db_to_gain(background_gain_db)
    mix = _peak_guard(mix, peak_limit_dbfs)
    write_audio(output_path, mix, sample_rate)
    return output_path
