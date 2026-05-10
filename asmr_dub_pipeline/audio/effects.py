from __future__ import annotations

from typing import Any

import numpy as np

from asmr_dub_pipeline.audio.features import ensure_stereo
from asmr_dub_pipeline.schemas import Segment

EFFECT_ALIASES: dict[str, str] = {
    "phone": "telephone",
    "telephone_filter": "telephone",
    "telephone_voice": "telephone",
    "radio_voice": "radio",
    "walkie_talkie": "radio",
    "robot_voice": "robot",
    "robotic": "robot",
    "voice_effect": "effect",
    "effected_voice": "effect",
}

EFFECT_ORDER = ("telephone", "radio", "robot", "distortion", "reverb", "echo")
EFFECT_TARGETS_FOR_DIALOGUE = {"voice", "mixed"}


def apply_segment_effects(
    data: np.ndarray,
    sample_rate: int,
    segment: Segment,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Apply deterministic mix-time voice effects requested by segment metadata."""

    out = ensure_stereo(data).astype(np.float32, copy=True)
    applied: list[dict[str, Any]] = []
    if not len(out):
        return out, applied

    for spec in _segment_effect_specs(segment):
        tag = spec["tag"]
        target = spec.get("target", "voice")
        if target not in EFFECT_TARGETS_FOR_DIALOGUE:
            continue
        start = max(0, int(round(float(spec.get("start_sec", 0.0)) * sample_rate)))
        end_sec = spec.get("end_sec")
        end = len(out) if end_sec is None else int(round(float(end_sec) * sample_rate))
        end = min(len(out), max(start, end))
        if end <= start:
            continue
        intensity = _clamp_float(spec.get("intensity"), 1.0, 0.0, 1.0)
        settings = _effect_settings(tag, spec.get("params") or {}, intensity)
        dry = out[start:end].copy()
        wet = _apply_effect(tag, dry, sample_rate, settings)
        out[start:end] = dry * (1.0 - intensity) + wet * intensity
        event: dict[str, Any] = {
            "name": tag,
            "target": target,
            "settings": settings,
            "start_sec": round(start / sample_rate, 6),
            "end_sec": round(end / sample_rate, 6),
        }
        if spec.get("confidence") is not None:
            event["confidence"] = _clamp_float(spec.get("confidence"), 0.0, 0.0, 1.0)
        applied.append(event)

    return np.clip(out, -1.0, 1.0).astype(np.float32), applied


def _segment_effect_specs(segment: Segment) -> list[dict[str, Any]]:
    event_specs = _segment_effect_event_specs(segment)
    if event_specs:
        return event_specs
    return [
        {
            "tag": tag,
            "target": "voice",
            "start_sec": 0.0,
            "end_sec": None,
            "intensity": 1.0,
            "confidence": None,
            "params": {},
        }
        for tag in EFFECT_ORDER
        if tag in _segment_effect_tags(segment)
    ]


def _segment_effect_event_specs(segment: Segment) -> list[dict[str, Any]]:
    analysis = segment.analysis or {}
    containers: list[Any] = []
    audio_style = analysis.get("audio_style")
    if isinstance(audio_style, dict):
        containers.append(audio_style)
    containers.append(analysis)
    specs: list[dict[str, Any]] = []
    for container in containers:
        raw_events = container.get("effect_events") if isinstance(container, dict) else None
        if not isinstance(raw_events, list):
            continue
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue
            tag = EFFECT_ALIASES.get(
                _normalize_token(raw_event.get("tag") or raw_event.get("name")),
                _normalize_token(raw_event.get("tag") or raw_event.get("name")),
            )
            if tag == "none" or tag not in EFFECT_ORDER:
                continue
            target = _normalize_token(raw_event.get("target") or "voice")
            params = raw_event.get("params")
            specs.append(
                {
                    "tag": tag,
                    "target": target,
                    "start_sec": raw_event.get("start_sec", 0.0),
                    "end_sec": raw_event.get("end_sec"),
                    "intensity": raw_event.get("intensity", 1.0),
                    "confidence": raw_event.get("confidence"),
                    "params": dict(params) if isinstance(params, dict) else {},
                }
            )
        if specs:
            return specs
    return specs


def _segment_effect_tags(segment: Segment) -> set[str]:
    values: list[Any] = []
    if segment.script:
        values.extend(segment.script.style_tags)
        values.append(segment.script.ref_style)
        values.append(segment.script.spatial_style)
        values.extend(_cue_values(segment.script.nonverbal_cues))
    analysis = segment.analysis or {}
    for key in ("style_tags", "risk_flags", "voice_flags", "quality_flags", "training_flags", "speech_style"):
        raw = analysis.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif raw is not None:
            values.append(raw)
    voice_training = analysis.get("voice_training")
    if isinstance(voice_training, dict):
        effect_tags = voice_training.get("effect_tags")
        if isinstance(effect_tags, list):
            values.extend(effect_tags)
        elif effect_tags is not None:
            values.append(effect_tags)
    values.extend(_cue_values(analysis.get("nonverbal_cues") or []))
    tags = {_normalize_token(value) for value in values}
    tags.discard("")
    canonical = {EFFECT_ALIASES.get(tag, tag) for tag in tags}
    return {effect for effect in EFFECT_ORDER if effect in canonical}


def _cue_values(cues: Any) -> list[Any]:
    values: list[Any] = []
    for cue in cues or []:
        if isinstance(cue, dict):
            for key in ("kind", "type", "category", "label", "source_text", "normalized_text", "notes"):
                if key in cue:
                    values.append(cue[key])
            continue
        for key in ("kind", "source_text", "normalized_text", "notes"):
            values.append(getattr(cue, key, ""))
    return values


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _float_list(value: Any, default: list[float], low: float, high: float) -> list[float]:
    if not isinstance(value, list):
        return list(default)
    numbers = [_clamp_float(item, default[0], low, high) for item in value]
    return numbers or list(default)


def _effect_settings(tag: str, params: dict[str, Any], intensity: float) -> dict[str, Any]:
    if tag == "telephone":
        return {
            "intensity": intensity,
            "low_hz": _clamp_float(params.get("low_hz"), 300.0, 20.0, 20_000.0),
            "high_hz": _clamp_float(params.get("high_hz"), 3400.0, 20.0, 20_000.0),
            "drive": _clamp_float(params.get("drive"), 1.25, 1.0, 12.0),
        }
    if tag == "radio":
        return {
            "intensity": intensity,
            "low_hz": _clamp_float(params.get("low_hz"), 220.0, 20.0, 20_000.0),
            "high_hz": _clamp_float(params.get("high_hz"), 5000.0, 20.0, 20_000.0),
            "drive": _clamp_float(params.get("drive"), 1.45, 1.0, 12.0),
        }
    if tag == "robot":
        return {
            "intensity": intensity,
            "modulation_hz": _clamp_float(params.get("modulation_hz"), 35.0, 0.1, 500.0),
            "depth": _clamp_float(params.get("depth"), 0.22, 0.0, 1.0),
        }
    if tag == "distortion":
        return {
            "intensity": intensity,
            "drive": _clamp_float(params.get("drive"), 1.8, 1.0, 24.0),
        }
    if tag in {"reverb", "echo"}:
        delays_ms = _float_list(params.get("delays_ms"), [34.0, 71.0], 1.0, 2000.0)
        decays = _float_list(params.get("decays"), [0.18, 0.10], 0.0, 1.0)
        return {
            "intensity": intensity,
            "delays_ms": delays_ms,
            "decays": decays[: len(delays_ms)],
        }
    return {"intensity": intensity}


def _apply_effect(tag: str, data: np.ndarray, sample_rate: int, settings: dict[str, Any]) -> np.ndarray:
    if tag == "telephone":
        return _telephone(
            data,
            sample_rate,
            low_hz=settings["low_hz"],
            high_hz=settings["high_hz"],
            drive=settings["drive"],
        )
    if tag == "radio":
        return _radio(
            data,
            sample_rate,
            low_hz=settings["low_hz"],
            high_hz=settings["high_hz"],
            drive=settings["drive"],
        )
    if tag == "robot":
        return _robot(
            data,
            sample_rate,
            modulation_hz=settings["modulation_hz"],
            depth=settings["depth"],
        )
    if tag == "distortion":
        return _distortion(data, drive=settings["drive"])
    if tag in {"reverb", "echo"}:
        return _echo(data, sample_rate, settings["delays_ms"], settings["decays"])
    return data


def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or len(data) <= 1:
        return data.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    channels = [
        np.convolve(data[:, channel], kernel, mode="same")
        for channel in range(data.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _telephone(data: np.ndarray, sample_rate: int, low_hz: float = 300.0, high_hz: float = 3400.0, drive: float = 1.25) -> np.ndarray:
    high_pass_window = max(3, int(round(sample_rate / max(1.0, low_hz))))
    low_pass_window = max(3, int(round(sample_rate / max(1.0, high_hz))))
    high_passed = data - _moving_average(data, high_pass_window)
    band_limited = _moving_average(high_passed, low_pass_window)
    return _distortion(band_limited, drive=drive) * 0.92


def _radio(data: np.ndarray, sample_rate: int, low_hz: float = 220.0, high_hz: float = 5000.0, drive: float = 1.45) -> np.ndarray:
    high_pass_window = max(3, int(round(sample_rate / max(1.0, low_hz))))
    low_pass_window = max(3, int(round(sample_rate / max(1.0, high_hz))))
    high_passed = data - _moving_average(data, high_pass_window)
    band_limited = _moving_average(high_passed, low_pass_window)
    return _distortion(band_limited, drive=drive) * 0.96


def _robot(data: np.ndarray, sample_rate: int, modulation_hz: float = 35.0, depth: float = 0.22) -> np.ndarray:
    t = np.arange(len(data), dtype=np.float32) / float(sample_rate)
    depth = _clamp_float(depth, 0.22, 0.0, 1.0)
    modulation = (1.0 - depth) + depth * np.sin(2.0 * np.pi * modulation_hz * t)
    return data * modulation[:, None]


def _distortion(data: np.ndarray, drive: float = 1.8) -> np.ndarray:
    drive = max(1.0, float(drive))
    return (np.tanh(data * drive) / np.tanh(drive)).astype(np.float32)


def _echo(data: np.ndarray, sample_rate: int, delays_ms: list[float] | None = None, decays: list[float] | None = None) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    delays = delays_ms or [34.0, 71.0]
    gains = decays or [0.18, 0.10]
    for delay_ms, decay in zip(delays, gains, strict=False):
        delay = int(round(sample_rate * delay_ms / 1000.0))
        if 0 < delay < len(out):
            out[delay:] += data[:-delay] * decay
    return out
