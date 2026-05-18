from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from asmr_dub_pipeline.audio.features import ensure_stereo, load_audio
from asmr_dub_pipeline.audio.quality import AudioQualityMetrics, measure_source_voice_quality
from asmr_dub_pipeline.rights import RightsError
from asmr_dub_pipeline.schemas import ProjectConfig, Segment

DISALLOWED_TRAINING_TAGS = {
    "background_music",
    "chorus",
    "distorted",
    "distortion",
    "echo",
    "effect",
    "effected_voice",
    "flanger",
    "multi_speaker",
    "multiple_speakers",
    "overlap",
    "overlapped_speech",
    "processed",
    "processed_voice",
    "radio",
    "reverb",
    "robot",
    "sfx",
    "sound_effect",
    "speaker_overlap",
    "telephone",
    "voice_effect",
}
SOFT_ASMR_TRAINING_TAGS = {
    "breathy",
    "close_mic",
    "close_miked",
    "quiet",
    "low_energy",
    "short_reaction",
    "whisper",
    "whispered",
}


@dataclass(frozen=True)
class VoiceTrainingCandidateCheck:
    accepted: bool
    source_audio_path: Path | None
    metrics: AudioQualityMetrics | None
    reject_reasons: tuple[str, ...]
    clean_source_metrics: dict[str, float | None] = field(default_factory=dict)
    soft_penalty_reasons: tuple[str, ...] = ()


def resolve_project_audio_path(project_dir: Path, raw_path: str, field_name: str) -> Path:
    path = Path(raw_path).expanduser()
    resolved = (project_dir / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(project_dir.resolve())
    except ValueError as exc:
        raise RightsError(f"{field_name} must stay inside the project directory: {resolved}") from exc
    return resolved


def evaluate_voice_training_candidate(
    project_dir: Path,
    segment: Segment,
    cfg: ProjectConfig,
    *,
    min_quality_score: float,
    require_source_script: bool,
    require_speaker_id: bool,
    source_language: str | None = None,
    soft_asmr_policy: Literal["config", "strict", "relaxed"] = "config",
) -> VoiceTrainingCandidateCheck:
    reasons: list[str] = []
    source_audio_path: Path | None = None
    metrics: AudioQualityMetrics | None = None

    manual_reasons, soft_penalty_reasons = _manual_training_reject_reasons(
        segment.analysis,
        cfg,
        soft_asmr_policy=soft_asmr_policy,
    )
    reasons.extend(manual_reasons)
    if require_speaker_id and not segment.speaker_id:
        reasons.append("missing_speaker_id")
    speaker_count = _analysis_speaker_count(segment.analysis)
    if speaker_count is not None and speaker_count != 1:
        reasons.append(f"speaker_count_not_one:{speaker_count}")
    reasons.extend(_analysis_training_reject_reasons(segment.analysis))

    if require_source_script:
        if not segment.source_script or not segment.source_script.text.strip():
            reasons.append("missing_source_script")
        elif source_language and _canonical_language(segment.source_script.language) != _canonical_language(source_language):
            reasons.append(
                f"source_language_mismatch:{_canonical_language(segment.source_script.language)}"
            )

    if not segment.audio_for_mix:
        reasons.append("missing_audio_for_mix")
    else:
        source_audio_path = resolve_project_audio_path(project_dir, segment.audio_for_mix, "audio_for_mix")
        if not source_audio_path.exists():
            reasons.append("missing_audio_file")
        else:
            metrics = measure_source_voice_quality(source_audio_path)
            clean_source_metrics = _clean_source_metrics(project_dir, source_audio_path, segment)
            metrics = _metrics_with_training_quality_score(metrics, clean_source_metrics, cfg)
            if metrics.score < min_quality_score:
                reasons.append(
                    f"quality_score_below_min:{metrics.score:.3f}<{min_quality_score:.3f}"
                )
            reasons.extend(_clean_source_reject_reasons(clean_source_metrics, cfg))
    if source_audio_path is None or metrics is None:
        clean_source_metrics = {}

    return VoiceTrainingCandidateCheck(
        accepted=not reasons,
        source_audio_path=source_audio_path,
        metrics=metrics,
        clean_source_metrics=clean_source_metrics,
        reject_reasons=tuple(dict.fromkeys(reasons)),
        soft_penalty_reasons=tuple(dict.fromkeys(soft_penalty_reasons)),
    )


def _clean_source_metrics(
    project_dir: Path,
    source_audio_path: Path,
    segment: Segment,
) -> dict[str, float | None]:
    source_audio, source_sample_rate = load_audio(source_audio_path)
    source_stereo = ensure_stereo(source_audio)
    source_rms = _rms(source_stereo)
    metrics: dict[str, float | None] = {
        "side_to_mid_db": round(_side_to_mid_db(source_stereo), 6),
        "background_bleed_db": None,
    }
    background_path = project_dir / "work" / "audio" / "background_only_48k.wav"
    if background_path.exists():
        background = _read_timeline_slice(
            background_path,
            segment.start,
            segment.end,
            fallback_frames=len(source_stereo),
            fallback_sample_rate=source_sample_rate,
        )
        if background is not None and background.size:
            metrics["background_bleed_db"] = round(_ratio_db(_rms(background), source_rms), 6)
    return metrics


def _clean_source_reject_reasons(metrics: dict[str, float | None], cfg: ProjectConfig) -> list[str]:
    if not cfg.gsv_few_shot_clean_source_filter:
        return []
    reasons: list[str] = []
    background_bleed_db = metrics.get("background_bleed_db")
    if background_bleed_db is not None and background_bleed_db > cfg.gsv_few_shot_max_background_bleed_db:
        reasons.append(
            "background_bleed_db_above_max:"
            f"{background_bleed_db:.3f}>{cfg.gsv_few_shot_max_background_bleed_db:.3f}"
        )
    side_to_mid_db = metrics.get("side_to_mid_db")
    if side_to_mid_db is not None and side_to_mid_db > cfg.gsv_few_shot_max_side_to_mid_db:
        reasons.append(
            "side_to_mid_db_above_max:"
            f"{side_to_mid_db:.3f}>{cfg.gsv_few_shot_max_side_to_mid_db:.3f}"
        )
    return reasons


def _metrics_with_training_quality_score(
    metrics: AudioQualityMetrics,
    clean_source_metrics: dict[str, float | None],
    cfg: ProjectConfig,
) -> AudioQualityMetrics:
    penalty = _edge_silence_penalty(metrics) + _activity_shape_penalty(metrics)
    if cfg.gsv_few_shot_clean_source_filter:
        penalty += _soft_upper_db_penalty(
            clean_source_metrics.get("background_bleed_db"),
            preferred_max=cfg.gsv_few_shot_max_background_bleed_db - 12.0,
            hard_max=cfg.gsv_few_shot_max_background_bleed_db,
            max_penalty=0.35,
        )
        penalty += _soft_upper_db_penalty(
            clean_source_metrics.get("side_to_mid_db"),
            preferred_max=cfg.gsv_few_shot_max_side_to_mid_db - 6.0,
            hard_max=cfg.gsv_few_shot_max_side_to_mid_db,
            max_penalty=0.20,
        )
    if penalty <= 0.0:
        return metrics
    return replace(metrics, score=max(0.0, min(1.0, metrics.score - penalty)))


def _soft_upper_db_penalty(
    value: float | None,
    *,
    preferred_max: float,
    hard_max: float,
    max_penalty: float,
) -> float:
    if value is None or value <= preferred_max:
        return 0.0
    preferred_to_hard = max(hard_max - preferred_max, 1e-6)
    if value <= hard_max:
        return max_penalty * 0.65 * ((value - preferred_max) / preferred_to_hard)
    overflow = min(1.0, (value - hard_max) / max(abs(hard_max), 1.0))
    return min(max_penalty, max_penalty * (0.65 + 0.35 * overflow))


def _edge_silence_penalty(metrics: AudioQualityMetrics) -> float:
    if metrics.duration_sec <= 0:
        return 0.0
    edge_ratio = (metrics.leading_silence_sec + metrics.trailing_silence_sec) / metrics.duration_sec
    if edge_ratio <= 0.20:
        return 0.0
    return min(0.18, (edge_ratio - 0.20) * 0.30)


def _activity_shape_penalty(metrics: AudioQualityMetrics) -> float:
    if metrics.active_ratio < 0.30:
        return min(0.12, (0.30 - metrics.active_ratio) * 0.40)
    if metrics.active_ratio > 0.98 and metrics.estimated_snr_db is None and metrics.rms_dbfs > -45.0:
        return 0.04
    return 0.0


def _read_timeline_slice(
    path: Path,
    start_sec: float,
    end_sec: float,
    *,
    fallback_frames: int,
    fallback_sample_rate: int,
) -> np.ndarray | None:
    try:
        info = sf.info(str(path))
        start = max(0, int(round(start_sec * info.samplerate)))
        end = max(start, int(round(end_sec * info.samplerate)))
        frames = max(0, end - start)
        if frames == 0:
            frames = max(1, int(round(fallback_frames * info.samplerate / fallback_sample_rate)))
        data, _ = sf.read(str(path), start=start, frames=frames, always_2d=True, dtype="float32")
    except (OSError, RuntimeError, sf.LibsndfileError):
        return None
    if data.size == 0:
        return None
    return ensure_stereo(data)


def _rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0


def _ratio_db(numerator: float, denominator: float) -> float:
    if numerator <= 0:
        return -120.0
    return 20.0 * float(np.log10(numerator / max(denominator, 1e-8)))


def _side_to_mid_db(stereo: np.ndarray) -> float:
    mid = (stereo[:, 0] + stereo[:, 1]) * 0.5
    side = (stereo[:, 0] - stereo[:, 1]) * 0.5
    return _ratio_db(_rms(side), _rms(mid))


def _soft_asmr_relaxation_allowed(
    cfg: ProjectConfig,
    soft_asmr_policy: Literal["config", "strict", "relaxed"],
) -> bool:
    if soft_asmr_policy == "strict":
        return False
    if soft_asmr_policy == "relaxed":
        return True
    return bool(cfg.rvc_train_soft_allow_asmr_texture_for_low_data)


def _manual_training_reject_reasons(
    analysis: dict[str, Any],
    cfg: ProjectConfig,
    *,
    soft_asmr_policy: Literal["config", "strict", "relaxed"] = "config",
) -> tuple[list[str], list[str]]:
    voice_training = analysis.get("voice_training")
    if not isinstance(voice_training, dict):
        return [], []
    reasons: list[str] = []
    soft_penalties: list[str] = []
    hard_effect_tags, soft_effect_tags = _voice_training_effect_tags(voice_training.get("effect_tags"))
    allow_soft_asmr = _soft_asmr_relaxation_allowed(cfg, soft_asmr_policy)
    if voice_training.get("exclude") is True and cfg.rvc_train_respect_manual_exclude:
        if hard_effect_tags:
            reasons.extend(f"manual_training_exclude:hard_effect_tag:{tag}" for tag in hard_effect_tags)
        elif allow_soft_asmr and soft_effect_tags:
            soft_penalties.extend(f"manual_training_exclude_soft_allowed:{tag}" for tag in soft_effect_tags)
        elif soft_effect_tags:
            reasons.extend(
                "manual_training_exclude:soft_effect_tag_requires_low_data_relaxation:"
                f"{tag}"
                for tag in soft_effect_tags
            )
        else:
            reasons.append("manual_training_exclude")
    if voice_training.get("eligible") is False:
        reasons.append("manual_training_ineligible")
    if voice_training.get("clean_voice") is False:
        reasons.append("manual_not_clean_voice")
    reasons.extend(f"voice_training_effect_tag_not_none:{tag}" for tag in hard_effect_tags)
    if allow_soft_asmr:
        soft_penalties.extend(f"voice_training_soft_effect_tag:{tag}" for tag in soft_effect_tags)
    else:
        reasons.extend(
            f"voice_training_soft_effect_tag_requires_low_data_relaxation:{tag}"
            for tag in soft_effect_tags
        )
    return reasons, soft_penalties


def _voice_training_effect_tag_reject_reasons(raw_tags: Any) -> list[str]:
    hard_tags, _soft_tags = _voice_training_effect_tags(raw_tags)
    return [f"voice_training_effect_tag_not_none:{tag}" for tag in hard_tags]


def _voice_training_effect_tags(raw_tags: Any) -> tuple[list[str], list[str]]:
    if isinstance(raw_tags, list):
        values = raw_tags
    elif raw_tags is not None:
        values = [raw_tags]
    else:
        values = []
    tags = [
        _normalize_token(value)
        for value in values
        if _normalize_token(value) and _normalize_token(value) != "none"
    ]
    hard_tags: list[str] = []
    soft_tags: list[str] = []
    for tag in dict.fromkeys(tags):
        if tag in SOFT_ASMR_TRAINING_TAGS:
            soft_tags.append(tag)
        else:
            hard_tags.append(tag)
    return hard_tags, soft_tags


def _analysis_speaker_count(analysis: dict[str, Any]) -> int | None:
    raw = analysis.get("speaker_count")
    if raw is None:
        raw = analysis.get("speakers")
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return None


def _analysis_training_reject_reasons(analysis: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for token in _analysis_tokens(analysis):
        if token in DISALLOWED_TRAINING_TAGS:
            reasons.append(f"disallowed_training_tag:{token}")
    return list(dict.fromkeys(reasons))


def _analysis_tokens(analysis: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in (
        "risk_flags",
        "style_tags",
        "quality_flags",
        "voice_flags",
        "training_flags",
        "speech_style",
    ):
        raw = analysis.get(key)
        if isinstance(raw, list):
            values.extend(raw)
        elif raw is not None:
            values.append(raw)
    for cue in analysis.get("nonverbal_cues") or []:
        if isinstance(cue, dict):
            for key in ("type", "kind", "category", "label"):
                if key in cue:
                    values.append(cue[key])
        else:
            values.append(cue)
    voice_training = analysis.get("voice_training")
    if isinstance(voice_training, dict):
        effect_tags = voice_training.get("effect_tags")
        if isinstance(effect_tags, list):
            values.extend(effect_tags)
        elif effect_tags is not None:
            values.append(effect_tags)
    return [_normalize_token(value) for value in values if _normalize_token(value)]


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_language(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"ja", "jp", "jpn", "japanese"}:
        return "ja"
    if normalized in {"ko", "kr", "kor", "korean"}:
        return "ko"
    return normalized
