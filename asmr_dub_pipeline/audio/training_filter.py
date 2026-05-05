from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class VoiceTrainingCandidateCheck:
    accepted: bool
    source_audio_path: Path | None
    metrics: AudioQualityMetrics | None
    reject_reasons: tuple[str, ...]


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
) -> VoiceTrainingCandidateCheck:
    reasons: list[str] = []
    source_audio_path: Path | None = None
    metrics: AudioQualityMetrics | None = None

    reasons.extend(_manual_training_reject_reasons(segment.analysis))
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
            if metrics.score < min_quality_score:
                reasons.append(
                    f"quality_score_below_min:{metrics.score:.3f}<{min_quality_score:.3f}"
                )

    return VoiceTrainingCandidateCheck(
        accepted=not reasons,
        source_audio_path=source_audio_path,
        metrics=metrics,
        reject_reasons=tuple(dict.fromkeys(reasons)),
    )


def _manual_training_reject_reasons(analysis: dict[str, Any]) -> list[str]:
    voice_training = analysis.get("voice_training")
    if not isinstance(voice_training, dict):
        return []
    reasons: list[str] = []
    if voice_training.get("exclude") is True:
        reasons.append("manual_training_exclude")
    if voice_training.get("eligible") is False:
        reasons.append("manual_training_ineligible")
    if voice_training.get("clean_voice") is False:
        reasons.append("manual_not_clean_voice")
    return reasons


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
