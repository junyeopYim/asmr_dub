from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml

from asmr_dub_pipeline.asr import create_asr_backend, map_chunks_to_segments
from asmr_dub_pipeline.audio import ffmpeg
from asmr_dub_pipeline.audio.features import (
    duration_sec,
    ensure_stereo,
    load_audio,
    resample_linear,
    write_audio,
)
from asmr_dub_pipeline.audio.quality import measure_source_voice_quality
from asmr_dub_pipeline.audio.separation import SourceSeparationUnavailable, separate_source_audio
from asmr_dub_pipeline.config import (
    create_project_structure,
    load_project_config,
    save_project_config,
)
from asmr_dub_pipeline.gpt_sovits.few_shot import train_few_shot
from asmr_dub_pipeline.pipeline.manifest_io import write_json_atomic
from asmr_dub_pipeline.rights import (
    ensure_inside_project,
    require_confirmed_rights,
    sha256_file,
)
from asmr_dub_pipeline.rvc import (
    RVCTrainCommandClient,
    resolve_config_path,
    validate_rvc_training_config,
)
from asmr_dub_pipeline.schemas import (
    GSVSpeakerConfig,
    PipelineManifest,
    ProjectConfig,
    RVCSpeakerConfig,
    Segment,
    SourceScript,
    VoiceBankManifest,
    VoiceBankSourceSegment,
    VoiceBankSpeaker,
    validate_file_safe_id,
)


class VoiceBankError(RuntimeError):
    pass


_MIN_EMBEDDING_AUDIO_SEC = 0.05
_MIN_EMBEDDING_CROP_SEC = 0.50
_SOURCE_SPEAKER_MIN_OVERLAP_RATIO = 0.50
_SOURCE_SPEAKER_MULTI_OVERLAP_RATIO = 0.20
_MISSING = object()
_T = TypeVar("_T")


@dataclass
class DiarizationTurn:
    source_id: str
    source_path: Path
    local_speaker_label: str
    start: float
    end: float
    embedding: np.ndarray | None = None
    text: str | None = None
    language: str | None = None
    quality_score: float | None = None
    speaker_id: str | None = None
    audio_path: Path | None = None
    analysis_audio_path: Path | None = None
    segment_id: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


class MockDiarizationBackend:
    name = "mock"

    def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
        _ = cfg
        duration = _audio_duration(audio_path)
        return [
            DiarizationTurn(
                source_id=source_id,
                source_path=audio_path,
                local_speaker_label="SPEAKER_00",
                start=0.0,
                end=duration,
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            )
        ]

    def embed_clip(self, audio_path: Path) -> np.ndarray:
        _ = audio_path
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class PyannoteDiarizationBackend:
    name = "pyannote"

    def __init__(self, cfg: ProjectConfig) -> None:
        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
        _ensure_hf_cache_env()
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        diarization_model = _resolve_pyannote_model(
            cfg.diarization_model_id,
            cfg,
            label="diarization",
            token=token,
        )
        dependency_models = _ensure_pyannote_pipeline_dependencies(diarization_model, cfg, token)
        embedding_model = _resolve_pyannote_model(
            cfg.diarization_embedding_model_id,
            cfg,
            label="speaker embedding",
            token=token,
        )
        if _all_local_paths([diarization_model, embedding_model, *dependency_models]):
            _force_hf_offline_for_local_cache()
        try:
            import torch  # type: ignore[import-not-found]
            from pyannote.audio import Inference, Model, Pipeline  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional heavy dependency
            raise VoiceBankError(
                "pyannote.audio is not installed. Run `uv sync --extra diarization`, or use "
                "`uv run --extra diarization asmr-dub ...`."
            ) from exc
        _disable_pyannote_tf32(torch)
        try:
            with _block_broken_pyannote_optional_nemo():
                self.pipeline = _pyannote_from_pretrained(
                    Pipeline,
                    diarization_model,
                    token=token,
                )
        except Exception as exc:  # pragma: no cover - depends on local model access
            raise VoiceBankError(
                "Unable to load pyannote diarization model. Accept model terms and set HF_TOKEN, "
                f"or configure a local model path. model={diarization_model}"
            ) from exc
        try:
            with _block_broken_pyannote_optional_nemo():
                model = _pyannote_from_pretrained(
                    Model,
                    embedding_model,
                    token=token,
                )
        except Exception as exc:  # pragma: no cover - depends on local model access
            raise VoiceBankError(
                "Unable to load pyannote speaker embedding model. Accept model terms and set HF_TOKEN, "
                f"or configure a local model path. model={embedding_model}"
            ) from exc
        self.inference = Inference(model, window="whole")
        if "cuda" in cfg.rvc_device.lower() and torch.cuda.is_available():
            device = torch.device("cuda")
            self.pipeline.to(device)
            self.inference.to(device)

    def diarize(self, audio_path: Path, source_id: str, cfg: ProjectConfig) -> list[DiarizationTurn]:
        kwargs: dict[str, Any] = {}
        if cfg.diarization_min_speakers is not None:
            kwargs["min_speakers"] = cfg.diarization_min_speakers
        if cfg.diarization_max_speakers is not None:
            kwargs["max_speakers"] = cfg.diarization_max_speakers
        output = self.pipeline(str(audio_path), **kwargs)
        annotation = getattr(output, "speaker_diarization", output)
        turns: list[DiarizationTurn] = []
        min_turn_duration = _minimum_diarization_turn_sec(cfg)
        skipped_short_turns = 0
        for index, (segment, _, speaker) in enumerate(annotation.itertracks(yield_label=True), start=1):
            start = round(float(segment.start), 3)
            end = round(float(segment.end), 3)
            if end <= start:
                continue
            if end - start < min_turn_duration:
                skipped_short_turns += 1
                continue
            turns.append(
                DiarizationTurn(
                    source_id=source_id,
                    source_path=audio_path,
                    local_speaker_label=str(speaker) or f"SPEAKER_{index:02d}",
                    start=start,
                    end=end,
                    embedding=self.embed_excerpt(audio_path, start, end),
                )
            )
        if not turns:
            detail = (
                f" after dropping {skipped_short_turns} turn(s) shorter than "
                f"{min_turn_duration:.3f}s"
                if skipped_short_turns
                else ""
            )
            raise VoiceBankError(f"pyannote produced no diarization turns for {audio_path}{detail}")
        return turns

    def embed_excerpt(self, audio_path: Path, start: float, end: float) -> np.ndarray:
        from pyannote.core import Segment as PyannoteSegment  # type: ignore[import-not-found]

        crop_start, crop_end = _embedding_crop_bounds(audio_path, start, end)
        embedding = self.inference.crop(str(audio_path), PyannoteSegment(crop_start, crop_end))
        return _normalize_embedding(np.asarray(embedding, dtype=np.float32).reshape(-1))

    def embed_clip(self, audio_path: Path) -> np.ndarray:
        embedding = self.inference(str(audio_path))
        return _normalize_embedding(np.asarray(embedding, dtype=np.float32).reshape(-1))


def resolve_voice_bank_path(project_dir: Path, cfg: ProjectConfig, path: Path | str | None = None) -> Path:
    raw = Path(path or cfg.voice_bank_path).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (project_dir / raw).resolve()
    ensure_inside_project(project_dir, resolved)
    return resolved


def load_voice_bank(project_dir: Path, cfg: ProjectConfig, path: Path | str | None = None) -> VoiceBankManifest:
    voice_bank_path = resolve_voice_bank_path(project_dir, cfg, path)
    if not voice_bank_path.exists():
        raise VoiceBankError(f"Voice bank manifest does not exist: {voice_bank_path}")
    data = json.loads(voice_bank_path.read_text("utf-8"))
    return VoiceBankManifest.model_validate(data)


def save_voice_bank(path: Path, manifest: VoiceBankManifest) -> Path:
    manifest.mark_updated()
    return write_json_atomic(path, manifest.model_dump(mode="json"))


def create_diarization_backend(kind: str, cfg: ProjectConfig):
    normalized = kind.replace("-", "_")
    if normalized == "mock":
        return MockDiarizationBackend()
    if normalized == "pyannote":
        return PyannoteDiarizationBackend(cfg)
    raise VoiceBankError(f"Unsupported speaker assignment backend: {kind}")


def _minimum_diarization_turn_sec(cfg: ProjectConfig) -> float:
    return max(float(cfg.segmentation_min_segment_sec), _MIN_EMBEDDING_AUDIO_SEC)


def _disable_pyannote_tf32(torch_module: Any) -> None:
    try:
        torch_module.backends.cuda.matmul.allow_tf32 = False
        torch_module.backends.cudnn.allow_tf32 = False
    except AttributeError:
        return


@contextmanager
def _block_broken_pyannote_optional_nemo():
    """Make pyannote treat NeMo as unavailable for non-NeMo speaker embeddings."""

    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "nemo" or name.startswith("nemo.")
    }
    had_nemo = "nemo" in sys.modules
    previous_nemo = sys.modules.get("nemo", _MISSING)
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.modules["nemo"] = None
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "nemo" or name.startswith("nemo."):
                sys.modules.pop(name, None)
        if had_nemo:
            sys.modules["nemo"] = previous_nemo
        for name, module in saved_modules.items():
            if name != "nemo":
                sys.modules[name] = module


def _pyannote_from_pretrained(cls: type[_T], model_id_or_path: str, *, token: str | None) -> _T:
    loader = cls.from_pretrained
    try:
        return loader(model_id_or_path, token=token, cache_dir=str(_hf_hub_cache()))
    except TypeError as exc:
        if not _is_unexpected_keyword_error(exc, "token", "cache_dir"):
            raise
        return loader(model_id_or_path, use_auth_token=token)


def _is_unexpected_keyword_error(exc: TypeError, *keywords: str) -> bool:
    message = str(exc)
    return any(f"unexpected keyword argument '{keyword}'" in message for keyword in keywords)


def _embedding_crop_bounds(audio_path: Path, start: float, end: float) -> tuple[float, float]:
    source_duration = max(0.0, _audio_duration(audio_path))
    if source_duration <= 0.0:
        return max(0.0, start), max(0.0, end)
    crop_start = min(max(0.0, start), source_duration)
    crop_end = min(max(crop_start, end), source_duration)
    target_duration = min(_MIN_EMBEDDING_CROP_SEC, source_duration)
    if crop_end - crop_start >= target_duration:
        return crop_start, crop_end
    midpoint = (crop_start + crop_end) / 2.0
    crop_start = max(0.0, midpoint - target_duration / 2.0)
    crop_end = min(source_duration, crop_start + target_duration)
    crop_start = max(0.0, crop_end - target_duration)
    return crop_start, crop_end


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", _repo_root() / ".cache" / "huggingface")).expanduser().resolve()


def _hf_hub_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE", _hf_home() / "hub")).expanduser().resolve()


def _ensure_hf_cache_env() -> None:
    hf_home = _hf_home()
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))


def _force_hf_offline_for_local_cache() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _all_local_paths(paths: list[str]) -> bool:
    return all(Path(path).expanduser().exists() for path in paths)


def _hf_repo_cache_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _cached_hf_snapshot(model_id: str) -> Path | None:
    snapshot_root = _hf_hub_cache() / _hf_repo_cache_name(model_id) / "snapshots"
    if not snapshot_root.exists():
        return None
    snapshots = [path for path in snapshot_root.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    with_config = [path for path in snapshots if (path / "config.yaml").exists()]
    candidates = with_config or snapshots
    return max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve()


def _download_hf_snapshot(model_id: str, token: str | None) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - dependency of pyannote.audio
        raise VoiceBankError(
            "huggingface_hub is required to download pyannote models. "
            "Run `uv sync --extra diarization` first."
        ) from exc
    try:
        return Path(
            snapshot_download(
                repo_id=model_id,
                token=token,
                cache_dir=str(_hf_hub_cache()),
                local_files_only=False,
            )
        ).resolve()
    except Exception as exc:  # pragma: no cover - network/token dependent
        raise VoiceBankError(
            f"Unable to download pyannote model {model_id!r}. Accept the model terms on Hugging Face "
            "and set HF_TOKEN if the repository requires gated access."
        ) from exc


def _resolve_pyannote_model(
    model_id_or_path: str,
    cfg: ProjectConfig,
    *,
    label: str,
    token: str | None,
) -> str:
    raw = model_id_or_path.strip()
    if not raw:
        raise VoiceBankError(f"pyannote {label} model id/path must not be empty.")
    path = Path(raw).expanduser()
    if path.exists():
        return str(path.resolve())
    cached = _cached_hf_snapshot(raw)
    if cached is not None:
        return str(cached)
    if not cfg.diarization_auto_download:
        return raw
    return str(_download_hf_snapshot(raw, token))


def _ensure_pyannote_pipeline_dependencies(
    diarization_model: str,
    cfg: ProjectConfig,
    token: str | None,
) -> list[str]:
    config_path = _pyannote_config_path(diarization_model)
    if config_path is None:
        return []
    try:
        data = yaml.safe_load(config_path.read_text("utf-8")) or {}
    except Exception:
        return []
    params = data.get("pipeline", {}).get("params", {}) if isinstance(data, dict) else {}
    if not isinstance(params, dict):
        return []
    resolved: list[str] = []
    for key in ("segmentation", "embedding"):
        value = params.get(key)
        if isinstance(value, str) and _is_hf_model_id(value):
            resolved.append(
                _resolve_pyannote_model(
                    value,
                    cfg,
                    label=f"diarization {key}",
                    token=token,
                )
            )
    return resolved


def _pyannote_config_path(model_path: str) -> Path | None:
    path = Path(model_path).expanduser()
    if path.is_dir() and (path / "config.yaml").exists():
        return path / "config.yaml"
    if path.is_file() and path.name == "config.yaml":
        return path
    return None


def _is_hf_model_id(value: str) -> bool:
    return "/" in value and not value.startswith("$model/") and not Path(value).expanduser().exists()


def cluster_turns(turns: list[DiarizationTurn], threshold: float = 0.78) -> list[DiarizationTurn]:
    centroids: list[np.ndarray] = []
    labels: list[str] = []
    for turn in turns:
        embedding = _normalize_embedding(turn.embedding) if turn.embedding is not None else None
        if embedding is None:
            key = turn.local_speaker_label
            if key not in labels:
                labels.append(key)
                centroids.append(np.zeros(1, dtype=np.float32))
            turn.speaker_id = f"speaker_{labels.index(key) + 1:04d}"
            continue
        best_index = -1
        best_score = -1.0
        for index, centroid in enumerate(centroids):
            if centroid.shape != embedding.shape:
                continue
            score = float(np.dot(centroid, embedding))
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= threshold:
            turn.speaker_id = f"speaker_{best_index + 1:04d}"
            centroids[best_index] = _normalize_embedding((centroids[best_index] + embedding) / 2.0)
        else:
            centroids.append(embedding)
            turn.speaker_id = f"speaker_{len(centroids):04d}"
    return turns


def apply_voice_bank_to_config(
    project_dir: Path,
    cfg: ProjectConfig,
    bank: VoiceBankManifest,
) -> ProjectConfig:
    gsv_speakers = {
        speaker_id: GSVSpeakerConfig(
            gpt_weights_path=str(_resolve_project_or_absolute(project_dir, speaker.gsv.gpt_weights_path))
            if speaker.gsv.gpt_weights_path
            else None,
            sovits_weights_path=str(_resolve_project_or_absolute(project_dir, speaker.gsv.sovits_weights_path)),
            refs_path=str(_resolve_project_or_absolute(project_dir, speaker.gsv.refs_path)),
            default_ref_style=speaker.gsv.default_ref_style,
        )
        for speaker_id, speaker in bank.speakers.items()
    }
    rvc_speakers = {
        speaker_id: RVCSpeakerConfig(
            model_path=str(_resolve_project_or_absolute(project_dir, speaker.rvc.model_path)),
            index_path=str(_resolve_project_or_absolute(project_dir, speaker.rvc.index_path))
            if speaker.rvc.index_path
            else None,
            f0_method=speaker.rvc.f0_method,
            index_rate=speaker.rvc.index_rate,
            f0_up_key=speaker.rvc.f0_up_key,
            filter_radius=speaker.rvc.filter_radius,
            resample_sr=speaker.rvc.resample_sr,
            rms_mix_rate=speaker.rvc.rms_mix_rate,
            protect=speaker.rvc.protect,
        )
        for speaker_id, speaker in bank.speakers.items()
    }
    return type(cfg).model_validate(
        {
            **cfg.model_dump(mode="json"),
            "gsv_speaker_models": {key: value.model_dump(mode="json") for key, value in gsv_speakers.items()},
            "rvc_speaker_models": {key: value.model_dump(mode="json") for key, value in rvc_speakers.items()},
            "rvc_train_required": False,
            "speaker_assignment_backend": "pyannote",
            "diarization_auto_download": True,
            "diarization_embedding_match_threshold": cfg.diarization_embedding_match_threshold,
        }
    )


def validate_voice_bank_models(project_dir: Path, bank: VoiceBankManifest) -> None:
    errors: list[str] = []
    if not bank.speakers:
        errors.append("voice bank contains no speakers")
    for speaker_id, speaker in bank.speakers.items():
        if not speaker.gsv.gpt_weights_path:
            errors.append(f"{speaker_id} GPT weights missing: <not configured>")
        for label, raw_path in (
            ("GPT weights", speaker.gsv.gpt_weights_path),
            ("SoVITS weights", speaker.gsv.sovits_weights_path),
            ("refs", speaker.gsv.refs_path),
            ("RVC model", speaker.rvc.model_path),
        ):
            if not raw_path:
                continue
            path = _resolve_project_or_absolute(project_dir, raw_path)
            if not path.exists():
                errors.append(f"{speaker_id} {label} missing: {path}")
        if speaker.rvc.index_path:
            index_path = _resolve_project_or_absolute(project_dir, speaker.rvc.index_path)
            if not index_path.exists():
                errors.append(f"{speaker_id} RVC index missing: {index_path}")
    if errors:
        raise VoiceBankError("Invalid voice bank: " + "; ".join(errors))


def assign_speakers_to_manifest(
    project_dir: Path,
    manifest: PipelineManifest,
    bank: VoiceBankManifest,
    *,
    backend_kind: str = "mock",
    require_all: bool = True,
) -> PipelineManifest:
    cfg = manifest.project_config
    backend = create_diarization_backend(backend_kind, cfg) if backend_kind == "pyannote" else MockDiarizationBackend()
    source_path = Path(manifest.source_info.path).resolve() if manifest.source_info else None
    source_segments = [
        segment
        for speaker in bank.speakers.values()
        for segment in speaker.source_segments
    ]
    matched = 0
    for segment in manifest.segments:
        if segment.status in {"failed", "needs_manual_review"}:
            continue
        speaker_id = _speaker_by_source_overlap(source_path, segment, source_segments)
        if speaker_id is None and backend_kind == "pyannote":
            speaker_id = _speaker_by_embedding(
                project_dir,
                segment,
                bank,
                backend,
                threshold=cfg.diarization_embedding_match_threshold,
            )
        if speaker_id is None and not require_all and bank.speakers:
            speaker_id = sorted(bank.speakers)[0]
        if speaker_id is not None:
            segment.speaker_id = speaker_id
            matched += 1
    missing = [
        segment.id
        for segment in manifest.segments
        if segment.status not in {"failed", "needs_manual_review"}
        and (not segment.speaker_id or segment.speaker_id not in bank.speakers)
    ]
    if missing and require_all:
        raise VoiceBankError(
            "Voice bank speaker assignment failed for segments: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )
    manifest.artifacts["voice_bank"] = str(resolve_voice_bank_path(project_dir, cfg))
    manifest.stage_state["speaker-assign"] = {
        "status": "completed",
        "backend": backend_kind,
        "matched_segments": matched,
        "missing_segments": missing,
        "speaker_count": len(bank.speakers),
    }
    return manifest


def assign_source_speakers_to_manifest(
    project_dir: Path,
    manifest: PipelineManifest,
    *,
    backend_kind: str = "mock",
) -> PipelineManifest:
    cfg = manifest.project_config
    audio_path = _source_speaker_audio_path(project_dir, manifest)
    backend = create_diarization_backend(backend_kind, cfg) if backend_kind == "pyannote" else MockDiarizationBackend()
    turns = backend.diarize(audio_path, "source", cfg)
    cluster_turns(turns, threshold=cfg.diarization_embedding_match_threshold)
    assigned = 0
    excluded = 0
    for segment in manifest.segments:
        if segment.status in {"failed", "needs_manual_review"}:
            continue
        assignment = _source_speaker_assignment(segment, turns)
        segment.analysis["speaker_count"] = assignment["speaker_count"]
        segment.analysis["source_speaker_assignment"] = assignment
        if assignment["speaker_count"] == 1 and assignment["speaker_id"]:
            segment.speaker_id = str(assignment["speaker_id"])
            assigned += 1
            voice_training = dict(segment.analysis.get("voice_training") or {})
            if voice_training.get("reason") in {"multi_speaker_overlap", "no_source_speaker_match"}:
                voice_training.pop("exclude", None)
                voice_training.pop("reason", None)
            if voice_training:
                segment.analysis["voice_training"] = voice_training
            else:
                segment.analysis.pop("voice_training", None)
            continue
        segment.speaker_id = None
        excluded += 1
        reason = "multi_speaker_overlap" if assignment["speaker_count"] > 1 else "no_source_speaker_match"
        segment.analysis["voice_training"] = {"exclude": True, "reason": reason}
    manifest.artifacts["source_speaker_audio"] = str(audio_path)
    manifest.stage_state["source-speakers"] = {
        "status": "completed",
        "backend": backend_kind,
        "turn_count": len(turns),
        "assigned_segments": assigned,
        "excluded_segments": excluded,
    }
    return manifest


def _source_speaker_audio_path(project_dir: Path, manifest: PipelineManifest) -> Path:
    for key in ("source_vocals_48k", "original_stereo_48k"):
        raw_path = manifest.artifacts.get(key)
        if raw_path:
            path = _resolve_project_or_absolute(project_dir, raw_path)
            if path.exists():
                return path
    if manifest.source_info:
        path = Path(manifest.source_info.path).expanduser().resolve()
        if path.exists():
            return path
    raise VoiceBankError("source-speakers requires source_vocals_48k, original_stereo_48k, or source_info.path.")


def _source_speaker_assignment(segment: Segment, turns: list[DiarizationTurn]) -> dict[str, Any]:
    overlaps: dict[str, float] = {}
    for turn in turns:
        if not turn.speaker_id:
            continue
        overlap = max(0.0, min(segment.end, turn.end) - max(segment.start, turn.start))
        if overlap <= 0:
            continue
        overlaps[turn.speaker_id] = overlaps.get(turn.speaker_id, 0.0) + overlap
    duration = max(segment.duration, 0.001)
    active = {
        speaker_id: overlap
        for speaker_id, overlap in overlaps.items()
        if overlap / duration >= _SOURCE_SPEAKER_MULTI_OVERLAP_RATIO
    }
    if not active:
        return {
            "speaker_id": None,
            "speaker_count": 0,
            "overlaps": {},
            "dominant_overlap_ratio": 0.0,
        }
    speaker_id, overlap = max(active.items(), key=lambda item: (item[1], item[0]))
    ratio = overlap / duration
    speaker_count = len(active)
    selected_speaker_id = (
        speaker_id if speaker_count == 1 and ratio >= _SOURCE_SPEAKER_MIN_OVERLAP_RATIO else None
    )
    return {
        "speaker_id": selected_speaker_id,
        "speaker_count": speaker_count,
        "overlaps": {key: round(value, 6) for key, value in sorted(active.items())},
        "dominant_overlap_ratio": round(ratio, 6),
    }


def build_voice_bank(
    input_paths: list[Path],
    project_dir: Path,
    *,
    confirm_rights: bool,
    backend_kind: str = "pyannote",
    mock_training: bool = False,
    force: bool = False,
) -> VoiceBankManifest:
    if not input_paths:
        raise VoiceBankError("voice-bank-build requires at least one input media path.")
    create_project_structure(project_dir)
    cfg = load_project_config(project_dir)
    backend = create_diarization_backend(backend_kind, cfg)
    voice_bank_path = resolve_voice_bank_path(project_dir, cfg)
    previous_bank = load_voice_bank(project_dir, cfg) if voice_bank_path.exists() and not force else None
    if force and voice_bank_path.parent.exists():
        shutil.rmtree(voice_bank_path.parent)
    voice_bank_path.parent.mkdir(parents=True, exist_ok=True)
    rights_history: list[dict[str, Any]] = []
    all_turns: list[DiarizationTurn] = []
    for source_index, input_path in enumerate(input_paths, start=1):
        input_path = input_path.expanduser().resolve()
        audit = require_confirmed_rights(
            confirm_rights,
            "voice-bank-build",
            input_path,
            metadata={"backend": backend_kind},
        )
        rights_history.extend(audit.history)
        source_id = _safe_id(f"src_{source_index:04d}_{input_path.stem}")
        source_dir = ensure_inside_project(project_dir, project_dir / "voice_bank" / "sources" / source_id)
        source_audio = source_dir / "source_stereo_48k.wav"
        _extract_stereo_48k(input_path, source_audio)
        analysis_audio = _voice_bank_analysis_audio(
            source_audio,
            source_dir,
            cfg,
            mock=mock_training or backend_kind == "mock",
            force=True,
        )
        turns = backend.diarize(analysis_audio, source_id, cfg)
        for turn in turns:
            turn.analysis_audio_path = analysis_audio
            turn.source_path = input_path
        _transcribe_turns(analysis_audio, turns, cfg, mock_training or backend_kind == "mock")
        all_turns.extend(turns)
    cluster_turns(all_turns, threshold=cfg.diarization_embedding_match_threshold)
    speakers: dict[str, VoiceBankSpeaker] = {}
    for turn_index, turn in enumerate(all_turns, start=1):
        if not turn.speaker_id:
            raise VoiceBankError("Internal error: diarization turn was not assigned a global speaker_id.")
        speaker_id = validate_file_safe_id(turn.speaker_id, "speaker_id")
        segment_id = _safe_id(f"{turn.source_id}_{turn.local_speaker_label}_{turn_index:04d}")
        turn.segment_id = segment_id
        speaker_dir = ensure_inside_project(project_dir, project_dir / "voice_bank" / "speakers" / speaker_id)
        clip_path = speaker_dir / "clips" / f"{segment_id}.wav"
        _write_audio_slice(turn.analysis_audio_path or turn.source_path, turn.start, turn.end, clip_path)
        turn.audio_path = clip_path
        try:
            turn.quality_score = measure_source_voice_quality(clip_path).score
        except Exception:
            turn.quality_score = None
    for speaker_id in sorted({turn.speaker_id for turn in all_turns if turn.speaker_id}):
        speaker_turns = [turn for turn in all_turns if turn.speaker_id == speaker_id]
        if not speaker_turns:
            continue
        speaker = _build_speaker_artifacts(
            project_dir,
            cfg,
            speaker_id,
            speaker_turns,
            mock_training=mock_training or backend_kind == "mock",
            previous_speaker=previous_bank.speakers.get(speaker_id) if previous_bank else None,
        )
        speakers[speaker_id] = speaker
    bank = VoiceBankManifest(
        speakers=speakers,
        source_paths=[str(path.expanduser().resolve()) for path in input_paths],
        backend=backend_kind,
        rights_audit={"confirmed": True, "history": rights_history},
    )
    save_voice_bank(voice_bank_path, bank)
    next_cfg = apply_voice_bank_to_config(project_dir, cfg, bank)
    save_project_config(next_cfg, project_dir / "pipeline.yaml")
    return bank


def _voice_bank_analysis_audio(
    source_audio: Path,
    source_dir: Path,
    cfg: ProjectConfig,
    *,
    mock: bool,
    force: bool,
) -> Path:
    backend = "mock" if mock else cfg.source_separation_backend
    try:
        result = separate_source_audio(
            source_audio,
            source_dir,
            backend=backend,
            model=cfg.source_separation_model,
            device=cfg.source_separation_device,
            sample_rate=cfg.mix_sample_rate,
            mono_sample_rate=cfg.gemma_sample_rate,
            force=force,
        )
    except SourceSeparationUnavailable as exc:
        raise VoiceBankError(f"voice-bank source separation failed for {source_audio}: {exc}") from exc
    return result.vocals_path if result is not None else source_audio


def _build_speaker_artifacts(
    project_dir: Path,
    cfg: ProjectConfig,
    speaker_id: str,
    turns: list[DiarizationTurn],
    *,
    mock_training: bool,
    previous_speaker: VoiceBankSpeaker | None = None,
) -> VoiceBankSpeaker:
    speaker_dir = ensure_inside_project(project_dir, project_dir / "voice_bank" / "speakers" / speaker_id)
    dataset_fingerprint = _speaker_fingerprint(turns)
    if (
        previous_speaker is not None
        and previous_speaker.dataset_fingerprint == dataset_fingerprint
        and _speaker_artifacts_exist(project_dir, previous_speaker)
    ):
        return previous_speaker
    embeddings = [turn.embedding for turn in turns if turn.embedding is not None]
    centroid_path: Path | None = None
    if embeddings:
        centroid = _normalize_embedding(np.mean(np.stack([_normalize_embedding(e) for e in embeddings]), axis=0))
        centroid_path = speaker_dir / "embedding.npy"
        centroid_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(centroid_path), centroid)
    refs_json = _write_speaker_refs(project_dir, speaker_dir, turns, cfg)
    gsv_gpt_weights = speaker_dir / "gsv" / "v001" / "gpt.ckpt"
    gsv_sovits_weights = speaker_dir / "gsv" / "v001" / "final.pth"
    rvc_model = speaker_dir / "rvc" / "v001" / "model.pth"
    rvc_index = speaker_dir / "rvc" / "v001" / "added.index"
    if mock_training:
        gsv_sovits_weights.parent.mkdir(parents=True, exist_ok=True)
        rvc_model.parent.mkdir(parents=True, exist_ok=True)
        gsv_gpt_weights.write_bytes(f"mock gpt weights for {speaker_id}\n".encode())
        gsv_sovits_weights.write_bytes(f"mock sovits weights for {speaker_id}\n".encode())
        rvc_model.write_bytes(f"mock rvc model for {speaker_id}\n".encode())
        rvc_index.write_bytes(f"mock rvc index for {speaker_id}\n".encode())
    else:
        gsv_gpt_weights, gsv_sovits_weights = _train_speaker_gsv(project_dir, cfg, speaker_id, turns)
        rvc_model, rvc_index = _train_speaker_rvc(project_dir, cfg, speaker_id, turns)
    source_segments = [
        VoiceBankSourceSegment(
            source_id=turn.source_id,
            source_path=str(turn.source_path),
            local_speaker_label=turn.local_speaker_label,
            segment_id=turn.segment_id,
            speaker_id=speaker_id,
            start=turn.start,
            end=turn.end,
            duration=turn.duration,
            audio_path=str(_relative_to_project(project_dir, turn.audio_path or Path(""))),
            text=turn.text,
            language=turn.language,
            quality_score=turn.quality_score,
        )
        for turn in turns
    ]
    return VoiceBankSpeaker(
        speaker_id=speaker_id,
        display_name=speaker_id,
        source_segments=source_segments,
        embedding_centroid_path=str(_relative_to_project(project_dir, centroid_path)) if centroid_path else None,
        gsv=GSVSpeakerConfig(
            gpt_weights_path=str(_relative_to_project(project_dir, gsv_gpt_weights)),
            sovits_weights_path=str(_relative_to_project(project_dir, gsv_sovits_weights)),
            refs_path=str(_relative_to_project(project_dir, refs_json)),
        ),
        rvc=RVCSpeakerConfig(
            model_path=str(_relative_to_project(project_dir, rvc_model)),
            index_path=str(_relative_to_project(project_dir, rvc_index)),
        ),
        dataset_fingerprint=dataset_fingerprint,
        rights_audit={"source_derived_voice_model": True},
    )


def _speaker_artifacts_exist(project_dir: Path, speaker: VoiceBankSpeaker) -> bool:
    paths = [
        speaker.gsv.gpt_weights_path,
        speaker.gsv.sovits_weights_path,
        speaker.gsv.refs_path,
        speaker.rvc.model_path,
    ]
    if speaker.rvc.index_path:
        paths.append(speaker.rvc.index_path)
    return all(bool(path) and _resolve_project_or_absolute(project_dir, path).exists() for path in paths)


def _train_speaker_gsv(
    project_dir: Path,
    cfg: ProjectConfig,
    speaker_id: str,
    turns: list[DiarizationTurn],
) -> tuple[Path, Path]:
    speaker_project = project_dir / "voice_bank" / "training" / speaker_id / "gsv"
    create_project_structure(speaker_project)
    save_project_config(cfg.model_copy(update={"project_name": f"{cfg.project_name}-{speaker_id}"}), speaker_project / "pipeline.yaml")
    segments: list[Segment] = []
    cursor = 0.0
    for index, turn in enumerate(turns, start=1):
        if not turn.audio_path:
            continue
        target = speaker_project / "work" / "segments" / "audio" / f"seg_{index:04d}_mix.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(turn.audio_path, target)
        duration = _audio_duration(target)
        segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=cursor,
                end=cursor + duration,
                duration=duration,
                audio_for_gemma=str(target.relative_to(speaker_project)),
                audio_for_mix=str(target.relative_to(speaker_project)),
                source_script=SourceScript(
                    text=turn.text or f"voice bank clip {index}",
                    language=turn.language or cfg.source_language,
                    backend="voice-bank",
                    start=cursor,
                    end=cursor + duration,
                ),
            )
        )
        cursor += duration
    manifest = PipelineManifest(project_config=cfg, segments=segments)
    result = train_few_shot(speaker_project, manifest, cfg)
    return result.gpt_weights_path, result.sovits_weights_path


def _train_speaker_rvc(
    project_dir: Path,
    cfg: ProjectConfig,
    speaker_id: str,
    turns: list[DiarizationTurn],
) -> tuple[Path, Path]:
    train_cfg = cfg.model_copy(
        update={
            "rvc_train_experiment_name": speaker_id,
            "rvc_train_output_model_path": str(project_dir / "voice_bank" / "speakers" / speaker_id / "rvc" / "v001" / "model.pth"),
            "rvc_train_output_index_path": str(project_dir / "voice_bank" / "speakers" / speaker_id / "rvc" / "v001" / "added.index"),
        }
    )
    validate_rvc_training_config(project_dir, train_cfg, real=True)
    dataset_dir = project_dir / "voice_bank" / "speakers" / speaker_id / "rvc" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for index, turn in enumerate(turns, start=1):
        if turn.audio_path:
            shutil.copy2(turn.audio_path, dataset_dir / f"seg_{index:04d}.wav")
    model_path = Path(train_cfg.rvc_train_output_model_path or "")
    index_path = Path(train_cfg.rvc_train_output_index_path or "")
    client = RVCTrainCommandClient(
        train_cfg.rvc_train_command,
        working_dir=resolve_config_path(project_dir, train_cfg.rvc_train_working_dir),
        timeout_sec=train_cfg.rvc_train_timeout_sec,
        stream_output=True,
        log_prefix=f"voice-bank-rvc-{speaker_id}",
    )
    result = client.train(
        project_dir=project_dir,
        dataset_dir=dataset_dir,
        work_dir=project_dir / "voice_bank" / "speakers" / speaker_id / "rvc",
        model_path=model_path,
        index_path=index_path,
        cfg=train_cfg,
        force=False,
    )
    return result.model_path, result.index_path or index_path


def _speaker_by_source_overlap(
    source_path: Path | None,
    segment: Segment,
    source_segments: list[VoiceBankSourceSegment],
) -> str | None:
    best_speaker: str | None = None
    best_overlap = 0.0
    for bank_segment in source_segments:
        if source_path is not None:
            try:
                if Path(bank_segment.source_path).resolve() != source_path:
                    continue
            except OSError:
                continue
        overlap = max(0.0, min(segment.end, bank_segment.end) - max(segment.start, bank_segment.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = bank_segment.speaker_id
    return best_speaker if best_overlap > 0 else None


def _speaker_by_embedding(
    project_dir: Path,
    segment: Segment,
    bank: VoiceBankManifest,
    backend: Any,
    *,
    threshold: float,
) -> str | None:
    audio_path = _resolve_project_or_absolute(project_dir, segment.audio_for_mix)
    embedding = _normalize_embedding(backend.embed_clip(audio_path))
    best_speaker: str | None = None
    best_score = -1.0
    for speaker_id, speaker in bank.speakers.items():
        if not speaker.embedding_centroid_path:
            continue
        centroid_path = _resolve_project_or_absolute(project_dir, speaker.embedding_centroid_path)
        if not centroid_path.exists():
            continue
        centroid = _normalize_embedding(np.load(str(centroid_path)))
        if centroid.shape != embedding.shape:
            continue
        score = float(np.dot(centroid, embedding))
        if score > best_score:
            best_score = score
            best_speaker = speaker_id
    return best_speaker if best_score >= threshold else None


def _transcribe_turns(audio_path: Path, turns: list[DiarizationTurn], cfg: ProjectConfig, mock: bool) -> None:
    segments = [
        Segment(
            id=_safe_id(f"{turn.source_id}_{index:04d}"),
            start=turn.start,
            end=turn.end,
            duration=turn.duration,
            audio_for_gemma=str(audio_path),
            audio_for_mix=str(audio_path),
        )
        for index, turn in enumerate(turns, start=1)
        if turn.duration > 0.01
    ]
    backend_kind = "mock" if mock else cfg.asr_backend
    backend = create_asr_backend(
        backend_kind,
        {
            "model_id": cfg.asr_model_id,
            "language": cfg.asr_language,
            "local_files_only": cfg.asr_local_files_only,
            "qwen_model_id": cfg.qwen_asr_model_id,
            "qwen_forced_aligner_model_id": cfg.qwen_asr_forced_aligner_model_id,
            "qwen_device_map": cfg.qwen_asr_device_map,
            "qwen_dtype": cfg.qwen_asr_dtype,
            "qwen_return_timestamps": cfg.qwen_asr_return_timestamps,
            "qwen_context": cfg.qwen_asr_context,
            "qwen_max_inference_batch_size": cfg.qwen_asr_max_inference_batch_size,
            "qwen_max_new_tokens": cfg.qwen_asr_max_new_tokens,
        },
    )
    chunks = backend.transcribe(audio_path, segments)
    mapped = map_chunks_to_segments(segments, chunks, backend=backend.name, fallback_language=cfg.asr_language)
    for turn, segment in zip(turns, segments, strict=False):
        source_script = mapped.get(segment.id)
        if source_script:
            turn.text = source_script.text
            turn.language = source_script.language


def _write_speaker_refs(project_dir: Path, speaker_dir: Path, turns: list[DiarizationTurn], cfg: ProjectConfig) -> Path:
    if not turns or not turns[0].audio_path:
        raise VoiceBankError("Cannot write refs without speaker clips.")
    refs_dir = speaker_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_audio = refs_dir / "whisper_close.wav"
    shutil.copy2(turns[0].audio_path, ref_audio)
    prompt_text = " ".join(turn.text or "" for turn in turns if turn.text).strip() or "voice reference"
    refs_json = refs_dir / "refs.json"
    write_json_atomic(
        refs_json,
        {
            "whisper_close": {
                "ref_audio_path": str(_relative_to_project(project_dir, ref_audio)),
                "prompt_text": prompt_text,
                "prompt_lang": turns[0].language or cfg.source_language,
            }
        },
    )
    return refs_json


def _extract_stereo_48k(input_path: Path, output_path: Path) -> None:
    try:
        ffmpeg.extract_stereo_48k(input_path, output_path)
    except ffmpeg.FFmpegError:
        data, sample_rate = load_audio(input_path)
        write_audio(output_path, ensure_stereo(resample_linear(data, sample_rate, 48_000)), 48_000)


def _write_audio_slice(input_path: Path, start: float, end: float, output_path: Path) -> None:
    data, sample_rate = load_audio(input_path)
    start_idx = max(0, int(round(start * sample_rate)))
    end_idx = min(len(data), int(round(end * sample_rate)))
    write_audio(output_path, data[start_idx:end_idx], sample_rate)


def _audio_duration(path: Path) -> float:
    return duration_sec(path)


def _normalize_embedding(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.zeros(1, dtype=np.float32)
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return vector
    return vector / norm


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return validate_file_safe_id(safe or "speaker", "id")


def _speaker_fingerprint(turns: list[DiarizationTurn]) -> str:
    digest = hashlib.sha256()
    for turn in sorted(turns, key=lambda item: (item.source_id, item.start, item.end, item.local_speaker_label)):
        digest.update(turn.source_id.encode("utf-8"))
        digest.update(turn.local_speaker_label.encode("utf-8"))
        digest.update(f"{turn.start:.3f}:{turn.end:.3f}".encode())
        if turn.audio_path and turn.audio_path.exists():
            digest.update(sha256_file(turn.audio_path).encode("utf-8"))
        if turn.text:
            digest.update(turn.text.encode("utf-8"))
    return digest.hexdigest()


def _relative_to_project(project_dir: Path, path: Path | None) -> Path:
    if path is None:
        return Path("")
    try:
        return path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        return path.resolve()


def _resolve_project_or_absolute(project_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_dir / path).resolve()
