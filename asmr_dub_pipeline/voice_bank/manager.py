from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import yaml

from asmr_dub_pipeline.asr import create_asr_backend, map_chunks_to_segments
from asmr_dub_pipeline.audio import ffmpeg
from asmr_dub_pipeline.audio.features import (
    AudioProcessingError,
    duration_sec,
    ensure_stereo,
    load_audio,
    peak_dbfs,
    resample_linear,
    rms_dbfs,
    write_audio,
)
from asmr_dub_pipeline.audio.preprocess import folder_asr_part_skip_reason
from asmr_dub_pipeline.audio.quality import measure_source_voice_quality
from asmr_dub_pipeline.audio.separation import SourceSeparationUnavailable, separate_source_audio
from asmr_dub_pipeline.audio.training_filter import evaluate_voice_training_candidate
from asmr_dub_pipeline.config import (
    create_project_structure,
    load_project_config,
    save_project_config,
)
from asmr_dub_pipeline.gpt_sovits.few_shot import few_shot_min_total_sec, train_few_shot
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
from asmr_dub_pipeline.script.normalizer import normalize_japanese_kana_text


class VoiceBankError(RuntimeError):
    pass


_MIN_EMBEDDING_AUDIO_SEC = 0.05
_MIN_EMBEDDING_CROP_SEC = 0.50
_SOURCE_SPEAKER_MIN_OVERLAP_RATIO = 0.50
_SOURCE_SPEAKER_MULTI_OVERLAP_RATIO = 0.20
_SOURCE_SPEAKER_BORDERLINE_OVERLAP_RATIO = 0.45
_SOURCE_SPEAKER_CONTEXT_OVERLAP_RATIO = 0.30
_SOURCE_SPEAKER_CONTEXT_MAX_GAP_SEC = 3.0
_SOURCE_SPEAKER_NEIGHBOR_MAX_SEGMENT_SEC = 12.0
_SOURCE_SPEAKER_SHORT_TEXTURE_MAX_SEC = 1.2
_SOURCE_DIARIZATION_CACHE_VERSION = 2
_SOURCE_EFFECT_AUGMENTED_MAX_SEGMENTS_PER_SPEAKER = 2
_SOURCE_EFFECT_AUGMENTED_MIN_MATCH_THRESHOLD = 0.60
_SOURCE_EFFECT_AUGMENTED_PROFILES = ("sfx_center", "sfx_left", "sfx_right", "echo")
_SOURCE_DISTINCT_MINOR_MAX_CENTROID_SIMILARITY = 0.35
_SOURCE_DIARIZATION_SILENT_RMS_DBFS = -85.0
_SOURCE_DIARIZATION_SILENT_PEAK_DBFS = -75.0
_SOURCE_SPEAKER_SKIP_STATUSES = {
    "absorbed",
    "failed",
    "needs_manual_review",
    "no_speech_detected",
    "non_speech_texture",
}
_SOURCE_SPEAKER_SHORT_TEXTURE_TOKENS = frozenset(
    (
        "あ",
        "あー",
        "ああ",
        "あっ",
        "あれ",
        "え",
        "えー",
        "ええ",
        "えっ",
        "うふ",
        "うふふ",
        "ふふ",
        "ふふふ",
        "ん",
        "んー",
        "んふ",
        "んふふ",
        "はぁ",
        "はあ",
        "へへ",
        "ほ",
        "ほら",
        "まあ",
        "ね",
        "ねえ",
        "そう",
        "そうそう",
    )
)
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


@dataclass(frozen=True)
class SourceDiarizationPart:
    part_index: int
    source_id: str
    start: float
    end: float
    clip_path: Path
    cache_path: Path
    skip_reason: str | None = None


@dataclass(frozen=True)
class SourceDiarizationPartResult:
    part: SourceDiarizationPart
    turns: list[DiarizationTurn]
    cache_status: str


class MockDiarizationBackend:
    name = "mock"

    def diarize(
        self,
        audio_path: Path,
        source_id: str,
        cfg: ProjectConfig,
        *,
        include_embeddings: bool = True,
    ) -> list[DiarizationTurn]:
        _ = cfg
        duration = _audio_duration(audio_path)
        return [
            DiarizationTurn(
                source_id=source_id,
                source_path=audio_path,
                local_speaker_label="SPEAKER_00",
                start=0.0,
                end=duration,
                embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32)
                if include_embeddings
                else None,
            )
        ]

    def embed_clip(self, audio_path: Path) -> np.ndarray:
        _ = audio_path
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class PyannoteDiarizationBackend:
    name = "pyannote"

    def __init__(self, cfg: ProjectConfig, *, load_embedding_model: bool = True) -> None:
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
        embedding_model = (
            _resolve_pyannote_model(
                cfg.diarization_embedding_model_id,
                cfg,
                label="speaker embedding",
                token=token,
            )
            if load_embedding_model
            else None
        )
        local_paths = [diarization_model, *dependency_models]
        if embedding_model is not None:
            local_paths.append(embedding_model)
        if _all_local_paths(local_paths):
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
        self.inference = None
        if embedding_model is not None:
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
            if self.inference is not None:
                self.inference.to(device)

    def diarize(
        self,
        audio_path: Path,
        source_id: str,
        cfg: ProjectConfig,
        *,
        include_embeddings: bool = True,
    ) -> list[DiarizationTurn]:
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
                    embedding=self.embed_excerpt(audio_path, start, end)
                    if include_embeddings
                    else None,
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
        if self.inference is None:
            raise VoiceBankError("pyannote speaker embedding inference was not loaded.")
        from pyannote.core import Segment as PyannoteSegment  # type: ignore[import-not-found]

        crop_start, crop_end = _embedding_crop_bounds(audio_path, start, end)
        embedding = self.inference.crop(str(audio_path), PyannoteSegment(crop_start, crop_end))
        return _normalize_embedding(np.asarray(embedding, dtype=np.float32).reshape(-1))

    def embed_clip(self, audio_path: Path) -> np.ndarray:
        if self.inference is None:
            raise VoiceBankError("pyannote speaker embedding inference was not loaded.")
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


def create_diarization_backend(kind: str, cfg: ProjectConfig, *, load_embedding_model: bool = True):
    normalized = kind.replace("-", "_")
    if normalized == "mock":
        return MockDiarizationBackend()
    if normalized == "pyannote":
        return PyannoteDiarizationBackend(cfg, load_embedding_model=load_embedding_model)
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


def cluster_turns(turns: list[DiarizationTurn], threshold: float = 0.75) -> list[DiarizationTurn]:
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


def _cluster_source_turns(turns: list[DiarizationTurn], threshold: float = 0.75) -> list[DiarizationTurn]:
    groups: dict[tuple[str, str], list[DiarizationTurn]] = {}
    for turn in turns:
        groups.setdefault((turn.source_id, turn.local_speaker_label), []).append(turn)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            min(turn.start for turn in item[1]),
            item[0][0],
            item[0][1],
        ),
    )
    clusters: list[dict[str, Any]] = []
    fallback_labels: dict[str, str] = {}
    speaker_index = 0

    def next_speaker_id() -> str:
        nonlocal speaker_index
        speaker_index += 1
        return f"speaker_{speaker_index:04d}"

    for (_source_id, local_speaker_label), group_turns in ordered_groups:
        embeddings = [
            _normalize_embedding(turn.embedding)
            for turn in group_turns
            if turn.embedding is not None
        ]
        if not embeddings:
            speaker_id = fallback_labels.get(local_speaker_label)
            if speaker_id is None:
                speaker_id = next_speaker_id()
                fallback_labels[local_speaker_label] = speaker_id
            for turn in group_turns:
                turn.speaker_id = speaker_id
            continue

        centroid = _normalize_embedding(np.mean(np.stack(embeddings), axis=0))
        best_index = -1
        best_score = -1.0
        for index, cluster in enumerate(clusters):
            score = float(np.dot(cluster["centroid"], centroid))
            if score > best_score:
                best_index = index
                best_score = score
        if best_index >= 0 and best_score >= threshold:
            cluster = clusters[best_index]
            speaker_id = str(cluster["speaker_id"])
            cluster["centroid"] = _normalize_embedding((cluster["centroid"] + centroid) / 2.0)
        else:
            speaker_id = next_speaker_id()
            clusters.append({"speaker_id": speaker_id, "centroid": centroid})
        for turn in group_turns:
            turn.speaker_id = speaker_id
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
    jobs: int = 4,
) -> PipelineManifest:
    cfg = manifest.project_config
    audio_path = _source_speaker_audio_path(project_dir, manifest)
    normalized_jobs = max(1, int(jobs))
    part_specs = (
        _source_diarization_part_specs(project_dir, manifest, audio_path)
        if normalized_jobs > 1
        else []
    )
    if part_specs:
        part_results = _diarize_source_parts(
            audio_path,
            cfg,
            backend_kind,
            part_specs,
            jobs=normalized_jobs,
            include_embeddings=backend_kind == "pyannote",
        )
        turns = [turn for result in part_results for turn in result.turns]
        cache_status = _combined_cache_status([result.cache_status for result in part_results])
        cache_path = project_dir / "work" / "diarization" / "source_parts"
        skipped_silent_parts = sum(1 for result in part_results if result.cache_status == "skipped_silent")
        skipped_effect_only_parts = sum(
            1 for result in part_results if result.cache_status == "skipped_effect_only"
        )
        skipped_no_diarization_parts = sum(
            1 for result in part_results if result.cache_status == "skipped_no_diarization"
        )
        skipped_asr_silenced_parts = sum(
            1 for result in part_results if result.cache_status == "skipped_asr_silenced"
        )
        skipped_part_count = sum(
            1 for result in part_results if result.cache_status.startswith("skipped_")
        )
    else:
        cache_path = _source_diarization_cache_path(project_dir)
        cache_signature = _source_diarization_cache_signature(
            audio_path,
            cfg,
            backend_kind,
            include_embeddings=False,
        )
        turns = _load_source_diarization_cache(cache_path, cache_signature, audio_path)
        cache_status = "hit" if turns is not None else "miss"
        if turns is None:
            backend = _create_source_diarization_backend(backend_kind, cfg)
            turns = _diarize_source_audio(backend, audio_path, cfg)
            _write_source_diarization_cache(cache_path, cache_signature, turns)
        skipped_silent_parts = 0
        skipped_effect_only_parts = 0
        skipped_no_diarization_parts = 0
        skipped_asr_silenced_parts = 0
        skipped_part_count = 0
    _cluster_source_turns(turns, threshold=cfg.diarization_embedding_match_threshold)
    assigned = 0
    excluded = 0
    for segment in manifest.segments:
        if segment.status in _SOURCE_SPEAKER_SKIP_STATUSES:
            continue
        assignment = _source_speaker_assignment(segment, turns)
        segment.analysis["speaker_count"] = assignment["speaker_count"]
        segment.analysis["source_speaker_assignment"] = assignment
        if assignment["speaker_count"] == 1 and assignment["speaker_id"]:
            segment.speaker_id = str(assignment["speaker_id"])
            assigned += 1
            training_exclusion_reason = _source_speaker_training_exclusion_reason(segment, cfg)
            voice_training = dict(segment.analysis.get("voice_training") or {})
            if training_exclusion_reason:
                voice_training["exclude"] = True
                voice_training["reason"] = training_exclusion_reason
            elif voice_training.get("reason") in {
                "multi_speaker_overlap",
                "no_source_speaker_match",
                "minor_bucket_auto_merged",
                "low_dominant_source_speaker_overlap",
            }:
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
    embedding_backend_factory = None
    if backend_kind == "pyannote":
        def embedding_backend_factory() -> Any:
            return _create_source_diarization_backend(
                backend_kind,
                cfg,
                load_embedding_model=True,
            )

    bucket_normalization = _normalize_source_speaker_buckets(
        project_dir,
        manifest,
        cfg,
        turns,
        embedding_backend_factory=embedding_backend_factory,
    )
    speaker_routing = _resolve_source_speaker_null_segments(manifest, bucket_normalization)
    manifest.artifacts["source_speaker_audio"] = str(audio_path)
    bucket_qc_path = project_dir / "work" / "diarization" / "source_speaker_bucket_qc.json"
    write_json_atomic(bucket_qc_path, bucket_normalization)
    manifest.artifacts["source_speaker_bucket_qc"] = str(bucket_qc_path)
    training_qc = _source_speaker_training_qc_payload(project_dir, manifest, cfg)
    training_qc_path = project_dir / "work" / "diarization" / "source_speaker_training_qc.json"
    write_json_atomic(training_qc_path, training_qc)
    manifest.artifacts["source_speaker_training_qc"] = str(training_qc_path)
    manifest.stage_state["source-speakers"] = {
        "status": "completed",
        "backend": backend_kind,
        "diarization_cache": cache_status,
        "diarization_cache_path": str(cache_path),
        "parallel_jobs": normalized_jobs,
        "parallel_tracks": len(part_specs),
        "skipped_part_count": skipped_part_count,
        "skipped_silent_parts": skipped_silent_parts,
        "skipped_effect_only_parts": skipped_effect_only_parts,
        "skipped_asr_silenced_parts": skipped_asr_silenced_parts,
        "skipped_no_diarization_parts": skipped_no_diarization_parts,
        "turn_count": len(turns),
        "assigned_segments": assigned,
        "excluded_segments": excluded,
        "bucket_normalization": bucket_normalization["summary"],
        "speaker_routing": speaker_routing,
        "training_eligible_counts": training_qc["summary"]["training_eligible_counts"],
        "training_excluded_counts": training_qc["summary"]["training_excluded_counts"],
        "possible_overlap_counts": training_qc["summary"]["possible_overlap_counts"],
    }
    return manifest


def _source_speaker_audio_path(project_dir: Path, manifest: PipelineManifest) -> Path:
    for key in ("source_vocals_mono_16k", "source_vocals_48k", "gemma_mono_16k", "original_stereo_48k"):
        raw_path = manifest.artifacts.get(key)
        if raw_path:
            path = _resolve_project_or_absolute(project_dir, raw_path)
            if path.exists():
                return path
    if manifest.source_info:
        path = Path(manifest.source_info.path).expanduser().resolve()
        if path.exists():
            return path
    raise VoiceBankError(
        "source-speakers requires source_vocals_mono_16k, source_vocals_48k, "
        "gemma_mono_16k, original_stereo_48k, or source_info.path."
    )


def _create_source_diarization_backend(
    backend_kind: str,
    cfg: ProjectConfig,
    *,
    load_embedding_model: bool = False,
):
    if backend_kind != "pyannote":
        return MockDiarizationBackend()
    try:
        parameters = inspect.signature(create_diarization_backend).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "load_embedding_model" in parameters:
        return create_diarization_backend(
            backend_kind,
            cfg,
            load_embedding_model=load_embedding_model,
        )
    return create_diarization_backend(backend_kind, cfg)


def _diarize_source_audio(
    backend: Any,
    audio_path: Path,
    cfg: ProjectConfig,
    *,
    source_id: str = "source",
    include_embeddings: bool = False,
) -> list[DiarizationTurn]:
    resolved_audio_path = audio_path.resolve()
    try:
        parameters = inspect.signature(backend.diarize).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "include_embeddings" in parameters:
        return backend.diarize(
            resolved_audio_path,
            source_id,
            cfg,
            include_embeddings=include_embeddings,
        )
    return backend.diarize(resolved_audio_path, source_id, cfg)


def _source_diarization_part_specs(
    project_dir: Path,
    manifest: PipelineManifest,
    audio_path: Path,
) -> list[SourceDiarizationPart]:
    if not manifest.source_info:
        return []
    folder_input = manifest.source_info.raw.get("folder_input")
    if not isinstance(folder_input, dict):
        return []
    raw_parts = folder_input.get("asr_parts")
    if not isinstance(raw_parts, list) or len(raw_parts) < 2:
        return []
    audio_duration = _audio_duration(audio_path)
    specs: list[SourceDiarizationPart] = []
    seen_indices: set[int] = set()
    for fallback_index, raw_part in enumerate(raw_parts, start=1):
        if not isinstance(raw_part, dict):
            return []
        try:
            start = float(raw_part["start_sec"])
            end = float(raw_part["end_sec"])
        except (KeyError, TypeError, ValueError):
            return []
        if end <= start:
            return []
        start = max(0.0, min(start, audio_duration))
        end = max(start, min(end, audio_duration))
        if end - start < _MIN_EMBEDDING_AUDIO_SEC:
            continue
        try:
            part_index = int(raw_part.get("part_index") or fallback_index)
        except (TypeError, ValueError):
            part_index = fallback_index
        if part_index in seen_indices:
            part_index = fallback_index
        seen_indices.add(part_index)
        skip_reason = _source_diarization_part_skip_reason(raw_part)
        part_dir = ensure_inside_project(project_dir, project_dir / "work" / "diarization" / "source_parts")
        specs.append(
            SourceDiarizationPart(
                part_index=part_index,
                source_id=f"source_part_{part_index:04d}",
                start=start,
                end=end,
                clip_path=part_dir / f"part_{part_index:04d}.wav",
                cache_path=part_dir / f"part_{part_index:04d}_turns.json",
                skip_reason=skip_reason,
            )
        )
    return sorted(specs, key=lambda part: (part.start, part.part_index)) if len(specs) >= 2 else []


def _source_diarization_part_skip_reason(raw_part: dict[str, Any]) -> str | None:
    raw_reason = str(raw_part.get("asr_skip_reason") or "").strip()
    path_text = str(raw_part.get("path") or raw_part.get("stem") or "")
    detected_reason = folder_asr_part_skip_reason(path_text) if path_text else None
    if bool(raw_part.get("asr_silenced")):
        return raw_reason or detected_reason or "asr_silenced"
    return detected_reason


def _diarize_source_parts(
    audio_path: Path,
    cfg: ProjectConfig,
    backend_kind: str,
    parts: list[SourceDiarizationPart],
    *,
    jobs: int,
    include_embeddings: bool = False,
) -> list[SourceDiarizationPartResult]:
    audio_sha256 = sha256_file(audio_path)
    max_workers = min(max(1, jobs), len(parts))
    local_state = threading.local()

    def backend_for_thread():
        backend = getattr(local_state, "backend", None)
        if backend is None:
            backend = _create_source_diarization_backend(
                backend_kind,
                cfg,
                load_embedding_model=include_embeddings,
            )
            local_state.backend = backend
        return backend

    def run_part(part: SourceDiarizationPart) -> SourceDiarizationPartResult:
        signature = _source_diarization_cache_signature(
            audio_path,
            cfg,
            backend_kind,
            audio_sha256=audio_sha256,
            part=part,
            include_embeddings=include_embeddings,
        )
        if part.skip_reason:
            _write_source_diarization_cache(
                part.cache_path,
                signature,
                [],
                empty_reason=part.skip_reason,
            )
            return SourceDiarizationPartResult(
                part=part,
                turns=[],
                cache_status=f"skipped_{part.skip_reason}",
            )
        cached = _load_source_diarization_cache(part.cache_path, signature, part.clip_path)
        if cached is not None:
            cache_status = "hit"
            if not cached:
                cache_status = _source_diarization_empty_cache_status(part.cache_path, signature) or cache_status
            return SourceDiarizationPartResult(
                part=part,
                turns=_offset_source_part_turns(cached, part, audio_path),
                cache_status=cache_status,
            )
        _write_source_part_audio(audio_path, part)
        if _source_diarization_part_is_silent(part.clip_path):
            return SourceDiarizationPartResult(
                part=part,
                turns=[],
                cache_status="skipped_silent",
            )
        try:
            turns = _diarize_source_audio(
                backend_for_thread(),
                part.clip_path,
                cfg,
                source_id=part.source_id,
                include_embeddings=include_embeddings,
            )
        except VoiceBankError as exc:
            if _is_no_diarization_turns_error(exc):
                _write_source_diarization_cache(
                    part.cache_path,
                    signature,
                    [],
                    empty_reason="no_diarization",
                )
                return SourceDiarizationPartResult(
                    part=part,
                    turns=[],
                    cache_status="skipped_no_diarization",
                )
            raise
        _write_source_diarization_cache(part.cache_path, signature, turns)
        return SourceDiarizationPartResult(
            part=part,
            turns=_offset_source_part_turns(turns, part, audio_path),
            cache_status="miss",
        )

    if max_workers == 1:
        return [run_part(part) for part in parts]
    results: list[SourceDiarizationPartResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_part, part) for part in parts]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: (result.part.start, result.part.part_index))


def _is_no_diarization_turns_error(exc: Exception) -> bool:
    return "pyannote produced no diarization turns" in str(exc)


def _source_diarization_empty_cache_status(
    cache_path: Path,
    signature: dict[str, Any],
) -> str | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("signature") != signature or data.get("turns") != []:
        return None
    reason = str(data.get("empty_reason") or "").strip()
    if reason == "no_diarization_turns":
        reason = "no_diarization"
    return f"skipped_{reason}" if reason else None


def _source_diarization_part_is_silent(audio_path: Path) -> bool:
    try:
        return (
            rms_dbfs(audio_path) <= _SOURCE_DIARIZATION_SILENT_RMS_DBFS
            and peak_dbfs(audio_path) <= _SOURCE_DIARIZATION_SILENT_PEAK_DBFS
        )
    except AudioProcessingError:
        return False


def _write_source_part_audio(audio_path: Path, part: SourceDiarizationPart) -> None:
    try:
        ffmpeg.slice_audio(
            audio_path,
            part.start,
            part.end,
            part.clip_path,
            sample_rate=16_000,
            channels=1,
        )
    except ffmpeg.FFmpegError:
        _write_audio_slice(audio_path, part.start, part.end, part.clip_path)


def _offset_source_part_turns(
    turns: list[DiarizationTurn],
    part: SourceDiarizationPart,
    audio_path: Path,
) -> list[DiarizationTurn]:
    offset_turns: list[DiarizationTurn] = []
    for turn in turns:
        offset_turns.append(
            DiarizationTurn(
                source_id=turn.source_id,
                source_path=audio_path.resolve(),
                local_speaker_label=turn.local_speaker_label,
                start=round(part.start + turn.start, 3),
                end=round(part.start + turn.end, 3),
                embedding=turn.embedding,
                text=turn.text,
                language=turn.language,
                quality_score=turn.quality_score,
                audio_path=turn.audio_path,
                analysis_audio_path=turn.analysis_audio_path,
                segment_id=turn.segment_id,
            )
        )
    return offset_turns


def _combined_cache_status(statuses: list[str]) -> str:
    unique = set(statuses)
    if unique == {"hit"}:
        return "hit"
    if unique == {"miss"}:
        return "miss"
    return "partial"


def _source_diarization_cache_path(project_dir: Path) -> Path:
    return ensure_inside_project(project_dir, project_dir / "work" / "diarization" / "source_turns.json")


def _source_diarization_cache_signature(
    audio_path: Path,
    cfg: ProjectConfig,
    backend_kind: str,
    *,
    audio_sha256: str | None = None,
    part: SourceDiarizationPart | None = None,
    include_embeddings: bool = False,
) -> dict[str, Any]:
    resolved_audio_path = audio_path.resolve()
    signature: dict[str, Any] = {
        "version": _SOURCE_DIARIZATION_CACHE_VERSION,
        "backend": backend_kind,
        "audio_path": str(resolved_audio_path),
        "audio_sha256": audio_sha256 or sha256_file(resolved_audio_path),
        "diarization_model_id": cfg.diarization_model_id,
        "diarization_embedding_model_id": cfg.diarization_embedding_model_id,
        "include_embeddings": include_embeddings,
        "diarization_min_speakers": cfg.diarization_min_speakers,
        "diarization_max_speakers": cfg.diarization_max_speakers,
        "minimum_turn_sec": round(_minimum_diarization_turn_sec(cfg), 6),
    }
    if part is not None:
        signature["part_index"] = part.part_index
        signature["part_start_sec"] = round(part.start, 6)
        signature["part_end_sec"] = round(part.end, 6)
    return signature


def _load_source_diarization_cache(
    cache_path: Path,
    signature: dict[str, Any],
    audio_path: Path,
) -> list[DiarizationTurn] | None:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return None
    if data.get("signature") != signature:
        return None
    raw_turns = data.get("turns")
    if not isinstance(raw_turns, list):
        return None
    try:
        turns = [_source_turn_from_cache(item, audio_path) for item in raw_turns]
    except (KeyError, TypeError, ValueError):
        return None
    return turns


def _write_source_diarization_cache(
    cache_path: Path,
    signature: dict[str, Any],
    turns: list[DiarizationTurn],
    *,
    empty_reason: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "signature": signature,
        "turns": [_source_turn_to_cache(turn) for turn in turns],
    }
    if empty_reason is not None:
        payload["empty_reason"] = empty_reason
    write_json_atomic(cache_path, payload)


def _source_turn_to_cache(turn: DiarizationTurn) -> dict[str, Any]:
    item: dict[str, Any] = {
        "source_id": turn.source_id,
        "local_speaker_label": turn.local_speaker_label,
        "start": round(float(turn.start), 6),
        "end": round(float(turn.end), 6),
    }
    if turn.embedding is not None:
        item["embedding"] = [float(value) for value in np.asarray(turn.embedding, dtype=np.float32).reshape(-1)]
    return item


def _source_turn_from_cache(item: dict[str, Any], audio_path: Path) -> DiarizationTurn:
    start = float(item["start"])
    end = float(item["end"])
    if end <= start:
        raise ValueError("cached diarization turn has non-positive duration")
    raw_embedding = item.get("embedding")
    embedding = (
        np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
        if isinstance(raw_embedding, list)
        else None
    )
    return DiarizationTurn(
        source_id=str(item["source_id"]),
        source_path=audio_path.resolve(),
        local_speaker_label=str(item["local_speaker_label"]),
        start=start,
        end=end,
        embedding=embedding,
    )


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


def _source_speaker_training_exclusion_reason(segment: Segment, cfg: ProjectConfig) -> str | None:
    assignment = segment.analysis.get("source_speaker_assignment")
    if not isinstance(assignment, dict):
        return None
    if int(assignment.get("speaker_count") or 0) != 1:
        return None
    ratio = _source_speaker_float(assignment.get("dominant_overlap_ratio"))
    threshold = cfg.voice_bank.source_speaker_training_min_dominant_overlap_ratio
    if ratio is None or ratio < threshold:
        return "low_dominant_source_speaker_overlap"
    return None


def _source_speaker_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _resolve_source_speaker_null_segments(
    manifest: PipelineManifest,
    bucket_normalization: dict[str, Any],
) -> dict[str, Any]:
    merge_map = _source_speaker_merge_map(bucket_normalization)
    routed = 0
    textures = 0
    keep_original = 0
    unresolved = 0
    reason_counts: dict[str, int] = {}

    def record_reason(reason: str) -> None:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for index, segment in enumerate(manifest.segments):
        if segment.status in _SOURCE_SPEAKER_SKIP_STATUSES or segment.speaker_id:
            continue
        assignment = segment.analysis.get("source_speaker_assignment")
        if not isinstance(assignment, dict):
            continue
        speaker_count = int(assignment.get("speaker_count") or 0)
        routed_speaker_id: str | None = None
        reason: str | None = None
        overlap_ratio: float | None = None
        if speaker_count == 1:
            candidate_id, overlap_ratio = _source_speaker_single_overlap_candidate(assignment)
            candidate_id = _source_speaker_model_id(candidate_id, merge_map)
            if candidate_id and overlap_ratio is not None:
                if overlap_ratio >= _SOURCE_SPEAKER_BORDERLINE_OVERLAP_RATIO:
                    routed_speaker_id = candidate_id
                    reason = "borderline_single_speaker_overlap"
                elif overlap_ratio >= _SOURCE_SPEAKER_CONTEXT_OVERLAP_RATIO:
                    context_speaker_id = _source_speaker_neighbor_context(
                        manifest.segments,
                        index,
                    )
                    if context_speaker_id == candidate_id:
                        routed_speaker_id = candidate_id
                        reason = "neighbor_confirmed_single_speaker_overlap"
                    else:
                        routed_speaker_id = candidate_id
                        reason = "single_speaker_overlap_tts_routing"
                elif overlap_ratio >= _SOURCE_SPEAKER_MULTI_OVERLAP_RATIO:
                    routed_speaker_id = candidate_id
                    reason = "single_speaker_overlap_tts_routing"
        elif speaker_count == 0:
            if _source_segment_is_short_no_overlap_texture(segment):
                _mark_source_segment_texture(segment, "source_speaker_no_overlap_short_texture")
                textures += 1
                record_reason("source_speaker_no_overlap_short_texture")
                continue
        elif speaker_count > 1:
            routed_speaker_id, overlap_ratio, reason = _source_speaker_multi_overlap_route(
                assignment,
                merge_map,
                segment.duration,
            )
            if not routed_speaker_id:
                context_speaker_id = _source_speaker_neighbor_context(manifest.segments, index)
                if context_speaker_id:
                    routed_speaker_id = context_speaker_id
                    reason = "neighbor_confirmed_multi_speaker_overlap"
                else:
                    _mark_source_segment_keep_original_overlap(segment, "multi_speaker_overlap")
                    keep_original += 1
                    record_reason("multi_speaker_overlap")
                    continue
        if routed_speaker_id and reason:
            _route_source_segment_to_speaker(
                segment,
                routed_speaker_id,
                reason=reason,
                overlap_ratio=overlap_ratio,
            )
            routed += 1
            record_reason(reason)

    for index, segment in enumerate(manifest.segments):
        if segment.status in _SOURCE_SPEAKER_SKIP_STATUSES or segment.speaker_id:
            continue
        assignment = segment.analysis.get("source_speaker_assignment")
        if not isinstance(assignment, dict):
            unresolved += 1
            continue
        speaker_count = int(assignment.get("speaker_count") or 0)
        if speaker_count == 0:
            context_speaker_id = _source_speaker_neighbor_context(
                manifest.segments,
                index,
                require_close_gap=False,
            )
            if context_speaker_id and _source_segment_can_inherit_neighbor_speaker(segment):
                _route_source_segment_to_speaker(
                    segment,
                    context_speaker_id,
                    reason="neighbor_same_speaker_context",
                    overlap_ratio=0.0,
                )
                routed += 1
                record_reason("neighbor_same_speaker_context")
                continue
        unresolved += 1
    return {
        "routed_segments": routed,
        "texture_segments": textures,
        "keep_original_segments": keep_original,
        "unresolved_segments": unresolved,
        "reason_counts": dict(sorted(reason_counts.items())),
    }


def _source_speaker_merge_map(bucket_normalization: dict[str, Any]) -> dict[str, str]:
    merge_map: dict[str, str] = {}
    merges = bucket_normalization.get("merges")
    if not isinstance(merges, list):
        return merge_map
    for merge in merges:
        if not isinstance(merge, dict):
            continue
        original = merge.get("original_speaker_id")
        target = merge.get("merged_into_speaker_id")
        if original and target:
            merge_map[str(original)] = str(target)
    return merge_map


def _source_speaker_model_id(speaker_id: str | None, merge_map: dict[str, str]) -> str | None:
    if not speaker_id:
        return None
    seen: set[str] = set()
    current = speaker_id
    while current in merge_map and current not in seen:
        seen.add(current)
        current = merge_map[current]
    return current


def _source_speaker_single_overlap_candidate(assignment: dict[str, Any]) -> tuple[str | None, float | None]:
    overlaps = assignment.get("overlaps")
    if not isinstance(overlaps, dict) or len(overlaps) != 1:
        return None, None
    speaker_id, overlap = next(iter(overlaps.items()))
    try:
        overlap_value = float(overlap)
    except (TypeError, ValueError):
        return str(speaker_id), None
    ratio = assignment.get("dominant_overlap_ratio")
    try:
        return str(speaker_id), float(ratio)
    except (TypeError, ValueError):
        return str(speaker_id), overlap_value


def _source_speaker_multi_overlap_route(
    assignment: dict[str, Any],
    merge_map: dict[str, str],
    duration: float,
) -> tuple[str | None, float | None, str | None]:
    normalized_overlaps = _source_speaker_normalized_overlaps(assignment, merge_map)
    if not normalized_overlaps:
        return None, None, None
    if len(normalized_overlaps) == 1:
        speaker_id, overlap = next(iter(normalized_overlaps.items()))
        return (
            speaker_id,
            overlap / max(duration, 0.001),
            "merged_overlap_candidates_tts_routing",
        )
    speaker_id, overlap = max(normalized_overlaps.items(), key=lambda item: (item[1], item[0]))
    ratio = overlap / max(duration, 0.001)
    if ratio >= 0.60:
        return speaker_id, ratio, "dominant_multi_speaker_overlap_tts_routing"
    return None, ratio, None


def _source_speaker_normalized_overlaps(
    assignment: dict[str, Any],
    merge_map: dict[str, str],
) -> dict[str, float]:
    overlaps = assignment.get("overlaps")
    if not isinstance(overlaps, dict):
        return {}
    normalized: dict[str, float] = {}
    for raw_speaker_id, raw_overlap in overlaps.items():
        speaker_id = _source_speaker_model_id(str(raw_speaker_id), merge_map)
        if not speaker_id:
            continue
        try:
            overlap = float(raw_overlap)
        except (TypeError, ValueError):
            continue
        normalized[speaker_id] = normalized.get(speaker_id, 0.0) + overlap
    return normalized


def _source_speaker_neighbor_context(
    segments: list[Segment],
    index: int,
    *,
    require_close_gap: bool = True,
) -> str | None:
    current = segments[index]
    previous = _nearest_routed_source_neighbor(segments, index, step=-1)
    following = _nearest_routed_source_neighbor(segments, index, step=1)
    if previous is None or following is None:
        return None
    if previous.speaker_id != following.speaker_id:
        return None
    if require_close_gap:
        if current.start - previous.end > _SOURCE_SPEAKER_CONTEXT_MAX_GAP_SEC:
            return None
        if following.start - current.end > _SOURCE_SPEAKER_CONTEXT_MAX_GAP_SEC:
            return None
    return previous.speaker_id


def _nearest_routed_source_neighbor(segments: list[Segment], index: int, *, step: int) -> Segment | None:
    cursor = index + step
    while 0 <= cursor < len(segments):
        candidate = segments[cursor]
        if candidate.status not in _SOURCE_SPEAKER_SKIP_STATUSES and candidate.speaker_id:
            return candidate
        if candidate.status not in _SOURCE_SPEAKER_SKIP_STATUSES and not candidate.speaker_id:
            return None
        cursor += step
    return None


def _source_segment_can_inherit_neighbor_speaker(segment: Segment) -> bool:
    if segment.duration > _SOURCE_SPEAKER_NEIGHBOR_MAX_SEGMENT_SEC:
        return False
    if segment.source_script is None or not segment.source_script.text.strip():
        return False
    return not _source_segment_is_short_no_overlap_texture(segment)


def _source_segment_is_short_no_overlap_texture(segment: Segment) -> bool:
    if segment.duration > _SOURCE_SPEAKER_SHORT_TEXTURE_MAX_SEC:
        return False
    source_script = segment.source_script
    if source_script is None:
        return False
    compact = _compact_source_speaker_short_text(source_script.text)
    if not compact:
        return bool(segment.keep_original_texture)
    if compact in _SOURCE_SPEAKER_SHORT_TEXTURE_TOKENS:
        return True
    return _source_speaker_repeated_short_texture(compact)


def _compact_source_speaker_short_text(text: str) -> str:
    return re.sub(r"[\s　、。,.，．!！?？…・♪♡❤「」『』（）()［］\[\]【】:：;；\"'`]+", "", text.strip())


def _source_speaker_repeated_short_texture(compact: str) -> bool:
    for token in _SOURCE_SPEAKER_SHORT_TEXTURE_TOKENS:
        if len(token) == 0 or len(compact) <= len(token) or len(compact) % len(token) != 0:
            continue
        if token * (len(compact) // len(token)) == compact:
            return True
    return False


def _mark_source_segment_texture(segment: Segment, reason: str) -> None:
    segment.status = "non_speech_texture"
    segment.speaker_id = None
    segment.keep_original_texture = True
    segment.script = None
    segment.tts = None
    if reason not in segment.errors:
        segment.errors.append(reason)
    segment.analysis["source_speaker_routing"] = {
        "decision": "texture",
        "reason": reason,
    }
    segment.analysis["asr_quality_gate"] = {
        "decision": "texture",
        "reasons": [reason],
        "tts_blocked": True,
    }
    segment.analysis["voice_training"] = {
        "exclude": True,
        "reason": reason,
    }


def _mark_source_segment_keep_original_overlap(segment: Segment, reason: str) -> None:
    segment.status = "absorbed"
    segment.speaker_id = None
    segment.keep_original_texture = True
    segment.script = None
    segment.tts = None
    segment.analysis["source_speaker_routing"] = {
        "decision": "keep_original_texture",
        "reason": reason,
        "tts_blocked": True,
    }
    segment.analysis["voice_training"] = {
        "exclude": True,
        "reason": reason,
    }


def _route_source_segment_to_speaker(
    segment: Segment,
    speaker_id: str,
    *,
    reason: str,
    overlap_ratio: float | None,
) -> None:
    segment.speaker_id = speaker_id
    assignment = segment.analysis.get("source_speaker_assignment")
    if isinstance(assignment, dict):
        assignment["routing_speaker_id"] = speaker_id
        assignment["routing_reason"] = reason
        if overlap_ratio is not None:
            assignment["routing_overlap_ratio"] = round(float(overlap_ratio), 6)
    routing = {
        "decision": "speaker_routed_training_excluded",
        "speaker_id": speaker_id,
        "reason": reason,
    }
    if overlap_ratio is not None:
        routing["overlap_ratio"] = round(float(overlap_ratio), 6)
    segment.analysis["source_speaker_routing"] = routing
    segment.analysis["voice_training"] = {
        "exclude": True,
        "reason": reason,
    }


def _normalize_source_speaker_buckets(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    turns: list[DiarizationTurn],
    *,
    embedding_backend_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    stats = _source_speaker_bucket_stats(project_dir, manifest, cfg)
    if len(stats) <= 1:
        return _source_speaker_bucket_qc_payload(stats, [])
    target_sec = few_shot_min_total_sec(cfg)
    major_ids = [
        speaker_id
        for speaker_id, row in stats.items()
        if float(row["clean_training_duration_sec"]) >= target_sec
    ]
    if not major_ids:
        return _source_speaker_bucket_qc_payload(stats, [])
    centroids = _speaker_centroids_from_turns(turns)
    direct_matches: dict[str, tuple[str | None, float | None]] = {}
    low_similarity_minor_ids: list[str] = []
    for speaker_id in sorted(stats):
        if speaker_id in major_ids:
            continue
        target_id, similarity = _best_source_bucket_merge_target(
            speaker_id,
            major_ids,
            stats,
            centroids,
        )
        direct_matches[speaker_id] = (target_id, similarity)
        if similarity is None or similarity < cfg.diarization_embedding_match_threshold:
            low_similarity_minor_ids.append(speaker_id)
    effect_matches = _source_effect_augmented_bucket_matches(
        project_dir,
        manifest,
        cfg,
        stats,
        major_ids,
        low_similarity_minor_ids,
        centroids,
        embedding_backend_factory,
    )
    merges: list[dict[str, Any]] = []
    preserves: list[dict[str, Any]] = []
    for speaker_id, row in sorted(stats.items()):
        if speaker_id in major_ids:
            continue
        target_id, similarity = direct_matches.get(speaker_id, (None, None))
        if target_id is None:
            continue
        effect_match = effect_matches.get(speaker_id)
        match_basis = "clean_centroid"
        merge_confidence = "high"
        selected_similarity = similarity
        if similarity is None or similarity < cfg.diarization_embedding_match_threshold:
            match_basis = "major_bucket_fallback"
            merge_confidence = "low"
        if effect_match is not None and effect_match.get("accepted") is True:
            target_id = str(effect_match["speaker_id"])
            selected_similarity = float(effect_match["similarity"])
            match_basis = "effect_augmented"
            merge_confidence = "medium"
        if _source_minor_bucket_should_remain_distinct(row, similarity, cfg) and match_basis == "major_bucket_fallback":
            preserve = {
                "speaker_id": speaker_id,
                "model_fallback_speaker_id": target_id,
                "reason": "distinct_minor_bucket_insufficient_training_data",
                "clean_training_duration_sec": row["clean_training_duration_sec"],
                "target_clean_training_duration_sec": stats[target_id]["clean_training_duration_sec"],
                "centroid_similarity": round(selected_similarity, 6) if selected_similarity is not None else None,
                "clean_centroid_similarity": round(similarity, 6) if similarity is not None else None,
                "effect_augmented_similarity": round(float(effect_match["similarity"]), 6)
                if effect_match is not None
                else None,
                "effect_profile": effect_match.get("profile") if effect_match is not None else None,
                "effect_augmented_match_threshold": effect_match.get("threshold") if effect_match is not None else None,
                "effect_augmented_accepted": effect_match.get("accepted") if effect_match is not None else None,
                "match_basis": "distinct_low_similarity",
                "merge_confidence": "distinct",
            }
            preserves.append(preserve)
            _apply_source_bucket_model_fallback(manifest, preserve)
            continue
        merge = {
            "original_speaker_id": speaker_id,
            "merged_into_speaker_id": target_id,
            "reason": "minor_bucket_auto_merged",
            "clean_training_duration_sec": row["clean_training_duration_sec"],
            "target_clean_training_duration_sec": stats[target_id]["clean_training_duration_sec"],
            "centroid_similarity": round(selected_similarity, 6) if selected_similarity is not None else None,
            "clean_centroid_similarity": round(similarity, 6) if similarity is not None else None,
            "effect_augmented_similarity": round(float(effect_match["similarity"]), 6)
            if effect_match is not None
            else None,
            "effect_profile": effect_match.get("profile") if effect_match is not None else None,
            "effect_augmented_match_threshold": effect_match.get("threshold") if effect_match is not None else None,
            "effect_augmented_accepted": effect_match.get("accepted") if effect_match is not None else None,
            "match_basis": match_basis,
            "merge_confidence": merge_confidence,
        }
        merges.append(merge)
        _apply_source_bucket_merge(manifest, merge)
    return _source_speaker_bucket_qc_payload(stats, merges, preserves)


def _source_speaker_bucket_stats(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for segment in manifest.segments:
        speaker_id = segment.speaker_id
        if not speaker_id or segment.status in {"failed", "needs_manual_review"}:
            continue
        row = stats.setdefault(
            speaker_id,
            {
                "speaker_id": speaker_id,
                "segment_count": 0,
                "duration_sec": 0.0,
                "clean_training_segment_count": 0,
                "clean_training_duration_sec": 0.0,
                "clean_training_segment_ids": [],
            },
        )
        row["segment_count"] += 1
        row["duration_sec"] += segment.duration
        if _source_segment_is_clean_training_candidate(project_dir, segment, cfg):
            row["clean_training_segment_count"] += 1
            row["clean_training_duration_sec"] += segment.duration
            row["clean_training_segment_ids"].append(segment.id)
    for row in stats.values():
        row["duration_sec"] = round(float(row["duration_sec"]), 6)
        row["clean_training_duration_sec"] = round(float(row["clean_training_duration_sec"]), 6)
    return dict(sorted(stats.items()))


def _source_speaker_training_qc_payload(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    keep_original_segment_ids: list[str] = []
    for segment in manifest.segments:
        routing = segment.analysis.get("source_speaker_routing")
        if isinstance(routing, dict) and routing.get("decision") == "keep_original_texture":
            keep_original_segment_ids.append(segment.id)
        speaker_id = segment.speaker_id
        if not speaker_id or segment.status in {"failed", "needs_manual_review"}:
            continue
        row = rows.setdefault(
            speaker_id,
            {
                "speaker_id": speaker_id,
                "eligible_training_segment_ids": [],
                "excluded_segment_ids": [],
                "exclude_reason_counts": {},
                "representative_wav_candidates": [],
            },
        )
        duration_reasons = _source_speaker_training_qc_duration_reasons(segment, cfg)
        check = evaluate_voice_training_candidate(
            project_dir,
            segment,
            cfg,
            min_quality_score=cfg.gsv_few_shot_min_quality_score,
            require_source_script=True,
            require_speaker_id=True,
            source_language=cfg.source_language,
        )
        if not duration_reasons and check.accepted:
            row["eligible_training_segment_ids"].append(segment.id)
            if len(row["representative_wav_candidates"]) < 8:
                row["representative_wav_candidates"].append(
                    {
                        "segment_id": segment.id,
                        "audio_for_mix": segment.audio_for_mix,
                        "duration_sec": round(float(segment.duration), 6),
                    }
                )
            continue
        row["excluded_segment_ids"].append(segment.id)
        reasons = _source_speaker_training_qc_reject_reasons(segment, check.reject_reasons, duration_reasons)
        reason_counts = row["exclude_reason_counts"]
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    speakers = []
    for row in rows.values():
        row["exclude_reason_counts"] = dict(sorted(row["exclude_reason_counts"].items()))
        speakers.append(row)
    speakers.sort(key=lambda item: item["speaker_id"])
    training_eligible_counts = {
        row["speaker_id"]: len(row["eligible_training_segment_ids"])
        for row in speakers
        if row["eligible_training_segment_ids"]
    }
    training_excluded_counts = {
        row["speaker_id"]: len(row["excluded_segment_ids"])
        for row in speakers
        if row["excluded_segment_ids"]
    }
    possible_overlap_counts = {
        row["speaker_id"]: sum(
            1
            for segment_id in row["excluded_segment_ids"]
            if _source_speaker_segment_has_possible_overlap_reason(manifest, segment_id)
        )
        for row in speakers
    }
    possible_overlap_counts = {
        speaker_id: count for speaker_id, count in possible_overlap_counts.items() if count
    }
    return {
        "speakers": speakers,
        "keep_original_segment_ids": keep_original_segment_ids,
        "summary": {
            "speaker_count": len(speakers),
            "eligible_training_segment_count": sum(training_eligible_counts.values()),
            "excluded_training_segment_count": sum(training_excluded_counts.values()),
            "keep_original_segment_count": len(keep_original_segment_ids),
            "training_eligible_counts": dict(sorted(training_eligible_counts.items())),
            "training_excluded_counts": dict(sorted(training_excluded_counts.items())),
            "possible_overlap_counts": dict(sorted(possible_overlap_counts.items())),
        },
    }


def _source_speaker_training_qc_duration_reasons(segment: Segment, cfg: ProjectConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if segment.duration < cfg.gsv_few_shot_min_clip_sec:
        reasons.append(f"duration_below_min:{segment.duration:.3f}<{cfg.gsv_few_shot_min_clip_sec:.3f}")
    if segment.duration > cfg.gsv_few_shot_max_clip_sec:
        reasons.append(f"duration_above_max:{segment.duration:.3f}>{cfg.gsv_few_shot_max_clip_sec:.3f}")
    return tuple(reasons)


def _source_speaker_training_qc_reject_reasons(
    segment: Segment,
    check_reasons: tuple[str, ...],
    duration_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    reasons = [*duration_reasons, *check_reasons]
    voice_training = segment.analysis.get("voice_training")
    if isinstance(voice_training, dict):
        voice_reason = str(voice_training.get("reason") or "").strip()
        if voice_training.get("exclude") is True and voice_reason:
            replaced = False
            normalized: list[str] = []
            for reason in reasons:
                if reason == "manual_training_exclude":
                    normalized.append(voice_reason)
                    replaced = True
                else:
                    normalized.append(reason)
            if not replaced:
                normalized.insert(0, voice_reason)
            reasons = normalized
    if not reasons:
        reasons = ["not_clean_training_candidate"]
    return tuple(dict.fromkeys(reasons))


def _source_speaker_segment_has_possible_overlap_reason(
    manifest: PipelineManifest,
    segment_id: str,
) -> bool:
    segment = next((item for item in manifest.segments if item.id == segment_id), None)
    if segment is None:
        return False
    voice_training = segment.analysis.get("voice_training")
    reason = ""
    if isinstance(voice_training, dict):
        reason = str(voice_training.get("reason") or "")
    routing = segment.analysis.get("source_speaker_routing")
    routing_reason = str(routing.get("reason") or "") if isinstance(routing, dict) else ""
    return "overlap" in reason or "overlap" in routing_reason


def _source_segment_is_clean_training_candidate(
    project_dir: Path,
    segment: Segment,
    cfg: ProjectConfig,
) -> bool:
    if segment.duration < cfg.gsv_few_shot_min_clip_sec:
        return False
    if segment.duration > cfg.gsv_few_shot_max_clip_sec:
        return False
    check = evaluate_voice_training_candidate(
        project_dir,
        segment,
        cfg,
        min_quality_score=cfg.gsv_few_shot_min_quality_score,
        require_source_script=True,
        require_speaker_id=True,
        source_language=cfg.source_language,
    )
    return check.accepted


def _source_minor_bucket_should_remain_distinct(
    row: dict[str, Any],
    similarity: float | None,
    cfg: ProjectConfig,
) -> bool:
    clean_duration = float(row["clean_training_duration_sec"])
    min_distinct_sec = max(float(cfg.gsv_ref_min_sec), few_shot_min_total_sec(cfg) * 0.15)
    if clean_duration < min_distinct_sec:
        return False
    if similarity is None:
        return True
    return similarity <= _SOURCE_DISTINCT_MINOR_MAX_CENTROID_SIMILARITY


def _source_effect_augmented_bucket_matches(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    stats: dict[str, dict[str, Any]],
    major_ids: list[str],
    minor_ids: list[str],
    centroids: dict[str, np.ndarray],
    embedding_backend_factory: Callable[[], Any] | None,
) -> dict[str, dict[str, Any]]:
    if not minor_ids or embedding_backend_factory is None:
        return {}
    threshold = _source_effect_augmented_match_threshold(cfg)
    try:
        backend = embedding_backend_factory()
    except Exception:
        return {}
    prototypes = _source_effect_augmented_prototypes(project_dir, manifest, stats, major_ids, backend)
    if not prototypes:
        return {}
    matches: dict[str, dict[str, Any]] = {}
    for minor_id in minor_ids:
        minor_centroid = centroids.get(minor_id)
        if minor_centroid is None:
            continue
        best: dict[str, Any] | None = None
        for target_id, target_prototypes in prototypes.items():
            for prototype in target_prototypes:
                embedding = prototype["embedding"]
                if embedding.shape != minor_centroid.shape:
                    continue
                similarity = float(np.dot(minor_centroid, embedding))
                if best is None or similarity > float(best["similarity"]):
                    best = {
                        "speaker_id": target_id,
                        "profile": prototype["profile"],
                        "similarity": similarity,
                    }
        if best is not None:
            best["threshold"] = round(threshold, 6)
            best["accepted"] = float(best["similarity"]) >= threshold
            matches[minor_id] = best
    return matches


def _source_effect_augmented_match_threshold(cfg: ProjectConfig) -> float:
    return max(
        _SOURCE_EFFECT_AUGMENTED_MIN_MATCH_THRESHOLD,
        float(cfg.diarization_embedding_match_threshold) - 0.15,
    )


def _source_effect_augmented_prototypes(
    project_dir: Path,
    manifest: PipelineManifest,
    stats: dict[str, dict[str, Any]],
    major_ids: list[str],
    backend: Any,
) -> dict[str, list[dict[str, Any]]]:
    by_id = {segment.id: segment for segment in manifest.segments}
    output_root = ensure_inside_project(project_dir, project_dir / "work" / "diarization" / "effect_augmented")
    prototypes: dict[str, list[dict[str, Any]]] = {}
    for speaker_id in major_ids:
        selected_ids = list(stats[speaker_id].get("clean_training_segment_ids") or [])
        for segment_id in selected_ids[:_SOURCE_EFFECT_AUGMENTED_MAX_SEGMENTS_PER_SPEAKER]:
            segment = by_id.get(str(segment_id))
            if segment is None or not segment.audio_for_mix:
                continue
            try:
                audio_path = _resolve_project_or_absolute(project_dir, segment.audio_for_mix)
                data, sample_rate = load_audio(audio_path)
            except Exception:
                continue
            safe_segment_id = validate_file_safe_id(segment.id, "segment_id")
            speaker_dir = ensure_inside_project(project_dir, output_root / validate_file_safe_id(speaker_id, "speaker_id"))
            for profile, augmented in _source_effect_augmented_audio_variants(data, sample_rate):
                output_path = ensure_inside_project(
                    project_dir,
                    speaker_dir / f"{safe_segment_id}_{profile}.wav",
                )
                try:
                    write_audio(output_path, augmented, sample_rate)
                    embedding = _normalize_embedding(np.asarray(backend.embed_clip(output_path), dtype=np.float32).reshape(-1))
                except Exception:
                    continue
                prototypes.setdefault(speaker_id, []).append(
                    {
                        "profile": profile,
                        "embedding": embedding,
                    }
                )
    return prototypes


def _source_effect_augmented_audio_variants(data: np.ndarray, sample_rate: int) -> list[tuple[str, np.ndarray]]:
    stereo = ensure_stereo(data).astype(np.float32, copy=True)
    variants = {
        "sfx_center": _mix_source_sfx_bed(stereo, sample_rate, pan="center"),
        "sfx_left": _mix_source_sfx_bed(stereo, sample_rate, pan="left"),
        "sfx_right": _mix_source_sfx_bed(stereo, sample_rate, pan="right"),
        "echo": _source_echo_effect(stereo, sample_rate),
    }
    return [(profile, variants[profile]) for profile in _SOURCE_EFFECT_AUGMENTED_PROFILES]


def _mix_source_sfx_bed(data: np.ndarray, sample_rate: int, *, pan: str) -> np.ndarray:
    frames = data.shape[0]
    if frames <= 0:
        return data
    bed = _source_sfx_texture(frames, sample_rate)
    speech_rms = max(float(np.sqrt(np.mean(np.square(data)))), 1e-5)
    bed_rms = max(float(np.sqrt(np.mean(np.square(bed)))), 1e-5)
    bed = bed * (speech_rms * (10 ** (-12.0 / 20.0)) / bed_rms)
    if pan == "left":
        stereo_bed = np.stack([bed, bed * 0.18], axis=1)
    elif pan == "right":
        stereo_bed = np.stack([bed * 0.18, bed], axis=1)
    else:
        stereo_bed = np.repeat(bed[:, None], 2, axis=1)
    return _peak_guard(data + stereo_bed)


def _source_sfx_texture(frames: int, sample_rate: int) -> np.ndarray:
    rng = np.random.default_rng(9173)
    noise = rng.standard_normal(frames).astype(np.float32)
    rustle_window = max(1, int(round(sample_rate / 240.0)))
    if rustle_window > 1 and frames > 1:
        kernel = np.ones(rustle_window, dtype=np.float32) / float(rustle_window)
        noise = np.convolve(noise, kernel, mode="same").astype(np.float32)
    t = np.arange(frames, dtype=np.float32) / float(max(1, sample_rate))
    breath = 0.35 * np.sin(2.0 * np.pi * 7.0 * t).astype(np.float32)
    texture = noise + breath
    texture -= float(np.mean(texture))
    return texture.astype(np.float32)


def _source_echo_effect(data: np.ndarray, sample_rate: int) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    for delay_ms, decay in ((26.0, 0.16), (58.0, 0.09)):
        delay = int(round(sample_rate * delay_ms / 1000.0))
        if 0 < delay < len(out):
            out[delay:] += data[:-delay] * decay
    return _peak_guard(out)


def _peak_guard(data: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > peak_limit > 0:
        data = data * (peak_limit / peak)
    return np.clip(data, -1.0, 1.0).astype(np.float32)


def _speaker_centroids_from_turns(turns: list[DiarizationTurn]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for turn in turns:
        if turn.speaker_id and turn.embedding is not None:
            grouped.setdefault(turn.speaker_id, []).append(_normalize_embedding(turn.embedding))
    return {
        speaker_id: _normalize_embedding(np.mean(np.stack(embeddings), axis=0))
        for speaker_id, embeddings in grouped.items()
        if embeddings
    }


def _best_source_bucket_merge_target(
    speaker_id: str,
    major_ids: list[str],
    stats: dict[str, dict[str, Any]],
    centroids: dict[str, np.ndarray],
) -> tuple[str | None, float | None]:
    minor_centroid = centroids.get(speaker_id)
    best_id: str | None = None
    best_similarity: float | None = None
    for target_id in major_ids:
        target_centroid = centroids.get(target_id)
        similarity = (
            float(np.dot(minor_centroid, target_centroid))
            if minor_centroid is not None and target_centroid is not None
            else None
        )
        if best_id is None:
            best_id = target_id
            best_similarity = similarity
            continue
        if similarity is not None and (best_similarity is None or similarity > best_similarity):
            best_id = target_id
            best_similarity = similarity
            continue
        if similarity is None and best_similarity is None:
            current_duration = float(stats[target_id]["clean_training_duration_sec"])
            best_duration = float(stats[best_id]["clean_training_duration_sec"])
            if current_duration > best_duration:
                best_id = target_id
    return best_id, best_similarity


def _apply_source_bucket_merge(manifest: PipelineManifest, merge: dict[str, Any]) -> None:
    original = str(merge["original_speaker_id"])
    target = str(merge["merged_into_speaker_id"])
    for segment in manifest.segments:
        if segment.speaker_id != original:
            continue
        segment.speaker_id = target
        segment.analysis["source_speaker_bucket_normalization"] = dict(merge)
        _set_source_speaker_voice_training_exclusion(segment, "minor_bucket_auto_merged")


def _apply_source_bucket_model_fallback(manifest: PipelineManifest, preserve: dict[str, Any]) -> None:
    speaker_id = str(preserve["speaker_id"])
    fallback_id = str(preserve["model_fallback_speaker_id"])
    for segment in manifest.segments:
        if segment.speaker_id != speaker_id:
            continue
        segment.analysis["source_speaker_bucket_normalization"] = dict(preserve)
        segment.analysis["source_speaker_model_fallback"] = {
            "speaker_id": fallback_id,
            "reason": "insufficient_distinct_speaker_training_data",
            "match_basis": preserve["match_basis"],
            "centroid_similarity": preserve["centroid_similarity"],
        }
        _set_source_speaker_voice_training_exclusion(segment, "insufficient_distinct_speaker_training_data")


def _set_source_speaker_voice_training_exclusion(segment: Segment, reason: str) -> None:
    voice_training = dict(segment.analysis.get("voice_training") or {})
    existing_reason = str(voice_training.get("reason") or "")
    voice_training["exclude"] = True
    if not _source_speaker_preserve_training_exclusion_reason(existing_reason):
        voice_training["reason"] = reason
    segment.analysis["voice_training"] = voice_training


def _source_speaker_preserve_training_exclusion_reason(reason: str) -> bool:
    return "overlap" in reason


def _source_speaker_bucket_qc_payload(
    stats: dict[str, dict[str, Any]],
    merges: list[dict[str, Any]],
    preserves: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    preserves = preserves or []
    return {
        "speakers": list(stats.values()),
        "merges": merges,
        "preserves": preserves,
        "summary": {
            "speaker_bucket_count": len(stats),
            "major_bucket_count": len(stats) - len(merges) - len(preserves),
            "merged_bucket_count": len(merges),
            "preserved_bucket_count": len(preserves),
        },
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
    prompt_text_original = " ".join(turn.text or "" for turn in turns if turn.text).strip() or "voice reference"
    prompt_lang = turns[0].language or cfg.source_language
    if str(prompt_lang or "").strip().lower().replace("-", "_") in {"ja", "jp", "jpn", "japanese"}:
        normalized = normalize_japanese_kana_text(prompt_text_original)
        prompt_text = normalized.text
        text_normalization = {"policy": "ja_hiragana", "risk_flags": normalized.risk_flags}
    else:
        prompt_text = prompt_text_original
        text_normalization = {"policy": "none", "risk_flags": []}
    refs_json = refs_dir / "refs.json"
    write_json_atomic(
        refs_json,
        {
            "whisper_close": {
                "ref_audio_path": str(_relative_to_project(project_dir, ref_audio)),
                "prompt_text": prompt_text,
                "prompt_text_original": prompt_text_original,
                "prompt_lang": prompt_lang,
                "text_normalization": text_normalization,
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
