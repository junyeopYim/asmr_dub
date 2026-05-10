from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import yaml

from asmr_dub_pipeline.audio.training_filter import (
    VoiceTrainingCandidateCheck,
    evaluate_voice_training_candidate,
)
from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSError
from asmr_dub_pipeline.gpt_sovits.server import SHIM_DIR, _default_gsv_command
from asmr_dub_pipeline.pipeline.manifest_io import write_json_atomic
from asmr_dub_pipeline.rights import (
    ensure_inside_project,
    ensure_not_same_path,
    sha256_file,
)
from asmr_dub_pipeline.schemas import PipelineManifest, ProjectConfig, Segment
from asmr_dub_pipeline.script.duration_rewrite import japanese_pronunciation_count

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
FEW_SHOT_PROGRESS_LOG_SECONDS = 10.0

FEW_SHOT_STAGE = "gsv-few-shot"
FEW_SHOT_ARTIFACT_GPT = "gsv_few_shot_gpt_weights"
FEW_SHOT_ARTIFACT_SOVITS = "gsv_few_shot_sovits_weights"
SUPPORTED_VERSIONS = {"v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"}
TRAINING_IMPORTANT_MARKERS = (
    "error",
    "exception",
    "save",
    "saved",
    "saving",
    "traceback",
    "warning",
)
GSV_TRAINING_REQUIRED_MODULES = (
    "transformers",
    "torch",
    "librosa",
    "scipy",
    "numpy",
    "pytorch_lightning",
    "peft",
    "torchaudio",
    "tqdm",
    "ffmpeg",
    "pyopenjtalk",
    "x_transformers",
)
GSV_TRAINING_REQUIRED_MODULE_ATTRS = {"ffmpeg": "input"}
SKIP_TRAINING_STATUSES = {"needs_manual_review", "no_speech_detected", "non_speech_texture"}
DEFAULT_FEW_SHOT_MIN_TOTAL_SEC = 60.0
FEW_SHOT_TIMING_DEFICIT_BONUS = 0.45
FEW_SHOT_TIMING_OVERREPRESENTED_PENALTY = 0.50
FEW_SHOT_TIMING_BUCKET_TARGET_SHARES = {
    "very_fast": 0.05,
    "fast": 0.15,
    "normal_fast": 0.30,
    "normal": 0.25,
    "slow": 0.18,
    "very_slow": 0.07,
    "extra_slow": 0.0,
    "unknown": 0.0,
}
FEW_SHOT_TARGET_PACING_WEIGHT = 0.45
FEW_SHOT_TARGET_PACING_FACTOR_WINDOW = 2.0
FEW_SHOT_CLEAN_SOURCE_REJECT_PREFIXES = (
    "background_bleed_db_above_max",
    "side_to_mid_db_above_max",
)
_KOREAN_SYLLABLE_RE = re.compile(r"[가-힣]")


def few_shot_min_total_sec(cfg: ProjectConfig) -> float:
    configured = getattr(cfg, "gsv_few_shot_min_total_sec", None)
    if configured is not None:
        return float(configured)
    return DEFAULT_FEW_SHOT_MIN_TOTAL_SEC


@dataclass(frozen=True)
class FewShotTrainingItem:
    segment_id: str
    speaker_id: str
    source_audio_path: Path
    training_filename: str
    text: str
    language: str
    duration_sec: float
    quality_score: float = 0.0
    quality_issues: tuple[str, ...] = ()
    source_chars_per_sec: float = 0.0
    source_pronunciation_count: int = 0
    source_sec_per_pronunciation: float | None = None
    timing_bucket: str = "unknown"
    target_sec_per_syllable: float | None = None
    target_pacing_ratio: float | None = None
    target_pacing_score: float | None = None
    training_selection_score: float = 0.0
    selection_penalties: tuple[str, ...] = ()


@dataclass(frozen=True)
class FewShotDataset:
    items: list[FewShotTrainingItem]
    wav_dir: Path
    list_path: Path
    total_duration_sec: float


@dataclass(frozen=True)
class FewShotTrainingSelection:
    items: list[FewShotTrainingItem]
    diagnostics: list[dict[str, Any]]
    candidate_speaker_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _PacingSelectionState:
    items: tuple[FewShotTrainingItem, ...]
    total_duration_sec: float = 0.0
    speed_weighted_sum: float = 0.0
    speed_sq_weighted_sum: float = 0.0
    quality_weighted_sum: float = 0.0
    score_weighted_sum: float = 0.0


@dataclass(frozen=True)
class GPTSoVITSInstall:
    root: Path
    api_path: Path
    tts_config_path: Path
    version: str
    bert_base_path: Path
    cnhubert_base_path: Path
    pretrained_gpt_path: Path
    pretrained_sovits_path: Path
    sv_pretrained_path: Path | None
    s1_config_path: Path
    s2_config_path: Path
    s2_train_script: str
    needs_sv_features: bool
    checkout: str | None


@dataclass(frozen=True)
class FewShotTrainingResult:
    status: str
    fingerprint: str
    dataset: FewShotDataset
    install: GPTSoVITSInstall
    metadata_path: Path
    gpt_weights_path: Path
    sovits_weights_path: Path
    gpt_weights_sha256: str
    sovits_weights_sha256: str
    reused_existing: bool
    log_path: Path


@dataclass(frozen=True)
class FewShotTrainingProgress:
    phase: str
    status: str
    index: int
    total: int
    detail: str | None = None
    log_path: Path | None = None


FewShotProgressCallback = Callable[[FewShotTrainingProgress], None]


def _project_path(project_dir: Path, *parts: str) -> Path:
    return ensure_inside_project(project_dir, project_dir.joinpath(*parts))


def _command_parts(command: Sequence[str] | str | None) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if command:
        return [str(part) for part in command]
    return []


def _command_parts_or_default(command: Sequence[str] | str | None, base_url: str) -> list[str]:
    return _command_parts(command) or _default_gsv_command(base_url)


def _find_api_path_from_parts(parts: Sequence[str]) -> Path:
    for part in parts:
        path = Path(part).expanduser()
        if path.name == "api_v2.py" and path.exists():
            return path.resolve()
    raise GPTSoVITSError(
        "Cannot locate GPT-SoVITS api_v2.py for few-shot training. "
        "Set gsv_server_command or install GPT-SoVITS in a known local path."
    )


def _find_api_path(command: Sequence[str] | str | None, base_url: str) -> Path:
    return _find_api_path_from_parts(_command_parts_or_default(command, base_url))


def _is_python_executable(part: str) -> bool:
    name = Path(part).name.lower()
    return name in {"python", "python3", "python.exe"} or (
        name.startswith("python") and name[6:7].isdigit()
    )


def _command_python(parts: Sequence[str]) -> str | None:
    if parts and _is_python_executable(parts[0]):
        return str(parts[0])
    return None


def _dedupe_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _candidate_training_pythons(
    cfg: ProjectConfig,
    install: GPTSoVITSInstall,
    command: Sequence[str] | str | None,
) -> list[str]:
    parts = _command_parts_or_default(command or cfg.gsv_server_command, cfg.gsv_url)
    candidates: list[str] = []
    if os.environ.get("ASMR_DUB_GSV_PYTHON"):
        candidates.append(os.environ["ASMR_DUB_GSV_PYTHON"])
    command_python = _command_python(parts)
    if command_python:
        candidates.append(command_python)
    for candidate in (
        install.root / ".venv" / "bin" / "python",
        install.root / "venv" / "bin" / "python",
        install.root / "runtime" / "bin" / "python",
    ):
        if candidate.exists():
            candidates.append(str(candidate))
    base_prefix = Path(getattr(sys, "base_prefix", "") or "")
    if base_prefix:
        candidates.append(str(base_prefix / "bin" / "python"))
    for name in ("python", "python3"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    candidates.append(sys.executable or "python")
    return _dedupe_text(candidates)


def _python_missing_imports(python: str, modules: Sequence[str]) -> list[str] | None:
    code = f"""
import importlib
import importlib.util

mods = {list(modules)!r}
required_attrs = {GSV_TRAINING_REQUIRED_MODULE_ATTRS!r}
missing = []
for mod in mods:
    if importlib.util.find_spec(mod) is None:
        missing.append(mod)
        continue
    attr = required_attrs.get(mod)
    if attr is None:
        continue
    try:
        loaded = importlib.import_module(mod)
    except Exception:
        missing.append(mod)
        continue
    if not hasattr(loaded, attr):
        missing.append(mod)
print("\\n".join(missing))
raise SystemExit(1 if missing else 0)
"""
    try:
        result = subprocess.run(
            [python, "-s", "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode == 0:
        return []
    missing = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return missing or list(modules)


def _select_training_python(
    cfg: ProjectConfig,
    install: GPTSoVITSInstall,
    command: Sequence[str] | str | None,
    *,
    require_modules: bool,
) -> str:
    candidates = _candidate_training_pythons(cfg, install, command)
    if not require_modules:
        return candidates[0] if candidates else sys.executable or "python"
    failures: list[str] = []
    for candidate in candidates:
        missing = _python_missing_imports(candidate, GSV_TRAINING_REQUIRED_MODULES)
        if missing == []:
            return candidate
        if missing is None:
            failures.append(f"{candidate}: could not execute")
        else:
            failures.append(f"{candidate}: missing {', '.join(missing[:8])}")
    detail = "; ".join(failures)
    raise GPTSoVITSError(
        "Could not find a Python environment with GPT-SoVITS training dependencies. "
        "Install the GPT-SoVITS requirements in the selected environment or set "
        f"ASMR_DUB_GSV_PYTHON=/path/to/python. Checked: {detail}"
    )


def _resolve_command_path(root: Path, raw_path: str) -> Path | None:
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else [root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _find_tts_config_path_from_parts(parts: Sequence[str], root: Path) -> Path | None:
    for index, part in enumerate(parts):
        raw_path: str | None = None
        if part in {"-c", "--config", "--tts-config", "--tts_config"} and index + 1 < len(parts):
            raw_path = parts[index + 1]
        elif part.startswith(("-c=", "--config=")):
            raw_path = part.split("=", 1)[1]
        elif part.endswith((".yaml", ".yml")):
            raw_path = part
        if raw_path:
            resolved = _resolve_command_path(root, raw_path)
            if resolved is not None:
                return resolved
    return None


def _resolve_install_path(root: Path, raw_path: str | Path, label: str) -> Path:
    path = Path(raw_path).expanduser()
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.exists():
        raise GPTSoVITSError(f"GPT-SoVITS {label} does not exist: {resolved}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    if not isinstance(data, dict):
        raise GPTSoVITSError(f"GPT-SoVITS config must be a mapping: {path}")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text("utf-8"))
    if not isinstance(data, dict):
        raise GPTSoVITSError(f"GPT-SoVITS config must be a mapping: {path}")
    return data


def _git_checkout(root: Path) -> str | None:
    git_dir = root / ".git"
    if not git_dir.exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def discover_install(
    cfg: ProjectConfig,
    *,
    command: Sequence[str] | str | None = None,
) -> GPTSoVITSInstall:
    parts = _command_parts_or_default(command or cfg.gsv_server_command, cfg.gsv_url)
    api_path = _find_api_path_from_parts(parts)
    root = Path(cfg.gsv_server_cwd).expanduser().resolve() if cfg.gsv_server_cwd else api_path.parent
    tts_config_path = (
        _find_tts_config_path_from_parts(parts, root)
        or root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    )
    if not tts_config_path.exists():
        raise GPTSoVITSError(f"GPT-SoVITS tts_infer.yaml not found: {tts_config_path}")
    tts_config = _read_yaml(tts_config_path)
    requested_version = cfg.gsv_few_shot_version
    active_section = "custom" if "custom" in tts_config else requested_version
    if requested_version != "auto":
        active_section = requested_version
    raw_active = tts_config.get(active_section)
    if not isinstance(raw_active, dict):
        raise GPTSoVITSError(f"GPT-SoVITS config section not found: {active_section}")
    version = str(raw_active.get("version") or ("v2" if requested_version == "auto" else requested_version))
    if version not in SUPPORTED_VERSIONS:
        raise GPTSoVITSError(f"Unsupported GPT-SoVITS few-shot version: {version}")
    s2_config_name = f"s2{version}.json" if version in {"v2Pro", "v2ProPlus"} else "s2.json"
    s2_train_script = (
        "GPT_SoVITS/s2_train_v3_lora.py" if version in {"v3", "v4"} else "GPT_SoVITS/s2_train.py"
    )
    needs_sv_features = "Pro" in version
    sv_pretrained_path = (
        _resolve_install_path(
            root,
            "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt",
            "speaker verification weights",
        )
        if needs_sv_features
        else None
    )
    return GPTSoVITSInstall(
        root=root,
        api_path=api_path,
        tts_config_path=tts_config_path,
        version=version,
        bert_base_path=_resolve_install_path(root, raw_active["bert_base_path"], "BERT path"),
        cnhubert_base_path=_resolve_install_path(root, raw_active["cnhuhbert_base_path"], "CNHuBERT path"),
        pretrained_gpt_path=_resolve_install_path(root, raw_active["t2s_weights_path"], "GPT weights"),
        pretrained_sovits_path=_resolve_install_path(root, raw_active["vits_weights_path"], "SoVITS weights"),
        sv_pretrained_path=sv_pretrained_path,
        s1_config_path=_resolve_install_path(
            root,
            "GPT_SoVITS/configs/s1longer.yaml" if version == "v1" else "GPT_SoVITS/configs/s1longer-v2.yaml",
            "s1 training config",
        ),
        s2_config_path=_resolve_install_path(root, f"GPT_SoVITS/configs/{s2_config_name}", "s2 training config"),
        s2_train_script=s2_train_script,
        needs_sv_features=needs_sv_features,
        checkout=_git_checkout(root),
    )


def select_training_items(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    *,
    speaker_id: str | None = None,
) -> list[FewShotTrainingItem]:
    return _select_training_items_with_diagnostics(
        project_dir,
        manifest,
        cfg,
        speaker_id=speaker_id,
    ).items


def select_training_speaker_ids(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
) -> list[str]:
    return list(
        _select_training_items_with_diagnostics(
            project_dir,
            manifest,
            cfg,
            enforce_single_speaker=False,
        ).candidate_speaker_ids
    )


def _select_training_items_with_diagnostics(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    *,
    speaker_id: str | None = None,
    enforce_single_speaker: bool = True,
) -> FewShotTrainingSelection:
    candidates: list[FewShotTrainingItem] = []
    diagnostics: list[dict[str, Any]] = []
    source_language = _canonical_language(cfg.source_language)
    target_sec_per_syllable = _target_korean_sec_per_syllable(manifest, cfg)
    for segment in sorted(manifest.segments, key=lambda item: (item.start, item.id)):
        if segment.status in SKIP_TRAINING_STATUSES:
            continue
        if speaker_id is not None and segment.speaker_id != speaker_id:
            continue
        check = evaluate_voice_training_candidate(
            project_dir,
            segment,
            cfg,
            min_quality_score=cfg.gsv_few_shot_min_quality_score,
            require_source_script=True,
            require_speaker_id=True,
            source_language=source_language,
        )
        text = segment.source_script.text.strip() if segment.source_script else ""
        source_chars_per_sec = _source_chars_per_sec(text, segment.duration)
        source_pronunciation_count = japanese_pronunciation_count(text)
        source_sec_per_pronunciation = _source_sec_per_pronunciation(
            source_pronunciation_count,
            segment.duration,
        )
        timing_bucket = _few_shot_timing_bucket(source_sec_per_pronunciation)
        target_pacing_ratio, target_pacing_score = _target_pacing_metrics(
            source_sec_per_pronunciation,
            target_sec_per_syllable,
        )
        training_selection_score, selection_penalties = _training_selection_score(
            check.metrics.score if check.metrics else None,
            source_chars_per_sec,
            text,
            cfg,
            target_pacing_score=target_pacing_score,
        )
        reject_reasons = [
            *check.reject_reasons,
            *_few_shot_duration_reject_reasons(segment, cfg),
            *_few_shot_text_reject_reasons(source_chars_per_sec, cfg),
        ]
        soft_reject_reasons = _few_shot_soft_reject_reasons(
            reject_reasons,
            cfg,
            target_sec_per_syllable,
        )
        hard_reject_reasons = [
            reason for reason in reject_reasons if reason not in soft_reject_reasons
        ]
        if hard_reject_reasons or not check.source_audio_path or not check.metrics:
            diagnostics.append(
                _few_shot_diagnostic_row(
                    segment,
                    check,
                    cfg,
                    selected=False,
                    reject_reasons=hard_reject_reasons or ("missing_source_audio",),
                    source_chars_per_sec=source_chars_per_sec,
                    source_pronunciation_count=source_pronunciation_count,
                    source_sec_per_pronunciation=source_sec_per_pronunciation,
                    timing_bucket=timing_bucket,
                    target_sec_per_syllable=target_sec_per_syllable,
                    target_pacing_ratio=target_pacing_ratio,
                    target_pacing_score=target_pacing_score,
                    training_selection_score=training_selection_score,
                    selection_penalties=selection_penalties,
                )
            )
            continue
        source_audio_path = check.source_audio_path
        language = segment.source_script.language if segment.source_script else cfg.asr_language
        training_filename = f"{segment.id}.wav"
        if soft_reject_reasons:
            relaxed_penalty = float(getattr(cfg, "gsv_few_shot_relaxed_clean_source_penalty", 0.15))
            training_selection_score = max(
                0.0,
                float(training_selection_score or 0.0) - relaxed_penalty,
            )
            selection_penalties = (
                *selection_penalties,
                *(
                    f"relaxed_clean_source_filter:{reason}"
                    for reason in soft_reject_reasons
                ),
            )
        candidates.append(
            FewShotTrainingItem(
                segment_id=segment.id,
                speaker_id=segment.speaker_id or "",
                source_audio_path=source_audio_path,
                training_filename=training_filename,
                text=text,
                language=_canonical_language(language or cfg.asr_language),
                duration_sec=segment.duration,
                quality_score=check.metrics.score,
                quality_issues=tuple(check.metrics.issues),
                source_chars_per_sec=source_chars_per_sec,
                source_pronunciation_count=source_pronunciation_count,
                source_sec_per_pronunciation=source_sec_per_pronunciation,
                timing_bucket=timing_bucket,
                target_sec_per_syllable=target_sec_per_syllable,
                target_pacing_ratio=target_pacing_ratio,
                target_pacing_score=target_pacing_score,
                training_selection_score=training_selection_score,
                selection_penalties=selection_penalties,
            )
        )
        diagnostics.append(
            _few_shot_diagnostic_row(
                segment,
                check,
                cfg,
                selected=False,
                reject_reasons=("not_selected_target_reached",),
                source_chars_per_sec=source_chars_per_sec,
                source_pronunciation_count=source_pronunciation_count,
                source_sec_per_pronunciation=source_sec_per_pronunciation,
                timing_bucket=timing_bucket,
                target_sec_per_syllable=target_sec_per_syllable,
                target_pacing_ratio=target_pacing_ratio,
                target_pacing_score=target_pacing_score,
                training_selection_score=training_selection_score,
                selection_penalties=selection_penalties,
            )
        )
    items: list[FewShotTrainingItem] = []
    candidates, duplicate_reasons = _dedupe_training_items(candidates)
    items = _select_training_items_for_pacing(candidates, cfg)
    total = sum(item.duration_sec for item in items)
    min_total_sec = few_shot_min_total_sec(cfg)
    candidate_speaker_ids = tuple(sorted({item.speaker_id for item in candidates if item.speaker_id}))
    if total < min_total_sec:
        raise GPTSoVITSError(
            "Not enough source voice data for GPT-SoVITS few-shot training after clean-source filtering: "
            f"selected {total:.2f}s, need at least {min_total_sec:.2f}s."
        )
    if enforce_single_speaker:
        _ensure_single_few_shot_speaker(items)
    selected_ids = {item.segment_id for item in items}
    normalized_diagnostics: list[dict[str, Any]] = []
    for row in diagnostics:
        if row["segment_id"] in selected_ids:
            row = {**row, "selected_for_training": True, "reject_reasons": []}
        elif row["segment_id"] in duplicate_reasons:
            row = {
                **row,
                "selected_for_training": False,
                "reject_reasons": [duplicate_reasons[row["segment_id"]]],
            }
        normalized_diagnostics.append(row)
    return FewShotTrainingSelection(
        items=items,
        diagnostics=normalized_diagnostics,
        candidate_speaker_ids=candidate_speaker_ids,
    )


def _dedupe_training_items(
    candidates: Sequence[FewShotTrainingItem],
) -> tuple[list[FewShotTrainingItem], dict[str, str]]:
    kept_by_source: dict[Path, FewShotTrainingItem] = {}
    duplicate_reasons: dict[str, str] = {}
    for item in sorted(candidates, key=lambda candidate: (-candidate.quality_score, candidate.duration_sec, candidate.segment_id)):
        source_key = item.source_audio_path.resolve()
        kept = kept_by_source.get(source_key)
        if kept is None:
            kept_by_source[source_key] = item
            continue
        duplicate_reasons[item.segment_id] = f"duplicate_source_audio:{kept.segment_id}"
    kept_ids = {item.segment_id for item in kept_by_source.values()}
    return [item for item in candidates if item.segment_id in kept_ids], duplicate_reasons


def _few_shot_soft_reject_reasons(
    reject_reasons: Sequence[str],
    cfg: ProjectConfig,
    target_sec_per_syllable: float | None,
) -> tuple[str, ...]:
    if not getattr(cfg, "gsv_few_shot_relax_clean_source_for_pacing", True):
        return ()
    if target_sec_per_syllable is None:
        return ()
    soft_reasons = [
        reason
        for reason in reject_reasons
        if reason.startswith(FEW_SHOT_CLEAN_SOURCE_REJECT_PREFIXES)
    ]
    if len(soft_reasons) != len(reject_reasons):
        return ()
    return tuple(soft_reasons)


def _few_shot_duration_reject_reasons(segment: Segment, cfg: ProjectConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if segment.duration < cfg.gsv_few_shot_min_clip_sec:
        reasons.append(f"duration_below_min:{segment.duration:.3f}<{cfg.gsv_few_shot_min_clip_sec:.3f}")
    if segment.duration > cfg.gsv_few_shot_max_clip_sec:
        reasons.append(f"duration_above_max:{segment.duration:.3f}>{cfg.gsv_few_shot_max_clip_sec:.3f}")
    return tuple(reasons)


def _source_chars_per_sec(text: str, duration_sec: float) -> float:
    if duration_sec <= 0:
        return 0.0
    normalized = "".join(text.split())
    return len(normalized) / duration_sec


def _source_sec_per_pronunciation(
    source_pronunciation_count: int,
    duration_sec: float,
) -> float | None:
    if duration_sec <= 0 or source_pronunciation_count <= 0:
        return None
    return duration_sec / float(source_pronunciation_count)


def _korean_syllable_count(text: str) -> int:
    return len(_KOREAN_SYLLABLE_RE.findall(text or ""))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ranked = sorted(values)
    midpoint = len(ranked) // 2
    if len(ranked) % 2:
        return ranked[midpoint]
    return (ranked[midpoint - 1] + ranked[midpoint]) / 2.0


def _target_korean_sec_per_syllable(
    manifest: PipelineManifest,
    cfg: ProjectConfig,
) -> float | None:
    if _canonical_language(cfg.target_language) != "ko":
        return None
    values: list[float] = []
    for segment in manifest.segments:
        script = segment.script
        if script is None or _canonical_language(script.tts_language) != "ko":
            continue
        syllable_count = _korean_syllable_count(script.tts_text)
        if syllable_count <= 0 or segment.duration <= 0:
            continue
        values.append(segment.duration / float(syllable_count))
    return _median(values)


def _target_pacing_metrics(
    source_sec_per_pronunciation: float | None,
    target_sec_per_syllable: float | None,
) -> tuple[float | None, float | None]:
    if (
        source_sec_per_pronunciation is None
        or target_sec_per_syllable is None
        or source_sec_per_pronunciation <= 0
        or target_sec_per_syllable <= 0
    ):
        return None, None
    ratio = source_sec_per_pronunciation / target_sec_per_syllable
    log_distance = abs(math.log(ratio) / math.log(FEW_SHOT_TARGET_PACING_FACTOR_WINDOW))
    score = max(0.0, min(1.0, 1.0 - log_distance))
    return ratio, score


def _few_shot_timing_bucket(source_sec_per_pronunciation: float | None) -> str:
    if source_sec_per_pronunciation is None:
        return "unknown"
    if source_sec_per_pronunciation < 0.16:
        return "very_fast"
    if source_sec_per_pronunciation < 0.20:
        return "fast"
    if source_sec_per_pronunciation < 0.25:
        return "normal_fast"
    if source_sec_per_pronunciation < 0.30:
        return "normal"
    if source_sec_per_pronunciation < 0.40:
        return "slow"
    if source_sec_per_pronunciation < 0.55:
        return "very_slow"
    return "extra_slow"


def _few_shot_base_sort_key(candidate: FewShotTrainingItem) -> tuple[float, float, float, str]:
    return (
        -candidate.training_selection_score,
        -candidate.quality_score,
        candidate.duration_sec,
        candidate.segment_id,
    )


def _timing_balance_adjustment(
    candidate: FewShotTrainingItem,
    selected_duration_by_bucket: dict[str, float],
    selected_total_sec: float,
) -> float:
    target_share = FEW_SHOT_TIMING_BUCKET_TARGET_SHARES.get(candidate.timing_bucket, 0.0)
    current_share = (
        selected_duration_by_bucket.get(candidate.timing_bucket, 0.0) / selected_total_sec
        if selected_total_sec > 0
        else 0.0
    )
    deficit = target_share - current_share
    target_pacing_factor = (
        1.0 - candidate.target_pacing_score
        if candidate.target_pacing_score is not None
        else 1.0
    )
    if deficit >= 0:
        return deficit * FEW_SHOT_TIMING_DEFICIT_BONUS * target_pacing_factor
    return deficit * FEW_SHOT_TIMING_OVERREPRESENTED_PENALTY * target_pacing_factor


def _timing_balanced_sort_key(
    candidate: FewShotTrainingItem,
    selected_duration_by_bucket: dict[str, float],
    selected_total_sec: float,
) -> tuple[float, float, float, float, str]:
    balanced_score = candidate.training_selection_score + _timing_balance_adjustment(
        candidate,
        selected_duration_by_bucket,
        selected_total_sec,
    )
    return (
        -balanced_score,
        -candidate.training_selection_score,
        -candidate.quality_score,
        candidate.duration_sec,
        candidate.segment_id,
    )


def _select_timing_balanced_training_items(
    candidates: Sequence[FewShotTrainingItem],
    cfg: ProjectConfig,
) -> list[FewShotTrainingItem]:
    ranked = sorted(candidates, key=_few_shot_base_sort_key)
    target_total_sec = few_shot_min_total_sec(cfg)
    if sum(item.duration_sec for item in ranked) <= target_total_sec:
        return ranked
    selected: list[FewShotTrainingItem] = []
    selected_duration_by_bucket: dict[str, float] = {}
    selected_total_sec = 0.0
    remaining = list(ranked)
    while remaining and selected_total_sec < target_total_sec:
        remaining.sort(
            key=lambda candidate: _timing_balanced_sort_key(
                candidate,
                selected_duration_by_bucket,
                selected_total_sec,
            )
        )
        item = remaining.pop(0)
        selected.append(item)
        selected_total_sec += item.duration_sec
        selected_duration_by_bucket[item.timing_bucket] = (
            selected_duration_by_bucket.get(item.timing_bucket, 0.0) + item.duration_sec
        )
    return selected


def _select_training_items_for_pacing(
    candidates: Sequence[FewShotTrainingItem],
    cfg: ProjectConfig,
) -> list[FewShotTrainingItem]:
    if not getattr(cfg, "gsv_few_shot_pacing_target_enabled", True):
        return _select_timing_balanced_training_items(candidates, cfg)
    if not candidates:
        return []
    target_sec_per_syllable = next(
        (
            item.target_sec_per_syllable
            for item in candidates
            if item.target_sec_per_syllable is not None and item.target_sec_per_syllable > 0
        ),
        None,
    )
    if target_sec_per_syllable is None:
        return _select_timing_balanced_training_items(candidates, cfg)
    selected = _select_target_mean_variance_training_items(
        candidates,
        cfg,
        target_speed=1.0 / target_sec_per_syllable,
    )
    if selected:
        return selected
    return _select_timing_balanced_training_items(candidates, cfg)


def _select_target_mean_variance_training_items(
    candidates: Sequence[FewShotTrainingItem],
    cfg: ProjectConfig,
    *,
    target_speed: float,
) -> list[FewShotTrainingItem]:
    viable = [item for item in candidates if _few_shot_pronunciation_per_sec(item) is not None]
    if not viable:
        return []
    target_total_sec = few_shot_min_total_sec(cfg)
    base_tolerance = float(getattr(cfg, "gsv_few_shot_pacing_target_tolerance", 0.10))
    max_tolerance = max(
        base_tolerance,
        float(getattr(cfg, "gsv_few_shot_pacing_max_target_tolerance", 0.30)),
    )
    tolerance_steps = _pacing_tolerance_steps(base_tolerance, max_tolerance)
    max_overage_ratio = float(getattr(cfg, "gsv_few_shot_pacing_max_duration_overage_ratio", 0.10))
    normal_max_duration = target_total_sec + max(1.0, target_total_sec * max_overage_ratio)
    fallback_max_duration = target_total_sec + max(item.duration_sec for item in viable)
    for max_duration_sec in (normal_max_duration, fallback_max_duration):
        for tolerance in tolerance_steps:
            selected = _beam_select_target_pacing_items(
                viable,
                cfg,
                target_speed=target_speed,
                target_total_sec=target_total_sec,
                max_duration_sec=max_duration_sec,
                tolerance=tolerance,
            )
            if selected:
                return selected
    return []


def _pacing_tolerance_steps(base_tolerance: float, max_tolerance: float) -> tuple[float, ...]:
    values = [max(0.0, base_tolerance)]
    while values[-1] + 1e-9 < max_tolerance:
        values.append(min(max_tolerance, values[-1] * 2.0 if values[-1] > 0 else 0.05))
    return tuple(dict.fromkeys(round(value, 6) for value in values))


def _beam_select_target_pacing_items(
    candidates: Sequence[FewShotTrainingItem],
    cfg: ProjectConfig,
    *,
    target_speed: float,
    target_total_sec: float,
    max_duration_sec: float,
    tolerance: float,
) -> list[FewShotTrainingItem]:
    beam_size = int(getattr(cfg, "gsv_few_shot_pacing_beam_size", 768))
    beam_size = max(32, min(4096, beam_size))
    search_iterations = int(getattr(cfg, "gsv_few_shot_pacing_search_iterations", 12000))
    ordered = sorted(candidates, key=lambda item: _pacing_candidate_sort_key(item, target_speed))
    states: list[_PacingSelectionState] = [_PacingSelectionState(items=())]
    best_state: _PacingSelectionState | None = None
    for item in ordered:
        additions: list[_PacingSelectionState] = []
        for state in states:
            next_state = _pacing_state_add(state, item)
            if next_state.total_duration_sec <= max_duration_sec + 1e-6:
                additions.append(next_state)
        states = _prune_pacing_states(
            [*states, *additions],
            beam_size=beam_size,
            target_speed=target_speed,
            target_total_sec=target_total_sec,
        )
        for state in states:
            if not _pacing_state_is_valid(
                state,
                target_speed=target_speed,
                target_total_sec=target_total_sec,
                tolerance=tolerance,
            ):
                continue
            if best_state is None or _pacing_final_sort_key(
                state,
                target_speed=target_speed,
                target_total_sec=target_total_sec,
                cfg=cfg,
            ) > _pacing_final_sort_key(
                best_state,
                target_speed=target_speed,
                target_total_sec=target_total_sec,
                cfg=cfg,
            ):
                best_state = state
    if best_state:
        return list(best_state.items)
    if search_iterations <= 0:
        return []
    return _local_search_target_pacing_items(
        ordered,
        cfg,
        target_speed=target_speed,
        target_total_sec=target_total_sec,
        max_duration_sec=max_duration_sec,
        tolerance=tolerance,
        iterations=search_iterations,
    )


def _local_search_target_pacing_items(
    candidates: Sequence[FewShotTrainingItem],
    cfg: ProjectConfig,
    *,
    target_speed: float,
    target_total_sec: float,
    max_duration_sec: float,
    tolerance: float,
    iterations: int,
) -> list[FewShotTrainingItem]:
    if not candidates:
        return []
    rng = random.Random(_pacing_search_seed(candidates, cfg, target_speed, target_total_sec))
    pool = list(candidates)
    seeds = _pacing_local_search_seeds(pool, target_speed, target_total_sec)
    current = _repair_pacing_selection(
        seeds[0] if seeds else [],
        pool,
        rng,
        target_total_sec=target_total_sec,
        max_duration_sec=max_duration_sec,
    )
    best = _best_valid_pacing_selection(
        seeds,
        pool,
        rng,
        target_speed=target_speed,
        target_total_sec=target_total_sec,
        max_duration_sec=max_duration_sec,
        tolerance=tolerance,
        cfg=cfg,
    )
    for index in range(max(0, iterations)):
        if index % 250 == 0 and seeds:
            current = _repair_pacing_selection(
                list(seeds[(index // 250) % len(seeds)]),
                pool,
                rng,
                target_total_sec=target_total_sec,
                max_duration_sec=max_duration_sec,
            )
        trial = _repair_pacing_selection(
            _mutate_pacing_selection(current, pool, rng),
            pool,
            rng,
            target_total_sec=target_total_sec,
            max_duration_sec=max_duration_sec,
        )
        trial_key = _valid_pacing_selection_key(
            trial,
            target_speed=target_speed,
            target_total_sec=target_total_sec,
            tolerance=tolerance,
            cfg=cfg,
        )
        current_key = _valid_pacing_selection_key(
            current,
            target_speed=target_speed,
            target_total_sec=target_total_sec,
            tolerance=tolerance,
            cfg=cfg,
        )
        best_key = (
            _valid_pacing_selection_key(
                best,
                target_speed=target_speed,
                target_total_sec=target_total_sec,
                tolerance=tolerance,
                cfg=cfg,
            )
            if best is not None
            else None
        )
        if trial_key is not None and (best_key is None or trial_key > best_key):
            best = trial
        accept_trial = trial_key is not None and (
            current_key is None or trial_key > current_key or rng.random() < 0.02
        )
        if not accept_trial and current_key is None and rng.random() < 0.50:
            accept_trial = True
        if accept_trial:
            current = trial
    return best or []


def _pacing_search_seed(
    candidates: Sequence[FewShotTrainingItem],
    cfg: ProjectConfig,
    target_speed: float,
    target_total_sec: float,
) -> int:
    payload = "|".join(item.segment_id for item in candidates)
    payload += f"|{getattr(cfg, 'base_seed', 0)}|{target_speed:.6f}|{target_total_sec:.6f}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _pacing_local_search_seeds(
    pool: Sequence[FewShotTrainingItem],
    target_speed: float,
    target_total_sec: float,
) -> list[list[FewShotTrainingItem]]:
    seeds: list[list[FewShotTrainingItem]] = []
    sorters = [
        _few_shot_base_sort_key,
        lambda item: (abs((_few_shot_pronunciation_per_sec(item) or target_speed) - target_speed), -item.training_selection_score),
        lambda item: (_few_shot_pronunciation_per_sec(item) or target_speed, -item.training_selection_score),
        lambda item: (-( _few_shot_pronunciation_per_sec(item) or target_speed), -item.training_selection_score),
        lambda item: (-abs((_few_shot_pronunciation_per_sec(item) or target_speed) - target_speed), -item.training_selection_score),
    ]
    for sorter in sorters:
        selected: list[FewShotTrainingItem] = []
        total = 0.0
        for item in sorted(pool, key=sorter):
            selected.append(item)
            total += item.duration_sec
            if total >= target_total_sec:
                break
        seeds.append(selected)
    slow = sorted(
        [item for item in pool if (_few_shot_pronunciation_per_sec(item) or target_speed) < target_speed],
        key=lambda item: (_few_shot_pronunciation_per_sec(item) or target_speed, -item.training_selection_score),
    )[:8]
    fast = sorted(
        [item for item in pool if (_few_shot_pronunciation_per_sec(item) or target_speed) > target_speed],
        key=lambda item: (-(_few_shot_pronunciation_per_sec(item) or target_speed), -item.training_selection_score),
    )[:8]
    for slow_count in range(1, min(4, len(slow)) + 1):
        for fast_count in range(1, min(4, len(fast)) + 1):
            seeds.append([*slow[:slow_count], *fast[:fast_count]])
    return seeds


def _best_valid_pacing_selection(
    seeds: Sequence[Sequence[FewShotTrainingItem]],
    pool: Sequence[FewShotTrainingItem],
    rng: random.Random,
    *,
    target_speed: float,
    target_total_sec: float,
    max_duration_sec: float,
    tolerance: float,
    cfg: ProjectConfig,
) -> list[FewShotTrainingItem] | None:
    best: list[FewShotTrainingItem] | None = None
    best_key: tuple[float, float, float, float, float] | None = None
    for seed in seeds:
        selected = _repair_pacing_selection(
            list(seed),
            pool,
            rng,
            target_total_sec=target_total_sec,
            max_duration_sec=max_duration_sec,
        )
        key = _valid_pacing_selection_key(
            selected,
            target_speed=target_speed,
            target_total_sec=target_total_sec,
            tolerance=tolerance,
            cfg=cfg,
        )
        if key is not None and (best_key is None or key > best_key):
            best = selected
            best_key = key
    return best


def _valid_pacing_selection_key(
    selected: Sequence[FewShotTrainingItem] | None,
    *,
    target_speed: float,
    target_total_sec: float,
    tolerance: float,
    cfg: ProjectConfig,
) -> tuple[float, float, float, float, float] | None:
    if selected is None:
        return None
    state = _pacing_state_from_items(selected)
    if not _pacing_state_is_valid(
        state,
        target_speed=target_speed,
        target_total_sec=target_total_sec,
        tolerance=tolerance,
    ):
        return None
    return _pacing_final_sort_key(
        state,
        target_speed=target_speed,
        target_total_sec=target_total_sec,
        cfg=cfg,
    )


def _mutate_pacing_selection(
    selected: Sequence[FewShotTrainingItem],
    pool: Sequence[FewShotTrainingItem],
    rng: random.Random,
) -> list[FewShotTrainingItem]:
    trial = list(selected)
    selected_ids = {item.segment_id for item in trial}
    operation = rng.random()
    if operation < 0.30 and len(trial) > 1:
        trial.pop(rng.randrange(len(trial)))
    elif operation < 0.65:
        choices = [item for item in pool if item.segment_id not in selected_ids]
        if choices:
            trial.append(rng.choice(choices))
    elif trial:
        choices = [item for item in pool if item.segment_id not in selected_ids]
        if choices:
            trial[rng.randrange(len(trial))] = rng.choice(choices)
    return _dedupe_pacing_selection(trial)


def _repair_pacing_selection(
    selected: Sequence[FewShotTrainingItem],
    pool: Sequence[FewShotTrainingItem],
    rng: random.Random,
    *,
    target_total_sec: float,
    max_duration_sec: float,
) -> list[FewShotTrainingItem]:
    trial = _dedupe_pacing_selection(selected)
    for _ in range(64):
        total = sum(item.duration_sec for item in trial)
        if total >= target_total_sec:
            break
        selected_ids = {item.segment_id for item in trial}
        choices = [
            item
            for item in pool
            if item.segment_id not in selected_ids
            and total + item.duration_sec <= max_duration_sec + 1e-6
        ]
        if not choices:
            choices = [item for item in pool if item.segment_id not in selected_ids]
        if not choices:
            break
        trial.append(rng.choice(choices))
    for _ in range(64):
        total = sum(item.duration_sec for item in trial)
        if total <= max_duration_sec + 1e-6 or len(trial) <= 1:
            break
        trial.pop(rng.randrange(len(trial)))
    return _dedupe_pacing_selection(trial)


def _dedupe_pacing_selection(
    selected: Sequence[FewShotTrainingItem],
) -> list[FewShotTrainingItem]:
    seen: set[str] = set()
    result: list[FewShotTrainingItem] = []
    for item in selected:
        if item.segment_id in seen:
            continue
        seen.add(item.segment_id)
        result.append(item)
    return result


def _pacing_state_from_items(
    items: Sequence[FewShotTrainingItem],
) -> _PacingSelectionState:
    state = _PacingSelectionState(items=())
    for item in items:
        state = _pacing_state_add(state, item)
    return state


def _pacing_candidate_sort_key(
    item: FewShotTrainingItem,
    target_speed: float,
) -> tuple[float, float, float, float, str]:
    speed = _few_shot_pronunciation_per_sec(item) or 0.0
    return (
        -abs(speed - target_speed),
        -item.training_selection_score,
        -item.quality_score,
        item.duration_sec,
        item.segment_id,
    )


def _pacing_state_add(
    state: _PacingSelectionState,
    item: FewShotTrainingItem,
) -> _PacingSelectionState:
    speed = _few_shot_pronunciation_per_sec(item)
    if speed is None:
        return state
    duration = item.duration_sec
    return _PacingSelectionState(
        items=(*state.items, item),
        total_duration_sec=state.total_duration_sec + duration,
        speed_weighted_sum=state.speed_weighted_sum + speed * duration,
        speed_sq_weighted_sum=state.speed_sq_weighted_sum + speed * speed * duration,
        quality_weighted_sum=state.quality_weighted_sum + item.quality_score * duration,
        score_weighted_sum=state.score_weighted_sum + item.training_selection_score * duration,
    )


def _prune_pacing_states(
    states: Sequence[_PacingSelectionState],
    *,
    beam_size: int,
    target_speed: float,
    target_total_sec: float,
) -> list[_PacingSelectionState]:
    best_by_bin: dict[tuple[int, int, int], _PacingSelectionState] = {}
    for state in states:
        mean = _pacing_state_mean(state) or 0.0
        key = (
            int(round(state.total_duration_sec * 2.0)),
            int(round((mean - target_speed) * 10.0)),
            len(state.items),
        )
        current = best_by_bin.get(key)
        if current is None or _pacing_partial_sort_key(
            state,
            target_speed=target_speed,
            target_total_sec=target_total_sec,
        ) > _pacing_partial_sort_key(
            current,
            target_speed=target_speed,
            target_total_sec=target_total_sec,
        ):
            best_by_bin[key] = state
    return sorted(
        best_by_bin.values(),
        key=lambda state: _pacing_partial_sort_key(
            state,
            target_speed=target_speed,
            target_total_sec=target_total_sec,
        ),
        reverse=True,
    )[:beam_size]


def _pacing_partial_sort_key(
    state: _PacingSelectionState,
    *,
    target_speed: float,
    target_total_sec: float,
) -> tuple[float, float, float, float, float]:
    mean = _pacing_state_mean(state)
    mean_distance = abs((mean if mean is not None else target_speed) - target_speed)
    duration_progress = min(state.total_duration_sec / max(target_total_sec, 1e-6), 1.0)
    return (
        duration_progress,
        _pacing_state_variance(state),
        _pacing_state_average_score(state),
        -mean_distance,
        -len(state.items),
    )


def _pacing_state_is_valid(
    state: _PacingSelectionState,
    *,
    target_speed: float,
    target_total_sec: float,
    tolerance: float,
) -> bool:
    mean = _pacing_state_mean(state)
    return (
        state.total_duration_sec + 1e-6 >= target_total_sec
        and mean is not None
        and abs(mean - target_speed) <= tolerance + 1e-9
    )


def _pacing_final_sort_key(
    state: _PacingSelectionState,
    *,
    target_speed: float,
    target_total_sec: float,
    cfg: ProjectConfig,
) -> tuple[float, float, float, float, float]:
    variance_weight = float(getattr(cfg, "gsv_few_shot_pacing_variance_weight", 1.0))
    quality_weight = float(getattr(cfg, "gsv_few_shot_pacing_quality_weight", 0.25))
    mean = _pacing_state_mean(state) or target_speed
    variance = _pacing_state_variance(state)
    average_score = _pacing_state_average_score(state)
    objective = variance * variance_weight + average_score * quality_weight
    return (
        objective,
        variance,
        average_score,
        -abs(mean - target_speed),
        -abs(state.total_duration_sec - target_total_sec),
    )


def _pacing_state_mean(state: _PacingSelectionState) -> float | None:
    if state.total_duration_sec <= 0:
        return None
    return state.speed_weighted_sum / state.total_duration_sec


def _pacing_state_variance(state: _PacingSelectionState) -> float:
    mean = _pacing_state_mean(state)
    if mean is None:
        return 0.0
    return max(0.0, state.speed_sq_weighted_sum / state.total_duration_sec - mean * mean)


def _pacing_state_average_score(state: _PacingSelectionState) -> float:
    if state.total_duration_sec <= 0:
        return 0.0
    return state.score_weighted_sum / state.total_duration_sec


def _few_shot_pronunciation_per_sec(item: FewShotTrainingItem) -> float | None:
    if item.source_sec_per_pronunciation is None or item.source_sec_per_pronunciation <= 0:
        return None
    return 1.0 / item.source_sec_per_pronunciation


def _few_shot_text_reject_reasons(source_chars_per_sec: float, cfg: ProjectConfig) -> tuple[str, ...]:
    max_chars_per_sec = getattr(cfg, "gsv_few_shot_max_chars_per_sec", None)
    if max_chars_per_sec is None or max_chars_per_sec <= 0:
        return ()
    if source_chars_per_sec <= max_chars_per_sec:
        return ()
    return (
        "source_chars_per_sec_above_max:"
        f"{source_chars_per_sec:.3f}>{float(max_chars_per_sec):.3f}",
    )


_SOURCE_COUNTER_PREFIX_RE = re.compile(r"^\s*(?:[0-9０-９]+|[一二三四五六七八九十]+)[\s、,.．)]")
_SOURCE_FRAGMENT_PREFIX_RE = re.compile(r"^\s*[。,.、]")
_SOURCE_EFFECT_LIKE_RE = re.compile(
    r"(ぎゅ|ギュ|じゅ|ジュ|ぷしゃ|プシャ|ピク|ぴく|ビンビン|びんびん|ざわ|ザワ|ぐちゅ|グチュ|びく|ビク)"
)
_SOURCE_INTENSE_STYLE_RE = re.compile(
    r"(触手|貫通|絶頂|変態|潮|母乳|子宮|乳首|おまんこ|おちんちん|おっぱい|チンポ|"
    r"息汁|愛液|快楽|喘ぎ|硬直|突き上げ|のけぞ|ぐちゃ|限界|泣き叫|股間|粘液|ぬめ|"
    r"下乳|馬乗り|極太|攻め|欲しい)"
)


def _training_selection_score(
    quality_score: float | None,
    source_chars_per_sec: float,
    source_text: str,
    cfg: ProjectConfig,
    *,
    target_pacing_score: float | None = None,
) -> tuple[float | None, tuple[str, ...]]:
    if quality_score is None:
        return None, ()
    score = quality_score
    penalties: list[str] = []
    preferred = getattr(cfg, "gsv_few_shot_preferred_chars_per_sec", None)
    maximum = getattr(cfg, "gsv_few_shot_max_chars_per_sec", None)
    if preferred is not None and preferred > 0 and source_chars_per_sec > preferred:
        if maximum is not None and maximum > preferred:
            span = maximum - preferred
            ratio = min(max((source_chars_per_sec - preferred) / span, 0.0), 1.0)
        else:
            ratio = min(max((source_chars_per_sec - preferred) / preferred, 0.0), 1.0)
        penalty = 0.35 * ratio
        score -= penalty
        penalties.append(f"source_chars_per_sec_penalty:{penalty:.3f}")
    if getattr(cfg, "gsv_few_shot_prefer_plain_text", True):
        style_penalty, style_penalties = _source_text_style_penalty(source_text)
        if style_penalty > 0:
            score -= style_penalty
            penalties.extend(style_penalties)
    if target_pacing_score is not None:
        weight = FEW_SHOT_TARGET_PACING_WEIGHT
        score = score * (1.0 - weight) + target_pacing_score * weight
    return max(0.0, min(1.0, score)), tuple(penalties)


def _source_text_style_penalty(source_text: str) -> tuple[float, tuple[str, ...]]:
    text = source_text.strip()
    if not text:
        return 0.0, ()
    penalty = 0.0
    penalties: list[str] = []
    if _SOURCE_COUNTER_PREFIX_RE.search(text) or _SOURCE_FRAGMENT_PREFIX_RE.search(text):
        penalty += 0.18
        penalties.append("source_text_counter_or_fragment_penalty:0.180")
    if _SOURCE_EFFECT_LIKE_RE.search(text):
        penalty += 0.22
        penalties.append("source_text_effect_like_penalty:0.220")
    if _SOURCE_INTENSE_STYLE_RE.search(text):
        penalty += 0.18
        penalties.append("source_text_intense_style_penalty:0.180")
    if text.count(" ") >= 2 and len(text.replace(" ", "")) / max(text.count(" ") + 1, 1) < 9:
        penalty += 0.10
        penalties.append("source_text_fragmented_phrase_penalty:0.100")
    return min(penalty, 0.65), tuple(penalties)


def _ensure_single_few_shot_speaker(items: Sequence[FewShotTrainingItem]) -> None:
    by_speaker: dict[str, list[str]] = {}
    for item in items:
        by_speaker.setdefault(item.speaker_id, []).append(item.segment_id)
    if len(by_speaker) <= 1:
        return
    details = ", ".join(
        f"{speaker_id}({','.join(segment_ids[:5])})"
        for speaker_id, segment_ids in sorted(by_speaker.items())
    )
    raise GPTSoVITSError(
        "speaker_id_mismatch: GPT-SoVITS few-shot training requires one speaker_id, "
        f"but selected segments include multiple speakers: {details}"
    )


def _few_shot_diagnostic_row(
    segment: Segment,
    check: VoiceTrainingCandidateCheck,
    cfg: ProjectConfig,
    *,
    selected: bool,
    reject_reasons: Sequence[str],
    source_chars_per_sec: float | None = None,
    source_pronunciation_count: int | None = None,
    source_sec_per_pronunciation: float | None = None,
    timing_bucket: str | None = None,
    target_sec_per_syllable: float | None = None,
    target_pacing_ratio: float | None = None,
    target_pacing_score: float | None = None,
    training_selection_score: float | None = None,
    selection_penalties: Sequence[str] = (),
) -> dict[str, Any]:
    metrics = check.metrics
    return {
        "segment_id": segment.id,
        "source_audio_path": str(check.source_audio_path) if check.source_audio_path else None,
        "training_filename": f"{segment.id}.wav",
        "language": _canonical_language(segment.source_script.language if segment.source_script else cfg.asr_language),
        "duration_sec": round(segment.duration, 6),
        "quality_score": round(metrics.score, 6) if metrics else None,
        "quality_issues": list(metrics.issues) if metrics else [],
        "source_chars_per_sec": round(source_chars_per_sec, 6) if source_chars_per_sec is not None else None,
        "source_pronunciation_count": source_pronunciation_count,
        "source_sec_per_pronunciation": (
            round(source_sec_per_pronunciation, 6)
            if source_sec_per_pronunciation is not None
            else None
        ),
        "timing_bucket": timing_bucket or "unknown",
        "target_sec_per_syllable": (
            round(target_sec_per_syllable, 6) if target_sec_per_syllable is not None else None
        ),
        "target_pacing_ratio": (
            round(target_pacing_ratio, 6) if target_pacing_ratio is not None else None
        ),
        "target_pacing_score": (
            round(target_pacing_score, 6) if target_pacing_score is not None else None
        ),
        "training_selection_score": (
            round(training_selection_score, 6) if training_selection_score is not None else None
        ),
        "selection_penalties": list(selection_penalties),
        "clean_source_metrics": check.clean_source_metrics,
        "source_language": cfg.source_language,
        "target_language": cfg.target_language,
        "cross_lingual_role": "ja_source_voice_for_ko_sovits_transfer",
        "speaker_id": segment.speaker_id,
        "analysis_speaker_count": segment.analysis.get("speaker_count"),
        "selected_for_training": selected,
        "reject_reasons": list(dict.fromkeys(reject_reasons)),
    }


def _canonical_language(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"ja", "jp", "jpn", "japanese"}:
        return "ja"
    if normalized in {"ko", "kr", "kor", "korean"}:
        return "ko"
    return normalized


def build_training_dataset(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    *,
    speaker_id: str | None = None,
    work_dir: Path | None = None,
) -> FewShotDataset:
    base_dir = _few_shot_work_dir(project_dir, work_dir)
    wav_dir = ensure_inside_project(project_dir, base_dir / "wavs")
    list_path = ensure_inside_project(project_dir, base_dir / "dataset.list")
    wav_dir.mkdir(parents=True, exist_ok=True)
    selection = _select_training_items_with_diagnostics(
        project_dir,
        manifest,
        cfg,
        speaker_id=speaker_id,
    )
    items = selection.items
    lines: list[str] = []
    total = 0.0
    for item in items:
        target = wav_dir / item.training_filename
        ensure_not_same_path(item.source_audio_path, target)
        shutil.copy2(item.source_audio_path, target)
        lines.append(f"{item.training_filename}|source_voice|{item.language}|{item.text}")
        total += item.duration_sec
    selected_filenames = {item.training_filename for item in items}
    for stale_wav in wav_dir.glob("*.wav"):
        if stale_wav.name not in selected_filenames:
            stale_wav.unlink()
    tmp = list_path.with_suffix(list_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", "utf-8")
    os.replace(tmp, list_path)
    write_json_atomic(
        ensure_inside_project(project_dir, base_dir / "source_clip_qc.json"),
        {"clips": selection.diagnostics},
    )
    return FewShotDataset(items=items, wav_dir=wav_dir, list_path=list_path, total_duration_sec=total)


def _few_shot_work_dir(project_dir: Path, work_dir: Path | None = None) -> Path:
    if work_dir is None:
        return _project_path(project_dir, "work", "gpt_sovits", "few_shot")
    path = Path(work_dir).expanduser()
    resolved = path.resolve() if path.is_absolute() else (project_dir / path).resolve()
    return ensure_inside_project(project_dir, resolved)


def _fingerprint_payload(
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    dataset: FewShotDataset,
    install: GPTSoVITSInstall,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_sha256": manifest.source_info.raw.get("source_sha256") if manifest.source_info else None,
        "rights_source_sha256": manifest.rights_audit.source_sha256,
        "segments": [
            {
                "id": item.segment_id,
                "speaker_id": item.speaker_id,
                "source_audio_path": str(item.source_audio_path),
                "source_audio_sha256": sha256_file(item.source_audio_path),
                "text": item.text,
                "language": item.language,
                "duration_sec": round(item.duration_sec, 6),
                "quality_score": round(item.quality_score, 6),
                "quality_issues": list(item.quality_issues),
                "source_chars_per_sec": round(item.source_chars_per_sec, 6),
                "source_pronunciation_count": item.source_pronunciation_count,
                "source_sec_per_pronunciation": (
                    round(item.source_sec_per_pronunciation, 6)
                    if item.source_sec_per_pronunciation is not None
                    else None
                ),
                "timing_bucket": item.timing_bucket,
                "target_sec_per_syllable": (
                    round(item.target_sec_per_syllable, 6)
                    if item.target_sec_per_syllable is not None
                    else None
                ),
                "target_pacing_ratio": (
                    round(item.target_pacing_ratio, 6)
                    if item.target_pacing_ratio is not None
                    else None
                ),
                "target_pacing_score": (
                    round(item.target_pacing_score, 6)
                    if item.target_pacing_score is not None
                    else None
                ),
                "training_selection_score": round(item.training_selection_score, 6),
                "selection_penalties": list(item.selection_penalties),
            }
            for item in dataset.items
        ],
        "source_language": cfg.source_language,
        "target_language": cfg.target_language,
        "cross_lingual_voice_transfer": cfg.source_language != cfg.target_language,
        "gpt_sovits": {
            "root": str(install.root),
            "checkout": install.checkout,
            "tts_config_path": str(install.tts_config_path),
            "tts_config_sha256": sha256_file(install.tts_config_path),
            "version": install.version,
            "pretrained_gpt_path": str(install.pretrained_gpt_path),
            "pretrained_gpt_sha256": sha256_file(install.pretrained_gpt_path),
            "pretrained_sovits_path": str(install.pretrained_sovits_path),
            "pretrained_sovits_sha256": sha256_file(install.pretrained_sovits_path),
        },
        "training_config": training_config,
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _latest_weight(root: Path, suffix: str) -> Path | None:
    candidates = [path for path in root.rglob(f"*{suffix}") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, str(path)))


def _load_reusable_result(
    metadata_path: Path,
    fingerprint: str,
    dataset: FewShotDataset,
    install: GPTSoVITSInstall,
    log_path: Path,
) -> FewShotTrainingResult | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return None
    if metadata.get("fingerprint") != fingerprint:
        return None
    gpt_path = Path(str(metadata.get("gpt_weights_path") or ""))
    sovits_path = Path(str(metadata.get("sovits_weights_path") or ""))
    if not gpt_path.exists() or not sovits_path.exists():
        return None
    return FewShotTrainingResult(
        status="skipped",
        fingerprint=fingerprint,
        dataset=dataset,
        install=install,
        metadata_path=metadata_path,
        gpt_weights_path=gpt_path,
        sovits_weights_path=sovits_path,
        gpt_weights_sha256=sha256_file(gpt_path),
        sovits_weights_sha256=sha256_file(sovits_path),
        reused_existing=True,
        log_path=log_path,
    )


def _metadata_fingerprint(metadata_path: Path) -> str | None:
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
    except json.JSONDecodeError:
        return None
    value = metadata.get("fingerprint")
    return str(value) if value else None


def _clean_training_outputs(
    base_dir: Path,
    install: GPTSoVITSInstall,
    weights_gpt_dir: Path,
    weights_sovits_dir: Path,
) -> None:
    for path in (
        base_dir / "dataset",
        base_dir / "logs" / f"s1_{install.version}",
        base_dir / "logs" / f"s2_{install.version}",
        weights_gpt_dir,
        weights_sovits_dir,
    ):
        shutil.rmtree(path, ignore_errors=True)
    train_log = base_dir / "logs" / "train.log"
    train_log.unlink(missing_ok=True)
    for path in (weights_gpt_dir, weights_sovits_dir, base_dir / "logs"):
        path.mkdir(parents=True, exist_ok=True)
    (base_dir / "dataset" / f"logs_s2_{install.version}").mkdir(parents=True, exist_ok=True)


def _default_batch_size(version: str) -> int:
    # Conservative defaults for a single local GPU/CPU. Users can still edit generated configs.
    if version in {"v3", "v4"}:
        return 2
    return 4


def _training_configs(
    project_dir: Path,
    cfg: ProjectConfig,
    dataset: FewShotDataset,
    install: GPTSoVITSInstall,
    work_dir: Path | None = None,
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    base_dir = _few_shot_work_dir(project_dir, work_dir)
    configs_dir = ensure_inside_project(project_dir, base_dir / "configs")
    weights_gpt_dir = ensure_inside_project(project_dir, base_dir / "weights" / "gpt")
    weights_sovits_dir = ensure_inside_project(project_dir, base_dir / "weights" / "sovits")
    logs_dir = ensure_inside_project(project_dir, base_dir / "logs")
    for path in (configs_dir, weights_gpt_dir, weights_sovits_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    exp_name = f"{project_dir.name}_few_shot"
    batch_size = _default_batch_size(install.version)

    s1_config = _read_yaml(install.s1_config_path)
    train_s1 = dict(s1_config.get("train") or {})
    train_s1.update(
        {
            "batch_size": batch_size,
            "epochs": 15,
            "save_every_n_epoch": 1,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "half_weights_save_dir": str(weights_gpt_dir),
            "exp_name": exp_name,
        }
    )
    s1_config["train"] = train_s1
    s1_config["pretrained_s1"] = str(install.pretrained_gpt_path)
    s1_config["train_semantic_path"] = str(base_dir / "dataset" / "6-name2semantic.tsv")
    s1_config["train_phoneme_path"] = str(base_dir / "dataset" / "2-name2text.txt")
    s1_config["output_dir"] = str(logs_dir / f"s1_{install.version}")
    s1_config_path = configs_dir / "tmp_s1.yaml"
    s1_config_path.write_text(yaml.safe_dump(s1_config, allow_unicode=True, sort_keys=True), "utf-8")

    s2_config = _read_json(install.s2_config_path)
    train_s2 = dict(s2_config.get("train") or {})
    train_s2.update(
        {
            "batch_size": batch_size,
            "epochs": 2 if install.version in {"v3", "v4"} else 8,
            "if_save_every_weights": True,
            "if_save_latest": True,
            "save_every_epoch": 1 if install.version in {"v3", "v4"} else 4,
            "pretrained_s2G": str(install.pretrained_sovits_path),
            "pretrained_s2D": str(install.pretrained_sovits_path).replace("s2G", "s2D"),
            "text_low_lr_rate": 0.4,
            "grad_ckpt": False,
            "lora_rank": 32,
            "gpu_numbers": (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").replace(",", "-"),
        }
    )
    s2_config["train"] = train_s2
    s2_config.setdefault("model", {})["version"] = install.version
    s2_config.setdefault("data", {})["exp_dir"] = str(base_dir / "dataset")
    s2_config["s2_ckpt_dir"] = str(logs_dir / f"s2_{install.version}")
    s2_config["save_weight_dir"] = str(weights_sovits_dir)
    s2_config["name"] = exp_name
    s2_config["version"] = install.version
    (base_dir / "dataset" / f"logs_s2_{install.version}").mkdir(parents=True, exist_ok=True)
    s2_config_path = configs_dir / "tmp_s2.json"
    s2_config_path.write_text(json.dumps(s2_config, ensure_ascii=False, indent=2, sort_keys=True), "utf-8")

    fingerprint_config = {
        "dataset_list": str(dataset.list_path),
        "dataset_wav_dir": str(dataset.wav_dir),
        "version": install.version,
        "s1_config": s1_config,
        "s2_config": s2_config,
    }
    return fingerprint_config, s1_config_path, s2_config_path, weights_gpt_dir, weights_sovits_dir


def _run_logged(
    runner: CommandRunner,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    phase: str,
    index: int,
    total: int,
    progress_callback: FewShotProgressCallback | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_callback:
        progress_callback(
            FewShotTrainingProgress(
                phase=phase,
                status="started",
                index=index,
                total=total,
                detail=" ".join(shlex.quote(part) for part in command),
                log_path=log_path,
            )
        )
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(shlex.quote(part) for part in command) + "\n")
        try:
            if runner is subprocess.run:
                _run_logged_streaming(
                    command,
                    cwd=cwd,
                    env=env,
                    log=log,
                    log_path=log_path,
                    phase=phase,
                    index=index,
                    total=total,
                    progress_callback=progress_callback,
                )
            else:
                runner(
                    command,
                    cwd=str(cwd),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
        except subprocess.CalledProcessError as exc:
            raise GPTSoVITSError(f"GPT-SoVITS few-shot command failed: {' '.join(command)}") from exc
    if progress_callback:
        progress_callback(
            FewShotTrainingProgress(
                phase=phase,
                status="completed",
                index=index,
                total=total,
                log_path=log_path,
            )
        )


def _run_logged_streaming(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: Any,
    log_path: Path,
    phase: str,
    index: int,
    total: int,
    progress_callback: FewShotProgressCallback | None,
) -> None:
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        raise GPTSoVITSError(f"GPT-SoVITS few-shot command failed to start: {' '.join(command)}") from exc
    last_progress_at = 0.0
    last_detail = ""
    pending_output: list[str] = []

    def maybe_emit_progress(raw_line: str) -> None:
        nonlocal last_detail, last_progress_at
        detail = _progress_detail(raw_line)
        if not detail or not progress_callback:
            return
        now = monotonic()
        if detail != last_detail and (
            _is_important_progress_line(detail)
            or last_progress_at == 0.0
            or now - last_progress_at >= FEW_SHOT_PROGRESS_LOG_SECONDS
        ):
            progress_callback(
                FewShotTrainingProgress(
                    phase=phase,
                    status="output",
                    index=index,
                    total=total,
                    detail=detail,
                    log_path=log_path,
                )
            )
            last_progress_at = now
            last_detail = detail

    if process.stdout is not None:
        while True:
            char = process.stdout.read(1)
            if char == "":
                break
            log.write(char)
            if char in {"\n", "\r"}:
                log.flush()
                maybe_emit_progress("".join(pending_output))
                pending_output = []
            else:
                pending_output.append(char)
        log.flush()
        if pending_output:
            maybe_emit_progress("".join(pending_output))
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _progress_detail(raw_line: str) -> str:
    parts = raw_line.replace("\r", "\n").splitlines()
    return (parts[-1] if parts else raw_line).strip()


def _is_important_progress_line(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in TRAINING_IMPORTANT_MARKERS)


def _merge_part_files(dataset_dir: Path, prefix: str, suffix: str, header: str | None = None) -> None:
    parts = sorted(dataset_dir.glob(f"{prefix}-*{suffix}"))
    if not parts:
        return
    lines: list[str] = [header] if header else []
    for part in parts:
        lines.extend(
            line
            for line in part.read_text("utf-8").splitlines()
            if line.strip() and line.strip() != header
        )
    (dataset_dir / f"{prefix}{suffix}").write_text("\n".join(lines) + ("\n" if lines else ""), "utf-8")


def _merge_dataset_part_outputs(dataset_dir: Path) -> None:
    _merge_part_files(dataset_dir, "2-name2text", ".txt")
    _merge_part_files(dataset_dir, "6-name2semantic", ".tsv", header="item_name\tsemantic_audio")


def _base_env(
    cfg: ProjectConfig,
    dataset: FewShotDataset,
    install: GPTSoVITSInstall,
    *,
    s2_config_path: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    gpu = env.get("CUDA_VISIBLE_DEVICES") or "0"
    dataset_dir = dataset.list_path.parent / "dataset"
    python_paths = [str(install.root), str(install.root / "GPT_SoVITS")]
    if SHIM_DIR.exists():
        python_paths.append(str(SHIM_DIR))
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    cuda_library_dirs = _candidate_cuda_library_dirs()
    library_paths = [str(path) for path in cuda_library_dirs]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    env.update(
        {
            "inp_text": str(dataset.list_path),
            "inp_wav_dir": str(dataset.wav_dir),
            "exp_name": "asmr_few_shot",
            "opt_dir": str(dataset_dir),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": gpu.split(",")[0],
            "is_half": "True",
            "version": install.version,
            "bert_pretrained_dir": str(install.bert_base_path),
            "cnhubert_base_dir": str(install.cnhubert_base_path),
            "pretrained_s2G": str(install.pretrained_sovits_path),
            "s2config_path": str(s2_config_path),
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(python_paths),
        }
    )
    if library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe_text(library_paths))
    if install.sv_pretrained_path is not None:
        env["sv_path"] = str(install.sv_pretrained_path)
    return env


def _candidate_cuda_library_dirs() -> list[Path]:
    roots: list[Path] = []
    for raw in (os.environ.get("VIRTUAL_ENV"), sys.prefix, sys.base_prefix):
        if raw:
            roots.append(Path(raw).expanduser())
    executable = Path(sys.executable).expanduser()
    if executable.name:
        roots.append(executable.parent.parent)
    dirs: list[Path] = []
    for root in roots:
        for pattern in (
            "lib/python*/site-packages/nvidia/cu*/lib",
            "lib/python*/site-packages/nvidia/cuda_nvrtc/lib",
        ):
            dirs.extend(path for path in root.glob(pattern) if _has_nvrtc_runtime(path))
    for path in (
        Path("/usr/local/cuda-13.2/targets/x86_64-linux/lib"),
        Path("/usr/local/lib/ollama/mlx_cuda_v13"),
    ):
        if _has_nvrtc_runtime(path):
            dirs.append(path)
    return _dedupe_paths(dirs)


def _has_nvrtc_runtime(path: Path) -> bool:
    return path.exists() and (
        any(path.glob("libnvrtc-builtins.so.13*")) or any(path.glob("libnvrtc.so.13*"))
    )


def _dedupe_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path.resolve())
    return result


def train_few_shot(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    *,
    speaker_id: str | None = None,
    work_dir: Path | None = None,
    force: bool | None = None,
    command: Sequence[str] | str | None = None,
    runner: CommandRunner = subprocess.run,
    progress_callback: FewShotProgressCallback | None = None,
) -> FewShotTrainingResult:
    project_dir = project_dir.expanduser().resolve()
    base_dir = _few_shot_work_dir(project_dir, work_dir)
    dataset = build_training_dataset(
        project_dir,
        manifest,
        cfg,
        speaker_id=speaker_id,
        work_dir=base_dir,
    )
    install = discover_install(cfg, command=command)
    fingerprint_config, s1_config_path, s2_config_path, weights_gpt_dir, weights_sovits_dir = _training_configs(
        project_dir,
        cfg,
        dataset,
        install,
        base_dir,
    )
    payload = _fingerprint_payload(manifest, cfg, dataset, install, fingerprint_config)
    fingerprint = _fingerprint(payload)
    metadata_path = base_dir / "training_manifest.json"
    log_path = base_dir / "logs" / "train.log"
    should_force = cfg.gsv_few_shot_force if force is None else force
    if not should_force:
        reusable = _load_reusable_result(metadata_path, fingerprint, dataset, install, log_path)
        if reusable is not None:
            if progress_callback:
                progress_callback(
                    FewShotTrainingProgress(
                        phase="reuse",
                        status="skipped",
                        index=0,
                        total=0,
                        detail="matching few-shot weights already exist",
                        log_path=log_path,
                    )
                )
            return reusable
    if should_force or _metadata_fingerprint(metadata_path) != fingerprint:
        _clean_training_outputs(base_dir, install, weights_gpt_dir, weights_sovits_dir)

    env = _base_env(cfg, dataset, install, s2_config_path=s2_config_path)
    semantic_s2_config_path = s2_config_path.with_name("tmp_s2_semantic.json")
    semantic_s2_config = json.loads(s2_config_path.read_text("utf-8"))
    if isinstance(semantic_s2_config.get("model"), dict):
        semantic_s2_config["model"].pop("version", None)
    semantic_s2_config_path.write_text(
        json.dumps(semantic_s2_config, ensure_ascii=False, indent=2, sort_keys=True),
        "utf-8",
    )
    semantic_env = dict(env)
    semantic_env["s2config_path"] = str(semantic_s2_config_path)
    py = _select_training_python(
        cfg,
        install,
        command,
        require_modules=runner is subprocess.run,
    )
    prep_commands: list[tuple[str, list[str], dict[str, str]]] = [
        ("prepare-text", [py, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], env),
        ("prepare-hubert", [py, "-s", "GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"], env),
    ]
    if install.needs_sv_features:
        prep_commands.append(("prepare-sv", [py, "-s", "GPT_SoVITS/prepare_datasets/2-get-sv.py"], env))
    prep_commands.append(
        (
            "prepare-semantic",
            [py, "-s", "GPT_SoVITS/prepare_datasets/3-get-semantic.py"],
            semantic_env,
        )
    )
    train_commands: list[tuple[str, list[str], dict[str, str]]] = [
        ("fine-tune-sovits", [py, "-s", install.s2_train_script, "--config", str(s2_config_path)], env),
        ("fine-tune-gpt", [py, "-s", "GPT_SoVITS/s1_train.py", "--config_file", str(s1_config_path)], env),
    ]
    total_commands = len(prep_commands) + len(train_commands)
    if progress_callback:
        progress_callback(
            FewShotTrainingProgress(
                phase="dataset",
                status="completed",
                index=0,
                total=total_commands,
                detail=f"selected={len(dataset.items)} duration={dataset.total_duration_sec:.2f}s",
                log_path=log_path,
            )
        )
    for command_index, (phase, training_command, command_env) in enumerate(prep_commands, start=1):
        _run_logged(
            runner,
            training_command,
            cwd=install.root,
            env=command_env,
            log_path=log_path,
            phase=phase,
            index=command_index,
            total=total_commands,
            progress_callback=progress_callback,
        )
    _merge_dataset_part_outputs(dataset.list_path.parent / "dataset")

    first_train_index = len(prep_commands) + 1
    for command_index, (phase, training_command, command_env) in enumerate(
        train_commands,
        start=first_train_index,
    ):
        _run_logged(
            runner,
            training_command,
            cwd=install.root,
            env=command_env,
            log_path=log_path,
            phase=phase,
            index=command_index,
            total=total_commands,
            progress_callback=progress_callback,
        )

    gpt_weights = _latest_weight(weights_gpt_dir, ".ckpt")
    sovits_weights = _latest_weight(weights_sovits_dir, ".pth")
    if gpt_weights is None or sovits_weights is None:
        raise GPTSoVITSError(
            "GPT-SoVITS few-shot training finished but expected .ckpt/.pth weights were not found."
        )
    result = FewShotTrainingResult(
        status="completed",
        fingerprint=fingerprint,
        dataset=dataset,
        install=install,
        metadata_path=metadata_path,
        gpt_weights_path=gpt_weights.resolve(),
        sovits_weights_path=sovits_weights.resolve(),
        gpt_weights_sha256=sha256_file(gpt_weights),
        sovits_weights_sha256=sha256_file(sovits_weights),
        reused_existing=False,
        log_path=log_path,
    )
    write_json_atomic(metadata_path, result_metadata(result, payload))
    return result


def result_metadata(result: FewShotTrainingResult, fingerprint_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.status,
        "fingerprint": result.fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "selected_duration_sec": result.dataset.total_duration_sec,
        "selected_speaker_id": result.dataset.items[0].speaker_id if result.dataset.items else None,
        "selected_segment_ids": [item.segment_id for item in result.dataset.items],
        "dataset_list_path": str(result.dataset.list_path),
        "dataset_wav_dir": str(result.dataset.wav_dir),
        "gpt_weights_path": str(result.gpt_weights_path),
        "gpt_weights_sha256": result.gpt_weights_sha256,
        "sovits_weights_path": str(result.sovits_weights_path),
        "sovits_weights_sha256": result.sovits_weights_sha256,
        "gpt_sovits_root": str(result.install.root),
        "gpt_sovits_checkout": result.install.checkout,
        "gpt_sovits_version": result.install.version,
        "reused_existing": result.reused_existing,
        "log_path": str(result.log_path),
    }
