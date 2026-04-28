from __future__ import annotations

import hashlib
import json
import os
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

from asmr_dub_pipeline.audio.quality import measure_source_voice_quality
from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSError
from asmr_dub_pipeline.gpt_sovits.server import SHIM_DIR, _default_gsv_command
from asmr_dub_pipeline.pipeline.manifest_io import write_json_atomic
from asmr_dub_pipeline.rights import (
    RightsError,
    ensure_inside_project,
    ensure_not_same_path,
    sha256_file,
)
from asmr_dub_pipeline.schemas import PipelineManifest, ProjectConfig, Segment

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


@dataclass(frozen=True)
class FewShotTrainingItem:
    segment_id: str
    source_audio_path: Path
    training_filename: str
    text: str
    language: str
    duration_sec: float
    quality_score: float = 0.0
    quality_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class FewShotDataset:
    items: list[FewShotTrainingItem]
    wav_dir: Path
    list_path: Path
    total_duration_sec: float


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


def _resolve_project_read_path(project_dir: Path, raw_path: str, field_name: str) -> Path:
    path = Path(raw_path).expanduser()
    resolved = (project_dir / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(project_dir.resolve())
    except ValueError as exc:
        raise RightsError(f"{field_name} must stay inside the project directory: {resolved}") from exc
    return resolved


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
) -> list[FewShotTrainingItem]:
    candidates: list[FewShotTrainingItem] = []
    source_language = _canonical_language(cfg.source_language)
    for segment in sorted(manifest.segments, key=lambda item: (item.start, item.id)):
        if not _segment_is_training_candidate(segment, cfg):
            continue
        source_audio_path = _resolve_project_read_path(project_dir, segment.audio_for_mix, "audio_for_mix")
        if not source_audio_path.exists():
            continue
        text = segment.source_script.text.strip() if segment.source_script else ""
        language = segment.source_script.language if segment.source_script else cfg.asr_language
        if _canonical_language(language) != source_language:
            continue
        metrics = measure_source_voice_quality(source_audio_path)
        if metrics.score < cfg.gsv_few_shot_min_quality_score:
            continue
        training_filename = f"{segment.id}.wav"
        candidates.append(
            FewShotTrainingItem(
                segment_id=segment.id,
                source_audio_path=source_audio_path,
                training_filename=training_filename,
                text=text,
                language=_canonical_language(language or cfg.asr_language),
                duration_sec=segment.duration,
                quality_score=metrics.score,
                quality_issues=tuple(metrics.issues),
            )
        )
    items: list[FewShotTrainingItem] = []
    total = 0.0
    for item in sorted(candidates, key=lambda candidate: (-candidate.quality_score, candidate.duration_sec)):
        items.append(item)
        total += item.duration_sec
        if total >= cfg.gsv_few_shot_target_sec:
            break
    if total < cfg.gsv_few_shot_target_sec:
        raise GPTSoVITSError(
            "Not enough source voice data for GPT-SoVITS few-shot training: "
            f"selected {total:.2f}s, need {cfg.gsv_few_shot_target_sec:.2f}s."
        )
    return items


def _segment_is_training_candidate(segment: Segment, cfg: ProjectConfig) -> bool:
    return bool(
        segment.source_script
        and segment.source_script.text.strip()
        and segment.audio_for_mix
        and cfg.gsv_few_shot_min_clip_sec <= segment.duration <= cfg.gsv_few_shot_max_clip_sec
    )


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
) -> FewShotDataset:
    wav_dir = _project_path(project_dir, "work", "gpt_sovits", "few_shot", "wavs")
    list_path = _project_path(project_dir, "work", "gpt_sovits", "few_shot", "dataset.list")
    wav_dir.mkdir(parents=True, exist_ok=True)
    items = select_training_items(project_dir, manifest, cfg)
    lines: list[str] = []
    total = 0.0
    qc_rows: list[dict[str, Any]] = []
    for item in items:
        target = wav_dir / item.training_filename
        ensure_not_same_path(item.source_audio_path, target)
        shutil.copy2(item.source_audio_path, target)
        lines.append(f"{item.training_filename}|source_voice|{item.language}|{item.text}")
        total += item.duration_sec
        qc_rows.append(
            {
                "segment_id": item.segment_id,
                "source_audio_path": str(item.source_audio_path),
                "training_filename": item.training_filename,
                "language": item.language,
                "duration_sec": round(item.duration_sec, 6),
                "quality_score": round(item.quality_score, 6),
                "quality_issues": list(item.quality_issues),
                "source_language": cfg.source_language,
                "target_language": cfg.target_language,
                "cross_lingual_role": "ja_source_voice_for_ko_sovits_transfer",
                "selected_for_training": True,
            }
        )
    tmp = list_path.with_suffix(list_path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", "utf-8")
    os.replace(tmp, list_path)
    write_json_atomic(
        _project_path(project_dir, "work", "gpt_sovits", "few_shot", "source_clip_qc.json"),
        {"clips": qc_rows},
    )
    return FewShotDataset(items=items, wav_dir=wav_dir, list_path=list_path, total_duration_sec=total)


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
                "source_audio_path": str(item.source_audio_path),
                "source_audio_sha256": sha256_file(item.source_audio_path),
                "text": item.text,
                "language": item.language,
                "duration_sec": round(item.duration_sec, 6),
                "quality_score": round(item.quality_score, 6),
                "quality_issues": list(item.quality_issues),
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
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    base_dir = _project_path(project_dir, "work", "gpt_sovits", "few_shot")
    configs_dir = _project_path(project_dir, "work", "gpt_sovits", "few_shot", "configs")
    weights_gpt_dir = _project_path(project_dir, "work", "gpt_sovits", "few_shot", "weights", "gpt")
    weights_sovits_dir = _project_path(project_dir, "work", "gpt_sovits", "few_shot", "weights", "sovits")
    logs_dir = _project_path(project_dir, "work", "gpt_sovits", "few_shot", "logs")
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
    if install.sv_pretrained_path is not None:
        env["sv_path"] = str(install.sv_pretrained_path)
    return env


def train_few_shot(
    project_dir: Path,
    manifest: PipelineManifest,
    cfg: ProjectConfig,
    *,
    force: bool | None = None,
    command: Sequence[str] | str | None = None,
    runner: CommandRunner = subprocess.run,
    progress_callback: FewShotProgressCallback | None = None,
) -> FewShotTrainingResult:
    project_dir = project_dir.expanduser().resolve()
    dataset = build_training_dataset(project_dir, manifest, cfg)
    install = discover_install(cfg, command=command)
    fingerprint_config, s1_config_path, s2_config_path, weights_gpt_dir, weights_sovits_dir = _training_configs(
        project_dir,
        cfg,
        dataset,
        install,
    )
    payload = _fingerprint_payload(manifest, cfg, dataset, install, fingerprint_config)
    fingerprint = _fingerprint(payload)
    base_dir = _project_path(project_dir, "work", "gpt_sovits", "few_shot")
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
    py = sys.executable or "python"
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
