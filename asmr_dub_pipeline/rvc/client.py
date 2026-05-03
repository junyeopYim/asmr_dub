from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from threading import Thread
from time import monotonic
from typing import Any

from rich.markup import escape

from asmr_dub_pipeline.logging import console
from asmr_dub_pipeline.process import (
    popen_process_group_kwargs,
    tail_text,
    terminate_process_group,
)
from asmr_dub_pipeline.schemas import ProjectConfig, RVCProfile, Segment

SUPPORTED_PLACEHOLDERS = {
    "input",
    "output",
    "jobs",
    "results",
    "model",
    "index",
    "device",
    "f0_up_key",
    "f0_method",
    "index_rate",
    "filter_radius",
    "resample_sr",
    "rms_mix_rate",
    "protect",
    "sid",
    "profile_name",
    "segment_id",
    "project",
    "dataset",
    "work_dir",
    "experiment_name",
    "output_model",
    "output_index",
    "sample_rate",
    "batch_size",
    "preprocess_processes",
    "f0_workers",
    "feature_workers",
    "save_every_epoch",
    "reuse_intermediate_cache",
}


class RVCCommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class RVCCommandResult:
    output_path: Path
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    elapsed_sec: float
    reused_existing: bool = False


@dataclass(frozen=True)
class RVCBatchJob:
    segment_id: str
    input_path: Path
    output_path: Path
    model_path: Path
    index_path: Path | None
    profile: RVCProfile
    sid: str = ""


@dataclass(frozen=True)
class RVCTrainResult:
    model_path: Path
    index_path: Path | None
    command: list[str]
    stdout: str
    stderr: str
    returncode: int
    elapsed_sec: float
    reused_existing: bool = False


def _tail(text: str | bytes | None, limit: int = 1200) -> str:
    return tail_text(text, limit=limit)


def _popen_process_group_kwargs() -> dict[str, bool]:
    return popen_process_group_kwargs()


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    terminate_process_group(process, terminate_timeout_sec=5)


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path | None,
    timeout_sec: float,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **_popen_process_group_kwargs(),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        stdout = _tail(exc.stdout, limit=10_000)
        stderr = _tail(exc.stderr, limit=10_000)
        try:
            collected_stdout, collected_stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        else:
            stdout = collected_stdout or stdout
            stderr = collected_stderr or stderr
        raise subprocess.TimeoutExpired(command, timeout_sec, output=stdout, stderr=stderr) from exc
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout, stderr=stderr)


def _stream_subprocess(
    command: list[str],
    *,
    cwd: Path | None,
    timeout_sec: float,
    log_prefix: str,
) -> subprocess.CompletedProcess[str]:
    output_parts: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            **_popen_process_group_kwargs(),
        )
    except FileNotFoundError:
        raise

    def read_output() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            output_parts.append(line)
            stripped = line.rstrip()
            if stripped:
                console.print(f"[dim]{escape(log_prefix)}: {escape(stripped)}[/dim]")

    reader = Thread(target=read_output, daemon=True)
    reader.start()
    try:
        returncode = process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        reader.join(timeout=5)
        raise subprocess.TimeoutExpired(command, timeout_sec, output="".join(output_parts), stderr="") from exc
    reader.join()
    return subprocess.CompletedProcess(command, returncode, stdout="".join(output_parts), stderr="")


def render_rvc_command(template: list[str], values: dict[str, object]) -> list[str]:
    if not template:
        raise RVCCommandError("rvc_command must be a non-empty list of argv strings.")
    formatter = Formatter()
    rendered: list[str] = []
    for raw_arg in template:
        if not isinstance(raw_arg, str):
            raise RVCCommandError("rvc_command entries must be strings.")
        try:
            parsed = list(formatter.parse(raw_arg))
        except ValueError as exc:
            raise RVCCommandError(f"Invalid RVC command placeholder syntax in {raw_arg!r}: {exc}") from exc
        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if field_name not in SUPPORTED_PLACEHOLDERS:
                raise RVCCommandError(f"Unsupported RVC command placeholder {{{field_name}}}.")
            if "." in field_name or "[" in field_name or "]" in field_name:
                raise RVCCommandError(f"Unsafe RVC command placeholder {{{field_name}}}.")
            if format_spec or conversion:
                raise RVCCommandError(f"RVC command placeholder {{{field_name}}} must not use format specs.")
        safe_values = {key: "" if value is None else str(value) for key, value in values.items()}
        try:
            rendered.append(raw_arg.format_map(safe_values))
        except KeyError as exc:
            raise RVCCommandError(f"Missing RVC command placeholder value: {exc.args[0]}") from exc
        except ValueError as exc:
            raise RVCCommandError(f"Invalid RVC command placeholder syntax in {raw_arg!r}: {exc}") from exc
    return rendered


def resolve_config_path(project_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_path = (project_dir / path).resolve()
    if project_path.exists():
        return project_path
    repo_path = (Path(__file__).resolve().parents[2] / path).resolve()
    if repo_path.exists():
        return repo_path
    return project_path


def _profile_values(profile: RVCProfile, cfg: ProjectConfig) -> dict[str, object]:
    return {
        "device": cfg.rvc_device,
        "f0_up_key": profile.f0_up_key,
        "f0_method": profile.f0_method,
        "index_rate": profile.index_rate,
        "filter_radius": profile.filter_radius,
        "resample_sr": profile.resample_sr,
        "rms_mix_rate": profile.rms_mix_rate,
        "protect": profile.protect,
        "profile_name": profile.name,
    }


def _rvc_sample_rate_arg(sample_rate: int) -> str:
    known_rates = {32_000: "32k", 40_000: "40k", 48_000: "48k"}
    return known_rates.get(sample_rate, str(sample_rate))


def _cuda_device_index(device: str) -> int:
    if not device.lower().startswith("cuda"):
        return 0
    parts = device.split(":", 1)
    if len(parts) == 1:
        return 0
    try:
        return max(0, int(parts[1]))
    except ValueError:
        return 0


def _gpu_total_memory_mib(device: str) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    values = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    index = _cuda_device_index(device)
    if index >= len(values):
        index = 0
    try:
        return int(float(values[index]))
    except (ValueError, IndexError):
        return None


def _auto_rvc_train_batch_size(device: str) -> int:
    total_mib = _gpu_total_memory_mib(device)
    if total_mib is None:
        return 4
    if total_mib >= 24_000:
        return 32
    if total_mib >= 16_000:
        return 22
    if total_mib >= 12_000:
        return 14
    if total_mib >= 8_000:
        return 8
    return 2


def effective_rvc_train_batch_size(cfg: ProjectConfig) -> int:
    if cfg.rvc_train_batch_size > 0:
        return cfg.rvc_train_batch_size
    return _auto_rvc_train_batch_size(cfg.rvc_device)


def command_values(
    *,
    input_path: Path,
    output_path: Path,
    model_path: Path | None,
    index_path: Path | None,
    cfg: ProjectConfig,
    profile: RVCProfile,
    segment_id: str,
    sid: str = "",
) -> dict[str, object]:
    return {
        "input": input_path,
        "output": output_path,
        "model": model_path,
        "index": index_path or "",
        "sid": sid,
        "segment_id": segment_id,
        **_profile_values(profile, cfg),
    }


def batch_command_values(
    *,
    jobs_path: Path,
    results_path: Path,
    model_path: Path | None,
    index_path: Path | None,
    cfg: ProjectConfig,
    profile: RVCProfile,
) -> dict[str, object]:
    return {
        "jobs": jobs_path,
        "results": results_path,
        "model": model_path,
        "index": index_path or "",
        **_profile_values(profile, cfg),
    }


def validate_rvc_config(
    project_dir: Path,
    cfg: ProjectConfig,
    *,
    real: bool,
    segments: list[Segment] | None = None,
    allow_trained_artifact: bool = False,
) -> None:
    errors: list[str] = []
    if cfg.rvc_required is not True:
        errors.append("rvc_required must be true for this pipeline.")
    if not real:
        if cfg.rvc_auto_profiles and not errors:
            return
        if not cfg.rvc_auto_profiles:
            errors.append("rvc_auto_profiles must contain at least one profile.")
        raise RVCCommandError("; ".join(errors))
    if cfg.rvc_backend != "command":
        errors.append("real RVC requires rvc_backend: command.")
    if not cfg.rvc_command:
        errors.append("real RVC requires a non-empty rvc_command list.")
    working_dir = resolve_config_path(project_dir, cfg.rvc_working_dir)
    if working_dir is not None and not working_dir.exists():
        errors.append(f"rvc_working_dir does not exist: {working_dir}")

    default_model = resolve_config_path(project_dir, cfg.rvc_model_path)
    default_index = resolve_config_path(project_dir, cfg.rvc_index_path)
    speaker_models = cfg.rvc_speaker_models
    if default_model:
        if not default_model.exists():
            errors.append(f"rvc_model_path does not exist: {default_model}")
    elif allow_trained_artifact:
        pass
    elif segments:
        missing = [
            segment.id
            for segment in segments
            if segment.status not in {"needs_manual_review", "failed"}
            and (not segment.speaker_id or segment.speaker_id not in speaker_models)
        ]
        if missing:
            errors.append(
                "rvc_model_path is required unless every segment speaker_id has "
                f"rvc_speaker_models mapping. Missing: {', '.join(missing[:10])}"
            )
    elif not speaker_models:
        errors.append("real RVC requires rvc_model_path or rvc_speaker_models.")
    if default_index and not default_index.exists():
        errors.append(f"rvc_index_path does not exist: {default_index}")
    for speaker_id, speaker_cfg in speaker_models.items():
        speaker_model = resolve_config_path(project_dir, speaker_cfg.model_path)
        speaker_index = resolve_config_path(project_dir, speaker_cfg.index_path)
        if speaker_model is None or not speaker_model.exists():
            errors.append(f"rvc_speaker_models.{speaker_id}.model_path does not exist: {speaker_model}")
        if speaker_index is not None and not speaker_index.exists():
            errors.append(f"rvc_speaker_models.{speaker_id}.index_path does not exist: {speaker_index}")

    if cfg.rvc_command and cfg.rvc_auto_profiles:
        dummy_profile = cfg.rvc_auto_profiles[0]
        try:
            render_rvc_command(
                cfg.rvc_command,
                command_values(
                    input_path=project_dir / "work" / "tts" / "dummy.wav",
                    output_path=project_dir / "work" / "rvc" / "dummy.wav",
                    model_path=default_model or Path("dummy_model.pth"),
                    index_path=default_index,
                    cfg=cfg,
                    profile=dummy_profile,
                    segment_id="seg_0000",
                ),
            )
        except RVCCommandError as exc:
            errors.append(str(exc))
    if cfg.rvc_batch_command and cfg.rvc_auto_profiles:
        dummy_profile = cfg.rvc_auto_profiles[0]
        try:
            render_rvc_command(
                cfg.rvc_batch_command,
                batch_command_values(
                    jobs_path=project_dir / "work" / "rvc" / "batch_jobs.jsonl",
                    results_path=project_dir / "work" / "rvc" / "batch_results.jsonl",
                    model_path=default_model or Path("dummy_model.pth"),
                    index_path=default_index,
                    cfg=cfg,
                    profile=dummy_profile,
                ),
            )
        except RVCCommandError as exc:
            errors.append(str(exc))
    if errors:
        raise RVCCommandError("Invalid RVC configuration: " + "; ".join(errors))


def rvc_train_output_paths(project_dir: Path, cfg: ProjectConfig) -> tuple[Path, Path]:
    model_path = resolve_config_path(project_dir, cfg.rvc_train_output_model_path)
    index_path = resolve_config_path(project_dir, cfg.rvc_train_output_index_path)
    if model_path is None:
        model_path = project_dir / "work" / "rvc_train" / "model" / f"{cfg.rvc_train_experiment_name}.pth"
    if index_path is None:
        index_path = (
            project_dir
            / "work"
            / "rvc_train"
            / "model"
            / f"added_{cfg.rvc_train_experiment_name}.index"
        )
    return model_path, index_path


def render_rvc_train_command(
    template: list[str],
    *,
    project_dir: Path,
    dataset_dir: Path,
    work_dir: Path,
    model_path: Path,
    index_path: Path,
    cfg: ProjectConfig,
) -> list[str]:
    return render_rvc_command(
        template,
        {
            "project": project_dir,
            "dataset": dataset_dir,
            "work_dir": work_dir,
            "experiment_name": cfg.rvc_train_experiment_name,
            "output_model": model_path,
            "output_index": index_path,
            "sample_rate": _rvc_sample_rate_arg(cfg.rvc_train_sample_rate),
            "device": cfg.rvc_device,
            "batch_size": effective_rvc_train_batch_size(cfg),
            "preprocess_processes": cfg.rvc_train_preprocess_processes,
            "f0_workers": cfg.rvc_train_f0_workers,
            "feature_workers": cfg.rvc_train_feature_workers,
            "save_every_epoch": cfg.rvc_train_save_every_epoch,
            "reuse_intermediate_cache": str(cfg.rvc_train_reuse_intermediate_cache).lower(),
        },
    )


def validate_rvc_training_config(project_dir: Path, cfg: ProjectConfig, *, real: bool) -> None:
    errors: list[str] = []
    if cfg.rvc_train_required is not True:
        errors.append("rvc_train_required must be true for real/full RVC.")
    if not real:
        if errors:
            raise RVCCommandError("; ".join(errors))
        return
    if cfg.rvc_train_backend != "command":
        errors.append("real train-rvc requires rvc_train_backend: command.")
    if not cfg.rvc_train_command:
        errors.append("real train-rvc requires a non-empty rvc_train_command list.")
    working_dir = resolve_config_path(project_dir, cfg.rvc_train_working_dir)
    if working_dir is not None and not working_dir.exists():
        errors.append(f"rvc_train_working_dir does not exist: {working_dir}")
    model_path, index_path = rvc_train_output_paths(project_dir, cfg)
    if cfg.rvc_train_command:
        try:
            render_rvc_train_command(
                cfg.rvc_train_command,
                project_dir=project_dir,
                dataset_dir=project_dir / "work" / "rvc_train" / "dataset",
                work_dir=project_dir / "work" / "rvc_train",
                model_path=model_path,
                index_path=index_path,
                cfg=cfg,
            )
        except RVCCommandError as exc:
            errors.append(str(exc))
    if errors:
        raise RVCCommandError("Invalid RVC training configuration: " + "; ".join(errors))


class RVCTrainCommandClient:
    def __init__(
        self,
        command_template: list[str],
        *,
        working_dir: Path | None = None,
        timeout_sec: float = 14400.0,
        runner: Any = subprocess.run,
        stream_output: bool = False,
        log_prefix: str = "train-rvc",
    ) -> None:
        self.command_template = command_template
        self.working_dir = working_dir
        self.timeout_sec = timeout_sec
        self.runner = runner
        self.stream_output = stream_output
        self.log_prefix = log_prefix

    def build_command(
        self,
        *,
        project_dir: Path,
        dataset_dir: Path,
        work_dir: Path,
        model_path: Path,
        index_path: Path,
        cfg: ProjectConfig,
    ) -> list[str]:
        return render_rvc_train_command(
            self.command_template,
            project_dir=project_dir,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            model_path=model_path,
            index_path=index_path,
            cfg=cfg,
        )

    def train(
        self,
        *,
        project_dir: Path,
        dataset_dir: Path,
        work_dir: Path,
        model_path: Path,
        index_path: Path,
        cfg: ProjectConfig,
        force: bool = False,
    ) -> RVCTrainResult:
        command = self.build_command(
            project_dir=project_dir,
            dataset_dir=dataset_dir,
            work_dir=work_dir,
            model_path=model_path,
            index_path=index_path,
            cfg=cfg,
        )
        outputs_exist = model_path.exists() and model_path.stat().st_size > 0
        outputs_exist = outputs_exist and index_path.exists() and index_path.stat().st_size > 0
        if outputs_exist and not force:
            return RVCTrainResult(
                model_path=model_path,
                index_path=index_path,
                command=command,
                stdout="",
                stderr="",
                returncode=0,
                elapsed_sec=0.0,
                reused_existing=True,
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        started = monotonic()
        try:
            if self.stream_output and self.runner is subprocess.run:
                completed = _stream_subprocess(
                    command,
                    cwd=self.working_dir,
                    timeout_sec=self.timeout_sec,
                    log_prefix=self.log_prefix,
                )
            elif self.runner is subprocess.run:
                completed = _run_subprocess(
                    command,
                    cwd=self.working_dir,
                    timeout_sec=self.timeout_sec,
                )
            else:
                completed = self.runner(
                    command,
                    cwd=str(self.working_dir) if self.working_dir else None,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
        except FileNotFoundError as exc:
            raise RVCCommandError(f"RVC training executable was not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RVCCommandError(
                f"RVC training timed out after {self.timeout_sec:g}s. "
                f"stdout={_tail(exc.stdout)} stderr={_tail(exc.stderr)}"
            ) from exc
        elapsed = monotonic() - started
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        returncode = int(getattr(completed, "returncode", 1))
        if returncode != 0:
            raise RVCCommandError(
                f"RVC training failed with exit code {returncode}: "
                f"stdout={_tail(stdout)} stderr={_tail(stderr)}"
            )
        if not model_path.exists() or model_path.stat().st_size <= 0:
            raise RVCCommandError(f"RVC training did not create model output: {model_path}")
        if not index_path.exists() or index_path.stat().st_size <= 0:
            raise RVCCommandError(f"RVC training did not create index output: {index_path}")
        return RVCTrainResult(
            model_path=model_path,
            index_path=index_path,
            command=command,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            elapsed_sec=elapsed,
        )


class RVCTrainMockClient:
    def train(
        self,
        *,
        project_dir: Path,
        dataset_dir: Path,
        work_dir: Path,
        model_path: Path,
        index_path: Path,
        cfg: ProjectConfig,
        force: bool = False,
    ) -> RVCTrainResult:
        _ = project_dir, dataset_dir, work_dir, cfg
        outputs_exist = model_path.exists() and model_path.stat().st_size > 0
        outputs_exist = outputs_exist and index_path.exists() and index_path.stat().st_size > 0
        if outputs_exist and not force:
            return RVCTrainResult(
                model_path=model_path,
                index_path=index_path,
                command=[],
                stdout="",
                stderr="",
                returncode=0,
                elapsed_sec=0.0,
                reused_existing=True,
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(b"mock rvc model\n")
        index_path.write_bytes(b"mock rvc index\n")
        return RVCTrainResult(
            model_path=model_path,
            index_path=index_path,
            command=[],
            stdout="mock train",
            stderr="",
            returncode=0,
            elapsed_sec=0.0,
            reused_existing=False,
        )


class RVCCommandClient:
    def __init__(
        self,
        command_template: list[str],
        *,
        working_dir: Path | None = None,
        timeout_sec: float = 180.0,
        runner: Any = subprocess.run,
        stream_output: bool = False,
        log_prefix: str = "rvc",
    ) -> None:
        self.command_template = command_template
        self.working_dir = working_dir
        self.timeout_sec = timeout_sec
        self.runner = runner
        self.stream_output = stream_output
        self.log_prefix = log_prefix

    def build_command(
        self,
        *,
        input_path: Path,
        output_path: Path,
        model_path: Path | None,
        index_path: Path | None,
        cfg: ProjectConfig,
        profile: RVCProfile,
        segment_id: str,
        sid: str = "",
    ) -> list[str]:
        return render_rvc_command(
            self.command_template,
            command_values(
                input_path=input_path,
                output_path=output_path,
                model_path=model_path,
                index_path=index_path,
                cfg=cfg,
                profile=profile,
                segment_id=segment_id,
                sid=sid,
            ),
        )

    def convert(
        self,
        input_path: Path,
        output_path: Path,
        *,
        model_path: Path | None,
        index_path: Path | None,
        cfg: ProjectConfig,
        profile: RVCProfile,
        segment_id: str,
        sid: str = "",
        force: bool = False,
    ) -> RVCCommandResult:
        command = self.build_command(
            input_path=input_path,
            output_path=output_path,
            model_path=model_path,
            index_path=index_path,
            cfg=cfg,
            profile=profile,
            segment_id=segment_id,
            sid=sid,
        )
        if output_path.exists() and output_path.stat().st_size > 0 and not force:
            return RVCCommandResult(
                output_path=output_path,
                command=command,
                stdout="",
                stderr="",
                returncode=0,
                elapsed_sec=0.0,
                reused_existing=True,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = monotonic()
        try:
            if self.stream_output and self.runner is subprocess.run:
                completed = _stream_subprocess(
                    command,
                    cwd=self.working_dir,
                    timeout_sec=self.timeout_sec,
                    log_prefix=self.log_prefix,
                )
            elif self.runner is subprocess.run:
                completed = _run_subprocess(
                    command,
                    cwd=self.working_dir,
                    timeout_sec=self.timeout_sec,
                )
            else:
                completed = self.runner(
                    command,
                    cwd=str(self.working_dir) if self.working_dir else None,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
        except FileNotFoundError as exc:
            raise RVCCommandError(f"RVC command executable was not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RVCCommandError(
                f"RVC command timed out after {self.timeout_sec:g}s. "
                f"stdout={_tail(exc.stdout)} stderr={_tail(exc.stderr)}"
            ) from exc
        elapsed = monotonic() - started
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        returncode = int(getattr(completed, "returncode", 1))
        result = RVCCommandResult(
            output_path=output_path,
            command=command,
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            elapsed_sec=elapsed,
        )
        if returncode != 0:
            raise RVCCommandError(
                f"RVC command failed with exit code {returncode}: "
                f"stdout={_tail(stdout)} stderr={_tail(stderr)}"
            )
        if not output_path.exists():
            raise RVCCommandError(
                f"RVC command completed but did not create output: {output_path}. "
                f"stdout={_tail(stdout)} stderr={_tail(stderr)}"
            )
        if output_path.stat().st_size <= 0:
            raise RVCCommandError(f"RVC command created an empty output file: {output_path}")
        return result


class RVCBatchCommandClient:
    def __init__(
        self,
        command_template: list[str],
        *,
        working_dir: Path | None = None,
        timeout_sec: float = 180.0,
        runner: Any = subprocess.run,
        stream_output: bool = False,
        log_prefix: str = "rvc-batch",
    ) -> None:
        self.command_template = command_template
        self.working_dir = working_dir
        self.timeout_sec = timeout_sec
        self.runner = runner
        self.stream_output = stream_output
        self.log_prefix = log_prefix

    def build_command(
        self,
        *,
        jobs_path: Path,
        results_path: Path,
        model_path: Path | None,
        index_path: Path | None,
        cfg: ProjectConfig,
        profile: RVCProfile,
    ) -> list[str]:
        return render_rvc_command(
            self.command_template,
            batch_command_values(
                jobs_path=jobs_path,
                results_path=results_path,
                model_path=model_path,
                index_path=index_path,
                cfg=cfg,
                profile=profile,
            ),
        )

    def convert_many(
        self,
        jobs: list[RVCBatchJob],
        *,
        jobs_path: Path,
        results_path: Path,
        model_path: Path,
        index_path: Path | None,
        cfg: ProjectConfig,
        profile: RVCProfile,
        force: bool = False,
    ) -> dict[str, RVCCommandResult]:
        command = self.build_command(
            jobs_path=jobs_path,
            results_path=results_path,
            model_path=model_path,
            index_path=index_path,
            cfg=cfg,
            profile=profile,
        )
        results: dict[str, RVCCommandResult] = {}
        pending: list[RVCBatchJob] = []
        for job in jobs:
            if job.output_path.exists() and job.output_path.stat().st_size > 0 and not force:
                results[job.segment_id] = RVCCommandResult(
                    output_path=job.output_path,
                    command=command,
                    stdout="",
                    stderr="",
                    returncode=0,
                    elapsed_sec=0.0,
                    reused_existing=True,
                )
            else:
                pending.append(job)
        if not pending:
            return results

        jobs_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_path.exists():
            results_path.unlink()
        with jobs_path.open("w", encoding="utf-8") as handle:
            for job in pending:
                handle.write(
                    json.dumps(
                        {
                            "segment_id": job.segment_id,
                            "input_path": str(job.input_path),
                            "output_path": str(job.output_path),
                            "f0_up_key": job.profile.f0_up_key,
                            "f0_method": job.profile.f0_method,
                            "index_rate": job.profile.index_rate,
                            "filter_radius": job.profile.filter_radius,
                            "resample_sr": job.profile.resample_sr,
                            "rms_mix_rate": job.profile.rms_mix_rate,
                            "protect": job.profile.protect,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        timeout_sec = self.timeout_sec * max(1, len(pending))
        started = monotonic()
        try:
            if self.stream_output and self.runner is subprocess.run:
                completed = _stream_subprocess(
                    command,
                    cwd=self.working_dir,
                    timeout_sec=timeout_sec,
                    log_prefix=self.log_prefix,
                )
            elif self.runner is subprocess.run:
                completed = _run_subprocess(
                    command,
                    cwd=self.working_dir,
                    timeout_sec=timeout_sec,
                )
            else:
                completed = self.runner(
                    command,
                    cwd=str(self.working_dir) if self.working_dir else None,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
        except FileNotFoundError as exc:
            raise RVCCommandError(f"RVC batch command executable was not found: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RVCCommandError(
                f"RVC batch command timed out after {timeout_sec:g}s for {len(pending)} job(s). "
                f"stdout={_tail(exc.stdout)} stderr={_tail(exc.stderr)}"
            ) from exc
        batch_elapsed = monotonic() - started
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        returncode = int(getattr(completed, "returncode", 1))

        parsed: dict[str, dict[str, object]] = {}
        if results_path.exists():
            for line in results_path.read_text("utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                segment_id = str(payload.get("segment_id") or "")
                if segment_id:
                    parsed[segment_id] = payload

        for job in pending:
            payload = parsed.get(job.segment_id)
            if payload is None:
                results[job.segment_id] = RVCCommandResult(
                    output_path=job.output_path,
                    command=command,
                    stdout=stdout,
                    stderr=f"RVC batch command did not report a result for {job.segment_id}. {_tail(stderr)}",
                    returncode=returncode or 1,
                    elapsed_sec=batch_elapsed,
                )
                continue
            job_returncode = int(payload.get("returncode") or 0)
            job_stdout = str(payload.get("stdout") or "")
            job_stderr = str(payload.get("stderr") or "")
            job_elapsed = payload.get("elapsed_sec")
            elapsed_sec = float(job_elapsed) if isinstance(job_elapsed, (int, float)) else batch_elapsed
            if job_returncode == 0 and (not job.output_path.exists() or job.output_path.stat().st_size <= 0):
                job_returncode = 1
                job_stderr = (
                    job_stderr
                    or f"RVC batch command completed but did not create output: {job.output_path}"
                )
            results[job.segment_id] = RVCCommandResult(
                output_path=job.output_path,
                command=command,
                stdout=job_stdout,
                stderr=job_stderr,
                returncode=job_returncode,
                elapsed_sec=elapsed_sec,
            )
        return results


class RVCMockClient:
    def convert(
        self,
        input_path: Path,
        output_path: Path,
        *,
        model_path: Path | None = None,
        index_path: Path | None = None,
        cfg: ProjectConfig | None = None,
        profile: RVCProfile | None = None,
        segment_id: str = "",
        sid: str = "",
        force: bool = False,
    ) -> RVCCommandResult:
        _ = model_path, index_path, cfg, profile, segment_id, sid
        if output_path.exists() and output_path.stat().st_size > 0 and not force:
            return RVCCommandResult(
                output_path=output_path,
                command=[],
                stdout="",
                stderr="",
                returncode=0,
                elapsed_sec=0.0,
                reused_existing=True,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        return RVCCommandResult(
            output_path=output_path,
            command=[],
            stdout="mock copy",
            stderr="",
            returncode=0,
            elapsed_sec=0.0,
            reused_existing=False,
        )
