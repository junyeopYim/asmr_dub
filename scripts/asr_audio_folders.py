#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.asr import create_asr_backend
from asmr_dub_pipeline.audio.preprocess import plan_folder_input
from asmr_dub_pipeline.config import load_project_config, save_project_config
from asmr_dub_pipeline.pipeline.steps import extract_step, transcribe_step

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class WorkItem:
    source_dir: Path
    project_dir: Path
    reason: str


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return normalized or "work"


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def discover_audio_folders(
    input_root: Path,
    runs_root: Path,
    *,
    batch_id: str,
    only: list[str],
    skip: list[str],
    limit: int | None,
    batch_children: bool,
) -> list[WorkItem]:
    input_root = input_root.expanduser().resolve()
    runs_root = runs_root.expanduser().resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input folder does not exist or is not a directory: {input_root}")

    items: list[WorkItem] = []
    candidate_dirs = (
        sorted(path for path in input_root.iterdir() if path.is_dir())
        if batch_children
        else [input_root]
    )

    for source_dir in candidate_dirs:
        name = source_dir.name
        if only and not _matches_any(name, only):
            continue
        if skip and _matches_any(name, skip):
            continue
        plan = plan_folder_input(source_dir)
        if not plan.should_prepare:
            print(f"[skip] {name}: {plan.reason}", file=sys.stderr)
            continue
        project_dir = runs_root / batch_id / _safe_name(name)
        items.append(WorkItem(source_dir=source_dir, project_dir=project_dir, reason=plan.reason))
        if limit is not None and len(items) >= limit:
            break
    return items


def command_for_extract(item: WorkItem) -> list[str]:
    return [
        "uv",
        "run",
        "asmr-dub",
        "extract",
        str(item.source_dir),
        "--project",
        str(item.project_dir),
        "--confirm-rights",
    ]


def command_for_transcribe(args: argparse.Namespace, item: WorkItem) -> list[str]:
    command = [
        "uv",
        "run",
        "asmr-dub",
        "transcribe",
        "--project",
        str(item.project_dir),
        "--asr-backend",
        args.asr_backend,
        "--asr-preset",
        args.asr_preset,
        "--asr-diagnostics" if args.asr_diagnostics else "--no-asr-diagnostics",
        "--confirm-rights",
    ]
    if args.asr_device:
        command.extend(["--asr-device", args.asr_device])
    if args.asr_compute_type:
        command.extend(["--asr-compute-type", args.asr_compute_type])
    command.append("--asr-batched" if args.asr_batched else "--no-asr-batched")
    command.extend(["--asr-batch-size", str(args.asr_batch_size)])
    command.append("--no-asr-repair" if args.disable_asr_repair else "--asr-repair")
    if args.asr_review:
        command.append("--asr-review")
    return command


class CachedASRBackendFactory:
    """Reuse ASR backend instances across an in-process batch."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Any] = {}

    def __call__(self, kind: str, config: dict[str, Any]) -> Any:
        normalized = kind.replace("-", "_")
        key = (normalized, json.dumps(config, ensure_ascii=False, sort_keys=True, default=str))
        if key not in self._cache:
            self._cache[key] = create_asr_backend(kind, config)
        return self._cache[key]


def _apply_asr_only_resource_config(project_dir: Path, args: argparse.Namespace) -> None:
    cfg = load_project_config(project_dir)
    payload = cfg.model_dump(mode="json")
    asr_payload = dict(payload.get("asr") or {})
    asr_payload["source_separation_backend"] = args.source_separation_backend
    asr_payload["diagnostics_enabled"] = bool(args.asr_diagnostics)
    if args.disable_asr_repair:
        asr_payload["repair_enabled"] = False
    payload["asr"] = asr_payload
    next_cfg = type(cfg).model_validate(payload)
    next_cfg.asr.correction_profile = cfg.asr.correction_profile
    save_project_config(next_cfg, project_dir / "pipeline.yaml")


def _manifest_transcribe_completed(project_dir: Path) -> bool:
    manifest_path = project_dir / "work" / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    stage = manifest.get("stage_state", {}).get("transcribe", {})
    return isinstance(stage, dict) and stage.get("status") == "completed"


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> int:
    printable = " ".join(_quote(part) for part in command)
    print(f"$ {printable}")
    if dry_run:
        return 0
    return subprocess.run(command, cwd=cwd, check=False).returncode


def _run_in_process_extract(item: WorkItem) -> int:
    printable = " ".join(_quote(part) for part in command_for_extract(item))
    print(f"$ {printable}  # in-process")
    try:
        extract_step(item.source_dir, item.project_dir, confirm_rights=True)
    except Exception as exc:
        print(f"extract failed for {item.source_dir}: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_in_process_transcribe(
    args: argparse.Namespace,
    item: WorkItem,
    backend_factory: CachedASRBackendFactory,
) -> int:
    printable = " ".join(_quote(part) for part in command_for_transcribe(args, item))
    print(f"$ {printable}  # in-process")
    try:
        transcribe_step(
            item.project_dir,
            args.asr_backend,
            confirm_rights=True,
            asr_review=True if args.asr_review else None,
            asr_preset=args.asr_preset,
            asr_diagnostics=bool(args.asr_diagnostics),
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            asr_batched_inference=bool(args.asr_batched),
            asr_batch_size=args.asr_batch_size,
            asr_repair_enabled=not bool(args.disable_asr_repair),
            asr_backend_factory=backend_factory,
        )
    except Exception as exc:
        print(f"transcribe failed for {item.project_dir}: {exc}", file=sys.stderr)
        return 1
    return 0


def _quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:=@+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def run_batch(args: argparse.Namespace) -> int:
    if not args.dry_run and not args.confirm_rights:
        raise SystemExit("real ASR batch requires --confirm-rights")
    batch_id = args.batch_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    batch_children = (
        args.batch_children
        if args.batch_children is not None
        else args.input_folder.expanduser().resolve().name == "audio"
    )
    items = discover_audio_folders(
        args.input_folder,
        args.runs_root,
        batch_id=batch_id,
        only=args.only,
        skip=args.skip,
        limit=args.limit,
        batch_children=batch_children,
    )
    if not items:
        print("No audio folders to process.")
        return 0

    summary: list[dict[str, str | int]] = []
    failures = 0
    backend_factory = CachedASRBackendFactory()
    for index, item in enumerate(items, start=1):
        print(f"\n[{index}/{len(items)}] {item.source_dir.name}")
        print(f"project: {item.project_dir}")
        print(f"plan: {item.reason}")
        if not args.force and _manifest_transcribe_completed(item.project_dir):
            print("skip: transcribe already completed")
            summary.append(
                {
                    "source_dir": str(item.source_dir),
                    "project_dir": str(item.project_dir),
                    "status": "skipped_completed",
                    "returncode": 0,
                }
            )
            continue

        status = "completed"
        returncode = 0
        if args.in_process and not args.dry_run:
            returncode = _run_in_process_extract(item)
        else:
            returncode = _run(command_for_extract(item), cwd=REPO_ROOT, dry_run=args.dry_run)
        if returncode == 0 and not args.dry_run:
            _apply_asr_only_resource_config(item.project_dir, args)
        if returncode == 0:
            if args.in_process and not args.dry_run:
                returncode = _run_in_process_transcribe(args, item, backend_factory)
            else:
                returncode = _run(command_for_transcribe(args, item), cwd=REPO_ROOT, dry_run=args.dry_run)
        if returncode != 0:
            status = "failed"
            failures += 1
        summary.append(
            {
                "source_dir": str(item.source_dir),
                "project_dir": str(item.project_dir),
                "status": status if not args.dry_run else "dry_run",
                "returncode": returncode,
            }
        )
        if returncode != 0 and args.stop_on_error:
            break

    if not args.dry_run:
        batch_root = args.runs_root.expanduser().resolve() / batch_id
        batch_root.mkdir(parents=True, exist_ok=True)
        (batch_root / "asr_batch_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            "utf-8",
        )
    return 1 if failures else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run extract + ASR transcribe only for an audio folder or its immediate work folders.",
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        type=Path,
        default=REPO_ROOT / "audio",
        help="Folder to process. Use 'audio' to batch immediate work folders under ./audio.",
    )
    parser.add_argument(
        "--single",
        dest="batch_children",
        action="store_false",
        default=None,
        help="Treat the input folder itself as one work instead of iterating its child folders.",
    )
    parser.add_argument(
        "--batch-children",
        dest="batch_children",
        action="store_true",
        help="Iterate immediate child folders of the input folder. This is the default.",
    )
    parser.add_argument("--runs-root", type=Path, default=REPO_ROOT / "runs" / "asr_only")
    parser.add_argument("--batch-id", default=None, help="Batch folder name under --runs-root.")
    parser.add_argument("--only", action="append", default=[], help="Folder name glob to include. Repeatable.")
    parser.add_argument("--skip", action="append", default=[], help="Folder name glob to skip. Repeatable.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N folders.")
    parser.add_argument("--force", action="store_true", help="Re-run even if transcribe already completed.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--confirm-rights", action="store_true", help="Confirm you have rights/consent for all inputs.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed folder.")
    parser.add_argument(
        "--in-process",
        dest="in_process",
        action="store_true",
        help="Run extract/transcribe through Python APIs so ASR models can be reused across projects.",
    )
    parser.add_argument(
        "--subprocess",
        dest="in_process",
        action="store_false",
        help="Run each extract/transcribe step through uv subprocesses for maximum process isolation.",
    )
    parser.set_defaults(in_process=True)
    parser.add_argument("--asr-backend", default="faster_whisper", choices=["faster_whisper", "qwen_asr", "mock"])
    parser.add_argument("--asr-preset", default="default")
    parser.add_argument("--asr-device", default=None)
    parser.add_argument("--asr-compute-type", default=None)
    parser.add_argument("--asr-batched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--asr-batch-size", type=int, default=16)
    parser.add_argument("--asr-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--source-separation-backend",
        choices=["none", "auto", "demucs", "mock"],
        default="none",
        help="Source separation backend to write into each project config before ASR.",
    )
    parser.add_argument(
        "--asr-repair",
        dest="disable_asr_repair",
        action="store_false",
        help="Keep ASR repair enabled. The default disables it to reduce temporary clips and extra passes.",
    )
    parser.add_argument(
        "--no-asr-repair",
        dest="disable_asr_repair",
        action="store_true",
        help="Disable ASR repair in each generated project config.",
    )
    parser.set_defaults(disable_asr_repair=True)
    parser.add_argument("--asr-review", action="store_true", help="Enable configured audio ASR review for suspicious chunks.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
