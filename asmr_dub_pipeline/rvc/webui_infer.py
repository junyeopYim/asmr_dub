from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_INFER_CLI_COMPAT = """
import runpy

try:
    import torch
    from fairseq.data.dictionary import Dictionary

    torch.serialization.add_safe_globals([Dictionary])
    if torch.cuda.is_available():
        torch.cuda.empty_cache = lambda: None
except Exception:
    pass

runpy.run_path("tools/infer_cli.py", run_name="__main__")
""".strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _link_or_copy_missing_assets(rvc_root: Path) -> None:
    source_root = _repo_root() / ".cache" / "rvc" / "assets"
    if not source_root.exists():
        return
    for source in source_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        target = rvc_root / "assets" / relative
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-root", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--f0-up-key", default="0")
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--index-rate", default="0.45")
    parser.add_argument("--filter-radius", default="3")
    parser.add_argument("--resample-sr", default="48000")
    parser.add_argument("--rms-mix-rate", default="0.25")
    parser.add_argument("--protect", default="0.33")
    parser.add_argument("--is-half", default="True")
    args = parser.parse_args()

    rvc_root = Path(args.rvc_root).expanduser().resolve()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve() if args.index else None

    if not rvc_root.exists():
        raise SystemExit(f"RVC-WebUI root does not exist: {rvc_root}")
    if not input_path.exists():
        raise SystemExit(f"RVC input does not exist: {input_path}")
    if not model_path.exists():
        raise SystemExit(f"RVC model does not exist: {model_path}")
    if index_path is not None and not index_path.exists():
        raise SystemExit(f"RVC index does not exist: {index_path}")
    _link_or_copy_missing_assets(rvc_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["weight_root"] = str(model_path.parent)
    env["index_root"] = str(index_path.parent if index_path else rvc_root / "logs")
    env["outside_index_root"] = str(index_path.parent if index_path else rvc_root / "assets" / "indices")
    env["rmvpe_root"] = str(rvc_root / "assets" / "rmvpe")

    command = [
        sys.executable,
        "-c",
        _INFER_CLI_COMPAT,
        "--input_path",
        str(input_path),
        "--opt_path",
        str(output_path),
        "--model_name",
        model_path.name,
        "--index_path",
        str(index_path or ""),
        "--f0method",
        str(args.f0_method),
        "--f0up_key",
        str(args.f0_up_key),
        "--index_rate",
        str(args.index_rate),
        "--filter_radius",
        str(args.filter_radius),
        "--resample_sr",
        str(args.resample_sr),
        "--rms_mix_rate",
        str(args.rms_mix_rate),
        "--protect",
        str(args.protect),
        "--device",
        str(args.device),
        "--is_half",
        str(args.is_half),
    ]
    completed = subprocess.run(command, cwd=str(rvc_root), env=env, check=False, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"RVC-WebUI infer_cli failed with exit code {completed.returncode}")
    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise SystemExit(f"RVC-WebUI infer_cli did not create output: {output_path}")


if __name__ == "__main__":
    main()
