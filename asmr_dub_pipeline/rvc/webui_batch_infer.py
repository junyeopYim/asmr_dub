from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from time import monotonic
from typing import Any


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


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _register_safe_globals() -> None:
    try:
        import torch
        from fairseq.data.dictionary import Dictionary

        torch.serialization.add_safe_globals([Dictionary])
        if torch.cuda.is_available():
            torch.cuda.empty_cache = lambda: None
    except Exception:
        pass


def _load_rvc_components(rvc_root: Path) -> tuple[type[Any], type[Any]]:
    os.chdir(rvc_root)
    sys.path.insert(0, str(rvc_root))
    from dotenv import load_dotenv

    _register_safe_globals()
    load_dotenv()
    from configs.config import Config
    from infer.modules.vc.modules import VC

    return Config, VC


def _write_wav(path: Path, sample_rate: int, audio: Any) -> None:
    from scipy.io import wavfile

    wavfile.write(str(path), sample_rate, audio)


def _read_jobs(jobs_path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for line_number, line in enumerate(jobs_path.read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"RVC batch job line {line_number} must be a JSON object.")
        jobs.append(payload)
    return jobs


def _convert_job(vc: Any, job: dict[str, Any], index_path: Path | None) -> dict[str, Any]:
    started = monotonic()
    segment_id = str(job.get("segment_id") or "")
    input_path = Path(str(job.get("input_path") or "")).expanduser().resolve()
    output_path = Path(str(job.get("output_path") or "")).expanduser().resolve()
    result: dict[str, Any] = {
        "segment_id": segment_id,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "returncode": 1,
        "stdout": "",
        "stderr": "",
    }
    try:
        if not segment_id:
            raise ValueError("RVC batch job is missing segment_id.")
        if not input_path.exists():
            raise FileNotFoundError(f"RVC input does not exist: {input_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        info, wav_opt = vc.vc_single(
            0,
            str(input_path),
            int(job.get("f0_up_key", 0)),
            None,
            str(job.get("f0_method", "rmvpe")),
            str(index_path or ""),
            None,
            float(job.get("index_rate", 0.45)),
            int(job.get("filter_radius", 3)),
            int(job.get("resample_sr", 48_000)),
            float(job.get("rms_mix_rate", 0.25)),
            float(job.get("protect", 0.33)),
        )
        result["stdout"] = str(info or "")
        if not isinstance(wav_opt, tuple) or len(wav_opt) != 2:
            raise RuntimeError(str(info or "RVC did not return audio."))
        sample_rate, audio = wav_opt
        if sample_rate is None or audio is None or "Success" not in str(info):
            raise RuntimeError(str(info or "RVC conversion failed."))
        _write_wav(output_path, int(sample_rate), audio)
        if not output_path.exists() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"RVC did not create output: {output_path}")
        result["returncode"] = 0
    except Exception:
        result["stderr"] = traceback.format_exc()
    finally:
        result["elapsed_sec"] = monotonic() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-root", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--is-half", default="True")
    args = parser.parse_args()

    rvc_root = Path(args.rvc_root).expanduser().resolve()
    jobs_path = Path(args.jobs).expanduser().resolve()
    results_path = Path(args.results).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve() if args.index else None

    if not rvc_root.exists():
        raise SystemExit(f"RVC-WebUI root does not exist: {rvc_root}")
    if not jobs_path.exists():
        raise SystemExit(f"RVC batch jobs file does not exist: {jobs_path}")
    if not model_path.exists():
        raise SystemExit(f"RVC model does not exist: {model_path}")
    if index_path is not None and not index_path.exists():
        raise SystemExit(f"RVC index does not exist: {index_path}")

    _link_or_copy_missing_assets(rvc_root)
    os.environ["weight_root"] = str(model_path.parent)  # noqa: SIM112 - RVC-WebUI expects lowercase.
    os.environ["index_root"] = str(index_path.parent if index_path else rvc_root / "logs")  # noqa: SIM112
    os.environ["outside_index_root"] = str(index_path.parent if index_path else rvc_root / "assets" / "indices")  # noqa: SIM112
    os.environ["rmvpe_root"] = str(rvc_root / "assets" / "rmvpe")  # noqa: SIM112

    Config, VC = _load_rvc_components(rvc_root)
    config = Config()
    config.device = str(args.device) if args.device else config.device
    config.is_half = _str_to_bool(str(args.is_half))
    vc = VC(config)
    vc.get_vc(model_path.name)

    jobs = _read_jobs(jobs_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with results_path.open("w", encoding="utf-8") as handle:
        for index, job in enumerate(jobs, start=1):
            result = _convert_job(vc, job, index_path)
            if result["returncode"] != 0:
                failures += 1
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[rvc-webui-batch] {index}/{len(jobs)} "
                f"segment={result.get('segment_id')} returncode={result['returncode']}",
                flush=True,
            )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
