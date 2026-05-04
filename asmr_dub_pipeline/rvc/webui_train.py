from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
CACHE_MANIFEST_NAME = "rvc_train_cache_manifest.json"
CACHE_VERSION = 1
WRAPPER_VERSION = "webui-train-cache-v1"


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


def _rvc_subprocess_env(*, trusted_checkpoint: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    if trusted_checkpoint:
        # RVC's Fairseq HuBERT checkpoint contains trusted pickled config objects.
        # PyTorch >=2.6 otherwise defaults torch.load to weights_only=True.
        env.setdefault(_TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, "1")
    else:
        env.pop(_TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, None)
    return env


def _run(command: list[str], *, cwd: Path, trusted_checkpoint: bool = False) -> None:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=_rvc_subprocess_env(trusted_checkpoint=trusted_checkpoint),
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _tail(text: str | bytes | None, limit: int = 1200) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text.strip()[-limit:]


def _run_parallel(commands: list[list[str]], *, cwd: Path, trusted_checkpoint: bool = False) -> None:
    if not commands:
        return

    def run_one(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(cwd),
            env=_rvc_subprocess_env(trusted_checkpoint=trusted_checkpoint),
            check=False,
            capture_output=True,
            text=True,
        )

    with ThreadPoolExecutor(max_workers=len(commands)) as executor:
        futures = {executor.submit(run_one, command): command for command in commands}
        for future in as_completed(futures):
            command = futures[future]
            completed = future.result()
            if completed.returncode != 0:
                raise SystemExit(
                    "Command failed with exit code "
                    f"{completed.returncode}: {' '.join(command)} "
                    f"stdout={_tail(completed.stdout)} stderr={_tail(completed.stderr)}"
                )


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def _auto_cpu_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _auto_gpu_workers(device: str) -> int:
    return 2 if "cuda" in device.lower() else 1


def _resolve_worker_count(requested: int, *, auto: int) -> int:
    if requested < 0:
        raise SystemExit("Worker counts must be >= 0.")
    return auto if requested == 0 else requested


def _log(message: str) -> None:
    print(f"[rvc-webui-train] {message}", flush=True)


def _copy_config(rvc_root: Path, exp_dir: Path, sample_rate: str, version: str) -> None:
    config_name = f"{version}/{sample_rate}.json" if version != "v1" or sample_rate != "40k" else "v1/40k.json"
    src = rvc_root / "configs" / config_name
    if not src.exists():
        raise SystemExit(f"RVC config template is missing: {src}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, exp_dir / "config.json")


def _write_filelist(rvc_root: Path, exp_dir: Path, sample_rate: str, version: str, if_f0: bool) -> None:
    gt_wavs_dir = exp_dir / "0_gt_wavs"
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    names = {path.stem for path in gt_wavs_dir.glob("*.wav")} & {path.stem for path in feature_dir.glob("*.npy")}
    if if_f0:
        f0_dir = exp_dir / "2a_f0"
        f0nsf_dir = exp_dir / "2b-f0nsf"
        names &= {path.name.removesuffix(".wav.npy") for path in f0_dir.glob("*.wav.npy")}
        names &= {path.name.removesuffix(".wav.npy") for path in f0nsf_dir.glob("*.wav.npy")}
    if not names:
        raise SystemExit(f"No RVC training feature rows were produced under {exp_dir}")
    fea_dim = 256 if version == "v1" else 768
    rows: list[str] = []
    for name in sorted(names):
        if if_f0:
            rows.append(
                f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|"
                f"{exp_dir / '2a_f0' / (name + '.wav.npy')}|"
                f"{exp_dir / '2b-f0nsf' / (name + '.wav.npy')}|0"
            )
        else:
            rows.append(f"{gt_wavs_dir / (name + '.wav')}|{feature_dir / (name + '.npy')}|0")
    mute_root = rvc_root / "logs" / "mute"
    for _ in range(2):
        if if_f0:
            rows.append(
                f"{mute_root / '0_gt_wavs' / ('mute' + sample_rate + '.wav')}|"
                f"{mute_root / ('3_feature' + str(fea_dim)) / 'mute.npy'}|"
                f"{mute_root / '2a_f0' / 'mute.wav.npy'}|"
                f"{mute_root / '2b-f0nsf' / 'mute.wav.npy'}|0"
            )
        else:
            rows.append(
                f"{mute_root / '0_gt_wavs' / ('mute' + sample_rate + '.wav')}|"
                f"{mute_root / ('3_feature' + str(fea_dim)) / 'mute.npy'}|0"
            )
    (exp_dir / "filelist.txt").write_text("\n".join(rows) + "\n", "utf-8")


def _reset_experiment_dir(exp_dir: Path) -> None:
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)


def _feature_dir(exp_dir: Path, version: str) -> Path:
    return exp_dir / ("3_feature256" if version == "v1" else "3_feature768")


def _dataset_fingerprint(dataset: Path, *, sample_rate: str, version: str, f0_method: str) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(dataset.glob("*.wav")):
        stat = path.stat()
        files.append({"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return {
        "cache_version": CACHE_VERSION,
        "wrapper_version": WRAPPER_VERSION,
        "sample_rate": sample_rate,
        "version": version,
        "f0_method": f0_method,
        "files": files,
    }


def _cache_manifest_path(exp_dir: Path) -> Path:
    return exp_dir / CACHE_MANIFEST_NAME


def _read_cache_manifest(exp_dir: Path) -> dict[str, object] | None:
    path = _cache_manifest_path(exp_dir)
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _write_cache_manifest(exp_dir: Path, fingerprint: dict[str, object], stages: dict[str, bool] | None = None) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": fingerprint,
        "stages": stages or {},
    }
    _cache_manifest_path(exp_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "utf-8",
    )


def _stage_done(exp_dir: Path, stage: str) -> bool:
    manifest = _read_cache_manifest(exp_dir) or {}
    stages = manifest.get("stages")
    return isinstance(stages, dict) and stages.get(stage) is True


def _mark_stage_done(exp_dir: Path, fingerprint: dict[str, object], stage: str) -> None:
    manifest = _read_cache_manifest(exp_dir) or {}
    stages = manifest.get("stages")
    next_stages = dict(stages) if isinstance(stages, dict) else {}
    next_stages[stage] = True
    _write_cache_manifest(exp_dir, fingerprint, next_stages)


def _matching_cache(exp_dir: Path, fingerprint: dict[str, object]) -> bool:
    manifest = _read_cache_manifest(exp_dir)
    return bool(manifest and manifest.get("fingerprint") == fingerprint)


def _wav_paths(exp_dir: Path) -> list[Path]:
    wav_root = exp_dir / "1_16k_wavs"
    return sorted(path for path in wav_root.glob("*.wav") if path.is_file())


def _preprocess_outputs_complete(exp_dir: Path) -> bool:
    gt_wavs = sorted((exp_dir / "0_gt_wavs").glob("*.wav"))
    wavs16k = _wav_paths(exp_dir)
    log = exp_dir / "preprocess.log"
    if not gt_wavs or not wavs16k or len(gt_wavs) != len(wavs16k):
        return False
    if not log.exists() or "end preprocess" not in log.read_text("utf-8", errors="ignore"):
        return False
    return all(path.stat().st_size > 0 for path in [*gt_wavs, *wavs16k])


def _f0_outputs_complete(exp_dir: Path) -> bool:
    wavs = _wav_paths(exp_dir)
    if not wavs:
        return False
    f0_dir = exp_dir / "2a_f0"
    f0nsf_dir = exp_dir / "2b-f0nsf"
    for wav in wavs:
        for path in (f0_dir / f"{wav.name}.npy", f0nsf_dir / f"{wav.name}.npy"):
            if not path.exists() or path.stat().st_size <= 0:
                return False
    return True


def _feature_outputs_complete(exp_dir: Path, version: str) -> bool:
    wavs = _wav_paths(exp_dir)
    if not wavs:
        return False
    feature_dir = _feature_dir(exp_dir, version)
    for wav in wavs:
        path = feature_dir / wav.with_suffix(".npy").name
        if not path.exists() or path.stat().st_size <= 0:
            return False
    return True


def _remove_stage_outputs(exp_dir: Path, *, f0: bool = False, feature: bool = False) -> None:
    if f0:
        for relative in ("2a_f0", "2b-f0nsf"):
            shutil.rmtree(exp_dir / relative, ignore_errors=True)
    if feature:
        for relative in ("3_feature256", "3_feature768"):
            shutil.rmtree(exp_dir / relative, ignore_errors=True)
        (exp_dir / "filelist.txt").unlink(missing_ok=True)


def _train_index(exp_dir: Path, experiment_name: str, version: str, output_index: Path) -> None:
    try:
        import faiss  # type: ignore[import-not-found]
        import numpy as np
    except Exception as exc:
        raise SystemExit(
            "RVC index training requires faiss and numpy in the Python environment used for train-rvc."
        ) from exc

    feature_dir = _feature_dir(exp_dir, version)
    npys = [np.load(str(path)) for path in sorted(feature_dir.glob("*.npy"))]
    if not npys:
        raise SystemExit(f"No feature npy files found for RVC index training: {feature_dir}")
    big_npy = np.concatenate(npys, 0)
    order = np.arange(big_npy.shape[0])
    np.random.shuffle(order)
    big_npy = big_npy[order]
    n_ivf = max(1, min(int(16 * np.sqrt(big_npy.shape[0])), max(1, big_npy.shape[0] // 39)))
    index = faiss.index_factory(256 if version == "v1" else 768, f"IVF{n_ivf},Flat")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    trained = exp_dir / f"trained_IVF{n_ivf}_Flat_nprobe_1_{experiment_name}_{version}.index"
    added = exp_dir / f"added_IVF{n_ivf}_Flat_nprobe_1_{experiment_name}_{version}.index"
    faiss.write_index(index, str(trained))
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start : start + 8192])
    faiss.write_index(index, str(added))
    output_index.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(added, output_index)


def _latest_exported_weight(weight_root: Path, experiment_name: str, min_mtime: float) -> Path:
    candidates = sorted(
        (path for path in weight_root.glob(f"{experiment_name}_e*_s*.pth") if path.stat().st_mtime >= min_mtime),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        direct = weight_root / f"{experiment_name}.pth"
        if direct.exists() and direct.stat().st_mtime >= min_mtime:
            return direct
        raise SystemExit(f"RVC training did not produce an exported weight for {experiment_name}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvc-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--sample-rate", default="48k")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--save-every-epoch", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--processes", type=int, default=0)
    parser.add_argument("--f0-workers", type=int, default=0)
    parser.add_argument("--feature-workers", type=int, default=0)
    parser.add_argument("--reuse-intermediate-cache", type=_parse_bool, default=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--version", default="v2")
    parser.add_argument("--f0-method", default="rmvpe")
    args = parser.parse_args()

    rvc_root = Path(args.rvc_root).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    output_model = Path(args.output_model).expanduser().resolve()
    output_index = Path(args.output_index).expanduser().resolve()
    exp_dir = rvc_root / "logs" / args.experiment_name
    sample_rate_int = {"32k": 32000, "40k": 40000, "48k": 48000}.get(args.sample_rate, 48000)
    gpu = args.device.split(":")[-1] if "cuda" in args.device else ""
    is_half = "True"
    preprocess_workers = _resolve_worker_count(args.processes, auto=_auto_cpu_workers())
    f0_workers = _resolve_worker_count(args.f0_workers, auto=_auto_gpu_workers(args.device))
    feature_workers = _resolve_worker_count(args.feature_workers, auto=_auto_gpu_workers(args.device))

    if not rvc_root.exists():
        raise SystemExit(f"RVC-WebUI root does not exist: {rvc_root}")
    if not dataset.exists():
        raise SystemExit(f"RVC training dataset does not exist: {dataset}")
    _link_or_copy_missing_assets(rvc_root)
    _log(
        "start "
        f"experiment={args.experiment_name} sample_rate={args.sample_rate} version={args.version} "
        f"preprocess_workers={preprocess_workers} f0_workers={f0_workers} feature_workers={feature_workers} "
        f"cache={'on' if args.reuse_intermediate_cache else 'off'}"
    )

    fingerprint = _dataset_fingerprint(
        dataset,
        sample_rate=args.sample_rate,
        version=args.version,
        f0_method=args.f0_method,
    )
    can_reuse = args.reuse_intermediate_cache and not args.rebuild_cache and _matching_cache(exp_dir, fingerprint)
    if not can_reuse:
        _log("cache miss; rebuilding preprocess/F0/feature intermediates")
        _reset_experiment_dir(exp_dir)
        _write_cache_manifest(exp_dir, fingerprint)
    elif not _preprocess_outputs_complete(exp_dir):
        _log("cache manifest matched but preprocess outputs are incomplete; rebuilding")
        _reset_experiment_dir(exp_dir)
        _write_cache_manifest(exp_dir, fingerprint)
        can_reuse = False

    if can_reuse and _preprocess_outputs_complete(exp_dir):
        _log("preprocess skip; cached outputs complete")
        if not _stage_done(exp_dir, "preprocess"):
            _mark_stage_done(exp_dir, fingerprint, "preprocess")
    else:
        _log("preprocess start")
        _run(
            [
                sys.executable,
                "infer/modules/train/preprocess.py",
                str(dataset),
                str(sample_rate_int),
                str(preprocess_workers),
                str(exp_dir),
                "False",
                "3.7",
            ],
            cwd=rvc_root,
        )
        if not _preprocess_outputs_complete(exp_dir):
            raise SystemExit(f"RVC preprocess did not create complete outputs under {exp_dir}")
        _mark_stage_done(exp_dir, fingerprint, "preprocess")
        can_reuse = True
        _log("preprocess complete")

    if can_reuse and _f0_outputs_complete(exp_dir):
        _log("F0 extraction skip; cached outputs complete")
        if not _stage_done(exp_dir, "f0"):
            _mark_stage_done(exp_dir, fingerprint, "f0")
    else:
        _log(f"F0 extraction start partitions={f0_workers}")
        _remove_stage_outputs(exp_dir, f0=True, feature=True)
        _run_parallel(
            [
                [
                    sys.executable,
                    "infer/modules/train/extract/extract_f0_rmvpe.py",
                    str(f0_workers),
                    str(worker_index),
                    gpu or "0",
                    str(exp_dir),
                    is_half,
                ]
                for worker_index in range(f0_workers)
            ],
            cwd=rvc_root,
        )
        if not _f0_outputs_complete(exp_dir):
            raise SystemExit(f"RVC F0 extraction did not create complete outputs under {exp_dir}")
        _mark_stage_done(exp_dir, fingerprint, "f0")
        _log("F0 extraction complete")

    if can_reuse and _feature_outputs_complete(exp_dir, args.version):
        _log("feature extraction skip; cached outputs complete")
        if not _stage_done(exp_dir, "feature"):
            _mark_stage_done(exp_dir, fingerprint, "feature")
    else:
        _log(f"feature extraction start partitions={feature_workers}")
        _remove_stage_outputs(exp_dir, feature=True)
        _run_parallel(
            [
                [
                    sys.executable,
                    "infer/modules/train/extract_feature_print.py",
                    args.device,
                    str(feature_workers),
                    str(worker_index),
                    gpu or "0",
                    str(exp_dir),
                    args.version,
                    is_half,
                ]
                for worker_index in range(feature_workers)
            ],
            cwd=rvc_root,
            trusted_checkpoint=True,
        )
        if not _feature_outputs_complete(exp_dir, args.version):
            raise SystemExit(f"RVC feature extraction did not create complete outputs under {exp_dir}")
        _mark_stage_done(exp_dir, fingerprint, "feature")
        _log("feature extraction complete")

    _log("training start")
    _copy_config(rvc_root, exp_dir, args.sample_rate, args.version)
    _write_filelist(rvc_root, exp_dir, args.sample_rate, args.version, if_f0=True)
    pretrained_g = rvc_root / "assets" / "pretrained_v2" / f"f0G{args.sample_rate}.pth"
    pretrained_d = rvc_root / "assets" / "pretrained_v2" / f"f0D{args.sample_rate}.pth"
    train_started = time.time() - 1.0
    _run(
        [
            sys.executable,
            "infer/modules/train/train.py",
            "-e",
            args.experiment_name,
            "-sr",
            args.sample_rate,
            "-f0",
            "1",
            "-bs",
            str(args.batch_size),
            "-g",
            gpu or "0",
            "-te",
            str(args.epochs),
            "-se",
            str(args.save_every_epoch),
            "-pg",
            str(pretrained_g),
            "-pd",
            str(pretrained_d),
            "-l",
            "1",
            "-c",
            "0",
            "-sw",
            "1",
            "-v",
            args.version,
        ],
        cwd=rvc_root,
    )
    _log("training complete; building FAISS index")
    _train_index(exp_dir, args.experiment_name, args.version, output_index)
    exported = _latest_exported_weight(rvc_root / "assets" / "weights", args.experiment_name, train_started)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, output_model)
    _log(f"export complete model={output_model} index={output_index}")
    metadata = {
        "rvc_root": str(rvc_root),
        "experiment_name": args.experiment_name,
        "source_model": str(exported),
        "output_model": str(output_model),
        "output_index": str(output_index),
    }
    (output_model.parent / "webui_train_result.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        "utf-8",
    )


if __name__ == "__main__":
    main()
