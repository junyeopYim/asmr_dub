from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yaml

from asmr_dub_pipeline.process import (
    popen_process_group_kwargs,
    tail_file,
    terminate_process_group,
)

from .client import GPTSoVITSError

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SHIM_DIR = Path(__file__).resolve().parent / "shims"
GSV_SERVER_REQUIRED_MODULES = (
    "numpy",
    "transformers",
    "torch",
    "torchaudio",
    "librosa",
    "fastapi",
    "uvicorn",
    "ffmpeg",
    "peft",
    "jieba",
    "jieba_fast",
    "fast_langdetect",
    "split_lang",
    "cn2an",
    "pypinyin",
    "pyopenjtalk",
    "g2p_en",
    "g2pk2",
    "ko_pron",
    "opencc",
    "wordsegment",
    "x_transformers",
)


def _host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if not parsed.hostname:
        raise GPTSoVITSError(f"GPT-SoVITS URL must include a host: {base_url}")
    if parsed.scheme not in {"http", "https"}:
        raise GPTSoVITSError(f"GPT-SoVITS URL must be http or https: {base_url}")
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, parsed.port or default_port


def is_local_gsv_url(base_url: str) -> bool:
    host, _ = _host_port(base_url)
    return host.lower() in LOCAL_HOSTS


def is_tcp_open(base_url: str, timeout_sec: float = 0.3) -> bool:
    host, port = _host_port(base_url)
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def is_http_ready(base_url: str, timeout_sec: float = 0.5) -> bool:
    if not is_tcp_open(base_url, timeout_sec=timeout_sec):
        return False
    with httpx.Client(timeout=timeout_sec) as client:
        for path in ("/docs", "/openapi.json", "/tts"):
            try:
                response = client.get(f"{base_url.rstrip('/')}{path}")
            except httpx.HTTPError:
                continue
            if response.status_code < 500 and response.status_code != 404:
                return True
    return False


def _dedupe_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _python_has_modules(python: str, modules: Sequence[str]) -> bool:
    code = (
        "import importlib.util, sys; "
        f"mods={list(modules)!r}; "
        "raise SystemExit(0 if all(importlib.util.find_spec(m) for m in mods) else 1)"
    )
    try:
        completed = subprocess.run(
            [python, "-s", "-c", code],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return completed.returncode == 0


def _default_gsv_python() -> str:
    candidates: list[str] = []
    if os.environ.get("ASMR_DUB_GSV_PYTHON"):
        candidates.append(os.environ["ASMR_DUB_GSV_PYTHON"])
    if sys.executable:
        candidates.append(sys.executable)
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(str(Path(os.environ["VIRTUAL_ENV"]) / "bin" / "python"))
    base_prefix = Path(getattr(sys, "base_prefix", "") or "")
    if base_prefix:
        candidates.append(str(base_prefix / "bin" / "python"))
    for name in ("python", "python3"):
        resolved = shutil.which(name)
        if resolved:
            candidates.append(resolved)
    for candidate in _dedupe_text(candidates):
        if _python_has_modules(candidate, GSV_SERVER_REQUIRED_MODULES):
            return candidate
    return candidates[0] if candidates else "python"


def _default_gsv_command(base_url: str) -> list[str]:
    host, port = _host_port(base_url)
    candidates = [
        REPO_ROOT / ".cache/third_party/GPT-SoVITS/api_v2.py",
        REPO_ROOT / ".cache/third_party/GPT_SoVITS/api_v2.py",
        REPO_ROOT / ".cache/gpt_sovits/GPT_SoVITS/api_v2.py",
        REPO_ROOT / ".cache/gpt_sovits/GPT-SoVITS/api_v2.py",
        Path.cwd() / "GPT_SoVITS/api_v2.py",
        Path.cwd() / "GPT-SoVITS/api_v2.py",
    ]
    api_path = next((path for path in candidates if path.exists()), None)
    if api_path is None:
        return []
    command = [_default_gsv_python(), str(api_path), "-a", host, "-p", str(port)]
    _repair_pretrained_model_links(api_path.parent)
    config_path = _local_tts_config(api_path.parent) or api_path.parent / "GPT_SoVITS/configs/tts_infer.yaml"
    if config_path.exists():
        command.extend(["-c", str(config_path)])
    return command


def _repair_pretrained_model_links(install_dir: Path) -> None:
    source = REPO_ROOT / ".cache" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models"
    target = install_dir / "GPT_SoVITS" / "pretrained_models"
    if not source.exists() or not target.exists():
        return
    for name in (
        "chinese-hubert-base",
        "chinese-roberta-wwm-ext-large",
        "fast_langdetect",
        "s1v3.ckpt",
        "sv",
        "v2Pro",
    ):
        source_path = source / name
        if name == "fast_langdetect":
            source_path.mkdir(parents=True, exist_ok=True)
        target_path = target / name
        if not source_path.exists():
            continue
        if target_path.is_symlink() and not target_path.exists():
            target_path.unlink()
        if not target_path.exists():
            target_path.symlink_to(source_path.resolve())


def _local_tts_config(install_dir: Path) -> Path | None:
    pretrained = REPO_ROOT / ".cache" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models"
    required = {
        "t2s_weights_path": pretrained / "s1v3.ckpt",
        "vits_weights_path": pretrained / "v2Pro" / "s2Gv2ProPlus.pth",
        "bert_base_path": pretrained / "chinese-roberta-wwm-ext-large",
        "cnhuhbert_base_path": pretrained / "chinese-hubert-base",
    }
    if not all(path.exists() for path in required.values()):
        return None
    config_path = REPO_ROOT / ".cache" / "gpt_sovits" / "tts_infer.local.yaml"
    custom = {
        "device": "cuda",
        "is_half": True,
        "version": "v2ProPlus",
        **{key: str(path.resolve()) for key, path in required.items()},
    }
    payload = {"custom": custom}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), "utf-8")
    return config_path


def _infer_gsv_cwd(command: Sequence[str]) -> Path | None:
    for part in command:
        path = Path(part).expanduser()
        if path.name == "api_v2.py" and path.exists():
            return path.resolve().parent
    return None


def _normalize_command(command: Sequence[str] | str | None, base_url: str) -> list[str]:
    host, port = _host_port(base_url)

    def format_part(part: object) -> str:
        text = str(part)
        return (
            text.replace("{base_url}", base_url)
            .replace("{host}", host)
            .replace("{port}", str(port))
        )

    if isinstance(command, str):
        return [format_part(part) for part in shlex.split(command)]
    if command:
        return [format_part(part) for part in command]
    return _default_gsv_command(base_url)


def _gsv_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if importlib.util.find_spec("mecab") is None and SHIM_DIR.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(SHIM_DIR) if not existing else os.pathsep.join((str(SHIM_DIR), existing))
        )
    cuda_library_dirs = _candidate_cuda_library_dirs()
    library_paths = [str(path) for path in cuda_library_dirs]
    if env.get("LD_LIBRARY_PATH"):
        library_paths.append(env["LD_LIBRARY_PATH"])
    if library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(_dedupe_text(library_paths))
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


def _tail(path: Path, max_chars: int = 2000) -> str:
    return tail_file(path, max_chars=max_chars)


class ManagedGPTSoVITSServer:
    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        command: Sequence[str] | str | None = None,
        cwd: str | Path | None = None,
        log_path: Path | None = None,
        startup_timeout_sec: float = 120.0,
        shutdown_timeout_sec: float = 10.0,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url
        self.command = _normalize_command(command, base_url)
        self.cwd = Path(cwd).expanduser().resolve() if cwd else _infer_gsv_cwd(self.command)
        self.log_path = log_path
        self.startup_timeout_sec = startup_timeout_sec
        self.shutdown_timeout_sec = shutdown_timeout_sec
        self.process: subprocess.Popen[str] | None = None
        self.started = False
        self.reused_existing = False
        self._log_file = None

    def _wait_until_ready(self, deadline: float) -> ManagedGPTSoVITSServer:
        while time.monotonic() < deadline:
            if is_http_ready(self.base_url):
                return self
            if self.process is not None and self.process.poll() is not None:
                details = f"\nServer log tail:\n{_tail(self.log_path)}" if self.log_path else ""
                self._close_log()
                raise GPTSoVITSError(
                    f"GPT-SoVITS server exited before becoming ready with code "
                    f"{self.process.returncode}.{details}"
                )
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.5, remaining))
        details = f"\nServer log tail:\n{_tail(self.log_path)}" if self.log_path else ""
        if self.process is not None:
            self.stop()
        raise GPTSoVITSError(
            f"GPT-SoVITS server did not become ready at {self.base_url} within "
            f"{self.startup_timeout_sec:g}s.{details}"
        )

    def start(self) -> ManagedGPTSoVITSServer:
        if not self.enabled:
            return self
        if is_http_ready(self.base_url):
            self.reused_existing = True
            return self
        if is_tcp_open(self.base_url):
            self.reused_existing = True
            return self._wait_until_ready(time.monotonic() + self.startup_timeout_sec)
        if not is_local_gsv_url(self.base_url):
            raise GPTSoVITSError(
                f"GPT-SoVITS auto-start only supports local URLs; got {self.base_url}"
            )
        if not self.command:
            raise GPTSoVITSError(
                "GPT-SoVITS auto-start requested, but no api_v2.py was found. "
                "Install GPT-SoVITS or set gsv_server_command / --gsv-server-command."
            )
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("a", encoding="utf-8")
            stdout = self._log_file
            stderr = subprocess.STDOUT
        else:
            stdout = subprocess.DEVNULL
            stderr = subprocess.DEVNULL
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=str(self.cwd) if self.cwd else None,
                env=_gsv_subprocess_env(),
                stdout=stdout,
                stderr=stderr,
                text=True,
                **popen_process_group_kwargs(),
            )
        except OSError as exc:
            self._close_log()
            raise GPTSoVITSError(f"Could not start GPT-SoVITS server: {exc}") from exc
        self.started = True
        return self._wait_until_ready(time.monotonic() + self.startup_timeout_sec)

    def stop(self) -> None:
        if self.process is None:
            self._close_log()
            return
        if self.process.poll() is None:
            terminate_process_group(
                self.process,
                terminate_timeout_sec=self.shutdown_timeout_sec,
                kill_timeout_sec=5,
            )
        self._close_log()

    def _close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> ManagedGPTSoVITSServer:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
