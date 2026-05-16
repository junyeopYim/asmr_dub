from __future__ import annotations

import shlex
import socket
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

import httpx

from asmr_dub_pipeline.process import (
    popen_process_group_kwargs,
    tail_file,
    terminate_process_group,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
DEFAULT_LLAMA_SERVER = (
    Path(".cache") / "llama_cpp" / "src" / "llama.cpp" / "build" / "bin" / "llama-server"
)


class GemmaTextServerError(RuntimeError):
    pass


def _host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    if not parsed.hostname:
        raise GemmaTextServerError(f"Gemma text server URL must include a host: {base_url}")
    if parsed.scheme not in {"http", "https"}:
        raise GemmaTextServerError(f"Gemma text server URL must be http or https: {base_url}")
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.hostname, parsed.port or default_port


def _resolve_existing_path(value: str | Path, label: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, REPO_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    locations = ", ".join(str(candidate) for candidate in candidates)
    raise GemmaTextServerError(f"llama-server {label} not found: {value} (tried {locations})")


def is_local_url(base_url: str) -> bool:
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
        for path in ("/health", "/v1/models"):
            try:
                response = client.get(f"{base_url.rstrip('/')}{path}")
            except httpx.HTTPError:
                continue
            if response.status_code == 200:
                return True
            if response.status_code in {401, 403}:
                return True
            if response.status_code == 404:
                continue
            if response.status_code >= 500:
                return False
    return False


def default_llama_server_command(
    *,
    base_url: str,
    model_path: str | Path,
    mmproj_path: str | Path | None = None,
    ctx_size: int,
    gpu_layers: int,
    n_predict: int,
    parallel_slots: int = 1,
) -> list[str]:
    host, port = _host_port(base_url)
    server_path = _resolve_existing_path(DEFAULT_LLAMA_SERVER, "binary")
    model = _resolve_existing_path(model_path, "model")
    effective_parallel_slots = max(1, int(parallel_slots))
    command = [
        str(server_path),
        "-m",
        str(model),
    ]
    if mmproj_path:
        mmproj = _resolve_existing_path(mmproj_path, "mmproj")
        command.extend(["--mmproj", str(mmproj)])
    command.extend(
        [
            "--host",
            host,
            "--port",
            str(port),
            "--jinja",
            "--reasoning",
            "off",
            "--reasoning-budget",
            "0",
            "--no-warmup",
            "--parallel",
            str(effective_parallel_slots),
            "--no-cache-prompt",
            "--cache-ram",
            "0",
            "--ctx-checkpoints",
            "0",
            "-c",
            str(ctx_size),
            "-ngl",
            str(gpu_layers),
            "-n",
            str(n_predict),
        ]
    )
    return command


def _normalize_command(command: Sequence[str] | str | None) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    if command:
        return [str(part) for part in command]
    return []


def _tail(path: Path, max_chars: int = 2000) -> str:
    return tail_file(path, max_chars=max_chars)


class ManagedGemmaTextServer:
    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        command: Sequence[str] | str | None,
        log_path: Path | None = None,
        startup_timeout_sec: float = 120.0,
        shutdown_timeout_sec: float = 10.0,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url
        self.command = _normalize_command(command)
        self.log_path = log_path
        self.startup_timeout_sec = startup_timeout_sec
        self.shutdown_timeout_sec = shutdown_timeout_sec
        self.process: subprocess.Popen[str] | None = None
        self.started = False
        self.reused_existing = False
        self._log_file = None

    def _wait_until_ready(self, deadline: float) -> ManagedGemmaTextServer:
        while time.monotonic() < deadline:
            if is_http_ready(self.base_url):
                return self
            if self.process is not None and self.process.poll() is not None:
                details = f"\nServer log tail:\n{_tail(self.log_path)}" if self.log_path else ""
                self._close_log()
                raise GemmaTextServerError(
                    f"Gemma text server exited before becoming ready with code "
                    f"{self.process.returncode}.{details}"
                )
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.5, remaining))
        details = f"\nServer log tail:\n{_tail(self.log_path)}" if self.log_path else ""
        if self.process is not None:
            self.stop()
        raise GemmaTextServerError(
            f"Gemma text server did not become ready at {self.base_url} within "
            f"{self.startup_timeout_sec:g}s.{details}"
        )

    def start(self) -> ManagedGemmaTextServer:
        if not self.enabled:
            return self
        if is_http_ready(self.base_url):
            self.reused_existing = True
            return self
        if is_tcp_open(self.base_url):
            self.reused_existing = True
            return self._wait_until_ready(time.monotonic() + self.startup_timeout_sec)
        if not is_local_url(self.base_url):
            raise GemmaTextServerError(
                f"Gemma text server auto-start only supports local URLs; got {self.base_url}"
            )
        if not self.command:
            raise GemmaTextServerError(
                "Gemma text server auto-start requested, but no command was configured."
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
                stdout=stdout,
                stderr=stderr,
                text=True,
                **popen_process_group_kwargs(),
            )
        except OSError as exc:
            self._close_log()
            raise GemmaTextServerError(f"Could not start Gemma text server: {exc}") from exc
        self.started = True
        return self._wait_until_ready(time.monotonic() + self.startup_timeout_sec)

    def stop(self) -> None:
        if self.process is None:
            self._close_log()
            return
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

    def __enter__(self) -> ManagedGemmaTextServer:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop()
