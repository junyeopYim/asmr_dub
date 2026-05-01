from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from asmr_dub_pipeline.gpt_sovits import server as gsv_server
from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSError
from asmr_dub_pipeline.gpt_sovits.server import (
    ManagedGPTSoVITSServer,
    _default_gsv_command,
    _gsv_subprocess_env,
    is_tcp_open,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_tiny_http_server(path: Path) -> None:
    path.write_text(
        """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""".strip()
        + "\n",
        "utf-8",
    )


def test_managed_gsv_server_starts_and_stops(tmp_path: Path) -> None:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    script = tmp_path / "tiny_server.py"
    write_tiny_http_server(script)

    manager = ManagedGPTSoVITSServer(
        enabled=True,
        base_url=base_url,
        command=[sys.executable, str(script), str(port)],
        log_path=tmp_path / "server.log",
        startup_timeout_sec=5,
        shutdown_timeout_sec=2,
    )

    with manager:
        assert manager.started is True
        assert manager.reused_existing is False
        assert manager.process is not None
        assert is_tcp_open(base_url)

    assert manager.process is not None
    assert manager.process.poll() is not None


def test_managed_gsv_server_reuses_existing_process() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = int(sock.getsockname()[1])
        manager = ManagedGPTSoVITSServer(
            enabled=True,
            base_url=f"http://127.0.0.1:{port}",
            command=["definitely-not-run"],
            startup_timeout_sec=1,
        )
        manager.start()

    assert manager.started is False
    assert manager.reused_existing is True
    assert manager.process is None


def test_managed_gsv_server_requires_command_when_missing(monkeypatch) -> None:
    monkeypatch.setattr(gsv_server, "_default_gsv_command", lambda base_url: [])
    port = free_port()
    manager = ManagedGPTSoVITSServer(
        enabled=True,
        base_url=f"http://127.0.0.1:{port}",
        command=None,
        startup_timeout_sec=1,
    )

    with pytest.raises(GPTSoVITSError, match="auto-start requested"):
        manager.start()


def test_default_gsv_command_discovers_third_party_install(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    install = root / ".cache" / "third_party" / "GPT-SoVITS"
    api = install / "api_v2.py"
    config = install / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    api.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    api.write_text("", "utf-8")
    config.write_text("", "utf-8")
    monkeypatch.setattr(gsv_server, "REPO_ROOT", root)
    monkeypatch.setattr(gsv_server, "_default_gsv_python", lambda: "python")

    command = _default_gsv_command("http://127.0.0.1:9880")

    assert command == [
        "python",
        str(api),
        "-a",
        "127.0.0.1",
        "-p",
        "9880",
        "-c",
        str(config),
    ]
    manager = ManagedGPTSoVITSServer(enabled=False, base_url="http://127.0.0.1:9880")
    assert manager.command == command
    assert manager.cwd == install.resolve()


def test_default_gsv_command_prepares_fast_langdetect_cache(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    install = root / ".cache" / "third_party" / "GPT-SoVITS"
    api = install / "api_v2.py"
    config = install / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    target = install / "GPT_SoVITS" / "pretrained_models"
    source = root / ".cache" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models"
    api.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    target.mkdir(parents=True)
    source.mkdir(parents=True)
    api.write_text("", "utf-8")
    config.write_text("", "utf-8")
    monkeypatch.setattr(gsv_server, "REPO_ROOT", root)
    monkeypatch.setattr(gsv_server, "_default_gsv_python", lambda: "python")

    _default_gsv_command("http://127.0.0.1:9880")

    cache_dir = source / "fast_langdetect"
    assert cache_dir.is_dir()
    assert (target / "fast_langdetect").resolve() == cache_dir.resolve()


def test_default_gsv_python_prefers_dependency_complete_env(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "gsv_env" / "bin" / "python"
    base = tmp_path / "base" / "bin" / "python"
    monkeypatch.setenv("ASMR_DUB_GSV_PYTHON", str(override))
    monkeypatch.setattr(gsv_server.sys, "base_prefix", str(base.parent.parent))
    monkeypatch.setattr(gsv_server.shutil, "which", lambda name: None)

    def fake_has_modules(python: str, modules) -> bool:
        return python == str(override)

    monkeypatch.setattr(gsv_server, "_python_has_modules", fake_has_modules)

    assert gsv_server._default_gsv_python() == str(override)


def test_default_gsv_python_skips_venv_without_server_deps(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "base" / "bin" / "python"
    venv = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.delenv("ASMR_DUB_GSV_PYTHON", raising=False)
    monkeypatch.setattr(gsv_server.sys, "base_prefix", str(base.parent.parent))
    monkeypatch.setattr(gsv_server.sys, "executable", str(venv))
    monkeypatch.setattr(gsv_server.shutil, "which", lambda name: str(venv))

    def fake_has_modules(python: str, modules) -> bool:
        return python == str(base)

    monkeypatch.setattr(gsv_server, "_python_has_modules", fake_has_modules)

    assert gsv_server._default_gsv_python() == str(base)


def test_gsv_subprocess_env_adds_mecab_shim_when_missing(monkeypatch, tmp_path: Path) -> None:
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    monkeypatch.setattr(gsv_server, "SHIM_DIR", shim_dir)
    monkeypatch.setattr(gsv_server.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setenv("PYTHONPATH", "existing")

    env = _gsv_subprocess_env()

    assert env["PYTHONPATH"].split(":")[:2] == [str(shim_dir), "existing"]
