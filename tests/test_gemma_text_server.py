from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import pytest

from asmr_dub_pipeline.gemma import text_server as text_server_module
from asmr_dub_pipeline.gemma.text_server import (
    ManagedGemmaTextServer,
    default_llama_server_command,
    is_http_ready,
)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_loading_then_ready_server(path: Path) -> None:
    path.write_text(
        """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys

class Handler(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):
        if self.path == "/health":
            Handler.hits += 1
            self.send_response(200 if Handler.hits >= 3 else 503)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""".strip()
        + "\n",
        "utf-8",
    )


def write_parent_with_child_ready_server(path: Path) -> None:
    path.write_text(
        """
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess
import sys

marker_path = sys.argv[2]
child_code = "import pathlib, sys, time; time.sleep(1); pathlib.Path(sys.argv[1]).write_text('alive', encoding='utf-8')"
subprocess.Popen([sys.executable, "-c", child_code, marker_path])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""".strip()
        + "\n",
        "utf-8",
    )


def test_default_llama_server_command_includes_mmproj_when_requested(
    monkeypatch,
    tmp_path: Path,
) -> None:
    server = tmp_path / "llama-server"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    for path in (server, model, mmproj):
        path.write_text("mock", "utf-8")
    monkeypatch.setattr(text_server_module, "DEFAULT_LLAMA_SERVER", server)

    command = default_llama_server_command(
        base_url="http://127.0.0.1:18080",
        model_path=model,
        mmproj_path=mmproj,
        ctx_size=4096,
        gpu_layers=999,
        n_predict=1024,
    )

    assert command[:5] == [str(server.resolve()), "-m", str(model.resolve()), "--mmproj", str(mmproj.resolve())]
    assert "--host" in command
    assert "--port" in command


def test_managed_gemma_text_server_waits_for_http_readiness(tmp_path: Path) -> None:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    script = tmp_path / "loading_server.py"
    write_loading_then_ready_server(script)

    manager = ManagedGemmaTextServer(
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
        assert is_http_ready(base_url)

    assert manager.process is not None
    assert manager.process.poll() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_managed_gemma_text_server_stop_kills_child_process_group(tmp_path: Path) -> None:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    marker_path = tmp_path / "child_survived.txt"
    script = tmp_path / "parent_server.py"
    write_parent_with_child_ready_server(script)

    manager = ManagedGemmaTextServer(
        enabled=True,
        base_url=base_url,
        command=[sys.executable, str(script), str(port), str(marker_path)],
        startup_timeout_sec=5,
        shutdown_timeout_sec=1,
    )

    with manager:
        assert manager.started is True
        assert is_http_ready(base_url)

    time.sleep(1.2)
    assert not marker_path.exists()
