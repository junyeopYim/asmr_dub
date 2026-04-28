from __future__ import annotations

import socket
import sys
from pathlib import Path

from asmr_dub_pipeline.gemma.text_server import ManagedGemmaTextServer, is_http_ready


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
