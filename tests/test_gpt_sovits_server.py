from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest
import yaml

from asmr_dub_pipeline.gpt_sovits import server as gsv_server
from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSError
from asmr_dub_pipeline.gpt_sovits.server import (
    ManagedGPTSoVITSServer,
    _default_gsv_command,
    _gsv_subprocess_env,
    _patch_gsv_korean_text_preprocessor,
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


def write_parent_with_child_http_server(path: Path) -> None:
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
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
""".strip()
        + "\n",
        "utf-8",
    )


def write_parent_exits_after_child_http_server(path: Path) -> None:
    path.write_text(
        """
import socket
import subprocess
import sys
import time

port = int(sys.argv[1])
child_code = r'''
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''
subprocess.Popen([sys.executable, "-c", child_code, str(port)])
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.1):
            break
    except OSError:
        time.sleep(0.05)
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


def test_managed_gsv_server_reuses_existing_http_ready_process(tmp_path: Path) -> None:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    script = tmp_path / "tiny_server.py"
    write_tiny_http_server(script)
    process = subprocess.Popen([sys.executable, str(script), str(port)])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not is_tcp_open(base_url):
            time.sleep(0.05)
        manager = ManagedGPTSoVITSServer(
            enabled=True,
            base_url=base_url,
            command=["definitely-not-run"],
            startup_timeout_sec=1,
        )
        manager.start()
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert manager.started is False
    assert manager.reused_existing is True
    assert manager.process is None


def test_managed_gsv_server_does_not_treat_tcp_only_socket_as_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gsv_server, "is_http_ready", lambda _base_url, **_kwargs: False, raising=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        port = int(sock.getsockname()[1])
        manager = ManagedGPTSoVITSServer(
            enabled=True,
            base_url=f"http://127.0.0.1:{port}",
            command=["definitely-not-run"],
            startup_timeout_sec=0.01,
        )
        with pytest.raises(GPTSoVITSError, match="ready"):
            manager.start()

    assert manager.started is False
    assert manager.reused_existing is True
    assert manager.process is None


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_managed_gsv_server_stop_kills_child_process_group(tmp_path: Path) -> None:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    marker_path = tmp_path / "child_survived.txt"
    script = tmp_path / "parent_server.py"
    write_parent_with_child_http_server(script)

    manager = ManagedGPTSoVITSServer(
        enabled=True,
        base_url=base_url,
        command=[sys.executable, str(script), str(port), str(marker_path)],
        startup_timeout_sec=5,
        shutdown_timeout_sec=1,
    )

    with manager:
        assert manager.started is True
        assert is_tcp_open(base_url)

    time.sleep(1.2)
    assert not marker_path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_managed_gsv_server_stop_kills_child_group_after_parent_exits(tmp_path: Path) -> None:
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    script = tmp_path / "parent_exits_after_child_server.py"
    write_parent_exits_after_child_http_server(script)
    manager = ManagedGPTSoVITSServer(
        enabled=True,
        base_url=base_url,
        command=[sys.executable, str(script), str(port)],
        startup_timeout_sec=5,
        shutdown_timeout_sec=1,
    )

    try:
        manager.start()
        assert manager.process is not None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and manager.process.poll() is None:
            time.sleep(0.05)
        assert manager.process.poll() is not None
        assert is_tcp_open(base_url)

        manager.stop()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and is_tcp_open(base_url):
            time.sleep(0.05)
        assert not is_tcp_open(base_url)
    finally:
        if manager.process is not None:
            with suppress(ProcessLookupError):
                os.killpg(manager.process.pid, signal.SIGKILL)


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


def test_default_gsv_command_writes_local_tts_config_atomically(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "repo"
    install = root / ".cache" / "third_party" / "GPT-SoVITS"
    api = install / "api_v2.py"
    pretrained = root / ".cache" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models"
    for path in (
        pretrained / "s1v3.ckpt",
        pretrained / "v2Pro" / "s2Gv2ProPlus.pth",
        pretrained / "chinese-roberta-wwm-ext-large" / "config.json",
        pretrained / "chinese-hubert-base" / "config.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", "utf-8")
    api.parent.mkdir(parents=True, exist_ok=True)
    api.write_text("", "utf-8")
    monkeypatch.setattr(gsv_server, "REPO_ROOT", root)
    monkeypatch.setattr(gsv_server, "_default_gsv_python", lambda: "python")

    command = _default_gsv_command("http://127.0.0.1:9880")
    config_path = root / ".cache" / "gpt_sovits" / "tts_infer.local.9880.yaml"
    config = yaml.safe_load(config_path.read_text("utf-8"))

    assert command[-2:] == ["-c", str(config_path)]
    assert isinstance(config, dict)
    assert config["custom"]["version"] == "v2ProPlus"
    assert not list(config_path.parent.glob("tts_infer.local.*.yaml.*.tmp"))


def test_default_gsv_command_uses_distinct_local_tts_config_per_port(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    install = root / ".cache" / "third_party" / "GPT-SoVITS"
    api = install / "api_v2.py"
    pretrained = root / ".cache" / "gpt_sovits" / "GPT_SoVITS" / "pretrained_models"
    for path in (
        pretrained / "s1v3.ckpt",
        pretrained / "v2Pro" / "s2Gv2ProPlus.pth",
        pretrained / "chinese-roberta-wwm-ext-large" / "config.json",
        pretrained / "chinese-hubert-base" / "config.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", "utf-8")
    api.parent.mkdir(parents=True, exist_ok=True)
    api.write_text("", "utf-8")
    monkeypatch.setattr(gsv_server, "REPO_ROOT", root)
    monkeypatch.setattr(gsv_server, "_default_gsv_python", lambda: "python")

    first = _default_gsv_command("http://127.0.0.1:9880")
    second = _default_gsv_command("http://127.0.0.1:9881")

    assert first[-1].endswith("tts_infer.local.9880.yaml")
    assert second[-1].endswith("tts_infer.local.9881.yaml")
    assert first[-1] != second[-1]
    assert Path(first[-1]).exists()
    assert Path(second[-1]).exists()


def test_patch_gsv_korean_text_preprocessor_disables_short_prefix(tmp_path: Path) -> None:
    install = tmp_path / "GPT-SoVITS"
    text_preprocessor = install / "GPT_SoVITS" / "TTS_infer_pack" / "TextPreprocessor.py"
    text_preprocessor.parent.mkdir(parents=True)
    text_preprocessor.write_text(
        """
def pre_seg_text(text, lang):
    if text[0] not in splits and len(get_first(text)) < 4:
        text = "。" + text if lang != "en" else "." + text
    return text
""".lstrip(),
        "utf-8",
    )

    assert _patch_gsv_korean_text_preprocessor(install) is True

    patched = text_preprocessor.read_text("utf-8")
    assert 'lang not in {"all_ko", "ko", "kr", "korean"}' in patched
    assert _patch_gsv_korean_text_preprocessor(install) is False


def test_managed_gsv_server_patches_korean_text_preprocessor_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install = tmp_path / "GPT-SoVITS"
    api = install / "api_v2.py"
    text_preprocessor = install / "GPT_SoVITS" / "TTS_infer_pack" / "TextPreprocessor.py"
    text_preprocessor.parent.mkdir(parents=True)
    api.write_text("", "utf-8")
    text_preprocessor.write_text(
        """
def pre_seg_text(text, lang):
    if text[0] not in splits and len(get_first(text)) < 4:
        text = "。" + text if lang != "en" else "." + text
    return text
""".lstrip(),
        "utf-8",
    )
    monkeypatch.setattr(gsv_server, "is_http_ready", lambda _base_url, **_kwargs: True)

    manager = ManagedGPTSoVITSServer(
        enabled=True,
        base_url="http://127.0.0.1:9880",
        command=[sys.executable, str(api)],
    )
    manager.start()

    assert manager.reused_existing is True
    assert 'lang not in {"all_ko", "ko", "kr", "korean"}' in text_preprocessor.read_text("utf-8")


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


def test_default_gsv_python_falls_back_to_current_venv_before_base(
    monkeypatch,
    tmp_path: Path,
) -> None:
    base = tmp_path / "base" / "bin" / "python"
    venv = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.delenv("ASMR_DUB_GSV_PYTHON", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(gsv_server.sys, "base_prefix", str(base.parent.parent))
    monkeypatch.setattr(gsv_server.sys, "executable", str(venv))
    monkeypatch.setattr(gsv_server.shutil, "which", lambda name: None)
    monkeypatch.setattr(gsv_server, "_python_has_modules", lambda _python, _modules: False)

    assert gsv_server._default_gsv_python() == str(venv)


def test_default_gsv_python_checks_api_startup_text_deps(monkeypatch, tmp_path: Path) -> None:
    override = tmp_path / "incomplete_gsv_env" / "bin" / "python"
    base = tmp_path / "base" / "bin" / "python"
    venv = tmp_path / ".venv" / "bin" / "python"
    monkeypatch.setenv("ASMR_DUB_GSV_PYTHON", str(override))
    monkeypatch.setattr(gsv_server.sys, "base_prefix", str(base.parent.parent))
    monkeypatch.setattr(gsv_server.sys, "executable", str(venv))
    monkeypatch.setattr(gsv_server.shutil, "which", lambda name: str(venv))

    def fake_has_modules(python: str, modules) -> bool:
        has_api_startup_deps = {"jieba", "fast_langdetect", "split_lang"}.issubset(modules)
        return python == str(base) and has_api_startup_deps

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


def test_gsv_subprocess_env_includes_python_nvrtc_lib_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    venv = tmp_path / "venv"
    nvrtc_dir = venv / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib"
    nvrtc_dir.mkdir(parents=True)
    (nvrtc_dir / "libnvrtc-builtins.so.13.0").write_bytes(b"")
    monkeypatch.setenv("VIRTUAL_ENV", str(venv))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    monkeypatch.setattr(gsv_server.sys, "prefix", str(venv))
    monkeypatch.setattr(gsv_server.sys, "base_prefix", str(tmp_path / "base"))

    env = _gsv_subprocess_env()

    ld_paths = env["LD_LIBRARY_PATH"].split(":")
    assert ld_paths[0] == str(nvrtc_dir)
    assert "/existing" in ld_paths
