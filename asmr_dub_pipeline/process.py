from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def tail_text(text: str | bytes | None, limit: int = 1200) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text.strip()[-limit:]


def tail_file(path: Path | None, max_chars: int = 2000) -> str:
    if path is None:
        return ""
    try:
        text = path.read_text("utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def popen_process_group_kwargs() -> dict[str, bool]:
    return {"start_new_session": True} if os.name != "nt" else {}


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(pgid: int, timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while time.monotonic() < deadline:
        if not _process_group_exists(pgid):
            return True
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return not _process_group_exists(pgid)


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    terminate_timeout_sec: float = 5.0,
    kill_timeout_sec: float = 5.0,
) -> None:
    if os.name != "nt":
        if process.poll() is not None and not _process_group_exists(process.pid):
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if _wait_for_process_group_exit(process.pid, terminate_timeout_sec):
            return

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        _wait_for_process_group_exit(process.pid, kill_timeout_sec)
        return

    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=terminate_timeout_sec)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        process.kill()
    except ProcessLookupError:
        return
    process.wait(timeout=kill_timeout_sec)
