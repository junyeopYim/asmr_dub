from __future__ import annotations

import os
import signal
import subprocess
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


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    terminate_timeout_sec: float = 5.0,
    kill_timeout_sec: float = 5.0,
) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=terminate_timeout_sec)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        return
    process.wait(timeout=kill_timeout_sec)
