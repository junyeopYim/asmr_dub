from __future__ import annotations

import logging as py_logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()

_NOISY_HTTP_LOGGERS = ("httpx", "httpcore")


def quiet_http_client_logs() -> None:
    """Keep per-request httpx/httpcore logs out of normal pipeline output."""
    for logger_name in _NOISY_HTTP_LOGGERS:
        py_logging.getLogger(logger_name).setLevel(py_logging.WARNING)


def configure_logging(level: int = py_logging.INFO) -> None:
    quiet_http_client_logs()
    py_logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


quiet_http_client_logs()
