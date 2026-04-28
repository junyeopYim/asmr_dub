from __future__ import annotations

import logging as py_logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def configure_logging(level: int = py_logging.INFO) -> None:
    py_logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
