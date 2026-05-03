from __future__ import annotations

# ruff: noqa: F403,F405,I001

from asmr_dub_pipeline.pipeline.stages.common import *


def init_project(project_dir: Path) -> None:
    create_project_structure(project_dir)


def inspect_input(input_path: Path) -> Any:
    return probe_with_fallback(input_path)
