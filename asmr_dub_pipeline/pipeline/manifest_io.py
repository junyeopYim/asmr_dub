from __future__ import annotations

import json
import os
from pathlib import Path

from asmr_dub_pipeline.schemas import SCHEMA_VERSION, PipelineManifest


class ManifestError(RuntimeError):
    pass


def manifest_path(project_dir: Path | str) -> Path:
    return Path(project_dir).expanduser().resolve() / "work" / "manifest.json"


def load_manifest(project_dir: Path | str) -> PipelineManifest:
    path = manifest_path(project_dir)
    if not path.exists():
        return PipelineManifest()
    try:
        data = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest is not valid JSON: {path}") from exc
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported manifest schema_version {data.get('schema_version')!r}; expected {SCHEMA_VERSION!r}."
        )
    return PipelineManifest.model_validate(data)


def save_manifest(project_dir: Path | str, manifest: PipelineManifest) -> Path:
    path = manifest_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.mark_updated()
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload + "\n", "utf-8")
    os.replace(tmp, path)
    return path


def write_json_atomic(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload + "\n", "utf-8")
    os.replace(tmp, path)
    return path
