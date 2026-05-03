from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from asmr_dub_pipeline.config import load_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.schemas import PipelineManifest


@dataclass
class PipelineContext:
    """Mutable per-project state passed through pipeline stages."""

    project_dir: Path
    manifest: PipelineManifest

    @classmethod
    def load(cls, project_dir: Path) -> PipelineContext:
        resolved_project_dir = project_dir.expanduser().resolve()
        return cls(resolved_project_dir, load_manifest(resolved_project_dir))

    def reload_manifest(self) -> PipelineManifest:
        self.manifest = load_manifest(self.project_dir)
        return self.manifest

    def refresh_config(self) -> PipelineManifest:
        self.manifest.project_config = load_project_config(self.project_dir)
        return self.manifest

    def save_manifest(self, manifest: PipelineManifest | None = None) -> PipelineManifest:
        if manifest is not None:
            self.manifest = manifest
        save_manifest(self.project_dir, self.manifest)
        return self.manifest

    def update_manifest(self, manifest: PipelineManifest) -> PipelineManifest:
        self.manifest = manifest
        return manifest
