from __future__ import annotations

import json

import pytest

from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.schemas import PipelineManifest, Segment


def test_segment_validates_duration() -> None:
    Segment(
        id="seg_0001",
        start=0,
        end=1,
        duration=1,
        audio_for_gemma="a.wav",
        audio_for_mix="b.wav",
    )
    with pytest.raises(ValueError):
        Segment(
            id="bad",
            start=1,
            end=0.5,
            duration=1,
            audio_for_gemma="a.wav",
            audio_for_mix="b.wav",
        )


def test_manifest_atomic_round_trip(tmp_project_dir) -> None:
    manifest = PipelineManifest()
    manifest.segments.append(
        Segment(
            id="seg_0001",
            start=0,
            end=1,
            duration=1,
            audio_for_gemma="a.wav",
            audio_for_mix="b.wav",
        )
    )
    path = save_manifest(tmp_project_dir, manifest)
    assert path.exists()
    text = path.read_text("utf-8")
    assert text.startswith("{\n")
    data = json.loads(text)
    assert data["schema_version"] == "1.0"
    loaded = load_manifest(tmp_project_dir)
    assert loaded.segments[0].id == "seg_0001"
