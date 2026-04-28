from __future__ import annotations

import json

import pytest

from asmr_dub_pipeline.config import create_project_structure
from asmr_dub_pipeline.gpt_sovits.refs import GPTSoVITSRefsError, load_refs
from asmr_dub_pipeline.pipeline.steps import analyze_step
from asmr_dub_pipeline.rights import RightsError, ensure_not_same_path


def test_refs_json_must_stay_inside_project(tmp_path) -> None:
    project = tmp_path / "project"
    create_project_structure(project)
    outside = tmp_path / "refs.json"
    outside.write_text("{}", "utf-8")
    with pytest.raises(GPTSoVITSRefsError):
        load_refs(outside, project_dir=project)


def test_hardlinked_output_is_rejected(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"data")
    linked = tmp_path / "linked.wav"
    linked.hardlink_to(source)
    with pytest.raises(RightsError):
        ensure_not_same_path(source, linked)


def test_ref_audio_path_must_stay_inside_project(tmp_path) -> None:
    project = tmp_path / "project"
    create_project_structure(project)
    refs = project / "refs" / "bad.json"
    refs.write_text(
        json.dumps({"bad": {"ref_audio_path": "../voice.wav", "prompt_text": "x", "prompt_lang": "ja"}}),
        "utf-8",
    )
    with pytest.raises(GPTSoVITSRefsError):
        load_refs(refs, project_dir=project)


def test_aux_ref_audio_paths_must_stay_inside_project(tmp_path) -> None:
    project = tmp_path / "project"
    create_project_structure(project)
    refs = project / "refs" / "bad_aux.json"
    refs.write_text(
        json.dumps(
            {
                "bad": {
                    "ref_audio_path": "refs/voice.wav",
                    "prompt_text": "x",
                    "prompt_lang": "ja",
                    "aux_ref_audio_paths": ["../aux.wav"],
                }
            }
        ),
        "utf-8",
    )
    with pytest.raises(GPTSoVITSRefsError):
        load_refs(refs, project_dir=project)


def test_ref_audio_paths_are_normalized_inside_project(tmp_path) -> None:
    project = tmp_path / "project"
    create_project_structure(project)
    refs = project / "refs" / "ok.json"
    refs.write_text(
        json.dumps(
            {
                "ok": {
                    "ref_audio_path": "refs/voice.wav",
                    "prompt_text": "x",
                    "prompt_lang": "ja",
                    "aux_ref_audio_paths": ["refs/aux.wav"],
                }
            }
        ),
        "utf-8",
    )
    loaded = load_refs(refs, project_dir=project)
    assert loaded["ok"].ref_audio_path == str((project / "refs/voice.wav").resolve())
    assert loaded["ok"].aux_ref_audio_paths == [str((project / "refs/aux.wav").resolve())]


def test_http_gemma_requires_existing_rights_audit(tmp_project_dir) -> None:
    create_project_structure(tmp_project_dir)
    config = tmp_project_dir / "pipeline.yaml"
    config.write_text(
        config.read_text("utf-8").replace("gemma_http_url: null", "gemma_http_url: http://gemma.local"),
        "utf-8",
    )
    with pytest.raises(RightsError):
        analyze_step(tmp_project_dir, "http")
