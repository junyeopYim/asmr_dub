from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from asmr_dub_pipeline.audio.features import write_audio
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import countdown_synth_step, synth_step
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
)


def _write_ref_fixture(project_dir: Path) -> Path:
    ref_audio = project_dir / "refs" / "ref.wav"
    samples = np.zeros((12_000, 2), dtype=np.float32)
    samples[:, 0] = np.sin(np.linspace(0.0, 180.0, len(samples), dtype=np.float32)) * 0.03
    samples[:, 1] = samples[:, 0]
    write_audio(ref_audio, samples, 24_000)
    refs_path = project_dir / "refs" / "refs.json"
    refs_path.write_text(
        json.dumps(
            {
                "whisper_close": {
                    "ref_audio_path": "refs/ref.wav",
                    "prompt_text": "テストです",
                    "prompt_lang": "ja",
                }
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        "utf-8",
    )
    return refs_path


def _embedded_countdown_segment() -> Segment:
    return Segment(
        id="seg_embed",
        start=10.0,
        end=15.0,
        duration=5.0,
        status="scripted",
        audio_for_gemma="work/segments/seg_embed_gemma.wav",
        audio_for_mix="work/segments/seg_embed_mix.wav",
        source_script=SourceScript(
            text="じゃあ 4 3 2 1 ゼロ いきます",
            language="ja",
            backend="mock",
            start=10.0,
            end=15.0,
        ),
        script=JapaneseScript(
            literal_ja="じゃあ 4 3 2 1 ゼロ いきます",
            ja_text="じゃあ 4 3 2 1 ゼロ いきます",
            tts_text="자, 사, 삼, 이, 일, 영, 갑니다.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=5.0,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "embedded_countdown",
                "values": [4, 3, 2, 1, 0],
                "token_timeline": [
                    {"value": 4, "source_text": "4", "start": 10.5, "end": 10.8},
                    {"value": 3, "source_text": "3", "start": 11.1, "end": 11.4},
                    {"value": 2, "source_text": "2", "start": 11.7, "end": 12.0},
                    {"value": 1, "source_text": "1", "start": 12.3, "end": 12.6},
                    {"value": 0, "source_text": "ゼロ", "start": 12.9, "end": 13.2},
                ],
            }
        },
    )


def _save_project(project_dir: Path) -> Path:
    cfg = ProjectConfig(project_name="embedded-hybrid")
    cfg.gsv.pronunciation_qc_enabled = False
    save_project_config(cfg, project_dir / "pipeline.yaml")
    refs_path = _write_ref_fixture(project_dir)
    manifest = PipelineManifest(project_config=cfg, segments=[_embedded_countdown_segment()])
    manifest.stage_state["korean-script"] = {"status": "completed"}
    save_manifest(project_dir, manifest)
    return refs_path


def test_countdown_synth_renders_embedded_countdown_bed_without_promoting_segment(
    tmp_path: Path,
) -> None:
    refs_path = _save_project(tmp_path)

    countdown_synth_step(tmp_path, None, refs_path, mock=True, confirm_rights=True)

    segment = load_manifest(tmp_path).segments[0]
    hybrid = segment.analysis.get("embedded_countdown_hybrid_renderer")
    assert segment.status == "scripted"
    assert segment.tts is None
    assert isinstance(hybrid, dict)
    assert hybrid["status"] == "rendered"
    assert hybrid["tokens"] == ["사", "삼", "이", "일", "영"]
    assert Path(hybrid["bed_path"]).exists()


def test_synth_applies_embedded_countdown_hybrid_overlay_to_selected_candidate(
    tmp_path: Path,
) -> None:
    refs_path = _save_project(tmp_path)

    synth_step(
        tmp_path,
        None,
        refs_path,
        mock=True,
        confirm_rights=True,
        render_countdowns=False,
    )

    segment = load_manifest(tmp_path).segments[0]
    assert segment.status == "synthesized"
    assert segment.script is not None
    assert segment.script.tts_text == "자, 사, 삼, 이, 일, 영, 갑니다."
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path is not None
    assert Path(segment.tts.selected_candidate_path).exists()
    hybrid = segment.analysis.get("embedded_countdown_hybrid_renderer")
    assert isinstance(hybrid, dict)
    assert hybrid["status"] == "rendered"
    selected = [candidate for candidate in segment.tts.candidates if candidate.selected]
    assert len(selected) == 1
    assert selected[0].selection_reason == "embedded_countdown_hybrid_overlay"
    assert selected[0].duration_sec == segment.duration
    assert selected[0].payload["embedded_countdown_hybrid"]["bed_path"] == hybrid["bed_path"]
    assert selected[0].payload["embedded_countdown_hybrid"]["base_ducking"]["regions"]
