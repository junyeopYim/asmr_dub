from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from conftest import write_tiny_wav

from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.qwen_tts.client import QwenTTSClient, QwenTTSRequest, qwen_language
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    RVCMetadata,
    Segment,
    TTSMetadata,
)


def _scripted_segment(project_dir: Path, *, selected_tts: Path | None = None) -> Segment:
    audio = project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"placeholder")
    return Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        status="ok",
        script=JapaneseScript(
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.2,
        ),
        tts=TTSMetadata(
            backend="gpt-sovits",
            selected_candidate_path=str(selected_tts) if selected_tts else None,
        )
        if selected_tts
        else None,
    )


def test_qwen_language_maps_pipeline_codes() -> None:
    assert qwen_language("ko") == "Korean"
    assert qwen_language("ja") == "Japanese"
    assert qwen_language("auto") == "Auto"


def test_qwen_tts_client_uses_voice_clone_prompt_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            calls.append(("from_pretrained", (model_id, kwargs)))
            return cls()

        def create_voice_clone_prompt(self, **kwargs: object) -> object:
            calls.append(("prompt", kwargs))
            return {"cached": True}

        def generate_voice_clone(self, **kwargs: object):
            calls.append(("generate", kwargs))
            return [np.zeros(2400, dtype=np.float32)], 24_000

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(bfloat16="bf16", manual_seed=lambda seed: calls.append(("seed", seed))),
    )
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeModel))
    client = QwenTTSClient(model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_files_only=False)
    request = QwenTTSRequest(
        text="안녕하세요",
        language="Korean",
        ref_audio_path="ref.wav",
        ref_text="こんにちは",
        seed=123,
    )

    output = tmp_path / "qwen.wav"
    result = client.synthesize_to_file(request, output)

    assert result.sample_rate == 24_000
    assert output.exists()
    generate_call = next(value for name, value in calls if name == "generate")
    assert isinstance(generate_call, dict)
    assert generate_call["voice_clone_prompt"] == {"cached": True}


def test_synth_qwen_compare_only_records_candidates_without_replacing_tts(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    steps.init_project(tmp_project_dir)
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav")
    old_tts = write_tiny_wav(tmp_project_dir / "work" / "tts" / "seg_0001_final.wav")
    save_manifest(tmp_project_dir, PipelineManifest(segments=[_scripted_segment(tmp_project_dir, selected_tts=old_tts)]))
    requests: list[QwenTTSRequest] = []

    class FakeQwenClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def synthesize_to_file(self, request: QwenTTSRequest, output_path: Path):
            requests.append(request)
            write_tiny_wav(output_path)
            return SimpleNamespace(sample_rate=48_000)

    monkeypatch.setattr(steps, "QwenTTSClient", FakeQwenClient)

    manifest = steps.synth_qwen_step(
        tmp_project_dir,
        Path("refs/refs.json"),
        confirm_rights=True,
        candidate_count=2,
        promote=False,
        local_files_only=False,
    )

    segment = manifest.segments[0]
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path == str(old_tts)
    assert segment.analysis["qwen_tts"]["selected_candidate_path"].endswith("_qwen_best.wav")
    assert len(segment.analysis["qwen_tts"]["candidates"]) == 2
    assert requests[0].language == "Korean"
    assert manifest.stage_state["synth-qwen"]["status"] == "completed"


def test_synth_qwen_promote_replaces_tts_and_invalidates_downstream(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    steps.init_project(tmp_project_dir)
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav")
    old_tts = write_tiny_wav(tmp_project_dir / "work" / "tts" / "seg_0001_final.wav")
    segment = _scripted_segment(tmp_project_dir, selected_tts=old_tts)
    segment.rvc = RVCMetadata(
        backend="mock",
        input_path=str(old_tts),
        output_path=str(tmp_project_dir / "work" / "rvc" / "seg_0001_final.wav"),
        accepted=True,
    )
    manifest = PipelineManifest(
        segments=[segment],
        stage_state={
            "rvc": {"status": "completed"},
            "qc": {"status": "completed"},
            "mix": {"status": "completed"},
        },
    )
    save_manifest(tmp_project_dir, manifest)

    class FakeQwenClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def synthesize_to_file(self, request: QwenTTSRequest, output_path: Path):
            write_tiny_wav(output_path)
            return SimpleNamespace(sample_rate=48_000)

    monkeypatch.setattr(steps, "QwenTTSClient", FakeQwenClient)

    steps.synth_qwen_step(
        tmp_project_dir,
        Path("refs/refs.json"),
        confirm_rights=True,
        candidate_count=1,
        promote=True,
        local_files_only=False,
    )
    promoted = load_manifest(tmp_project_dir)
    promoted_segment = promoted.segments[0]

    assert promoted_segment.tts is not None
    assert promoted_segment.tts.backend == "qwen-tts"
    assert promoted_segment.tts.selected_candidate_path.endswith("work/tts/seg_0001_final.wav")
    assert promoted_segment.rvc is None
    assert promoted_segment.qc is None
    assert promoted_segment.status == "synthesized"
    assert "rvc" not in promoted.stage_state
    assert "qc" not in promoted.stage_state
    assert "mix" not in promoted.stage_state
