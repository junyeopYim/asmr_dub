from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import soundfile as sf
from conftest import write_tiny_wav

from asmr_dub_pipeline.experimental_tts import (
    CosyVoiceTTSClient,
    ExperimentalTTSRequest,
    ExperimentalTTSResult,
    FishSpeechTTSClient,
)
from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.schemas import JapaneseScript, PipelineManifest, Segment, TTSMetadata


def _scripted_segment(project_dir: Path, *, selected_tts: Path | None = None) -> Segment:
    audio = project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
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


def test_fish_speech_client_posts_reference_audio_and_writes_wav(
    tmp_path: Path,
    tiny_wav_bytes: bytes,
) -> None:
    ref = write_tiny_wav(tmp_path / "ref.wav")
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content=tiny_wav_bytes, headers={"content-type": "audio/wav"})

    client = FishSpeechTTSClient(
        base_url="http://fish.local",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = ExperimentalTTSRequest(
        text="안녕하세요",
        language="ko",
        ref_audio_path=str(ref),
        ref_text="こんにちは",
        seed=123,
        generation_kwargs={"top_p": 0.9},
    )

    result = client.synthesize_to_file(request, tmp_path / "fish.wav")

    assert seen["url"] == "http://fish.local/v1/tts"
    payload = seen["payload"]
    assert isinstance(payload, dict)
    assert payload["text"] == "안녕하세요"
    assert payload["seed"] == 123
    assert payload["top_p"] == 0.9
    assert payload["references"][0]["text"] == "こんにちは"
    assert result.sample_rate == 48_000


def test_cosyvoice_client_posts_zero_shot_prompt_and_wraps_pcm(
    tmp_path: Path,
) -> None:
    ref = write_tiny_wav(tmp_path / "ref.wav")
    pcm = np.zeros(2205, dtype=np.int16).tobytes()
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        body = request.read()
        seen["body"] = body
        return httpx.Response(200, content=pcm)

    client = CosyVoiceTTSClient(
        base_url="http://cosy.local",
        sample_rate=22_050,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    request = ExperimentalTTSRequest(
        text="안녕하세요",
        language="ko",
        ref_audio_path=str(ref),
        ref_text="こんにちは",
        seed=123,
    )

    result = client.synthesize_to_file(request, tmp_path / "cosy.wav")

    assert seen["url"] == "http://cosy.local/inference_zero_shot"
    assert b'name="tts_text"' in seen["body"]
    assert b'name="prompt_text"' in seen["body"]
    assert result.sample_rate == 22_050
    assert sf.info(result.output_path).samplerate == 22_050


def test_synth_experimental_tts_records_fish_candidates(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    steps.init_project(tmp_project_dir)
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav")
    old_tts = write_tiny_wav(tmp_project_dir / "work" / "tts" / "seg_0001_final.wav")
    save_manifest(
        tmp_project_dir,
        PipelineManifest(segments=[_scripted_segment(tmp_project_dir, selected_tts=old_tts)]),
    )
    requests: list[ExperimentalTTSRequest] = []

    class FakeFishClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def load_model(self) -> None:
            return None

        def synthesize_to_file(self, request: ExperimentalTTSRequest, output_path: Path):
            requests.append(request)
            write_tiny_wav(output_path)
            return ExperimentalTTSResult(output_path=output_path, sample_rate=48_000)

    monkeypatch.setattr(steps, "FishSpeechTTSClient", FakeFishClient)

    manifest = steps.synth_experimental_tts_step(
        tmp_project_dir,
        Path("refs/refs.json"),
        backend="fish",
        confirm_rights=True,
        candidate_count=2,
        promote=False,
    )

    segment = manifest.segments[0]
    assert segment.tts is not None
    assert segment.tts.selected_candidate_path == str(old_tts)
    assert segment.analysis["fish_tts"]["selected_candidate_path"].endswith("_fish_best.wav")
    assert len(segment.analysis["fish_tts"]["candidates"]) == 2
    assert requests[0].text == "안녕하세요"
    assert manifest.stage_state["synth-fish"]["status"] == "completed"


def test_synth_experimental_tts_promote_cosyvoice_invalidates_downstream(
    tmp_project_dir: Path,
    monkeypatch,
) -> None:
    steps.init_project(tmp_project_dir)
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav")
    old_tts = write_tiny_wav(tmp_project_dir / "work" / "tts" / "seg_0001_final.wav")
    manifest = PipelineManifest(
        segments=[_scripted_segment(tmp_project_dir, selected_tts=old_tts)],
        stage_state={"rvc": {"status": "completed"}, "qc": {"status": "completed"}},
    )
    save_manifest(tmp_project_dir, manifest)

    class FakeCosyClient:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

        def load_model(self) -> None:
            return None

        def synthesize_to_file(self, request: ExperimentalTTSRequest, output_path: Path):
            write_tiny_wav(output_path)
            return SimpleNamespace(output_path=output_path, sample_rate=22_050)

    monkeypatch.setattr(steps, "CosyVoiceTTSClient", FakeCosyClient)

    promoted = steps.synth_experimental_tts_step(
        tmp_project_dir,
        Path("refs/refs.json"),
        backend="cosyvoice",
        confirm_rights=True,
        candidate_count=1,
        promote=True,
    )

    segment = promoted.segments[0]
    assert segment.tts is not None
    assert segment.tts.backend == "cosyvoice"
    assert segment.status == "synthesized"
    assert segment.rvc is None
    assert "rvc" not in promoted.stage_state
    assert "qc" not in promoted.stage_state
