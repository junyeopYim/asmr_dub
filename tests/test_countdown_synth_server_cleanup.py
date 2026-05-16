from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock, get_ident

import pytest
from conftest import write_tiny_wav

from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gpt_sovits.client import build_tts_request
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.stages import synth_gpt_sovits
from asmr_dub_pipeline.pipeline.steps import countdown_synth_step, init_project
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
)


def _write_countdown_project(tmp_project_dir: Path) -> Path:
    init_project(tmp_project_dir)
    save_project_config(
        ProjectConfig(
            project_name="test",
            gsv_auto_start=True,
            gsv_concurrency=2,
            gsv_countdown_renderer="compact",
            gsv_countdown_candidate_count=1,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    refs_dir = tmp_project_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    write_tiny_wav(refs_dir / "whisper_close.wav", duration=3.0)
    refs_path = refs_dir / "refs.json"
    refs_path.write_text(
        json.dumps(
            {
                "whisper_close": {
                    "prompt_lang": "ja",
                    "prompt_text": "ソレジャー、ユックリカゾエマスネ。",
                    "ref_audio_path": "refs/whisper_close.wav",
                }
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "utf-8",
    )
    audio_dir = tmp_project_dir / "work" / "segments" / "audio"
    segments: list[Segment] = []
    for number, values in enumerate(([3, 2, 1], [2, 1, 0]), start=1):
        segment_id = f"seg_{number:04d}"
        audio_path = audio_dir / f"{segment_id}_mix.wav"
        write_tiny_wav(audio_path, duration=3.0)
        text = " ".join(str(value) for value in values)
        segments.append(
            Segment(
                id=segment_id,
                start=float((number - 1) * 3),
                end=float(number * 3),
                duration=3.0,
                audio_for_gemma=str(audio_path),
                audio_for_mix=str(audio_path),
                source_script=SourceScript(
                    text=text,
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=3.0,
                ),
                script=JapaneseScript(
                    ja_text=text,
                    tts_text=text,
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                    expected_tts_duration_sec=3.0,
                    ref_style="whisper_close",
                ),
                analysis={
                    "countdown_event": {
                        "kind": "descending_countdown",
                        "values": values,
                    }
                },
                status="scripted",
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))
    return refs_path


def test_countdown_synth_stops_started_gsv_server_when_later_lane_start_fails(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs_path = _write_countdown_project(tmp_project_dir)
    instances: list[object] = []
    stop_calls: list[int] = []

    class FakeManagedGPTSoVITSServer:
        def __init__(self, **kwargs: object) -> None:
            self.base_url = str(kwargs["base_url"])
            self.log_path = kwargs.get("log_path")
            self.index = len(instances)
            self.started = False
            self.reused_existing = False
            instances.append(self)

        def start(self) -> object:
            if self.index == 1:
                raise RuntimeError("lane start failed")
            self.started = True
            return self

        def stop(self) -> None:
            if self.started:
                stop_calls.append(self.index)
                self.started = False

    monkeypatch.setattr(
        synth_gpt_sovits,
        "ManagedGPTSoVITSServer",
        FakeManagedGPTSoVITSServer,
    )

    with pytest.raises(RuntimeError, match="lane start failed"):
        countdown_synth_step(
            tmp_project_dir,
            gsv_url="http://127.0.0.1:9880",
            refs_path=refs_path,
            mock=False,
            confirm_rights=True,
            auto_gsv_server=True,
        )

    assert stop_calls == [0]


def test_countdown_synth_starts_gsv_lanes_in_parallel(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refs_path = _write_countdown_project(tmp_project_dir)
    start_barrier = Barrier(2)
    request_barrier = Barrier(2)
    starts_lock = Lock()
    requests_lock = Lock()
    instances: list[object] = []
    start_threads: set[int] = set()
    start_urls: list[str] = []
    request_threads: set[int] = set()
    request_urls: list[str] = []

    class FakeManagedGPTSoVITSServer:
        def __init__(self, **kwargs: object) -> None:
            self.base_url = str(kwargs["base_url"])
            self.log_path = kwargs.get("log_path")
            self.index = len(instances)
            self.started = False
            self.reused_existing = False
            instances.append(self)

        def start(self) -> object:
            with starts_lock:
                start_threads.add(get_ident())
                start_urls.append(self.base_url)
            try:
                start_barrier.wait(timeout=0.5)
            except BrokenBarrierError as exc:
                raise AssertionError("GSV lanes did not start concurrently") from exc
            self.started = True
            return self

        def stop(self) -> None:
            self.started = False

    class FakeClient:
        def __init__(self, base_url: str, *args: object, **kwargs: object) -> None:
            self.base_url = base_url

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            with requests_lock:
                request_threads.add(get_ident())
                request_urls.append(self.base_url)
            try:
                request_barrier.wait(timeout=0.5)
            except BrokenBarrierError as exc:
                raise AssertionError("countdown-synth did not use GSV lanes concurrently") from exc
            write_tiny_wav(output_path, duration=0.45)
            return output_path

    monkeypatch.setattr(
        synth_gpt_sovits,
        "ManagedGPTSoVITSServer",
        FakeManagedGPTSoVITSServer,
    )
    monkeypatch.setattr(synth_gpt_sovits, "GPTSoVITSClient", FakeClient)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=refs_path,
        mock=False,
        confirm_rights=True,
        auto_gsv_server=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    server_state = manifest.stage_state["countdown-synth"]["gsv_server"]
    assert len(start_threads) == 2
    assert start_urls == ["http://127.0.0.1:9880", "http://127.0.0.1:9881"]
    assert len(request_threads) == 2
    assert set(request_urls) == {"http://127.0.0.1:9880", "http://127.0.0.1:9881"}
    assert server_state["concurrency"] == 2
    assert [instance["base_url"] for instance in server_state["instances"]] == start_urls
