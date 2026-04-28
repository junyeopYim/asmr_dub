from __future__ import annotations

import json
import subprocess

import httpx
import pytest

from asmr_dub_pipeline.gemma.base import (
    GemmaResponseParseError,
    GemmaUnavailableError,
    create_gemma_backend,
)
from asmr_dub_pipeline.gemma.hf_client import HFGemmaBackend
from asmr_dub_pipeline.schemas import Segment


def sample_segment() -> Segment:
    return Segment(
        id="seg_0001",
        start=0,
        end=1,
        duration=1,
        audio_for_gemma="a.wav",
        audio_for_mix="b.wav",
    )


def test_mock_backend_deterministic(tiny_wav_path) -> None:
    backend = create_gemma_backend("mock")
    seg = sample_segment()
    first = backend.analyze_segment(tiny_wav_path, seg, {})
    second = backend.analyze_segment(tiny_wav_path, seg, {})
    assert first == second


def test_hf_backend_lazy_import_does_not_load_on_construct() -> None:
    backend = HFGemmaBackend()
    assert backend.model_id == "google/gemma-4-E4B-it"


def test_http_backend_uses_mock_transport(tiny_wav_path) -> None:
    from asmr_dub_pipeline.gemma.http_client import HTTPGemmaBackend

    payload = create_gemma_backend("mock").analyze_segment(tiny_wav_path, sample_segment(), {})
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    client = httpx.Client(transport=transport)
    backend = HTTPGemmaBackend("http://gemma.local", client=client)
    assert backend.analyze_segment(tiny_wav_path, sample_segment(), {})["source_language"] == "ja"


def test_http_backend_rejects_unstructured_json(tiny_wav_path) -> None:
    from asmr_dub_pipeline.gemma.http_client import HTTPGemmaBackend

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
    client = httpx.Client(transport=transport)
    backend = HTTPGemmaBackend("http://gemma.local", client=client, retries=0)
    with pytest.raises(GemmaResponseParseError):
        backend.analyze_segment(tiny_wav_path, sample_segment(), {})


def test_http_backend_repairs_text_json_after_parse_retry(monkeypatch, tiny_wav_path) -> None:
    from asmr_dub_pipeline.gemma import http_client as gemma_http_client
    from asmr_dub_pipeline.gemma.http_client import HTTPGemmaBackend

    monkeypatch.setattr(gemma_http_client.time, "sleep", lambda _: None)
    segment = sample_segment()
    payload = create_gemma_backend("mock").analyze_segment(tiny_wav_path, segment, {})
    repaired_body = json.dumps(payload, ensure_ascii=False)[:-1] + ",}"
    captured_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(json.loads(request.content))
        if len(captured_requests) == 1:
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="not json")
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text=f"Gemma result:\n```json\n{repaired_body}\n```\n",
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = HTTPGemmaBackend("http://gemma.local", client=client, retries=1)
    result = backend.analyze_segment(tiny_wav_path, segment, {"project": "contract"})

    assert result["transcript_original"] == payload["transcript_original"]
    assert result["source_language"] == "ja"
    assert len(captured_requests) == 2
    assert captured_requests[0]["task"] == "analyze"
    assert captured_requests[1]["segment"]["id"] == "seg_0001"  # type: ignore[index]


def test_llama_cpp_backend_invokes_mtmd_cli_and_parses_json(
    monkeypatch, tiny_wav_path, tmp_path
) -> None:
    from asmr_dub_pipeline.gemma import llama_cpp_client
    from asmr_dub_pipeline.gemma.llama_cpp_client import LlamaCppGemmaBackend

    cli = tmp_path / "llama-mtmd-cli"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    for path in (cli, model, mmproj):
        path.write_text("", "utf-8")
    payload = create_gemma_backend("mock").analyze_segment(tiny_wav_path, sample_segment(), {})
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"llama.cpp log\n{json.dumps(payload, ensure_ascii=False)}\n",
            stderr="",
        )

    monkeypatch.setattr(llama_cpp_client.subprocess, "run", fake_run)
    backend = LlamaCppGemmaBackend(
        cli_path=cli,
        model_path=model,
        mmproj_path=mmproj,
        timeout_sec=5,
        ctx_size=1024,
        n_predict=128,
        gpu_layers=0,
    )

    result = backend.analyze_segment(tiny_wav_path, sample_segment(), {})

    assert result["source_language"] == "ja"
    assert calls
    assert calls[0][:5] == [str(cli.resolve()), "-m", str(model.resolve()), "--mmproj", str(mmproj.resolve())]
    assert "--audio" in calls[0]
    assert str(tiny_wav_path.resolve()) in calls[0]
    assert "--json-schema" in calls[0]


def test_create_llama_cpp_backend_uses_config_paths(tmp_path) -> None:
    from asmr_dub_pipeline.gemma.llama_cpp_client import LlamaCppGemmaBackend

    cli = tmp_path / "llama-mtmd-cli"
    model = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    for path in (cli, model, mmproj):
        path.write_text("", "utf-8")

    backend = create_gemma_backend(
        "llama-cpp",
        {
            "llama_cpp_cli_path": str(cli),
            "llama_cpp_model_path": str(model),
            "llama_cpp_mmproj_path": str(mmproj),
            "llama_cpp_timeout_sec": 7,
        },
    )

    assert isinstance(backend, LlamaCppGemmaBackend)
    assert backend.cli_path == cli.resolve()
    assert backend.model_path == model.resolve()
    assert backend.mmproj_path == mmproj.resolve()
    assert backend.timeout_sec == 7


def test_unknown_backend_fails() -> None:
    with pytest.raises(GemmaUnavailableError):
        create_gemma_backend("wat")
