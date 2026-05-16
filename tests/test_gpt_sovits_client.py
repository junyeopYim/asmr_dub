from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSClient, GPTSoVITSError, build_tts_request
from asmr_dub_pipeline.gpt_sovits.refs import resolve_ref
from asmr_dub_pipeline.gpt_sovits.schemas import GPTSoVITSRef, GPTSoVITSTTSOptions


def test_payload_contains_required_fields() -> None:
    client = GPTSoVITSClient("http://example.invalid")
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは")
    payload = client.build_payload("テスト", ref, GPTSoVITSTTSOptions(seed=7, speed_factor=1.1))
    data = payload.as_payload()
    assert data["text"] == "テスト"
    assert data["text_lang"] == "all_ja"
    assert data["prompt_text"] == "こんにちわ"
    assert data["prompt_lang"] == "all_ja"
    assert data["seed"] == 7
    assert data["speed_factor"] == 1.1


def test_payload_can_request_korean_text_language() -> None:
    client = GPTSoVITSClient("http://example.invalid")
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは", prompt_lang="ja")

    payload = client.build_payload(
        "천천히 숨을 쉬어 주세요.",
        ref,
        GPTSoVITSTTSOptions(text_lang="ko"),
    )

    data = payload.as_payload()
    assert data["text"] == "천천히 숨을 쉬어 주세요."
    assert data["text_lang"] == "all_ko"
    assert data["prompt_lang"] == "all_ja"

    alias_payload = client.build_payload("괜찮아요.", ref, GPTSoVITSTTSOptions(text_lang="kr"))
    assert alias_payload.as_payload()["text_lang"] == "all_ko"


def test_korean_text_rejects_japanese_text_language() -> None:
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは", prompt_lang="ja")

    with pytest.raises(GPTSoVITSError, match="all_ko"):
        build_tts_request(
            "조금 더 가까이 갈게요.",
            ref,
            GPTSoVITSTTSOptions(text_lang="ja"),
        )


def test_tts_contract_posts_api_v2_payload_without_real_server(
    tmp_path: Path, tiny_wav_bytes: bytes
) -> None:
    captured_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/tts"
        captured_payloads.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=tiny_wav_bytes)

    client = GPTSoVITSClient("http://gsv.local/", transport=httpx.MockTransport(handler), retries=0)
    ref = GPTSoVITSRef(
        ref_audio_path="/project/refs/main.wav",
        prompt_text="近くで囁きます",
        prompt_lang="ja",
        aux_ref_audio_paths=["/project/refs/aux-left.wav"],
    )
    options = GPTSoVITSTTSOptions(
        top_k=8,
        top_p=0.9,
        temperature=0.75,
        text_split_method="cut0",
        batch_size=2,
        batch_threshold=0.5,
        split_bucket=False,
        speed_factor=0.96,
        fragment_interval=0.12,
        seed=1234,
        parallel_infer=False,
        repetition_penalty=1.1,
        sample_steps=16,
        super_sampling=True,
        overlap_length=3,
        min_chunk_length=10,
    )

    request = client.build_payload("ゆっくり……おやすみなさい。", ref, options)
    client.synthesize_to_file(request, tmp_path / "contract.wav")

    expected_payload = {
        "text": "ゆっくり……おやすみなさい。",
        "text_lang": "all_ja",
        "ref_audio_path": "/project/refs/main.wav",
        "prompt_text": "ちかくでささやきます",
        "prompt_lang": "all_ja",
        "aux_ref_audio_paths": ["/project/refs/aux-left.wav"],
        "top_k": 8,
        "top_p": 0.9,
        "temperature": 0.75,
        "text_split_method": "cut0",
        "batch_size": 2,
        "batch_threshold": 0.5,
        "split_bucket": False,
        "speed_factor": 0.96,
        "fragment_interval": 0.12,
        "seed": 1234,
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": False,
        "repetition_penalty": 1.1,
        "sample_steps": 16,
        "super_sampling": True,
        "overlap_length": 3,
        "min_chunk_length": 10,
    }
    assert captured_payloads == [expected_payload]
    assert request.as_payload() == expected_payload
    assert ref.aux_ref_audio_paths == ["/project/refs/aux-left.wav"]


def test_synthesize_writes_wav_bytes(tmp_path: Path, tiny_wav_bytes: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tts"
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=tiny_wav_bytes)

    transport = httpx.MockTransport(handler)
    client = GPTSoVITSClient("http://gsv.local", transport=transport)
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは")
    request = client.build_payload("テスト", ref)
    out = tmp_path / "out.wav"
    client.synthesize_to_file(request, out)
    assert out.read_bytes() == tiny_wav_bytes


def test_json_error_is_helpful(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(400, json={"message": "text is required"})
    )
    client = GPTSoVITSClient("http://gsv.local", transport=transport, retries=0)
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは")
    with pytest.raises(GPTSoVITSError, match="text is required"):
        client.synthesize_to_file(client.build_payload("テスト", ref), tmp_path / "out.wav")


def test_200_json_is_not_saved_as_wav(tmp_path: Path) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"message": "not audio"}))
    client = GPTSoVITSClient("http://gsv.local", transport=transport, retries=0)
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは")
    with pytest.raises(GPTSoVITSError, match="JSON instead of WAV"):
        client.synthesize_to_file(client.build_payload("テスト", ref), tmp_path / "out.wav")


def test_200_plain_text_is_not_saved_as_wav(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, content=b"not audio")
    )
    client = GPTSoVITSClient("http://gsv.local", transport=transport, retries=0)
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは")
    with pytest.raises(GPTSoVITSError, match="non-WAV"):
        client.synthesize_to_file(client.build_payload("テスト", ref), tmp_path / "out.wav")


def test_retry_then_success_writes_wav(tmp_path: Path, tiny_wav_bytes: bytes) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(500, text="busy")
        return httpx.Response(200, headers={"content-type": "audio/wav"}, content=tiny_wav_bytes)

    client = GPTSoVITSClient("http://gsv.local", transport=httpx.MockTransport(handler), backoff_sec=0)
    ref = GPTSoVITSRef(ref_audio_path="refs/a.wav", prompt_text="こんにちは")
    out = tmp_path / "out.wav"
    client.synthesize_to_file(client.build_payload("テスト", ref), out)
    assert calls == 2
    assert out.exists()


def test_set_weight_helpers() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"success")

    client = GPTSoVITSClient("http://gsv.local", transport=httpx.MockTransport(handler))
    assert client.set_gpt_weights("a.ckpt") == "success"
    assert client.set_sovits_weights("b.pth") == "success"
    assert seen == ["/set_gpt_weights", "/set_sovits_weights"]


def test_set_weight_helpers_accept_json_success() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"message": "success"}))
    client = GPTSoVITSClient("http://gsv.local", transport=transport)

    assert client.set_gpt_weights("a.ckpt") == "success"


def test_set_weight_cuda_illegal_memory_error_has_restart_hint() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            400,
            json={
                "message": "change gpt weight failed",
                "Exception": "CUDA error: an illegal memory access was encountered",
            },
        )
    )
    client = GPTSoVITSClient("http://gsv.local", transport=transport)

    with pytest.raises(GPTSoVITSError, match="restart the GPT-SoVITS api_v2 process"):
        client.set_gpt_weights("a.ckpt")


def test_resolve_ref_fallback() -> None:
    refs = {"whisper_close": GPTSoVITSRef(ref_audio_path="a.wav", prompt_text="x")}
    assert resolve_ref(refs, "missing").ref_audio_path == "a.wav"
