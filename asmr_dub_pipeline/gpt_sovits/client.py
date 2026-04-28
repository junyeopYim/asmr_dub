from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx

from .schemas import GPTSoVITSRef, GPTSoVITSTTSOptions, GPTSoVITSTTSRequest


class GPTSoVITSError(RuntimeError):
    pass


def normalize_api_language_code(language: str) -> str:
    normalized = language.strip().lower().replace("-", "_")
    if normalized in {"ja", "jp", "jpn", "japanese"}:
        return "all_ja"
    if normalized in {"ko", "kr", "kor", "korean"}:
        return "all_ko"
    if normalized in {"zh", "cn", "zho", "chinese", "mandarin"}:
        return "all_zh"
    return language.strip() or "all_ja"


def build_tts_request(
    text: str,
    ref: GPTSoVITSRef,
    options: GPTSoVITSTTSOptions | None = None,
) -> GPTSoVITSTTSRequest:
    """Build the deterministic api_v2 /tts JSON payload."""
    if not text.strip():
        raise GPTSoVITSError("GPT-SoVITS text must not be empty.")
    if not ref.ref_audio_path.strip():
        raise GPTSoVITSError("GPT-SoVITS ref_audio_path must not be empty.")
    options = options or GPTSoVITSTTSOptions()
    return GPTSoVITSTTSRequest(
        text=text,
        text_lang=normalize_api_language_code(options.text_lang),
        ref_audio_path=ref.ref_audio_path,
        prompt_text=ref.prompt_text,
        prompt_lang=normalize_api_language_code(ref.prompt_lang or "ja"),
        aux_ref_audio_paths=list(ref.aux_ref_audio_paths),
        top_k=options.top_k,
        top_p=options.top_p,
        temperature=options.temperature,
        text_split_method=options.text_split_method,
        batch_size=options.batch_size,
        batch_threshold=options.batch_threshold,
        split_bucket=options.split_bucket,
        speed_factor=options.speed_factor,
        fragment_interval=options.fragment_interval,
        seed=options.seed,
        media_type="wav",
        streaming_mode=options.streaming_mode,
        parallel_infer=options.parallel_infer,
        repetition_penalty=options.repetition_penalty,
        sample_steps=options.sample_steps,
        super_sampling=options.super_sampling,
        overlap_length=options.overlap_length,
        min_chunk_length=options.min_chunk_length,
    )


class GPTSoVITSClient:
    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 120.0,
        retries: int = 2,
        backoff_sec: float = 0.5,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.backoff_sec = backoff_sec
        self.client = client or httpx.Client(timeout=timeout_sec, transport=transport)

    def build_payload(
        self,
        text: str,
        ref: GPTSoVITSRef,
        options: GPTSoVITSTTSOptions | None = None,
    ) -> GPTSoVITSTTSRequest:
        return build_tts_request(text, ref, options)

    def _check_response(self, endpoint: str, response: httpx.Response, expect_wav: bool = False) -> bytes:
        content_type = response.headers.get("content-type", "")
        body = response.content
        if response.status_code >= 400:
            message = response.text
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    primary = str(parsed.get("message") or parsed)
                    detail = parsed.get("Exception")
                    message = f"{primary}: {detail}" if detail else primary
            except ValueError:
                pass
            raise GPTSoVITSError(f"{endpoint} failed with HTTP {response.status_code}: {message}")
        if "json" in content_type.lower():
            try:
                parsed = response.json()
            except ValueError as exc:
                raise GPTSoVITSError(f"{endpoint} returned invalid JSON error body") from exc
            raise GPTSoVITSError(f"{endpoint} returned JSON instead of WAV: {parsed}")
        if not body:
            raise GPTSoVITSError(f"{endpoint} returned an empty response")
        if expect_wav and "audio" not in content_type.lower() and not body.startswith(b"RIFF"):
            raise GPTSoVITSError(
                f"{endpoint} returned non-WAV content-type {content_type or '<missing>'}"
            )
        return body

    def synthesize_to_file(self, request: GPTSoVITSTTSRequest, output_path: Path) -> Path:
        endpoint = "/tts"
        payload = request.as_payload()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(f"{self.base_url}{endpoint}", json=payload)
                if response.status_code >= 500 and attempt < self.retries:
                    time.sleep(self.backoff_sec * (attempt + 1))
                    continue
                data = self._check_response(endpoint, response, expect_wav=True)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = output_path.with_suffix(output_path.suffix + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, output_path)
                return output_path
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.backoff_sec * (attempt + 1))
                    continue
                break
        raise GPTSoVITSError(f"{endpoint} request failed after retries: {last_error}")

    def _set_weights(self, endpoint: str, weights_path: str) -> str:
        response = self.client.get(f"{self.base_url}{endpoint}", params={"weights_path": weights_path})
        if response.status_code >= 400:
            self._check_response(endpoint, response)
        content_type = response.headers.get("content-type", "")
        if "json" in content_type.lower():
            try:
                parsed = response.json()
            except ValueError as exc:
                raise GPTSoVITSError(f"{endpoint} returned invalid JSON body") from exc
            if isinstance(parsed, dict):
                return str(parsed.get("message") or parsed.get("status") or parsed)
            return json.dumps(parsed, ensure_ascii=False)
        data = self._check_response(endpoint, response)
        return data.decode("utf-8", errors="replace")

    def set_gpt_weights(self, weights_path: str) -> str:
        return self._set_weights("/set_gpt_weights", weights_path)

    def set_sovits_weights(self, weights_path: str) -> str:
        return self._set_weights("/set_sovits_weights", weights_path)
