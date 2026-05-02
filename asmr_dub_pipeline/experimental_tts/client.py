from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

from asmr_dub_pipeline.audio.features import write_audio


class ExperimentalTTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentalTTSRequest:
    text: str
    language: str
    ref_audio_path: str
    ref_text: str
    seed: int
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "ref_audio_path": self.ref_audio_path,
            "ref_text": self.ref_text,
            "seed": self.seed,
            **self.generation_kwargs,
        }


@dataclass(frozen=True)
class ExperimentalTTSResult:
    output_path: Path
    sample_rate: int
    batch_size: int = 1
    batch_seed: int | None = None


def _strip_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ExperimentalTTSError("Experimental TTS base_url must not be empty.")
    return normalized


def _validate_request(request: ExperimentalTTSRequest, backend_name: str) -> None:
    if not request.text.strip():
        raise ExperimentalTTSError(f"{backend_name} text must not be empty.")
    if not request.ref_audio_path.strip():
        raise ExperimentalTTSError(f"{backend_name} ref_audio_path must not be empty.")
    if not Path(request.ref_audio_path).expanduser().exists():
        raise ExperimentalTTSError(f"{backend_name} reference audio does not exist: {request.ref_audio_path}")
    if not request.ref_text.strip():
        raise ExperimentalTTSError(f"{backend_name} ref_text must not be empty.")


def _wav_sample_rate(path: Path, fallback: int) -> int:
    try:
        return int(sf.info(str(path)).samplerate)
    except Exception:
        return fallback


class FishSpeechTTSClient:
    """Thin client for Fish Speech's local `/v1/tts` API server."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080",
        timeout_sec: float = 240.0,
        audio_format: str = "wav",
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _strip_base_url(base_url)
        self.timeout_sec = timeout_sec
        self.audio_format = audio_format
        self.http_client = http_client or httpx.Client(timeout=timeout_sec)

    def load_model(self) -> None:
        return None

    def synthesize_to_file(
        self,
        request: ExperimentalTTSRequest,
        output_path: Path,
    ) -> ExperimentalTTSResult:
        _validate_request(request, "Fish Speech")
        ref_path = Path(request.ref_audio_path).expanduser()
        references = [
            {
                "audio": base64.b64encode(ref_path.read_bytes()).decode("ascii"),
                "text": request.ref_text,
            }
        ]
        payload: dict[str, Any] = {
            "text": request.text,
            "format": self.audio_format,
            "references": references,
            "reference_id": None,
            "seed": request.seed if request.seed >= 0 else None,
            "streaming": False,
        }
        payload.update(request.generation_kwargs)
        payload = {key: value for key, value in payload.items() if value is not None}
        try:
            response = self.http_client.post(f"{self.base_url}/v1/tts", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExperimentalTTSError(f"Fish Speech synthesis request failed: {exc}") from exc
        if not response.content:
            raise ExperimentalTTSError("Fish Speech returned empty audio.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)
        return ExperimentalTTSResult(
            output_path=output_path,
            sample_rate=_wav_sample_rate(output_path, 44_100),
            batch_seed=request.seed,
        )

    def synthesize_many_to_files(
        self,
        requests: list[ExperimentalTTSRequest],
        output_paths: list[Path],
    ) -> list[ExperimentalTTSResult]:
        if len(requests) != len(output_paths):
            raise ExperimentalTTSError("Fish Speech batch request and output counts must match.")
        return [self.synthesize_to_file(request, path) for request, path in zip(requests, output_paths, strict=True)]


class CosyVoiceTTSClient:
    """Thin client for CosyVoice's FastAPI inference server."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:50000",
        mode: str = "zero_shot",
        sample_rate: int = 22_050,
        timeout_sec: float = 240.0,
        instruct_text: str = "",
        http_client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _strip_base_url(base_url)
        self.mode = mode.strip().lower().replace("-", "_")
        if self.mode not in {"zero_shot", "cross_lingual", "instruct2"}:
            raise ExperimentalTTSError("CosyVoice mode must be one of: zero_shot, cross_lingual, instruct2.")
        self.sample_rate = sample_rate
        self.timeout_sec = timeout_sec
        self.instruct_text = instruct_text
        self.http_client = http_client or httpx.Client(timeout=timeout_sec)

    def load_model(self) -> None:
        return None

    def synthesize_to_file(
        self,
        request: ExperimentalTTSRequest,
        output_path: Path,
    ) -> ExperimentalTTSResult:
        _validate_request(request, "CosyVoice")
        ref_path = Path(request.ref_audio_path).expanduser()
        data: dict[str, Any] = {"tts_text": request.text}
        if self.mode == "zero_shot":
            data["prompt_text"] = request.ref_text
        elif self.mode == "instruct2":
            data["instruct_text"] = str(request.generation_kwargs.get("instruct_text") or self.instruct_text)
        endpoint = f"{self.base_url}/inference_{self.mode}"
        try:
            with ref_path.open("rb") as prompt_wav:
                response = self.http_client.post(
                    endpoint,
                    data=data,
                    files={"prompt_wav": (ref_path.name, prompt_wav, "application/octet-stream")},
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExperimentalTTSError(f"CosyVoice synthesis request failed: {exc}") from exc
        if not response.content:
            raise ExperimentalTTSError("CosyVoice returned empty audio.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if response.content.startswith(b"RIFF"):
            output_path.write_bytes(response.content)
            sample_rate = _wav_sample_rate(output_path, self.sample_rate)
        else:
            pcm = np.frombuffer(response.content, dtype=np.int16)
            if pcm.size == 0:
                raise ExperimentalTTSError("CosyVoice returned no PCM samples.")
            audio = (pcm.astype(np.float32) / 32768.0)[:, None]
            write_audio(output_path, audio, self.sample_rate)
            sample_rate = self.sample_rate
        return ExperimentalTTSResult(output_path=output_path, sample_rate=sample_rate, batch_seed=request.seed)

    def synthesize_many_to_files(
        self,
        requests: list[ExperimentalTTSRequest],
        output_paths: list[Path],
    ) -> list[ExperimentalTTSResult]:
        if len(requests) != len(output_paths):
            raise ExperimentalTTSError("CosyVoice batch request and output counts must match.")
        return [self.synthesize_to_file(request, path) for request, path in zip(requests, output_paths, strict=True)]
