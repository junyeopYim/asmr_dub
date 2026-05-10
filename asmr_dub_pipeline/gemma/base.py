from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.schemas import Segment


class GemmaBackendError(RuntimeError):
    pass


class GemmaUnavailableError(GemmaBackendError):
    pass


class GemmaResponseParseError(GemmaBackendError):
    pass


class GemmaHTTPError(GemmaBackendError):
    pass


class GemmaBackend(ABC):
    supports_repair_prompt: bool = False

    @abstractmethod
    def analyze_segment(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def analyze_audio_style(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self.analyze_segment(audio_path, segment, context)

    @abstractmethod
    def generate_script(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def qc_audio(
        self,
        audio_path: Path,
        target_text: str,
        segment: Segment,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


def create_gemma_backend(kind: str, config: Mapping[str, Any] | None = None) -> GemmaBackend:
    config = config or {}
    normalized_kind = kind.replace("-", "_")
    if normalized_kind == "mock":
        from .mock import MockGemmaBackend

        return MockGemmaBackend()
    if normalized_kind == "hf":
        from .hf_client import HFGemmaBackend

        return HFGemmaBackend(
            model_id=str(config.get("model_id", "google/gemma-4-E4B-it")),
            local_files_only=bool(config.get("local_files_only", True)),
        )
    if normalized_kind == "http":
        from .http_client import HTTPGemmaBackend

        url = config.get("url")
        if not url:
            raise GemmaUnavailableError("HTTP Gemma backend requires an explicit URL.")
        return HTTPGemmaBackend(
            base_url=str(url),
            timeout=float(config.get("timeout", 120.0)),
            retries=int(config.get("retries", 2)),
            model_id=str(config.get("model_id", "google/gemma-4-E4B-it")),
            send_audio=bool(config.get("send_audio", False)),
            repair_on_parse_error=bool(config.get("repair_on_parse_error", True)),
        )
    if normalized_kind in {"llama_cpp", "llamacpp"}:
        from .llama_cpp_client import LlamaCppGemmaBackend

        model_path = config.get("llama_cpp_model_path") or config.get("model_path")
        mmproj_path = config.get("llama_cpp_mmproj_path") or config.get("mmproj_path")
        cli_path = config.get("llama_cpp_cli_path") or config.get("cli_path")
        kwargs: dict[str, Any] = {
            "timeout_sec": float(config.get("llama_cpp_timeout_sec", 600.0)),
            "ctx_size": int(config.get("llama_cpp_ctx_size", 4096)),
            "n_predict": int(config.get("llama_cpp_n_predict", 1024)),
            "gpu_layers": int(config.get("llama_cpp_gpu_layers", 999)),
            "temperature": float(config.get("llama_cpp_temperature", 0.0)),
            "seed": int(config.get("llama_cpp_seed", 12345)),
            "extra_args": config.get("llama_cpp_extra_args"),
        }
        if model_path:
            kwargs["model_path"] = str(model_path)
        if mmproj_path:
            kwargs["mmproj_path"] = str(mmproj_path)
        if cli_path:
            kwargs["cli_path"] = str(cli_path)
        return LlamaCppGemmaBackend(
            **kwargs,
        )
    raise GemmaUnavailableError(f"Unsupported Gemma backend: {kind}")
