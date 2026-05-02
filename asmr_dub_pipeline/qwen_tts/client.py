from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asmr_dub_pipeline.audio.features import write_audio

REPO_ROOT = Path(__file__).resolve().parents[2]

_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "korean": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}


class QwenTTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class QwenTTSRequest:
    text: str
    language: str
    ref_audio_path: str
    ref_text: str
    seed: int
    x_vector_only_mode: bool = False
    generation_kwargs: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "ref_audio_path": self.ref_audio_path,
            "ref_text": self.ref_text,
            "seed": self.seed,
            "x_vector_only_mode": self.x_vector_only_mode,
            **self.generation_kwargs,
        }


@dataclass(frozen=True)
class QwenTTSResult:
    output_path: Path
    sample_rate: int


def qwen_language(language: str | None) -> str:
    normalized = str(language or "").strip().lower().replace("-", "_")
    if not normalized or normalized in {"auto", "none", "null"}:
        return "Auto"
    return _LANGUAGE_NAMES.get(normalized, language or "Auto")


def _snapshot_from_hf_cache_root(root: Path) -> Path | None:
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None
    ref = root / "refs" / "main"
    if ref.exists():
        candidate = snapshots / ref.read_text("utf-8").strip()
        if _looks_like_qwen_snapshot(candidate):
            return candidate.resolve()
    candidates = sorted(
        (path for path in snapshots.iterdir() if _looks_like_qwen_snapshot(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _looks_like_qwen_snapshot(path: Path) -> bool:
    return path.exists() and (path / "config.json").exists() and any(path.glob("*.safetensors"))


def _hf_snapshot_for_model(model_id: str) -> Path | None:
    raw = Path(model_id).expanduser()
    if raw.exists():
        resolved = raw.resolve()
        return _snapshot_from_hf_cache_root(resolved) or resolved
    if "/" not in model_id:
        return None
    cache_name = "models--" + model_id.replace("/", "--")
    for root in (
        Path.cwd() / ".cache" / "huggingface" / "hub" / cache_name,
        REPO_ROOT / ".cache" / "huggingface" / "hub" / cache_name,
    ):
        candidate = _snapshot_from_hf_cache_root(root)
        if candidate is not None:
            return candidate
    return None


def _resolve_model(model_id: str, *, local_files_only: bool) -> str:
    snapshot = _hf_snapshot_for_model(model_id)
    if snapshot is not None:
        return str(snapshot)
    if local_files_only:
        raise QwenTTSError(f"Qwen TTS model not found in local Hugging Face cache: {model_id}")
    return model_id


def _dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    normalized = dtype_name.strip().lower()
    if normalized in {"auto", ""}:
        return "auto"
    dtype = getattr(torch_module, normalized, None)
    if dtype is None:
        raise QwenTTSError(f"Unsupported Qwen TTS torch dtype: {dtype_name}")
    return dtype


class QwenTTSClient:
    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.local_files_only = local_files_only
        self._model: Any | None = None
        self._prompt_cache: dict[tuple[str, str, bool], Any] = {}

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise QwenTTSError(
                "qwen-tts is not installed. Install it in an isolated environment, "
                "then run `asmr-dub synth-qwen --project ... --confirm-rights`."
            ) from exc

        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "dtype": _dtype_from_name(torch, self.dtype),
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        model_path = _resolve_model(self.model_id, local_files_only=self.local_files_only)
        try:
            self._model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            raise QwenTTSError(f"Qwen TTS model load failed: {exc}") from exc
        return self._model

    def _voice_clone_prompt(self, request: QwenTTSRequest) -> Any | None:
        model = self._load_model()
        if not hasattr(model, "create_voice_clone_prompt"):
            return None
        cache_key = (request.ref_audio_path, request.ref_text, request.x_vector_only_mode)
        if cache_key not in self._prompt_cache:
            self._prompt_cache[cache_key] = model.create_voice_clone_prompt(
                ref_audio=request.ref_audio_path,
                ref_text=request.ref_text,
                x_vector_only_mode=request.x_vector_only_mode,
            )
        return self._prompt_cache[cache_key]

    def synthesize_to_file(self, request: QwenTTSRequest, output_path: Path) -> QwenTTSResult:
        if not request.text.strip():
            raise QwenTTSError("Qwen TTS text must not be empty.")
        if not request.ref_audio_path.strip():
            raise QwenTTSError("Qwen TTS ref_audio_path must not be empty.")
        if not request.ref_text.strip() and not request.x_vector_only_mode:
            raise QwenTTSError("Qwen TTS ref_text must not be empty unless x_vector_only_mode is enabled.")
        try:
            import torch
        except ImportError as exc:
            raise QwenTTSError("torch is required for qwen-tts synthesis.") from exc

        model = self._load_model()
        if request.seed >= 0 and hasattr(torch, "manual_seed"):
            torch.manual_seed(request.seed)
        kwargs = dict(request.generation_kwargs)
        try:
            voice_clone_prompt = self._voice_clone_prompt(request)
            if voice_clone_prompt is not None:
                wavs, sample_rate = model.generate_voice_clone(
                    text=request.text,
                    language=request.language,
                    voice_clone_prompt=voice_clone_prompt,
                    **kwargs,
                )
            else:
                wavs, sample_rate = model.generate_voice_clone(
                    text=request.text,
                    language=request.language,
                    ref_audio=request.ref_audio_path,
                    ref_text=request.ref_text,
                    x_vector_only_mode=request.x_vector_only_mode,
                    **kwargs,
                )
        except Exception as exc:
            raise QwenTTSError(f"Qwen TTS synthesis failed: {exc}") from exc

        if not wavs:
            raise QwenTTSError("Qwen TTS returned no audio.")
        data = np.asarray(wavs[0], dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        write_audio(output_path, data, int(sample_rate))
        return QwenTTSResult(output_path=output_path, sample_rate=int(sample_rate))
