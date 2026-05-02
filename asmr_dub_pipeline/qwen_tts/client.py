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
    batch_size: int = 1
    batch_seed: int | None = None


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


def _set_nested_config_dtype(config: Any, dtype: Any) -> None:
    seen: set[int] = set()

    def visit(node: Any) -> None:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        try:
            node.dtype = dtype
        except Exception:
            return
        for key in getattr(node, "sub_configs", {}) or {}:
            try:
                visit(getattr(node, key))
            except AttributeError:
                continue

    visit(config)


def _qwen_config_with_dtype(model_path: str, dtype: Any, attn_implementation: str) -> Any | None:
    if dtype == "auto":
        return None
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from transformers import AutoConfig
    except ImportError:
        return None

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
    config = AutoConfig.from_pretrained(model_path)
    if attn_implementation:
        config._attn_implementation = attn_implementation
    _set_nested_config_dtype(config, dtype)
    return config


class QwenTTSClient:
    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        local_files_only: bool = True,
        target_vram_gb: float | None = 14.0,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.local_files_only = local_files_only
        self.target_vram_gb = target_vram_gb
        self._model: Any | None = None
        self._prompt_cache: dict[tuple[str, str, bool], Any] = {}
        self._cuda_memory_target_configured = False

    def load_model(self) -> Any:
        return self._load_model()

    def _cuda_device_index(self) -> int | None:
        normalized = str(self.device_map).strip().lower()
        if normalized == "cuda":
            return 0
        if normalized.startswith("cuda:"):
            suffix = normalized.split(":", 1)[1]
            return int(suffix) if suffix.isdigit() else 0
        return None

    def _configure_cuda_memory_target(self, torch_module: Any) -> None:
        if self._cuda_memory_target_configured:
            return
        self._cuda_memory_target_configured = True
        if not self.target_vram_gb or self.target_vram_gb <= 0:
            return
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not cuda.is_available():
            return
        device_index = self._cuda_device_index()
        if device_index is None:
            return
        try:
            total_memory = int(cuda.get_device_properties(device_index).total_memory)
            target_bytes = int(float(self.target_vram_gb) * 1024**3)
            fraction = min(max(target_bytes / max(total_memory, 1), 0.01), 1.0)
            cuda.set_per_process_memory_fraction(fraction, device_index)
        except Exception:
            return

    def cuda_memory_snapshot(self) -> dict[str, float | int | str] | None:
        try:
            import torch
        except ImportError:
            return None
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            return None
        device_index = self._cuda_device_index()
        if device_index is None:
            return None
        try:
            free_bytes, total_bytes = cuda.mem_get_info(device_index)
            return {
                "device": f"cuda:{device_index}",
                "allocated_gb": cuda.memory_allocated(device_index) / 1024**3,
                "reserved_gb": cuda.memory_reserved(device_index) / 1024**3,
                "free_gb": free_bytes / 1024**3,
                "total_gb": total_bytes / 1024**3,
                "target_vram_gb": float(self.target_vram_gb or 0.0),
            }
        except Exception:
            return None

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

        self._configure_cuda_memory_target(torch)
        dtype = _dtype_from_name(torch, self.dtype)
        model_path = _resolve_model(self.model_id, local_files_only=self.local_files_only)
        kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "dtype": dtype,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        config = _qwen_config_with_dtype(model_path, dtype, self.attn_implementation)
        if config is not None:
            kwargs["config"] = config
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

    @staticmethod
    def _validate_request(request: QwenTTSRequest) -> None:
        if not request.text.strip():
            raise QwenTTSError("Qwen TTS text must not be empty.")
        if not request.ref_audio_path.strip():
            raise QwenTTSError("Qwen TTS ref_audio_path must not be empty.")
        if not request.ref_text.strip() and not request.x_vector_only_mode:
            raise QwenTTSError("Qwen TTS ref_text must not be empty unless x_vector_only_mode is enabled.")

    @staticmethod
    def _empty_cuda_cache(torch_module: Any) -> None:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is None or not cuda.is_available():
            return
        try:
            cuda.empty_cache()
        except Exception:
            return

    def synthesize_to_file(self, request: QwenTTSRequest, output_path: Path) -> QwenTTSResult:
        self._validate_request(request)
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
            self._empty_cuda_cache(torch)
            raise QwenTTSError(f"Qwen TTS synthesis failed: {exc}") from exc

        if not wavs:
            raise QwenTTSError("Qwen TTS returned no audio.")
        data = np.asarray(wavs[0], dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        write_audio(output_path, data, int(sample_rate))
        return QwenTTSResult(output_path=output_path, sample_rate=int(sample_rate), batch_seed=request.seed)

    def synthesize_many_to_files(
        self,
        requests: list[QwenTTSRequest],
        output_paths: list[Path],
    ) -> list[QwenTTSResult]:
        if len(requests) != len(output_paths):
            raise QwenTTSError("Qwen TTS batch request and output counts must match.")
        if not requests:
            return []
        if len(requests) == 1:
            return [self.synthesize_to_file(requests[0], output_paths[0])]
        for request in requests:
            self._validate_request(request)
        first_generation_kwargs = dict(requests[0].generation_kwargs)
        if any(dict(request.generation_kwargs) != first_generation_kwargs for request in requests):
            raise QwenTTSError("Batched Qwen TTS requests must share the same generation kwargs.")
        try:
            import torch
        except ImportError as exc:
            raise QwenTTSError("torch is required for qwen-tts synthesis.") from exc

        model = self._load_model()
        batch_seed = next((request.seed for request in requests if request.seed >= 0), None)
        if batch_seed is not None and hasattr(torch, "manual_seed"):
            torch.manual_seed(batch_seed)
        try:
            prompt_items: list[Any] | None = []
            for request in requests:
                prompt = self._voice_clone_prompt(request)
                if prompt is None:
                    prompt_items = None
                    break
                if isinstance(prompt, list) and len(prompt) == 1:
                    prompt_items.append(prompt[0])
                    continue
                prompt_items = None
                break

            if prompt_items is not None:
                wavs, sample_rate = model.generate_voice_clone(
                    text=[request.text for request in requests],
                    language=[request.language for request in requests],
                    voice_clone_prompt=prompt_items,
                    **first_generation_kwargs,
                )
            else:
                wavs, sample_rate = model.generate_voice_clone(
                    text=[request.text for request in requests],
                    language=[request.language for request in requests],
                    ref_audio=[request.ref_audio_path for request in requests],
                    ref_text=[request.ref_text for request in requests],
                    x_vector_only_mode=[request.x_vector_only_mode for request in requests],
                    **first_generation_kwargs,
                )
        except Exception as exc:
            self._empty_cuda_cache(torch)
            raise QwenTTSError(f"Qwen TTS batch synthesis failed: {exc}") from exc

        if len(wavs) != len(requests):
            raise QwenTTSError(f"Qwen TTS batch returned {len(wavs)} audio items for {len(requests)} requests.")
        results: list[QwenTTSResult] = []
        for wav, output_path in zip(wavs, output_paths, strict=True):
            data = np.asarray(wav, dtype=np.float32)
            if data.ndim == 1:
                data = data[:, None]
            write_audio(output_path, data, int(sample_rate))
            results.append(
                QwenTTSResult(
                    output_path=output_path,
                    sample_rate=int(sample_rate),
                    batch_size=len(requests),
                    batch_seed=batch_seed,
                )
            )
        return results
