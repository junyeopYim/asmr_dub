from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import soundfile as sf

from asmr_dub_pipeline.schemas import Segment

from .base import ASRBackend, ASRChunk, ASRUnavailableError, ASRWord

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
    "th": "Thai",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh_cn": "Chinese",
    "zh_tw": "Chinese",
    "yue": "Cantonese",
}


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
    if candidates:
        return candidates[0].resolve()
    return None


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
    roots = [
        Path.cwd() / ".cache" / "huggingface" / "hub" / cache_name,
        REPO_ROOT / ".cache" / "huggingface" / "hub" / cache_name,
    ]
    for root in roots:
        candidate = _snapshot_from_hf_cache_root(root)
        if candidate is not None:
            return candidate
    return None


def _resolve_model_for_qwen(model_id: str, *, local_files_only: bool, label: str) -> str:
    snapshot = _hf_snapshot_for_model(model_id)
    if snapshot is not None:
        return str(snapshot)
    if local_files_only:
        raise ASRUnavailableError(
            f"Qwen ASR {label} not found in local Hugging Face cache: {model_id}"
        )
    return model_id


def _qwen_language(language: str | None) -> str | None:
    normalized = str(language or "").strip().lower().replace("-", "_")
    if not normalized or normalized in {"auto", "none", "null"}:
        return None
    return _LANGUAGE_NAMES.get(normalized, language)


def _attr_or_item(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _audio_duration_sec(audio_path: Path) -> float:
    try:
        info = sf.info(str(audio_path))
    except Exception:
        return 0.0
    if info.samplerate <= 0:
        return 0.0
    return max(0.0, float(info.frames) / float(info.samplerate))


def _dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    normalized = dtype_name.strip().lower()
    if normalized in {"auto", ""}:
        return "auto"
    dtype = getattr(torch_module, normalized, None)
    if dtype is None:
        raise ASRUnavailableError(f"Unsupported Qwen ASR torch dtype: {dtype_name}")
    return dtype


def _is_japanese_language(language: str | None) -> bool:
    return str(language or "").strip().lower() in {"ja", "japanese"}


def _join_qwen_timestamp_text(parts: list[str], *, language: str | None) -> str:
    cleaned = [part.strip() for part in parts if part.strip()]
    if not cleaned:
        return ""
    if _is_japanese_language(language):
        return "".join(cleaned)
    return " ".join(cleaned)


def _coalesce_qwen_timestamp_chunks(
    chunks: list[ASRChunk],
    *,
    language: str | None,
    max_gap_sec: float = 2.25,
    max_duration_sec: float = 18.0,
) -> list[ASRChunk]:
    if not chunks:
        return []
    coalesced: list[ASRChunk] = []
    current: list[ASRChunk] = []

    def chunk_words(chunk: ASRChunk) -> list[ASRWord]:
        if chunk.words:
            return chunk.words
        text = chunk.text.strip()
        if not text or chunk.end <= chunk.start:
            return []
        return [
            ASRWord(
                start=chunk.start,
                end=chunk.end,
                text=text,
                confidence=chunk.confidence,
            )
        ]

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = _join_qwen_timestamp_text([chunk.text for chunk in current], language=language)
        words = [word for chunk in current for word in chunk_words(chunk)]
        if text:
            coalesced.append(
                ASRChunk(
                    start=current[0].start,
                    end=current[-1].end,
                    text=text,
                    language=current[0].language,
                    confidence=None,
                    words=words,
                )
            )
        current = []

    for chunk in sorted(chunks, key=lambda item: (item.start, item.end)):
        if current:
            gap = max(0.0, chunk.start - current[-1].end)
            duration = chunk.end - current[0].start
            if gap > max_gap_sec or duration > max_duration_sec:
                flush()
        current.append(chunk)
        if chunk.text.endswith(("。", "！", "？", ".", "!", "?")):
            flush()
    flush()
    return coalesced


class QwenASRBackend(ASRBackend):
    name = "qwen_asr"

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-ASR-1.7B",
        language: str = "ja",
        local_files_only: bool = True,
        forced_aligner_model_id: str | None = "Qwen/Qwen3-ForcedAligner-0.6B",
        device_map: str = "cuda:0",
        dtype: str = "bfloat16",
        return_timestamps: bool = True,
        context: str = "",
        max_inference_batch_size: int = 8,
        max_new_tokens: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.local_files_only = local_files_only
        self.forced_aligner_model_id = forced_aligner_model_id
        self.device_map = device_map
        self.dtype = dtype
        self.return_timestamps = return_timestamps
        self.context = context
        self.max_inference_batch_size = max_inference_batch_size
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRUnavailableError(
                "qwen-asr is not installed. Install it in an isolated environment or with "
                "`uv pip install qwen-asr`, then run with --asr-backend qwen_asr."
            ) from exc

        dtype = _dtype_from_name(torch, self.dtype)
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": self.device_map,
            "max_inference_batch_size": self.max_inference_batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.return_timestamps and self.forced_aligner_model_id:
            kwargs["forced_aligner"] = _resolve_model_for_qwen(
                self.forced_aligner_model_id,
                local_files_only=self.local_files_only,
                label="forced aligner",
            )
            kwargs["forced_aligner_kwargs"] = {
                "dtype": dtype,
                "device_map": self.device_map,
            }

        model_path = _resolve_model_for_qwen(
            self.model_id,
            local_files_only=self.local_files_only,
            label="model",
        )
        self._model = Qwen3ASRModel.from_pretrained(model_path, **kwargs)
        return self._model

    def transcribe(self, audio_path: Path, segments: Sequence[Segment]) -> list[ASRChunk]:
        _ = segments
        try:
            model = self._load_model()
            raw_results = model.transcribe(
                audio=str(audio_path),
                context=self.context,
                language=_qwen_language(self.language),
                return_time_stamps=self.return_timestamps,
            )
        except ASRUnavailableError:
            raise
        except Exception as exc:
            raise ASRUnavailableError(f"Qwen ASR transcription failed: {exc}") from exc

        results = raw_results if isinstance(raw_results, list) else [raw_results]
        chunks: list[ASRChunk] = []
        duration = _audio_duration_sec(audio_path)
        for result in results:
            language = str(_attr_or_item(result, "language") or self.language)
            text = str(_attr_or_item(result, "text") or "").strip()
            timestamps = _attr_or_item(result, "time_stamps", "timestamps") or []
            result_chunks: list[ASRChunk] = []
            for timestamp in timestamps:
                chunk_text = str(_attr_or_item(timestamp, "text", "word") or "").strip()
                start = _attr_or_item(timestamp, "start_time", "start")
                end = _attr_or_item(timestamp, "end_time", "end")
                if not chunk_text or start is None or end is None:
                    continue
                start_sec = float(start)
                end_sec = float(end)
                if end_sec > start_sec:
                    result_chunks.append(
                        ASRChunk(
                            start=start_sec,
                            end=end_sec,
                            text=chunk_text,
                            language=language,
                            confidence=None,
                            words=[
                                ASRWord(
                                    start=start_sec,
                                    end=end_sec,
                                    text=chunk_text,
                                    confidence=None,
                                )
                            ],
                        )
                    )
            if result_chunks:
                chunks.extend(_coalesce_qwen_timestamp_chunks(result_chunks, language=language))
            elif text:
                chunks.append(
                    ASRChunk(
                        start=0.0,
                        end=duration,
                        text=text,
                        language=language,
                        confidence=None,
                    )
                )
        return chunks
