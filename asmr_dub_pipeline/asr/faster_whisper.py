from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.schemas import Segment

from .base import ASRBackend, ASRChunk, ASRUnavailableError, ASRWord

REPO_ROOT = Path(__file__).resolve().parents[2]


def _snapshot_from_cache_root(root: Path) -> Path | None:
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None
    ref = root / "refs" / "main"
    if ref.exists():
        candidate = snapshots / ref.read_text("utf-8").strip()
        if (candidate / "model.bin").exists():
            return candidate.resolve()
    candidates = sorted(
        (path for path in snapshots.iterdir() if (path / "model.bin").exists()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    return None


def _ctranslate2_snapshot_for_model(model_id: str) -> Path | None:
    raw = Path(model_id).expanduser()
    if raw.exists():
        resolved = raw.resolve()
        if (resolved / "model.bin").exists():
            return resolved
        return _snapshot_from_cache_root(resolved) or resolved
    if "/" not in model_id:
        return None
    cache_name = "models--" + model_id.replace("/", "--")
    roots = [
        Path.cwd() / ".cache" / "ctranslate2" / cache_name,
        REPO_ROOT / ".cache" / "ctranslate2" / cache_name,
    ]
    for root in roots:
        candidate = _snapshot_from_cache_root(root)
        if candidate is not None:
            return candidate
    return None


def _attr_or_item(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _word_confidence(value: Any) -> float | None:
    raw = _attr_or_item(value, "probability", "confidence")
    if raw is None:
        return None
    return max(0.0, min(1.0, float(raw)))


def _segment_words(item: Any) -> list[ASRWord]:
    words: list[ASRWord] = []
    for raw_word in _attr_or_item(item, "words") or []:
        text = str(_attr_or_item(raw_word, "word", "text") or "").strip()
        start = _attr_or_item(raw_word, "start")
        end = _attr_or_item(raw_word, "end")
        if not text or start is None or end is None:
            continue
        start_sec = float(start)
        end_sec = float(end)
        if end_sec <= start_sec:
            continue
        words.append(
            ASRWord(
                start=start_sec,
                end=end_sec,
                text=text,
                confidence=_word_confidence(raw_word),
            )
        )
    return words


class FasterWhisperASRBackend(ASRBackend):
    name = "faster_whisper"

    def __init__(
        self,
        *,
        model_id: str,
        language: str = "ja",
        local_files_only: bool = True,
        device: str = "auto",
        compute_type: str = "default",
        batched_inference: bool = False,
        batch_size: int = 8,
        beam_size: int = 5,
        best_of: int = 5,
        condition_on_previous_text: bool = False,
        vad_filter: bool = True,
        vad_parameters: dict[str, object] | None = None,
        word_timestamps: bool = False,
        hallucination_silence_threshold: float | None = None,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.local_files_only = local_files_only
        self.device = device
        self.compute_type = compute_type
        self.batched_inference = batched_inference
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.best_of = best_of
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {}
        self.word_timestamps = word_timestamps
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self.initial_prompt = initial_prompt.strip() if initial_prompt else None
        self.hotwords = hotwords.strip() if hotwords else None
        self._model: Any | None = None
        self._batched_model: Any | None = None

    def _get_model(self) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailableError(
                "faster-whisper is not installed. Install the ASR extras with "
                "python -m pip install -e .[asr], or use --asr-backend mock for tests."
            ) from exc

        if self._model is not None:
            return self._model
        model_size_or_path = str(_ctranslate2_snapshot_for_model(self.model_id) or self.model_id)
        try:
            self._model = WhisperModel(
                model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise ASRUnavailableError(f"faster-whisper model load failed: {exc}") from exc
        return self._model

    def _get_transcriber(self, use_batched: bool) -> Any:
        model = self._get_model()
        if not use_batched:
            return model
        if self._batched_model is not None:
            return self._batched_model
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError as exc:
            raise ASRUnavailableError(
                "faster-whisper BatchedInferencePipeline is not available in this installation."
            ) from exc
        self._batched_model = BatchedInferencePipeline(model=model)
        return self._batched_model

    def transcribe_with_options(
        self,
        audio_path: Path,
        segments: Sequence[Segment],
        **overrides: Any,
    ) -> list[ASRChunk]:
        _ = segments
        use_batched = bool(overrides.pop("batched_inference", self.batched_inference))
        batch_size = int(overrides.pop("batch_size", self.batch_size))
        options = {
            "language": self.language,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "condition_on_previous_text": self.condition_on_previous_text,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters or None,
            "word_timestamps": self.word_timestamps,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
            "initial_prompt": self.initial_prompt,
            "hotwords": self.hotwords,
        }
        options.update(overrides)
        if use_batched:
            options["batch_size"] = batch_size
            options.setdefault("without_timestamps", False)
        try:
            model = self._get_transcriber(use_batched)
            raw_segments, info = model.transcribe(
                str(audio_path),
                **options,
            )
        except Exception as exc:
            raise ASRUnavailableError(f"faster-whisper transcription failed: {exc}") from exc

        language = getattr(info, "language", None) or self.language
        chunks: list[ASRChunk] = []
        for item in raw_segments:
            avg_logprob = getattr(item, "avg_logprob", None)
            confidence = None
            if avg_logprob is not None:
                confidence = max(0.0, min(1.0, (float(avg_logprob) + 5.0) / 5.0))
            chunks.append(
                ASRChunk(
                    start=float(item.start),
                    end=float(item.end),
                    text=str(item.text).strip(),
                    language=language,
                    confidence=confidence,
                    words=_segment_words(item),
                )
            )
        return chunks

    def transcribe(self, audio_path: Path, segments: Sequence[Segment]) -> list[ASRChunk]:
        return self.transcribe_with_options(audio_path, segments)
