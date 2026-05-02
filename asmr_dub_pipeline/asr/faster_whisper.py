from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.schemas import Segment

from .base import ASRBackend, ASRChunk, ASRUnavailableError

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


class FasterWhisperASRBackend(ASRBackend):
    name = "faster_whisper"

    def __init__(
        self,
        *,
        model_id: str,
        language: str = "ja",
        local_files_only: bool = True,
        beam_size: int = 5,
        best_of: int = 5,
        condition_on_previous_text: bool = False,
        vad_filter: bool = True,
        vad_parameters: dict[str, object] | None = None,
        word_timestamps: bool = False,
        hallucination_silence_threshold: float | None = None,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.local_files_only = local_files_only
        self.beam_size = beam_size
        self.best_of = best_of
        self.condition_on_previous_text = condition_on_previous_text
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or {}
        self.word_timestamps = word_timestamps
        self.hallucination_silence_threshold = hallucination_silence_threshold
        self._model: Any | None = None

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
                device="auto",
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise ASRUnavailableError(f"faster-whisper model load failed: {exc}") from exc
        return self._model

    def transcribe_with_options(
        self,
        audio_path: Path,
        segments: Sequence[Segment],
        **overrides: Any,
    ) -> list[ASRChunk]:
        _ = segments
        options = {
            "language": self.language,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "condition_on_previous_text": self.condition_on_previous_text,
            "vad_filter": self.vad_filter,
            "vad_parameters": self.vad_parameters or None,
            "word_timestamps": self.word_timestamps,
            "hallucination_silence_threshold": self.hallucination_silence_threshold,
        }
        options.update(overrides)
        try:
            model = self._get_model()
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
                )
            )
        return chunks

    def transcribe(self, audio_path: Path, segments: Sequence[Segment]) -> list[ASRChunk]:
        return self.transcribe_with_options(audio_path, segments)
