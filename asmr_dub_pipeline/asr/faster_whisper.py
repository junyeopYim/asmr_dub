from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from asmr_dub_pipeline.schemas import Segment

from .base import ASRBackend, ASRChunk, ASRUnavailableError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _ctranslate2_snapshot_for_model(model_id: str) -> Path | None:
    raw = Path(model_id).expanduser()
    if raw.exists():
        return raw.resolve()
    if "/" not in model_id:
        return None
    cache_name = "models--" + model_id.replace("/", "--")
    roots = [
        Path.cwd() / ".cache" / "ctranslate2" / cache_name,
        REPO_ROOT / ".cache" / "ctranslate2" / cache_name,
    ]
    for root in roots:
        snapshots = root / "snapshots"
        if not snapshots.exists():
            continue
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


class FasterWhisperASRBackend(ASRBackend):
    name = "faster_whisper"

    def __init__(
        self,
        *,
        model_id: str,
        language: str = "ja",
        local_files_only: bool = True,
    ) -> None:
        self.model_id = model_id
        self.language = language
        self.local_files_only = local_files_only

    def transcribe(self, audio_path: Path, segments: Sequence[Segment]) -> list[ASRChunk]:
        _ = segments
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailableError(
                "faster-whisper is not installed. Install the ASR extras with "
                "python -m pip install -e .[asr], or use --asr-backend mock for tests."
            ) from exc

        model_size_or_path = str(_ctranslate2_snapshot_for_model(self.model_id) or self.model_id)
        try:
            model = WhisperModel(
                model_size_or_path,
                device="auto",
                local_files_only=self.local_files_only,
            )
            raw_segments, info = model.transcribe(str(audio_path), language=self.language)
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
