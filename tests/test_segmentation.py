from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from asmr_dub_pipeline.audio import segmentation
from asmr_dub_pipeline.audio.preprocess import translate_media_stem_to_korean
from asmr_dub_pipeline.schemas import Segment


def _write_segmentable_audio(path: Path, sample_rate: int, channels: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tone_len = int(sample_rate * 0.4)
    silence_len = int(sample_rate * 0.2)
    t = np.arange(tone_len, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    mono = np.concatenate([tone, np.zeros(silence_len, dtype=np.float32), tone])
    data = mono[:, None] if channels == 1 else np.stack([mono, mono * 0.8], axis=1)
    sf.write(str(path), data, sample_rate)
    return path


def test_energy_segments_streams_audio_for_clip_writes(monkeypatch, tmp_path: Path) -> None:
    gemma_audio = _write_segmentable_audio(tmp_path / "gemma.wav", 16_000, 1)
    mix_audio = _write_segmentable_audio(tmp_path / "mix.wav", 48_000, 2)

    def forbidden_load_audio(path: Path | str):
        raise AssertionError(f"full audio load should not be used for segmentation: {path}")

    monkeypatch.setattr(segmentation, "load_audio", forbidden_load_audio, raising=False)

    segments = segmentation.energy_segments(
        gemma_audio,
        mix_audio,
        tmp_path,
        min_segment_sec=0.1,
        min_silence_sec=0.1,
    )

    assert len(segments) == 2
    for segment in segments:
        assert Path(segment.audio_for_gemma).exists()
        assert Path(segment.audio_for_mix).exists()


def test_write_segment_audio_clips_streams_slices_without_full_load(monkeypatch, tmp_path: Path) -> None:
    gemma_audio = _write_segmentable_audio(tmp_path / "gemma.wav", 16_000, 1)
    mix_audio = _write_segmentable_audio(tmp_path / "mix.wav", 48_000, 2)
    segments = [
        Segment(
            id="seg_0001",
            start=0.0,
            end=0.3,
            duration=0.3,
            audio_for_gemma="",
            audio_for_mix="",
        ),
        Segment(
            id="seg_0002",
            start=0.55,
            end=0.8,
            duration=0.25,
            audio_for_gemma="",
            audio_for_mix="",
        ),
    ]

    def forbidden_load_audio(path: Path | str):
        raise AssertionError(f"full audio load should not be used for segment clips: {path}")

    monkeypatch.setattr(segmentation, "load_audio", forbidden_load_audio, raising=False)

    updated = segmentation.write_segment_audio_clips(segments, gemma_audio, mix_audio, tmp_path)

    assert [segment.id for segment in updated] == ["seg_0001", "seg_0002"]
    for segment in updated:
        assert Path(segment.audio_for_gemma).exists()
        assert Path(segment.audio_for_mix).exists()
        assert -1.0 <= segment.estimated_pan <= 1.0


def test_write_segment_audio_clips_opens_each_source_once(monkeypatch, tmp_path: Path) -> None:
    gemma_audio = _write_segmentable_audio(tmp_path / "gemma.wav", 16_000, 1)
    mix_audio = _write_segmentable_audio(tmp_path / "mix.wav", 48_000, 2)
    segments = [
        Segment(
            id="seg_0001",
            start=0.0,
            end=0.3,
            duration=0.3,
            audio_for_gemma="",
            audio_for_mix="",
        ),
        Segment(
            id="seg_0002",
            start=0.55,
            end=0.8,
            duration=0.25,
            audio_for_gemma="",
            audio_for_mix="",
        ),
    ]
    original_sound_file = segmentation.sf.SoundFile
    opened: list[Path] = []

    def counting_sound_file(path: str, *args: object, **kwargs: object):
        opened.append(Path(path))
        return original_sound_file(path, *args, **kwargs)

    monkeypatch.setattr(segmentation.sf, "SoundFile", counting_sound_file)

    segmentation.write_segment_audio_clips(segments, gemma_audio, mix_audio, tmp_path)

    assert opened.count(gemma_audio) == 1
    assert opened.count(mix_audio) == 1


def test_write_segment_audio_clips_falls_back_when_seek_fails(monkeypatch, tmp_path: Path) -> None:
    gemma_audio = _write_segmentable_audio(tmp_path / "gemma.wav", 16_000, 1)
    mix_audio = _write_segmentable_audio(tmp_path / "mix.wav", 48_000, 2)
    segment = Segment(
        id="seg_0001",
        start=0.1,
        end=0.3,
        duration=0.2,
        audio_for_gemma="",
        audio_for_mix="",
    )
    original_sound_file = segmentation.sf.SoundFile

    class SeekFailingSoundFile:
        def __init__(self, path: str, *args: object, **kwargs: object) -> None:
            self.path = Path(path)
            self._inner = original_sound_file(path, *args, **kwargs)

        def __enter__(self) -> SeekFailingSoundFile:
            self._inner.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._inner.__exit__(*args)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        def seek(self, frames: int) -> object:
            if self.path == mix_audio and frames > 0:
                raise RuntimeError("Internal psf_fseek() failed.")
            return self._inner.seek(frames)

        def read(self, *args: object, **kwargs: object) -> object:
            return self._inner.read(*args, **kwargs)

    monkeypatch.setattr(segmentation.sf, "SoundFile", SeekFailingSoundFile)

    [updated] = segmentation.write_segment_audio_clips([segment], gemma_audio, mix_audio, tmp_path)

    mix_clip = Path(updated.audio_for_mix)
    assert mix_clip.exists()
    data, sample_rate = sf.read(mix_clip, always_2d=True, dtype="float32")
    assert sample_rate == 48_000
    assert data.shape[1] == 2


def test_write_segment_audio_clips_uses_requested_end_for_seek_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    gemma_audio = _write_segmentable_audio(tmp_path / "gemma.wav", 16_000, 1)
    mix_audio = _write_segmentable_audio(tmp_path / "mix.wav", 48_000, 2)
    segment = Segment(
        id="seg_0001",
        start=0.2,
        end=0.4,
        duration=0.2,
        audio_for_gemma="",
        audio_for_mix="",
    )
    original_sound_file = segmentation.sf.SoundFile

    class TruncatedSeekFailingSoundFile:
        def __init__(self, path: str, *args: object, **kwargs: object) -> None:
            self.path = Path(path)
            self._inner = original_sound_file(path, *args, **kwargs)

        def __enter__(self) -> TruncatedSeekFailingSoundFile:
            self._inner.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._inner.__exit__(*args)

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

        @property
        def frames(self) -> int:
            if self.path == mix_audio:
                return 100
            return int(self._inner.frames)

        def seek(self, frames: int) -> object:
            if self.path == mix_audio and frames > self.frames:
                raise RuntimeError("Internal psf_fseek() failed.")
            return self._inner.seek(frames)

        def read(self, *args: object, **kwargs: object) -> object:
            return self._inner.read(*args, **kwargs)

    monkeypatch.setattr(segmentation.sf, "SoundFile", TruncatedSeekFailingSoundFile)

    [updated] = segmentation.write_segment_audio_clips([segment], gemma_audio, mix_audio, tmp_path)

    data, sample_rate = sf.read(updated.audio_for_mix, always_2d=True, dtype="float32")
    assert sample_rate == 48_000
    assert data.shape[0] > 0
    assert data.shape[1] == 2


def test_translate_media_stem_to_korean_handles_common_english_track_names() -> None:
    assert translate_media_stem_to_korean("01_Install") == "01_설치"
    assert translate_media_stem_to_korean("05_Nipple_exp") == "05_유두_확장"
    assert translate_media_stem_to_korean("10_Vagina_Anal") == "10_질_애널"
