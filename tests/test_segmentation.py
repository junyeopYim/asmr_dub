from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from asmr_dub_pipeline.audio import segmentation
from asmr_dub_pipeline.audio.preprocess import translate_media_stem_to_korean


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


def test_energy_segments_reuses_loaded_audio_for_clip_writes(monkeypatch, tmp_path: Path) -> None:
    gemma_audio = _write_segmentable_audio(tmp_path / "gemma.wav", 16_000, 1)
    mix_audio = _write_segmentable_audio(tmp_path / "mix.wav", 48_000, 2)
    calls: list[str] = []
    original_load_audio = segmentation.load_audio

    def counting_load_audio(path: Path | str):
        calls.append(Path(path).name)
        return original_load_audio(path)

    monkeypatch.setattr(segmentation, "load_audio", counting_load_audio)

    segments = segmentation.energy_segments(
        gemma_audio,
        mix_audio,
        tmp_path,
        min_segment_sec=0.1,
        min_silence_sec=0.1,
    )

    assert calls == ["gemma.wav", "mix.wav"]
    assert len(segments) == 2
    for segment in segments:
        assert Path(segment.audio_for_gemma).exists()
        assert Path(segment.audio_for_mix).exists()


def test_translate_media_stem_to_korean_handles_common_english_track_names() -> None:
    assert translate_media_stem_to_korean("01_Install") == "01_설치"
    assert translate_media_stem_to_korean("05_Nipple_exp") == "05_유두_확장"
    assert translate_media_stem_to_korean("10_Vagina_Anal") == "10_질_애널"
