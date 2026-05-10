from __future__ import annotations

from pathlib import Path

from asmr_dub_pipeline.audio import ffmpeg


def test_wav_writing_helpers_request_rf64_auto(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(ffmpeg, "run_ffmpeg", lambda args: calls.append(args))
    monkeypatch.setattr(ffmpeg, "probe_media", lambda path: type("Info", (), {"duration_sec": 1.0})())

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"

    ffmpeg.extract_stereo_48k(first, tmp_path / "stereo.wav")
    ffmpeg.extract_mono_16k(first, tmp_path / "mono.wav")
    ffmpeg.concat_audio_to_wav([first, second], tmp_path / "concat.wav")
    ffmpeg.concat_audio_to_wav_with_silence(
        [first, second],
        tmp_path / "concat_silence.wav",
        silent_paths=[second],
    )
    ffmpeg.slice_audio(first, 0.1, 0.3, tmp_path / "slice.wav")
    ffmpeg.fit_audio_duration(first, tmp_path / "fit.wav", target_duration_sec=0.5)

    assert calls
    for args in calls:
        output_index = len(args) - 1
        rf64_index = args.index("-rf64")
        assert args[rf64_index + 1] == "auto"
        assert rf64_index < output_index
