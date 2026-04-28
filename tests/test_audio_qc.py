from __future__ import annotations

import numpy as np

from asmr_dub_pipeline.audio.features import (
    duration_sec,
    leading_trailing_silence,
    peak_dbfs,
    rms_dbfs,
    trim_edge_silence,
    write_audio,
)
from asmr_dub_pipeline.audio.mixing import overlay, pan_stereo
from asmr_dub_pipeline.qc.audio_qc import measure_audio_qc
from asmr_dub_pipeline.qc.scoring import score_qc


def test_audio_features_and_qc(tiny_wav_path) -> None:
    assert duration_sec(tiny_wav_path) > 0.9
    assert peak_dbfs(tiny_wav_path) < 0
    assert rms_dbfs(tiny_wav_path) < 0
    metrics = measure_audio_qc(tiny_wav_path, target_duration_sec=1.0)
    qc = score_qc(metrics, {"recommendation": "pass"})
    assert qc.status in {"ok", "needs_regeneration"}


def test_qc_manual_review_not_downgraded(tiny_wav_path) -> None:
    metrics = measure_audio_qc(tiny_wav_path, target_duration_sec=10.0)
    qc = score_qc(metrics, {"recommendation": "manual_review", "issues": ["uncertain"]})
    assert qc.status == "needs_manual_review"
    assert qc.recommendation == "manual_review"


def test_qc_rejects_string_booleans(tiny_wav_path) -> None:
    metrics = measure_audio_qc(tiny_wav_path, target_duration_sec=1.0)
    qc = score_qc(
        metrics,
        {
            "recommendation": "pass",
            "unsafe_or_rights_issue": "false",
        },
    )
    assert qc.status == "needs_manual_review"
    assert qc.unsafe_or_rights_issue is True


def test_trim_edge_silence_keeps_small_pad(tmp_path) -> None:
    sr = 16_000
    silence_a = np.zeros(int(sr * 0.5), dtype=np.float32)
    tone = np.sin(2 * np.pi * 220.0 * np.arange(int(sr * 0.3), dtype=np.float32) / sr) * 0.05
    silence_b = np.zeros(int(sr * 0.4), dtype=np.float32)
    path = tmp_path / "tts.wav"
    write_audio(path, np.concatenate([silence_a, tone, silence_b])[:, None], sr)

    trim = trim_edge_silence(path, keep_sec=0.08)
    leading, trailing = leading_trailing_silence(path)

    assert trim["trimmed"] is True
    assert duration_sec(path) < 0.5
    assert leading <= 0.1
    assert trailing <= 0.1


def test_pan_and_overlay_shape() -> None:
    sr = 48_000
    base = np.zeros((sr, 2), dtype=np.float32)
    clip = np.ones((sr // 10, 1), dtype=np.float32) * 0.1
    left = pan_stereo(clip, -1.0)
    right = pan_stereo(clip, 1.0)
    assert left[:, 0].mean() > left[:, 1].mean()
    assert right[:, 1].mean() > right[:, 0].mean()
    mixed = overlay(base, clip, sr, start_sec=0.2)
    assert mixed.shape == base.shape
    assert mixed.sum() > 0
