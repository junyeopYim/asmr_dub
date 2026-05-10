from __future__ import annotations

import logging as py_logging
from io import StringIO

from rich.console import Console

from asmr_dub_pipeline.logging import configure_logging
from asmr_dub_pipeline.pipeline.stages import common, synth_gpt_sovits
from asmr_dub_pipeline.schemas import JapaneseScript, PipelineManifest, Segment


def test_configure_logging_suppresses_httpx_request_info(caplog) -> None:
    httpx_logger = py_logging.getLogger("httpx")
    httpcore_logger = py_logging.getLogger("httpcore")
    original_httpx_level = httpx_logger.level
    original_httpcore_level = httpcore_logger.level
    try:
        httpx_logger.setLevel(py_logging.NOTSET)
        httpcore_logger.setLevel(py_logging.NOTSET)

        configure_logging(py_logging.INFO)

        with caplog.at_level(py_logging.INFO):
            httpx_logger.info('HTTP Request: POST http://127.0.0.1:9880/tts "HTTP/1.1 200 OK"')

        assert not caplog.records
        assert httpx_logger.getEffectiveLevel() == py_logging.WARNING
        assert httpcore_logger.getEffectiveLevel() == py_logging.WARNING
    finally:
        httpx_logger.setLevel(original_httpx_level)
        httpcore_logger.setLevel(original_httpcore_level)


def test_segment_progress_can_distinguish_completed_jobs_from_segment_index(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        common,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=200),
    )
    manifest = PipelineManifest(
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="seg_0001.wav",
                audio_for_mix="seg_0001.wav",
                status="failed",
            ),
            Segment(
                id="seg_0002",
                start=1.0,
                end=2.0,
                duration=1.0,
                audio_for_gemma="seg_0002.wav",
                audio_for_mix="seg_0002.wav",
                status="scripted",
            ),
            Segment(
                id="seg_0250",
                start=2.0,
                end=3.0,
                duration=1.0,
                audio_for_gemma="seg_0250.wav",
                audio_for_mix="seg_0250.wav",
                status="synthesized",
            ),
        ]
    )

    common._log_segment_progress(
        "synth",
        250,
        1083,
        manifest.segments[2],
        manifest,
        started_at=0.0,
        last_logged_at=-1_000_000.0,
        progress_index=25,
        counts_label="status_counts",
    )

    rendered = output.getvalue()
    assert "synth: done=25/1083 (2.3%)" in rendered
    assert "latest=seg_0250 segment_index=250 status=synthesized" in rendered
    assert "status_counts=failed:1, scripted:1, synthesized:1" in rendered
    assert " counts=" not in rendered


def test_duration_rewrite_logging_shows_model_rewrite(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(
        synth_gpt_sovits,
        "console",
        Console(file=output, force_terminal=False, color_system=None, width=240),
    )

    synth_gpt_sovits._log_duration_rewrite_result(
        segment_id="seg_0042",
        metadata={
            "reason": "too_short",
            "accepted": True,
            "before": "짧아.",
            "after": "조금 더 길게 말해볼게요.",
            "current_speech_chars": 2,
            "speech_chars": 11,
            "target_speech_chars": 10,
            "min_speech_chars": 8,
            "max_speech_chars": 12,
        },
    )

    rendered = output.getvalue()
    assert "synth duration-rewrite seg_0042" in rendered
    assert "reason=too_short" in rendered
    assert "accepted=true" in rendered
    assert "chars=2->11 target=10 range=8-12" in rendered
    assert 'before="짧아."' in rendered
    assert 'after="조금 더 길게 말해볼게요."' in rendered


def test_duration_rewrite_relaxed_accepts_small_speech_char_shortfall() -> None:
    metadata = {
        "reason": "too_short",
        "accepted": False,
        "current_speech_chars": 20,
        "speech_chars": 27,
        "target_speech_chars": 34,
        "min_speech_chars": 28,
        "rejected_reasons": ["speech_chars_below_min:27<28"],
    }

    assert synth_gpt_sovits._maybe_relax_duration_rewrite_acceptance(metadata)
    assert metadata["accepted"] is True
    assert metadata["accepted_relaxed"] is True
    assert metadata["rejected_reasons"] == []
    assert metadata["original_rejected_reasons"] == ["speech_chars_below_min:27<28"]


def test_duration_rewrite_relaxed_rejects_preflight_and_large_shortfall() -> None:
    preflight_metadata = {
        "reason": "too_short",
        "accepted": False,
        "current_speech_chars": 20,
        "speech_chars": 27,
        "target_speech_chars": 34,
        "min_speech_chars": 28,
        "rejected_reasons": [
            "speech_chars_below_min:27<28",
            "preflight_blocked:korean_tts_suspicious_truncated_sentence",
        ],
    }
    large_shortfall_metadata = {
        "reason": "too_short",
        "accepted": False,
        "current_speech_chars": 20,
        "speech_chars": 22,
        "target_speech_chars": 34,
        "min_speech_chars": 28,
        "rejected_reasons": ["speech_chars_below_min:22<28"],
    }

    assert not synth_gpt_sovits._maybe_relax_duration_rewrite_acceptance(preflight_metadata)
    assert not synth_gpt_sovits._maybe_relax_duration_rewrite_acceptance(large_shortfall_metadata)
    assert preflight_metadata["accepted"] is False
    assert large_shortfall_metadata["accepted"] is False


def test_duration_rewrite_retry_requires_changed_rewrite() -> None:
    original = JapaneseScript(
        ja_text="原文",
        tts_text="조금 짧게 말할게요.",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    same_text = original.model_copy(deep=True)
    changed_text = original.model_copy(update={"tts_text": "조금 더 길게 다시 말할게요."}, deep=True)

    assert not synth_gpt_sovits._should_retry_duration_rewrite_result(None, original)
    assert not synth_gpt_sovits._should_retry_duration_rewrite_result(same_text, original)
    assert synth_gpt_sovits._should_retry_duration_rewrite_result(changed_text, original)
