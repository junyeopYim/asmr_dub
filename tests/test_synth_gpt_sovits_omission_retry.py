from __future__ import annotations

import json
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Lock, get_ident

import numpy as np
import pytest
from conftest import write_tiny_wav

from asmr_dub_pipeline.asr.base import ASRChunk, ASRWord
from asmr_dub_pipeline.audio.features import duration_sec, load_audio, write_audio
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gpt_sovits.client import build_tts_request
from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.stages import synth_gpt_sovits as synth_stage
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import (
    _countdown_anchor_aligned_start_frame,
    _countdown_apply_slice_start_backoff,
    _countdown_candidate_anchor_fit_score,
    _countdown_candidate_transcript_preference_score,
    _countdown_canonical_pack_prompt_specs,
    _countdown_carrier_full_sentence_prefilter_match,
    _countdown_carrier_templates_for_token_config,
    _countdown_extend_slice_end_to_energy_valley,
    _countdown_has_token_specific_carrier_templates,
    _countdown_sequence_qc,
    _countdown_token_alignment_from_chunks,
    _gsv_candidate_selection_score,
    _select_gsv_candidate_for_mix,
)
from asmr_dub_pipeline.pipeline.steps import countdown_synth_step, init_project, synth_step
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
    TTSCandidate,
)
from asmr_dub_pipeline.script.countdown import countdown_korean_tokens


def _write_tone_wav(
    path: Path,
    duration: float,
    sample_rate: int = 48_000,
    frequency_hz: float = 220.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    tone = 0.05 * np.sin(2 * np.pi * frequency_hz * t)
    write_audio(path, np.stack([tone, tone], axis=1), sample_rate)
    return path


def _write_chirp_wav(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    sweep = 180.0 + 220.0 * (t / max(duration, 0.001))
    phase = 2 * np.pi * np.cumsum(sweep) / sample_rate
    tone = 0.05 * np.sin(phase)
    write_audio(path, np.stack([tone, tone], axis=1), sample_rate)
    return path


def _write_tone_gap_tone_wav(
    path: Path,
    *,
    first_sec: float,
    gap_sec: float,
    second_sec: float,
    sample_rate: int = 48_000,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def tone(duration: float, frequency_hz: float) -> np.ndarray:
        t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
        return 0.05 * np.sin(2 * np.pi * frequency_hz * t)

    first = tone(first_sec, 220.0)
    gap = np.zeros(int(sample_rate * gap_sec), dtype=np.float32)
    second = tone(second_sec, 260.0)
    mono = np.concatenate([first, gap, second])
    write_audio(path, np.stack([mono, mono], axis=1), sample_rate)
    return path


def test_countdown_candidate_anchor_fit_score_penalizes_slack_and_overflow() -> None:
    clean_candidate = {
        "payload": {
            "countdown_carrier_bank": {
                "active_anchor_fit": {
                    "leading_silence_sec": 0.02,
                    "trailing_silence_sec": 0.03,
                    "next_anchor_overflow_sec": 0.0,
                    "pre_anchor_underflow_sec": 0.0,
                }
            }
        }
    }
    sloppy_candidate = {
        "payload": {
            "countdown_carrier_bank": {
                "active_anchor_fit": {
                    "leading_silence_sec": 0.22,
                    "trailing_silence_sec": 0.24,
                    "next_anchor_overflow_sec": 0.12,
                    "pre_anchor_underflow_sec": 0.0,
                }
            }
        }
    }

    assert _countdown_candidate_anchor_fit_score(clean_candidate) > 0.85
    assert _countdown_candidate_anchor_fit_score(sloppy_candidate) < 0.35


def test_countdown_transcript_preference_prefers_one_over_ye_alias() -> None:
    numeric_candidate = {"payload": {"pronunciation_qc": {"transcript": "1"}}}
    hangul_candidate = {"payload": {"pronunciation_qc": {"transcript": "일"}}}
    alias_candidate = {"payload": {"pronunciation_qc": {"transcript": "예"}}}

    assert _countdown_candidate_transcript_preference_score("일", numeric_candidate) == pytest.approx(1.0)
    assert _countdown_candidate_transcript_preference_score("일", hangul_candidate) == pytest.approx(1.0)
    assert _countdown_candidate_transcript_preference_score("일", alias_candidate) < 1.0
    assert _countdown_candidate_transcript_preference_score("일", alias_candidate) > 0.0


def test_countdown_full_sentence_prefilter_requires_exact_single_syllable_token() -> None:
    false_prefix_match = _countdown_carrier_full_sentence_prefilter_match(
        "이",
        "이번 숫자는 일 입니다",
    )
    exact_match = _countdown_carrier_full_sentence_prefilter_match(
        "이",
        "이번 숫자는 이 입니다",
    )
    digit_match = _countdown_carrier_full_sentence_prefilter_match(
        "이",
        "2 번만요",
    )

    assert false_prefix_match["matched"] is False
    assert false_prefix_match["coverage"] == 0.0
    assert exact_match["matched"] is True
    assert exact_match["coverage"] == 1.0
    assert digit_match["matched"] is True
    assert digit_match["coverage"] == 1.0


def test_countdown_token_alignment_ignores_single_syllable_inside_neighbor_words() -> None:
    chunks = [
        ASRChunk(
            start=0.0,
            end=0.9,
            text="이번 숫자는 일 입니다",
            language="ko",
            words=[
                ASRWord(start=0.0, end=0.18, text="이번"),
                ASRWord(start=0.22, end=0.42, text="숫자는"),
                ASRWord(start=0.46, end=0.58, text="일"),
                ASRWord(start=0.62, end=0.86, text="입니다"),
            ],
        )
    ]
    exact_chunks = [
        ASRChunk(
            start=0.0,
            end=0.9,
            text="이번 숫자는 이 입니다",
            language="ko",
            words=[
                ASRWord(start=0.0, end=0.18, text="이번"),
                ASRWord(start=0.22, end=0.42, text="숫자는"),
                ASRWord(start=0.46, end=0.58, text="이"),
                ASRWord(start=0.62, end=0.86, text="입니다"),
            ],
        )
    ]

    assert _countdown_token_alignment_from_chunks("이", chunks, 0.9) is None
    alignment = _countdown_token_alignment_from_chunks("이", exact_chunks, 0.9)
    assert alignment == {
        "source": "asr_word",
        "start_sec": 0.46,
        "end_sec": 0.58,
        "text": "이",
    }


def test_countdown_anchor_aligned_start_frame_subtracts_active_leading_silence() -> None:
    assert _countdown_anchor_aligned_start_frame(
        slot_start_frame=4_800,
        active_leading_frames=960,
        total_frames=48_000,
    ) == 3_840
    assert _countdown_anchor_aligned_start_frame(
        slot_start_frame=240,
        active_leading_frames=960,
        total_frames=48_000,
    ) == 0


def test_countdown_token_specific_templates_replace_generic_carriers() -> None:
    cfg = ProjectConfig(
        gsv_countdown_carrier_templates=["generic {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=True,
        gsv_countdown_carrier_numeric_unit_templates=["{token} 번만요."],
        gsv_countdown_carrier_token_templates={"구": ["구팔칠.", "구, 팔, 칠."]},
    )

    assert _countdown_carrier_templates_for_token_config(cfg, "구") == [
        (0, "구팔칠.", "구팔칠."),
        (1, "구, 팔, 칠.", "구, 팔, 칠."),
    ]


def test_countdown_token_specific_templates_disable_pack_warmup_for_token() -> None:
    cfg = ProjectConfig(gsv_countdown_carrier_token_templates={"사": ["사아 번만요."]})

    assert _countdown_has_token_specific_carrier_templates(cfg, "사") is True
    assert _countdown_has_token_specific_carrier_templates(cfg, "구") is False


def test_countdown_energy_boundary_extends_slice_to_tail_valley() -> None:
    sample_rate = 48_000
    total_frames = int(sample_rate * 0.55)
    audio = np.zeros((total_frames, 2), dtype=np.float32)
    vowel_end = int(sample_rate * 0.18)
    tail_end = int(sample_rate * 0.36)
    t = np.arange(total_frames, dtype=np.float32) / sample_rate
    wave = np.sin(2 * np.pi * 180.0 * t)
    audio[:vowel_end, :] = (0.18 * wave[:vowel_end])[:, None]
    tail = np.linspace(0.16, 0.015, tail_end - vowel_end, dtype=np.float32)
    audio[vowel_end:tail_end, :] = (tail * wave[vowel_end:tail_end])[:, None]

    initial_end = int(sample_rate * 0.22)
    result = _countdown_extend_slice_end_to_energy_valley(
        audio,
        start_frame=0,
        end_frame=initial_end,
        sample_rate=sample_rate,
        token_text="삼",
        enabled=True,
        max_extension_sec=0.08,
        coda_max_extension_sec=0.20,
    )

    assert result["end_frame"] > int(sample_rate * 0.34)
    assert result["extended_sec"] > 0.10
    assert result["max_extension_sec"] == pytest.approx(0.20)
    assert result["gate"] == "pass"
    assert result["reason"] == "extended_to_energy_valley"


def test_countdown_energy_boundary_keeps_quiet_boundary_unchanged() -> None:
    sample_rate = 48_000
    total_frames = int(sample_rate * 0.55)
    audio = np.zeros((total_frames, 2), dtype=np.float32)
    active_end = int(sample_rate * 0.20)
    t = np.arange(active_end, dtype=np.float32) / sample_rate
    wave = 0.16 * np.sin(2 * np.pi * 180.0 * t)
    audio[:active_end, :] = wave[:, None]

    initial_end = int(sample_rate * 0.30)
    result = _countdown_extend_slice_end_to_energy_valley(
        audio,
        start_frame=0,
        end_frame=initial_end,
        sample_rate=sample_rate,
        token_text="사",
        enabled=True,
    )

    assert result["end_frame"] == initial_end
    assert result["extended_sec"] == 0.0
    assert result["gate"] == "pass"


def test_countdown_retime_skips_even_internal_gap() -> None:
    sample_rate = 48_000
    first = np.sin(2 * np.pi * 220.0 * np.arange(int(sample_rate * 0.42)) / sample_rate)
    gap = np.zeros(int(sample_rate * 0.22), dtype=np.float32)
    second = np.sin(2 * np.pi * 260.0 * np.arange(int(sample_rate * 0.42)) / sample_rate)
    mono = (0.05 * np.concatenate([first, gap, second])).astype(np.float32)
    audio = np.stack([mono, mono], axis=1)

    retimed, qc = synth_stage._countdown_retime_phrase_audio(
        audio,
        sample_rate,
        expected_units=2,
    )

    assert qc["applied"] is False
    assert qc["reason"] == "timing_within_gate"
    assert retimed.shape == audio.shape


def test_countdown_slice_start_backoff_extends_onset_without_underflow() -> None:
    sample_rate = 48_000

    assert _countdown_apply_slice_start_backoff(
        1_200,
        sample_rate=sample_rate,
        backoff_sec=0.02,
    ) == 240
    assert _countdown_apply_slice_start_backoff(
        600,
        sample_rate=sample_rate,
        backoff_sec=0.02,
    ) == 0
    assert _countdown_apply_slice_start_backoff(
        1_200,
        sample_rate=sample_rate,
        backoff_sec=0.0,
    ) == 1_200


def test_gsv_candidate_selection_prefers_clean_delivery_over_exact_penalized_duration() -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=5.0,
        duration=5.0,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
        script=JapaneseScript(
            ja_text="ゆっくり聞いてください",
            tts_text="천천히 들어주세요.",
            tts_language="ko",
            ref_style="whisper_close",
            expected_tts_duration_sec=5.0,
        ),
    )
    exact_but_penalized = TTSCandidate(
        candidate_index=0,
        seed=1,
        output_path="exact.wav",
        duration_sec=5.0,
        backend="gpt-sovits",
        duration_ratio=1.0,
        duration_gate="pass",
        acceptable_for_mix=True,
        selection_reason="source_pause_padding_rescue",
        payload={
            "audio_qc": {"gate": "pass"},
            "speed_factor": 1.25,
            "fallback_used": True,
            "requested_ref_style": "whisper_close",
            "resolved_ref_style": "sleepy",
            "postprocess": {
                "edge_silence_trim": {
                    "leading_trim_sec": 0.5,
                    "trailing_trim_sec": 0.5,
                }
            },
        },
    )
    clean_delivery = TTSCandidate(
        candidate_index=1,
        seed=2,
        output_path="clean.wav",
        duration_sec=4.7,
        backend="gpt-sovits",
        duration_ratio=0.94,
        duration_gate="pass",
        acceptable_for_mix=True,
        selection_reason="duration_and_language_contract_pass",
        payload={
            "audio_qc": {"gate": "pass"},
            "speed_factor": 1.0,
            "fallback_used": False,
            "requested_ref_style": "whisper_close",
            "resolved_ref_style": "whisper_close",
            "postprocess": {
                "edge_silence_trim": {
                    "leading_trim_sec": 0.02,
                    "trailing_trim_sec": 0.02,
                }
            },
        },
    )

    assert _gsv_candidate_selection_score(clean_delivery, segment) > _gsv_candidate_selection_score(
        exact_but_penalized,
        segment,
    )
    assert _select_gsv_candidate_for_mix(
        [exact_but_penalized, clean_delivery],
        segment,
    ) is clean_delivery


def test_gsv_candidate_selection_prefers_clear_pronunciation_over_exact_duration() -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=5.0,
        duration=5.0,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
        script=JapaneseScript(
            ja_text="ゆっくり聞いてください",
            tts_text="천천히 들어주세요.",
            tts_language="ko",
            ref_style="whisper_close",
            expected_tts_duration_sec=5.0,
        ),
    )
    exact_but_mumbled = TTSCandidate(
        candidate_index=0,
        seed=1,
        output_path="exact.wav",
        duration_sec=5.0,
        backend="gpt-sovits",
        duration_ratio=1.0,
        duration_gate="pass",
        acceptable_for_mix=True,
        selection_reason="duration_and_language_contract_pass",
        payload={
            "audio_qc": {"gate": "pass"},
            "speed_factor": 1.0,
            "requested_ref_style": "whisper_close",
            "resolved_ref_style": "whisper_close",
            "pronunciation_qc": {
                "gate": "fail",
                "coverage": 0.31,
                "transcript": "천천히",
            },
        },
    )
    slightly_short_clear = TTSCandidate(
        candidate_index=1,
        seed=2,
        output_path="clear.wav",
        duration_sec=4.7,
        backend="gpt-sovits",
        duration_ratio=0.94,
        duration_gate="pass",
        acceptable_for_mix=True,
        selection_reason="duration_and_language_contract_pass",
        payload={
            "audio_qc": {"gate": "pass"},
            "speed_factor": 1.0,
            "requested_ref_style": "whisper_close",
            "resolved_ref_style": "whisper_close",
            "pronunciation_qc": {
                "gate": "pass",
                "coverage": 0.97,
                "transcript": "천천히 들어주세요",
            },
        },
    )

    assert _gsv_candidate_selection_score(slightly_short_clear, segment) > _gsv_candidate_selection_score(
        exact_but_mumbled,
        segment,
    )
    assert _select_gsv_candidate_for_mix(
        [exact_but_mumbled, slightly_short_clear],
        segment,
    ) is slightly_short_clear


def test_countdown_korean_tokens_use_sino_korean_readings() -> None:
    assert countdown_korean_tokens([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]) == [
        "십",
        "구",
        "팔",
        "칠",
        "육",
        "오",
        "사",
        "삼",
        "이",
        "일",
        "영",
    ]


def test_countdown_canonical_pack_prompt_specs_prefer_digit_space_for_long_runs() -> None:
    values = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    tokens = countdown_korean_tokens(values)

    specs = _countdown_canonical_pack_prompt_specs(values, tokens)

    assert specs[0]["prompt_kind"] == "digit_space"
    assert specs[0]["prompt_text"] == "9 8 7 6 5 4 3 2 1 0"
    assert specs[0]["expected_text"] == "구 팔 칠 육 오 사 삼 이 일 영"
    assert specs[0]["speed_factor"] == pytest.approx(1.0)
    assert any(
        spec["prompt_kind"] == "digit_space" and spec["speed_factor"] == pytest.approx(0.9)
        for spec in specs
    )


def test_countdown_sequence_qc_rejects_missing_or_duplicate_digit_tail() -> None:
    values = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    exact = _countdown_sequence_qc(values, {"transcript": "9876543210", "gate": "pass"})
    missing = _countdown_sequence_qc(values, {"transcript": "987654320", "gate": "pass"})
    duplicate = _countdown_sequence_qc(values, {"transcript": "98765432110", "gate": "pass"})

    assert exact["gate"] == "pass"
    assert exact["exact_match"] is True
    assert missing["gate"] == "fail"
    assert missing["exact_match"] is False
    assert "countdown_sequence_mismatch" in missing["issues"]
    assert duplicate["gate"] == "fail"
    assert duplicate["exact_match"] is False


def test_countdown_compact_phrase_uses_comma_separated_text(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="compact",
        gsv_countdown_candidate_count=1,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=6.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=6.0,
        duration=6.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3 2 1 0",
            language="ja",
            backend="mock",
            start=0.0,
            end=6.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3 2 1 0",
            ja_text="5 4 3 2 1 0",
            tts_text="오, 사, 삼, 이, 일, 영",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=3.0,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3, 2, 1, 0],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=3.0)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )
    assert requests == ["오, 사, 삼, 이, 일, 영"]
    assert metadata["compact_text"] == "오, 사, 삼, 이, 일, 영"


def test_countdown_chunk_bank_generates_context_windows_and_prefers_stable_slice(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="chunk_bank",
        gsv_countdown_chunk_candidate_count=10,
        gsv_countdown_chunk_max_size=10,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.05,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        status="scripted",
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            token_count = len([part for part in request.text.split(",") if part.strip()])
            duration = max(0.3, 0.3 * token_count)
            if "cand_06" in output_path.name:
                _write_chirp_wav(output_path, duration=duration)
            else:
                _write_tone_wav(output_path, duration=duration)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            stem = audio_path.stem
            token = stem.rsplit("_", 1)[-1]
            if "chunk_size_03" in stem and ("cand_06" in stem or "cand_07" in stem):
                return [ASRChunk(start=0.0, end=0.3, text=token, language="ko")]
            return [ASRChunk(start=0.0, end=0.3, text="잡음", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert metadata["renderer"] == "countdown_chunk_bank"
    assert len(requests) == 60
    assert {request for request in requests} == {
        "삼",
        "이",
        "일",
        "삼, 이",
        "이, 일",
        "삼, 이, 일",
    }
    assert metadata["chunk_bank"]["chunk_sizes"] == [1, 2, 3]
    assert {placement["selected_chunk_size"] for placement in metadata["token_placements"]} == {3}
    assert {placement["selected_candidate_index"] for placement in metadata["token_placements"]} == {7}
    assert all(placement["selected_pronunciation_gate"] == "pass" for placement in metadata["token_placements"])


def test_countdown_carrier_bank_tries_five_carriers_and_uses_segment_ref(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    carrier_templates = [
        "carrier zero {token}",
        "carrier one {token}",
        "carrier two {token}",
        "carrier three {token}",
        "carrier four {token}",
    ]
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_ref_mode="segment",
        gsv_ref_min_sec=1.0,
        gsv_ref_max_sec=8.0,
        gsv_ref_min_quality_score=0.0,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=carrier_templates,
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "speaker_count": 1,
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            },
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[tuple[str, str]] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            requests.append((request.text, request.ref_audio_path))
            duration = 0.9 if "carrier_02" in output_path.name else 1.8
            _write_tone_wav(output_path, duration=duration)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            stem = audio_path.stem
            token = stem.rsplit("_", 1)[-1]
            text = token if "carrier_02" in stem else "잡음"
            return [ASRChunk(start=0.0, end=0.3, text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert len(requests) == 9
    assert {Path(ref_path).name for _text, ref_path in requests} == {"seg_0001_mix.wav"}
    assert metadata["renderer"] == "countdown_carrier_bank"
    assert metadata["carrier_bank"]["carrier_template_count"] == 5
    assert metadata["carrier_bank"]["segment_ref"]["used"] is True
    assert {
        placement["selected_carrier_index"]
        for placement in metadata["token_placements"]
    } == {2}
    assert all(placement["selected_pronunciation_gate"] == "pass" for placement in metadata["token_placements"])


def test_countdown_carrier_bank_generates_multiple_pronunciation_profiles_per_template(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["plain {token}", "close {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=2,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.25,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[tuple[str, int, float, int]] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            assert options is not None
            requests.append((text, options.seed, options.temperature, options.top_k))
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.72)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            stem = audio_path.stem
            token = stem.rsplit("_", 1)[-1]
            text = token if "_cand_01_" in stem else "잡음"
            return [ASRChunk(start=0.0, end=0.3, text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert len(requests) == 6
    assert metadata["carrier_bank"]["carrier_template_count"] == 2
    assert metadata["carrier_bank"]["carrier_candidate_count"] == 2
    assert {request[2] for request in requests} != {0.55}
    assert {
        placement["selected_carrier_variant_index"]
        for placement in metadata["token_placements"]
    } == {1}
    assert all(placement["selected_pronunciation_gate"] == "pass" for placement in metadata["token_placements"])


def test_countdown_carrier_bank_searches_slice_windows_for_pronunciation_pass(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["carrier {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_slice_window_sec=[0.30, 0.42],
        gsv_countdown_carrier_slice_window_offset_sec=[-0.06, 0.0, 0.06],
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.9)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            stem = audio_path.stem
            token = stem.rsplit("_", 1)[-1]
            text = token if "_full_" in stem else token if "slice_window_03" in stem else "잡음"
            return [ASRChunk(start=0.0, end=0.3, text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert requests == ["carrier 삼", "carrier 이", "carrier 일"]
    assert segment.status == "synthesized"
    assert metadata["carrier_bank"]["slice_search_enabled"] is True
    assert metadata["carrier_bank"]["slice_window_count"] >= 4
    assert {
        placement["selected_slice_window_index"]
        for placement in metadata["token_placements"]
    } == {3}
    assert all(placement["selected_pronunciation_gate"] == "pass" for placement in metadata["token_placements"])


def test_countdown_carrier_bank_skips_slice_windows_when_full_sentence_lacks_token(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["bad {token}", "good {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_full_sentence_prefilter_enabled=True,
        gsv_countdown_carrier_full_sentence_prefilter_min_coverage=1.0,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_slice_window_sec=[0.30, 0.42],
        gsv_countdown_carrier_slice_window_offset_sec=[-0.06, 0.0, 0.06],
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []
    transcribed_stems: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.9)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            stem = audio_path.stem
            transcribed_stems.append(stem)
            token = stem.rsplit("_", 1)[-1]
            text = ("잡음" if "carrier_00" in stem else token) if "_full_" in stem else token
            return [ASRChunk(start=0.0, end=0.3, text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert requests == ["bad 삼", "good 삼", "bad 이", "good 이", "bad 일", "good 일"]
    assert segment.status == "synthesized"
    assert any("carrier_00" in stem and "_full_" in stem for stem in transcribed_stems)
    assert not any("carrier_00" in stem and "slice_window" in stem for stem in transcribed_stems)
    assert {
        placement["selected_carrier_index"]
        for placement in metadata["token_placements"]
    } == {1}
    assert metadata["carrier_bank"]["full_sentence_prefilter"]["enabled"] is True
    assert all(
        candidate["payload"]["countdown_carrier_bank"]["full_sentence_prefilter"]["gate"] == "pass"
        for candidate in metadata["token_candidates"]
    )


def test_countdown_carrier_bank_bulk_asr_stops_after_target_pass(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_candidate_count=1,
        gsv_countdown_carrier_templates=[
            "{token} 첫번째.",
            "{token} 두번째.",
            "{token} 세번째.",
        ],
        gsv_countdown_carrier_token_templates={},
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_carrier_bulk_asr_enabled=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    values = [5, 4, 3]
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        status="scripted",
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="다섯, 넷, 셋",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": values,
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    requests: list[str] = []

    class EarlyStopCountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.72)
            return output_path

    class PassingASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            return [ASRChunk(start=0.0, end=0.72, text=token, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PassingASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PassingASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", EarlyStopCountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert requests == ["오 첫번째.", "사 첫번째.", "삼 첫번째."]


def test_countdown_carrier_bank_bulk_asr_stops_slice_window_search_after_pass(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_candidate_count=1,
        gsv_countdown_carrier_templates=["carrier {token}"],
        gsv_countdown_carrier_token_templates={},
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_carrier_bulk_asr_enabled=True,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_max_slice_windows_per_candidate=5,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    transcribed_stems: list[str] = []

    class SliceStopCountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.72)
            return output_path

    class PassingASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            transcribed_stems.append(audio_path.stem)
            token = audio_path.stem.rsplit("_", 1)[-1]
            return [ASRChunk(start=0.0, end=0.72, text=token, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PassingASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PassingASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", SliceStopCountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert any("slice_window_00" in stem for stem in transcribed_stems)
    assert not any("slice_window_01" in stem for stem in transcribed_stems)
    assert {
        placement["selected_slice_window_index"]
        for placement in metadata["token_placements"]
    } == {0}


def test_countdown_carrier_bank_continues_window_search_after_active_boundary_pass(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=[],
        gsv_countdown_carrier_numeric_unit_templates=["{token} 번만요."],
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_slice_window_sec=[0.30, 0.42],
        gsv_countdown_carrier_slice_window_offset_sec=[-0.06, 0.0, 0.06],
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    transcribed_windows: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.9)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            stem = audio_path.stem
            transcribed_windows.append(stem)
            token = stem.rsplit("_", 1)[-1]
            text = token if "_full_" in stem else token if "slice_window_02" in stem else "잡음"
            return [ASRChunk(start=0.0, end=0.3, text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert any("slice_window_02" in stem for stem in transcribed_windows)
    assert any("slice_window_03" in stem for stem in transcribed_windows)
    assert {
        placement["selected_slice_window_index"]
        for placement in metadata["token_placements"]
    } == {2}
    assert {
        placement["selected_quality_tier"]
        for placement in metadata["token_placements"]
    } == {"B"}
    assert metadata["carrier_bank"]["early_stop"]["target_pronunciation_passes"] == 1


def test_countdown_carrier_bank_slices_token_by_carrier_text_position(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["{token} 그리고 아주 길게 뒤에서 속삭여요"],
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_slice_window_sec=[0.30, 0.42],
        gsv_countdown_carrier_slice_window_offset_sec=[-0.04, 0.0, 0.04],
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            silence = np.zeros(int(sample_rate * 0.08), dtype=np.float32)
            target_t = np.arange(int(sample_rate * 0.22), dtype=np.float32) / sample_rate
            suffix_t = np.arange(int(sample_rate * 0.90), dtype=np.float32) / sample_rate
            target = 0.08 * np.sin(2 * np.pi * 880.0 * target_t)
            suffix = 0.04 * np.sin(2 * np.pi * 220.0 * suffix_t)
            mono = np.concatenate([silence, target, suffix]).astype(np.float32)
            write_audio(output_path, np.stack([mono, mono], axis=1), sample_rate)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            if "_full_" in audio_path.stem:
                return [ASRChunk(start=0.0, end=1.2, text=token, language="ko")]
            data, sample_rate = load_audio(audio_path)
            mono = data[:, 0] if data.ndim == 2 else data
            if len(mono) == 0:
                text = "잡음"
            else:
                window = mono * np.hanning(len(mono))
                spectrum = np.abs(np.fft.rfft(window))
                freqs = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)
                target_power = float(np.sum(spectrum[(freqs >= 820.0) & (freqs <= 940.0)]))
                suffix_power = float(np.sum(spectrum[(freqs >= 180.0) & (freqs <= 260.0)]))
                text = token if target_power > suffix_power * 1.5 else "잡음"
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert {
        candidate["payload"]["countdown_carrier_bank"]["base_slice"]["strategy"]
        for candidate in metadata["token_candidates"]
        if candidate.get("quality_tier") == "A"
    } == {"carrier_text_position"}
    assert all(
        placement["selected_slice_start_sec"] < 0.3
        for placement in metadata["token_placements"]
    )


def test_countdown_carrier_bank_slices_sentence_final_token_from_onset(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["아주 길게 앞에서 속삭이고 마지막에 {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_slice_window_sec=[0.30],
        gsv_countdown_carrier_slice_window_offset_sec=[0.0],
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            prefix_t = np.arange(int(sample_rate * 0.70), dtype=np.float32) / sample_rate
            onset_t = np.arange(int(sample_rate * 0.10), dtype=np.float32) / sample_rate
            tail_t = np.arange(int(sample_rate * 0.25), dtype=np.float32) / sample_rate
            prefix = 0.04 * np.sin(2 * np.pi * 220.0 * prefix_t)
            onset = 0.10 * np.sin(2 * np.pi * 880.0 * onset_t)
            tail = 0.02 * np.sin(2 * np.pi * 220.0 * tail_t)
            mono = np.concatenate([prefix, onset, tail]).astype(np.float32)
            write_audio(output_path, np.stack([mono, mono], axis=1), sample_rate)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            if "_full_" in audio_path.stem:
                return [ASRChunk(start=0.0, end=1.05, text=token, language="ko")]
            data, sample_rate = load_audio(audio_path)
            mono = data[:, 0] if data.ndim == 2 else data
            if len(mono) == 0:
                text = "잡음"
            else:
                window = mono * np.hanning(len(mono))
                spectrum = np.abs(np.fft.rfft(window))
                freqs = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)
                onset_power = float(np.sum(spectrum[(freqs >= 820.0) & (freqs <= 940.0)]))
                prefix_power = float(np.sum(spectrum[(freqs >= 180.0) & (freqs <= 260.0)]))
                text = token if onset_power > prefix_power * 0.75 else "잡음"
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert all(
        placement["selected_slice_start_sec"] <= 0.72
        for placement in metadata["token_placements"]
    )
    assert {
        candidate["payload"]["countdown_carrier_bank"]["base_slice"].get("anchor")
        for candidate in metadata["token_candidates"]
        if candidate.get("quality_tier") == "A"
    } == {"suffix_end"}


def test_countdown_carrier_bank_scans_backward_for_sentence_final_token_with_long_tail(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["앞에서 아주 길게 속삭이고 마지막 숫자는 {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_slice_window_sec=[0.30],
        gsv_countdown_carrier_slice_window_offset_sec=[0.0],
        gsv_countdown_carrier_max_slice_windows_per_candidate=9,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            prefix_t = np.arange(int(sample_rate * 0.70), dtype=np.float32) / sample_rate
            onset_t = np.arange(int(sample_rate * 0.10), dtype=np.float32) / sample_rate
            tail_t = np.arange(int(sample_rate * 0.65), dtype=np.float32) / sample_rate
            prefix = 0.04 * np.sin(2 * np.pi * 220.0 * prefix_t)
            onset = 0.10 * np.sin(2 * np.pi * 880.0 * onset_t)
            tail = 0.02 * np.sin(2 * np.pi * 220.0 * tail_t)
            mono = np.concatenate([prefix, onset, tail]).astype(np.float32)
            write_audio(output_path, np.stack([mono, mono], axis=1), sample_rate)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            if "_full_" in audio_path.stem:
                return [ASRChunk(start=0.0, end=1.45, text=token, language="ko")]
            data, sample_rate = load_audio(audio_path)
            mono = data[:, 0] if data.ndim == 2 else data
            window = mono * np.hanning(len(mono)) if len(mono) else mono
            spectrum = np.abs(np.fft.rfft(window)) if len(window) else np.array([])
            freqs = np.fft.rfftfreq(len(window), d=1.0 / sample_rate) if len(window) else np.array([])
            onset_power = float(np.sum(spectrum[(freqs >= 820.0) & (freqs <= 940.0)]))
            prefix_power = float(np.sum(spectrum[(freqs >= 180.0) & (freqs <= 260.0)]))
            text = token if onset_power > prefix_power * 0.75 else "잡음"
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert all(
        0.65 <= placement["selected_slice_start_sec"] <= 0.82
        for placement in metadata["token_placements"]
    )


def test_countdown_carrier_bank_warmup_prefers_pack_token_over_sentence_carrier(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["긴 문장 사이에서 {token} 다시 긴 문장"],
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=True,
        gsv_countdown_token_bank_warmup_enabled=True,
        gsv_countdown_carrier_slice_search_enabled=False,
        gsv_countdown_carrier_quality_retry_max_rounds=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 0.9 if request.text.count(",") >= 2 else 1.2
            frequency = 880.0 if request.text.count(",") >= 2 else 220.0
            _write_tone_wav(output_path, duration=duration, frequency_hz=frequency)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            if "pack_token_bank" in str(audio_path):
                return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=token, language="ko")]
            if "_full_" in audio_path.stem:
                return [ASRChunk(start=0.0, end=duration_sec(audio_path), text="잡음", language="ko")]
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text="잡음", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert any(text == "삼, 삼, 삼" for text in requests)
    assert {
        placement["selected_candidate_source"]
        for placement in metadata["token_placements"]
    } == {"pack_take"}


def test_countdown_carrier_bank_uses_numeric_unit_onset_slice_without_prompted_asr(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=[],
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_quality_retry_max_rounds=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.25,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
        asr_batched_inference=True,
        asr_initial_prompt="SHOULD_NOT_LEAK_TO_SHORT_SLICE",
        asr_hotwords="SHOULD_NOT_LEAK_TO_SHORT_SLICE",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=1.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=1.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=0.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []
    asr_configs: list[dict[str, object]] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            target_t = np.arange(int(sample_rate * 0.18), dtype=np.float32) / sample_rate
            suffix_t = np.arange(int(sample_rate * 0.42), dtype=np.float32) / sample_rate
            target = 0.10 * np.sin(2 * np.pi * 880.0 * target_t)
            suffix = 0.04 * np.sin(2 * np.pi * 220.0 * suffix_t)
            mono = np.concatenate([target, suffix]).astype(np.float32)
            write_audio(output_path, np.stack([mono, mono], axis=1), sample_rate)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def __init__(self, config: dict[str, object]) -> None:
            self.config = config

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            digit_by_token = {"삼": "3", "이": "2", "일": "1"}
            digit = digit_by_token.get(token, token)
            if "_full_" in audio_path.stem:
                return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=f"{digit} 번만요", language="ko")]
            assert self.config.get("initial_prompt") in {"", None}
            assert self.config.get("hotwords") in {"", None}
            text = digit if duration_sec(audio_path) <= 0.23 else f"{digit}초"
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        asr_configs.append(dict(config))
        return PronunciationASRBackend(dict(config))

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert requests[0] == "삼 번만요."
    assert all("legacy carrier" not in text for text in requests)
    assert any(
        config.get("initial_prompt") in {"", None}
        and config.get("hotwords") in {"", None}
        and config.get("batched_inference") is False
        for config in asr_configs
    )
    placement = next(item for item in metadata["token_placements"] if item["text"] == "이")
    assert placement["selected_carrier_text"] == "이 번만요."
    assert placement["selected_slice_duration_sec"] == pytest.approx(0.22, abs=0.001)
    assert placement["selected_pronunciation_gate"] == "pass"
    assert placement["selected_slice_window_kind"] == "numeric_unit_onset_window"


def test_countdown_carrier_bank_treats_imnida_template_as_numeric_onset_slice(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=[],
        gsv_countdown_carrier_numeric_unit_templates=["{token} 입니다."],
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=True,
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_carrier_quality_retry_max_rounds=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.25,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=1.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=1.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=0.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            sample_rate = 48_000
            target_t = np.arange(int(sample_rate * 0.18), dtype=np.float32) / sample_rate
            suffix_t = np.arange(int(sample_rate * 0.42), dtype=np.float32) / sample_rate
            target = 0.10 * np.sin(2 * np.pi * 880.0 * target_t)
            suffix = 0.04 * np.sin(2 * np.pi * 220.0 * suffix_t)
            mono = np.concatenate([target, suffix]).astype(np.float32)
            write_audio(output_path, np.stack([mono, mono], axis=1), sample_rate)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            if "_full_" in audio_path.stem:
                return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=f"{token} 입니다", language="ko")]
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=token, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert requests[0] == "삼 입니다."
    placement = next(item for item in metadata["token_placements"] if item["text"] == "이")
    assert placement["selected_carrier_text"] == "이 입니다."
    assert placement["selected_slice_window_kind"] == "numeric_unit_onset_window"
    assert placement["selected_slice_window_index"] == 0


def test_countdown_synth_force_requeues_manual_review_countdown(
    tmp_project_dir: Path,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_countdown_renderer="compact",
        gsv_countdown_timing_mode="even_grid",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=1.2)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        status="needs_manual_review",
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=1.2,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.2,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            },
            "countdown_renderer_skip": {"reason": "previous_failure"},
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=True,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert metadata["renderer"] == "countdown_compact_phrase"


def test_countdown_carrier_bank_retries_generation_until_a_tier_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["carrier {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=False,
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=False,
        gsv_countdown_carrier_quality_retry_enabled=True,
        gsv_countdown_carrier_quality_retry_max_rounds=2,
        gsv_countdown_carrier_quality_retry_target_tier="A",
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=True,
        gsv_countdown_carrier_target_pronunciation_passes=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 1.95 if "_cand_00_" in output_path.name else 1.05
            _write_tone_wav(output_path, duration=duration)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            text = "잡음" if "_cand_00_" in audio_path.stem else token
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert requests == ["carrier 삼", "carrier 이", "carrier 일", "carrier 삼", "carrier 이", "carrier 일"]
    assert segment.status == "synthesized"
    assert metadata["carrier_bank"]["quality_retry"]["target_tier"] == "A"
    assert {
        placement["selected_carrier_variant_index"]
        for placement in metadata["token_placements"]
    } == {1}
    assert {
        placement["selected_quality_tier"]
        for placement in metadata["token_placements"]
    } == {"A"}
    assert {
        candidate["carrier_variant_index"]
        for candidate in metadata["token_candidates"]
        if candidate["quality_tier"] == "A"
    } == {1}


def test_countdown_carrier_bank_retries_a_tier_with_numeric_unit_only(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["carrier {token}"],
        gsv_countdown_carrier_numeric_unit_enabled=True,
        gsv_countdown_carrier_numeric_unit_templates=["{token} 번만요."],
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_token_bank_enabled=False,
        gsv_countdown_token_bank_warmup_enabled=False,
        gsv_countdown_token_bank_pack_warmup_enabled=False,
        gsv_countdown_carrier_slice_search_enabled=False,
        gsv_countdown_carrier_quality_retry_enabled=True,
        gsv_countdown_carrier_quality_retry_max_rounds=2,
        gsv_countdown_carrier_quality_retry_target_tier="A",
        gsv_countdown_carrier_stop_window_search_after_pronunciation_pass=False,
        gsv_countdown_carrier_target_pronunciation_passes=32,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 1.95 if "_cand_00_" in output_path.name else 1.05
            _write_tone_wav(output_path, duration=duration)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            text = "잡음" if "_cand_00_" in audio_path.stem else token
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert requests == [
        "삼 번만요.",
        "carrier 삼",
        "삼 번만요.",
        "이 번만요.",
        "carrier 이",
        "이 번만요.",
        "일 번만요.",
        "carrier 일",
        "일 번만요.",
    ]
    assert {
        placement["selected_candidate_source"]
        for placement in metadata["token_placements"]
    } == {"numeric_unit_carrier"}


def test_countdown_carrier_token_bank_warmup_selects_texture_matched_ref_sequence(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_token_bank_enabled=True,
        gsv_countdown_token_bank_warmup_enabled=True,
        gsv_countdown_token_bank_max_ref_count=2,
        gsv_countdown_token_bank_beam_width=4,
        gsv_countdown_carrier_templates=["carrier {token}"],
        gsv_countdown_carrier_candidate_count=1,
        gsv_countdown_carrier_slice_search_enabled=False,
        gsv_countdown_carrier_quality_retry_max_rounds=1,
        gsv_countdown_timing_mode="source_exact",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_dir = tmp_project_dir / "refs"
    write_tiny_wav(ref_dir / "whisper_close.wav", duration=4.0)
    write_tiny_wav(ref_dir / "bright.wav", duration=4.0)
    refs_path = ref_dir / "refs.json"
    refs = json.loads(refs_path.read_text("utf-8"))
    refs["bright"] = {
        **refs["whisper_close"],
        "ref_audio_path": "refs/bright.wav",
        "prompt_text": "ブライト",
    }
    refs_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")

    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0, frequency_hz=440.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
                "token_timeline": [
                    {
                        "source_text": str(value),
                        "korean_token": token,
                        "value": value,
                        "start": float(local_index),
                        "end": float(local_index) + 0.45,
                    }
                    for local_index, (value, token) in enumerate(
                        zip([3, 2, 1], ["삼", "이", "일"], strict=True)
                    )
                ],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[tuple[str, str]] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append((text, Path(ref.ref_audio_path).name))
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            frequency = 440.0 if Path(request.ref_audio_path).name == "bright.wav" else 220.0
            _write_tone_wav(output_path, duration=0.9, frequency_hz=frequency)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            token = audio_path.stem.rsplit("_", 1)[-1]
            return [ASRChunk(start=0.0, end=0.3, text=token, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert {ref_name for _text, ref_name in requests} == {"whisper_close.wav", "bright.wav"}
    assert metadata["carrier_bank"]["token_bank"]["warmup"]["ref_count"] == 2
    assert metadata["carrier_bank"]["span_beam_search"]["enabled"] is True
    assert {
        placement["selected_ref_style"]
        for placement in metadata["token_placements"]
    } == {"bright"}
    assert all(
        placement["selected_texture_score"] > 0.95
        for placement in metadata["token_placements"]
    )


def test_countdown_carrier_bank_prefers_source_matched_pitch_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="carrier_bank",
        gsv_countdown_carrier_templates=["low {token}", "high {token}"],
        gsv_countdown_timing_mode="source_exact",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0, frequency_hz=440.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
                "token_timeline": [
                    {
                        "source_text": str(value),
                        "korean_token": token,
                        "value": value,
                        "start": float(local_index),
                        "end": float(local_index) + 0.45,
                    }
                    for local_index, (value, token) in enumerate(
                        zip([3, 2, 1], ["삼", "이", "일"], strict=True)
                    )
                ],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class PitchCountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            frequency = 440.0 if "carrier_01" in output_path.name else 220.0
            _write_tone_wav(output_path, duration=1.35, frequency_hz=frequency)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", PitchCountdownClient)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert {
        placement["selected_carrier_index"]
        for placement in metadata["token_placements"]
    } == {1}
    assert all(
        placement["selected_prosody_score"] > 0.9
        for placement in metadata["token_placements"]
    )


def test_countdown_token_renderer_smooths_jittery_source_word_timing(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="source_smoothed",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="오, 사, 삼",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
                "token_timeline": [
                    {"source_text": "5", "korean_token": "오", "value": 5, "start": 0.0, "end": 0.2},
                    {"source_text": "4", "korean_token": "사", "value": 4, "start": 0.15, "end": 0.35},
                    {"source_text": "3", "korean_token": "삼", "value": 3, "start": 1.9, "end": 2.1},
                ],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.35)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    placements = manifest.segments[0].analysis["countdown_renderer"]["token_placements"]
    assert [placement["slot_start_sec"] for placement in placements] == pytest.approx(
        [0.0, 1.0, 2.0],
        abs=0.001,
    )
    assert {placement["placement_anchor"] for placement in placements} == {"source_smoothed_grid"}


def test_countdown_token_renderer_builds_source_anchors_when_timeline_missing(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="source_smoothed",
        gsv_countdown_source_anchor_enabled=True,
        gsv_countdown_source_anchor_smoothing_blend=0.70,
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=7.89)
    values = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=7.89,
        duration=7.89,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="9 8 7 6 5 4 3 2 1 絶頂します 0",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=7.89,
        ),
        script=JapaneseScript(
            literal_ja="9 8 7 6 5 4 3 2 1 絶頂します 0",
            ja_text="9 8 7 6 5 4 3 2 1 絶頂します 0",
            tts_text="구, 팔, 칠, 육, 오, 사, 삼, 이, 일, 영",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=4.0,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": values,
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.24)
            return output_path

    class SourceAnchorASRBackend:
        name = "fake_source_anchor"

        def transcribe_with_options(self, audio_path: Path, segments, **overrides):
            assert overrides["word_timestamps"] is True
            return [
                ASRChunk(
                    start=0.24,
                    end=7.33,
                    text="7 6 5 4 3 2 1 絶頂します。",
                    language="ja",
                    words=[
                        ASRWord(start=0.24, end=0.50, text="7"),
                        ASRWord(start=0.50, end=1.04, text="6"),
                        ASRWord(start=1.26, end=1.92, text="5"),
                        ASRWord(start=2.31, end=3.11, text="4"),
                        ASRWord(start=3.30, end=4.06, text="3"),
                        ASRWord(start=4.32, end=5.18, text="2"),
                        ASRWord(start=5.34, end=6.06, text="1"),
                    ],
                )
            ]

        def transcribe(self, audio_path: Path, segments):
            return self.transcribe_with_options(audio_path, segments, word_timestamps=True)

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> SourceAnchorASRBackend:
        assert kind == "faster_whisper"
        assert config["language"] == "ja"
        return SourceAnchorASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    placements = segment.analysis["countdown_renderer"]["token_placements"]
    assert [placement["slot_start_sec"] for placement in placements] == pytest.approx(
        [0.24, 0.76425, 1.4385, 2.19975, 2.781648, 3.38925, 4.002675, 4.58775, 5.34, 7.7322],
        abs=0.001,
    )
    assert {placement["placement_anchor"] for placement in placements} == {
        "source_anchor_smoothed"
    }
    event = segment.analysis["countdown_event"]
    assert event["source_anchor_policy"]["source_kind"] == "source_text_dp_fallback"
    assert len(event["source_anchor_timeline"]) == len(values)


def test_countdown_synth_uses_phrase_slice_when_strict_token_bank_has_no_approved_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_fallback_renderer="phrase_slice",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="오, 사, 삼",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.2 if " " in request.text else 0.35)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            text = "잡음"
            if "phrase_slice" in audio_path.name:
                for token in ("오", "사", "삼"):
                    if token in audio_path.name:
                        text = token
                        break
            return [ASRChunk(start=0.0, end=0.35, text=text, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_phrase_slice"
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )
    assert metadata["fallback_from"] == "countdown_token_timeline"
    assert requests == ["오", "사", "삼", "오, 사, 삼"]
    assert [
        placement["selected_pronunciation_gate"]
        for placement in segment.analysis["countdown_renderer"]["token_placements"]
    ] == ["pass", "pass", "pass"]


def test_countdown_phrase_slice_fallback_failure_marks_manual_review(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_fallback_renderer="phrase_slice",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        status="scripted",
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="오, 사, 삼",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.2 if " " in request.text else 0.35)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=0.35, text="잡음", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "needs_manual_review"
    assert "Countdown phrase slice fallback failed." in segment.errors
    assert segment.analysis["countdown_renderer_skip"]["reason"] == (
        "no_acceptable_countdown_phrase_slice_candidate"
    )


def test_countdown_phrase_slice_does_not_use_linear_resample_fit(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_fallback_renderer="phrase_slice",
        gsv_countdown_max_tempo=2.0,
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=2.0,
        gsv_countdown_token_single_syllable_max_sec=2.0,
        gsv_countdown_token_max_slot_occupancy=2.0,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="오, 사, 삼",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 4.5 if "," in request.text else 0.3
            _write_tone_wav(output_path, duration=duration)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            text = "잡음"
            if "phrase_slice" in audio_path.name:
                for token in ("오", "사", "삼"):
                    if token in audio_path.name:
                        text = token
                        break
            return [ASRChunk(start=0.0, end=0.3, text=text, language="ko")]

    def fail_linear_fit(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("countdown should not pitch-shift by linear resampling")

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)
    monkeypatch.setattr(synth_stage, "_fit_audio_frames", fail_linear_fit)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_phrase_slice"
    placements = segment.analysis["countdown_renderer"]["token_placements"]
    assert all(1.3 < placement["selected_duration_sec"] < 1.7 for placement in placements)


def test_countdown_synth_rejects_overlong_single_syllable_token_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_fallback_renderer="phrase_slice",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="오, 사, 삼",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.9 if " " in request.text else 0.72)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            if "phrase_slice" in audio_path.name:
                for token in ("오", "사", "삼"):
                    if token in audio_path.name:
                        return [ASRChunk(start=0.0, end=0.3, text=token, language="ko")]
            stem = audio_path.stem
            for token in ("오", "사", "삼"):
                if token in stem:
                    return [ASRChunk(start=0.0, end=0.72, text=token, language="ko")]
            return [ASRChunk(start=0.0, end=0.72, text="잡음", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_phrase_slice"
    skipped = segment.analysis["countdown_renderer_skip"]
    assert skipped["failed_tokens"][0]["candidates"][0]["duration_gate"] == "too_long"
    assert skipped["failed_tokens"][0]["candidates"][0]["payload"]["countdown_token_candidate"][
        "token_max_sec"
    ] == pytest.approx(0.55)


def test_countdown_synth_uses_canonical_pack_take_before_token_bank(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=3,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_pack_min_span_occupancy=0.3,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="오, 사, 삼",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.2)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=1.2, text="오 사 삼", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert requests == ["오 사 삼", "오 사 삼", "오 사 삼"]
    assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_canonical_pack"
    assert segment.analysis["countdown_renderer"]["placement_mode"] == "take"
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )
    assert metadata["renderer"] == "countdown_canonical_pack"
    assert metadata["selected_take"]["approved"] is True
    assert metadata["selected_take"]["reused_existing"] is False
    assert metadata["selected_take"]["prompt_kind"] == "sino_space"
    assert metadata["selected_take"]["pronunciation_gate"] == "pass"
    assert len(metadata["take_candidates"]) == 3
    assert Path(metadata["selected_take"]["phrase_path"]).exists()


def test_countdown_canonical_pack_retries_when_lcs_pass_misses_digit_tail(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=2,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_pack_min_span_occupancy=0.55,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=8.0)
    values = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=8.0,
        duration=8.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="9 8 7 6 5 4 3 2 1 0",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=8.0,
        ),
        script=JapaneseScript(
            literal_ja="9 8 7 6 5 4 3 2 1 0",
            ja_text="9 8 7 6 5 4 3 2 1 0",
            tts_text="구 팔 칠 육 오 사 삼 이 일 영",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=6.8,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": values,
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=6.8)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            transcript = "987654320" if "cand_00" in audio_path.stem else "9876543210"
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=transcript, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert requests == ["9 8 7 6 5 4 3 2 1 0", "구 팔 칠 육 오 사 삼 이 일 영"]
    assert metadata["take_candidates"][0]["pronunciation_gate"] == "pass"
    assert metadata["take_candidates"][0]["sequence_gate"] == "fail"
    assert metadata["take_candidates"][0]["reject_reason"] == "countdown_sequence_qc_failed"
    assert metadata["selected_take"]["candidate_index"] == 1
    assert metadata["selected_take"]["prompt_kind"] == "sino_space"
    assert metadata["selected_take"]["sequence_gate"] == "pass"
    assert metadata["placement_mode"] == "take"


def test_countdown_canonical_pack_uses_chunked_phrase_fallback_after_full_take_failures(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_pack_min_span_occupancy=0.55,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=4.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="4 3 2 1 0",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            literal_ja="4 3 2 1 0",
            ja_text="4 3 2 1 0",
            tts_text="사 삼 이 일 영",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=3.0,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [4, 3, 2, 1, 0],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.5)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            path_text = str(audio_path)
            if "canonical_pack_chunked" in path_text:
                transcript = "사 삼 이 일 영"
            elif "4_3_2_1_0" in path_text:
                transcript = "사 상 이 영"
            elif "사" in path_text or "/4_" in path_text:
                transcript = "사"
            elif "삼" in path_text or "/3_" in path_text:
                transcript = "삼"
            elif "이" in path_text or "/2_" in path_text:
                transcript = "이"
            elif "일" in path_text or "/1_" in path_text:
                transcript = "일"
            elif "영" in path_text or "/0_" in path_text:
                transcript = "영"
            else:
                transcript = ""
            return [ASRChunk(start=0.0, end=duration_sec(audio_path), text=transcript, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert metadata["selected_take"]["prompt_kind"] == "chunked_canonical_pack"
    assert metadata["selected_take"]["sequence_gate"] == "pass"
    assert len(metadata["selected_take"]["chunks"]) == 5
    assert metadata["placement_mode"] == "take"


def test_countdown_canonical_pack_selects_exact_take_with_better_internal_timing(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=2,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_pack_min_span_occupancy=0.3,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=8.0)
    values = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=8.0,
        duration=8.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="9 8 7 6 5 4 3 2 1 0",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=8.0,
        ),
        script=JapaneseScript(
            literal_ja="9 8 7 6 5 4 3 2 1 0",
            ja_text="9 8 7 6 5 4 3 2 1 0",
            tts_text="구 팔 칠 육 오 사 삼 이 일 영",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=4.0,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": values,
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            if output_path.stem == "cand_00_phrase":
                _write_tone_gap_tone_wav(
                    output_path,
                    first_sec=1.0,
                    gap_sec=1.4,
                    second_sec=1.6,
                )
            else:
                _write_tone_wav(output_path, duration=4.0)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=duration_sec(audio_path),
                    text="9876543210",
                    language="ko",
                )
            ]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert metadata["selected_take"]["candidate_index"] == 1
    assert metadata["selected_take"]["timing_qc"]["gap_gate"] == "pass"
    assert metadata["take_candidates"][0]["timing_qc"]["gap_gate"] == "fail"


def test_countdown_canonical_pack_keeps_exact_full_take_when_span_occupancy_is_low(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_pack_min_span_occupancy=0.8,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=8.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=8.0,
        duration=8.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="9 8 7 6 5 4 3 2 1 0",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=8.0,
        ),
        script=JapaneseScript(
            literal_ja="9 8 7 6 5 4 3 2 1 0",
            ja_text="9 8 7 6 5 4 3 2 1 0",
            tts_text="구 팔 칠 육 오 사 삼 이 일 영",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=4.0,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=4.0)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=duration_sec(audio_path),
                    text="9876543210",
                    language="ko",
                )
            ]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    assert segment.status == "synthesized"
    assert metadata["placement_mode"] == "take"
    assert metadata["direct_take_token_duration_qc"]["acceptable"] is True
    assert metadata["direct_take_token_duration_qc"]["low_span_occupancy"] is True
    assert metadata["direct_take_token_duration_qc"]["take_to_span_ratio"] == pytest.approx(0.5)


def test_countdown_canonical_pack_retimes_approved_take_with_long_internal_gap(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_pack_min_span_occupancy=0.3,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=4.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=4.0,
        duration=4.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1 0",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=4.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1 0",
            ja_text="3 2 1 0",
            tts_text="삼 이 일 영",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=2.0,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1, 0],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_gap_tone_wav(
                output_path,
                first_sec=0.42,
                gap_sec=1.15,
                second_sec=0.42,
            )
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            return [
                ASRChunk(
                    start=0.0,
                    end=duration_sec(audio_path),
                    text="3210",
                    language="ko",
                )
            ]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )

    retime_qc = metadata["selected_take"]["retime_qc"]
    assert segment.status == "synthesized"
    assert metadata["placement_mode"] == "take"
    assert retime_qc["applied"] is True
    assert retime_qc["method"] == "humanized_cap_280ms"
    assert retime_qc["pre_timing_qc"]["max_gap_sec"] > 1.0
    assert retime_qc["post_timing_qc"]["max_gap_sec"] <= 0.35
    assert retime_qc["duration_ratio"] < 0.8
    assert Path(retime_qc["retimed_phrase_path"]).exists()


def test_countdown_canonical_pack_rejects_take_with_overlong_single_syllable_slice(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_fallback_renderer="manual_review",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_single_syllable_max_sec=0.55,
        gsv_pronunciation_qc_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="3 2 1",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="3 2 1",
            ja_text="3 2 1",
            tts_text="삼, 이, 일",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [3, 2, 1],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.8)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "needs_manual_review"
    assert segment.analysis["countdown_renderer_skip"]["reason"] == "canonical_pack_slice_qc_failed"
    assert "Countdown canonical pack renderer failed." in segment.errors


def test_countdown_synth_reuses_canonical_pack_take_for_matching_sequence(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=2,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_pack_min_span_occupancy=0.3,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    segments: list[Segment] = []
    for index, start in enumerate((0.0, 8.0), start=1):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"seg_{index:04d}_mix.wav"
        _write_tone_wav(audio, duration=3.0)
        segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + 3.0,
                duration=3.0,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text="5 4 3",
                    language="ja",
                    backend="qwen_asr",
                    start=start,
                    end=start + 3.0,
                ),
                script=JapaneseScript(
                    literal_ja="5 4 3",
                    ja_text="5 4 3",
                    tts_text="오, 사, 삼",
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                    expected_tts_duration_sec=1.5,
                    ref_style="whisper_close",
                ),
                analysis={
                    "countdown_event": {
                        "kind": "descending_countdown",
                        "values": [5, 4, 3],
                    }
                },
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=1.2)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=1.2, text="오 사 삼", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert requests == ["오 사 삼", "오 사 삼"]
    assert [
        segment.analysis["countdown_renderer"]["selected_take"]["reused_existing"]
        for segment in manifest.segments
    ] == [False, True]


def test_countdown_synth_slices_short_canonical_pack_take_to_source_slots(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="canonical_pack",
        gsv_countdown_candidate_count=1,
        gsv_countdown_timing_mode="even_grid",
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_pack_min_span_occupancy=0.55,
        gsv_countdown_phrase_slice_edge_pad_sec=0.0,
        gsv_pronunciation_qc_enabled=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="qwen_asr",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="오, 사, 삼",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[str] = []

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.9)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            stem = audio_path.stem
            for token in ("오", "사", "삼"):
                if token in stem:
                    return [ASRChunk(start=0.0, end=0.3, text=token, language="ko")]
            return [ASRChunk(start=0.0, end=0.9, text="오 사 삼", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert requests == ["오 사 삼"]
    assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_canonical_pack"
    assert segment.analysis["countdown_renderer"]["placement_mode"] == "slice_grid"
    placements = segment.analysis["countdown_renderer"]["token_placements"]
    assert [placement["placed_start_sec"] for placement in placements] == pytest.approx(
        [0.35, 1.35, 2.35],
        abs=0.03,
    )
    assert {placement["selected_pronunciation_gate"] for placement in placements} == {"not_run"}


def test_synth_force_renders_countdown_as_timed_token_candidates(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=3,
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=3,
        gsv_countdown_timing_mode="source_exact",
        gsv_countdown_token_min_sec=0.25,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.85,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)

    segments: list[Segment] = []
    specs = [
        (
            "seg_0095",
            0.0,
            2.0,
            "4 3",
            "넷, 셋",
            1.0,
            [4, 3],
            [("4", "넷", 4, 0.22, 0.54), ("3", "셋", 3, 1.28, 1.6)],
        ),
        (
            "seg_0096",
            2.0,
            4.0,
            "2 1",
            "둘, 하나",
            1.0,
            [2, 1],
            [("2", "둘", 2, 2.18, 2.54), ("1", "하나", 1, 3.24, 3.62)],
        ),
        (
            "seg_0097",
            4.0,
            5.0,
            "0",
            "영",
            0.5,
            [0],
            [("0", "영", 0, 4.42, 4.78)],
        ),
    ]
    for segment_id, start, end, source_text, tts_text, expected, values, timeline in specs:
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        _write_tone_wav(audio, duration=end - start)
        segments.append(
            Segment(
                id=segment_id,
                start=start,
                end=end,
                duration=end - start,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text=source_text,
                    language="ja",
                    backend="mock",
                    start=start,
                    end=end,
                ),
                script=JapaneseScript(
                    literal_ja=source_text,
                    ja_text=source_text,
                    tts_text=tts_text,
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                    expected_tts_duration_sec=expected,
                    ref_style="whisper_close",
                ),
                analysis={
                    "countdown_event": {
                        "kind": "descending_countdown",
                        "values": values,
                        "korean_tokens": [item[1] for item in timeline],
                        "token_timeline": [
                            {
                                "source_text": source,
                                "korean_token": korean_token,
                                "value": value,
                                "start": token_start,
                                "end": token_end,
                            }
                            for source, korean_token, value, token_start, token_end in timeline
                        ],
                    }
                },
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))

    requests: list[str] = []
    token_calls: dict[str, int] = {}

    class CountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            token_calls[request.text] = token_calls.get(request.text, 0) + 1
            duration = 2.4 if token_calls[request.text] == 1 else 0.54
            _write_tone_wav(output_path, duration=duration)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", CountdownClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert "사삼이일영" not in requests
    assert set(requests) == {"사", "삼", "이", "일", "영"}
    assert all(count == 3 for count in token_calls.values())
    expected_slot_starts = {
        "seg_0095": [0.22, 1.28],
        "seg_0096": [2.18, 3.24],
        "seg_0097": [4.42],
    }
    for segment in manifest.segments:
        assert segment.status == "synthesized"
        assert segment.tts is not None
        assert segment.tts.backend == "gpt-sovits-countdown-renderer"
        assert segment.tts.selected_candidate_path is not None
        assert segment.tts.candidates[0].payload["renderer"] == "countdown_token_timeline"
        assert duration_sec(Path(segment.tts.selected_candidate_path)) == pytest.approx(
            segment.duration,
            abs=0.03,
        )
        assert segment.tts.retry_summary["countdown_renderer"] is True
        assert segment.tts.retry_summary["countdown_renderer_mode"] == "token"
        assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_token_timeline"
        assert all(
            not placement["text"].endswith(".")
            for placement in segment.analysis["countdown_renderer"]["token_placements"]
        )
        assert [
            placement["slot_start_sec"]
            for placement in segment.analysis["countdown_renderer"]["token_placements"]
        ] == pytest.approx(expected_slot_starts[segment.id], abs=0.001)
        assert all(
            placement["selected_duration_sec"] <= placement["slot_duration_sec"] * 0.85
            for placement in segment.analysis["countdown_renderer"]["token_placements"]
        )


def test_synth_parallelizes_independent_countdown_spans_across_gsv_lanes(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=2,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)

    specs = [
        ("seg_0001", 0.0, 3.0, "5 4 3", "다섯, 넷, 셋", [5, 4, 3]),
        ("seg_0002", 6.0, 9.0, "2 1 0", "둘, 하나, 영", [2, 1, 0]),
    ]
    segments: list[Segment] = []
    for segment_id, start, end, source_text, tts_text, values in specs:
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        _write_tone_wav(audio, duration=end - start)
        segments.append(
            Segment(
                id=segment_id,
                start=start,
                end=end,
                duration=end - start,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text=source_text,
                    language="ja",
                    backend="mock",
                    start=start,
                    end=end,
                ),
                script=JapaneseScript(
                    literal_ja=source_text,
                    ja_text=source_text,
                    tts_text=tts_text,
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                    expected_tts_duration_sec=1.0,
                    ref_style="whisper_close",
                ),
                analysis={
                    "countdown_event": {
                        "kind": "descending_countdown",
                        "values": values,
                        "token_timeline": [
                            {
                                "source_text": str(value),
                                "korean_token": token,
                                "value": value,
                                "start": start + local_index,
                                "end": start + local_index + 0.35,
                            }
                            for local_index, (value, token) in enumerate(
                                zip(
                                    values,
                                    ["다섯", "넷", "셋"]
                                    if values == [5, 4, 3]
                                    else ["둘", "하나", "영"],
                                    strict=True,
                                ),
                            )
                        ],
                    }
                },
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))

    first_token_barrier = Barrier(2)
    calls_lock = Lock()
    observed_threads: set[int] = set()
    observed_urls: list[str] = []
    token_calls: list[str] = []

    class ParallelCountdownClient:
        def __init__(self, base_url: str, *args: object, **kwargs: object) -> None:
            self.base_url = base_url

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            with calls_lock:
                observed_threads.add(get_ident())
                observed_urls.append(self.base_url)
                token_calls.append(request.text)
            try:
                first_token_barrier.wait(timeout=0.5)
            except BrokenBarrierError as exc:
                raise AssertionError("countdown spans did not synthesize concurrently") from exc
            _write_tone_wav(output_path, duration=0.45)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", ParallelCountdownClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert {segment.status for segment in manifest.segments} == {"synthesized"}
    assert len(observed_threads) == 2
    assert set(observed_urls) == {"http://127.0.0.1:9880", "http://127.0.0.1:9881"}
    assert set(token_calls) == {"오", "사", "삼", "이", "일", "영"}


def test_countdown_synth_reuses_token_bank_candidates_across_spans(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=1,
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)

    segments: list[Segment] = []
    for segment_id, start in (("seg_0001", 0.0), ("seg_0002", 10.0)):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        _write_tone_wav(audio, duration=3.0)
        values = [5, 4, 3]
        tokens = ["다섯", "넷", "셋"]
        segments.append(
            Segment(
                id=segment_id,
                start=start,
                end=start + 3.0,
                duration=3.0,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text="5 4 3",
                    language="ja",
                    backend="mock",
                    start=start,
                    end=start + 3.0,
                ),
                script=JapaneseScript(
                    literal_ja="5 4 3",
                    ja_text="5 4 3",
                    tts_text="다섯, 넷, 셋",
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                    expected_tts_duration_sec=1.5,
                    ref_style="whisper_close",
                ),
                analysis={
                    "countdown_event": {
                        "kind": "descending_countdown",
                        "values": values,
                        "token_timeline": [
                            {
                                "source_text": str(value),
                                "korean_token": token,
                                "value": value,
                                "start": start + local_index,
                                "end": start + local_index + 0.35,
                            }
                            for local_index, (value, token) in enumerate(zip(values, tokens, strict=True))
                        ],
                    }
                },
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))

    requests: list[str] = []

    class BankedCountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            requests.append(text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.45)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", BankedCountdownClient)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert [segment.status for segment in manifest.segments] == ["synthesized", "synthesized"]
    assert sorted(requests) == ["사", "삼", "오"]
    for segment in manifest.segments:
        assert segment.analysis["countdown_renderer"]["renderer"] == "countdown_token_timeline"
        metadata = json.loads(
            Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
        )
        assert all(
            candidate["payload"]["countdown_bank"]["reused_existing"] in {False, True}
            for candidate in metadata["token_candidates"]
        )


def test_countdown_synth_prefers_pronunciation_qc_pass_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        duration_tolerance=0.2,
        gsv_concurrency=1,
        gsv_countdown_renderer="token",
        gsv_countdown_candidate_count=2,
        gsv_countdown_token_min_sec=0.1,
        gsv_countdown_token_max_sec=0.95,
        gsv_countdown_token_max_slot_occupancy=0.95,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="다섯, 넷, 셋",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class PronunciationCountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=0.45)
            return output_path

    class PronunciationASRBackend:
        name = "mock"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            transcript = "잡음" if "_cand_00" in audio_path.name else audio_path.stem.split("_cand_")[0]
            return [ASRChunk(start=0.0, end=0.45, text=transcript, language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> PronunciationASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return PronunciationASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", PronunciationCountdownClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    countdown_synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    metadata = json.loads(
        Path(segment.analysis["countdown_renderer"]["span_metadata_path"]).read_text("utf-8")
    )
    assert {
        candidate["payload"]["pronunciation_qc"]["gate"]
        for candidate in metadata["token_candidates"]
    } == {"fail", "pass"}
    assert {
        placement["selected_candidate_index"]
        for placement in segment.analysis["countdown_renderer"]["token_placements"]
    } == {1}


def test_synth_without_countdown_renderer_leaves_countdown_for_countdown_synth(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    save_project_config(ProjectConfig(project_name="test"), tmp_project_dir / "pipeline.yaml")
    write_tiny_wav(tmp_project_dir / "refs" / "whisper_close.wav", duration=4.0)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.0)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        status="scripted",
        source_script=SourceScript(
            text="5 4 3",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        script=JapaneseScript(
            literal_ja="5 4 3",
            ja_text="5 4 3",
            tts_text="다섯, 넷, 셋",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.5,
            ref_style="whisper_close",
        ),
        analysis={
            "countdown_event": {
                "kind": "descending_countdown",
                "values": [5, 4, 3],
            }
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class NoCountdownClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            raise AssertionError("synth should not send countdown text to GPT-SoVITS")

    monkeypatch.setattr(steps, "GPTSoVITSClient", NoCountdownClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        force=True,
        render_countdowns=False,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "scripted"
    assert segment.tts is None
    assert manifest.stage_state["synth"]["countdown_skipped_segments"] == ["seg_0001"]


def test_synth_closes_open_korean_text_and_retries_omission_with_static_ref(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        duration_tolerance=0.2,
        gsv_ref_mode="segment",
        gsv_korean_segment_ref_enabled=True,
        gsv_max_attempts_per_candidate=2,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    static_ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(static_ref_audio, duration=4.0)
    segment_audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0010_mix.wav"
    write_tiny_wav(segment_audio, duration=8.04)
    original_text = "하지만 눈만은 아직 뜨고 있어야 해요. 멍하니…"
    closed_text = "하지만 눈만은 아직 뜨고 있어야 해요. 멍하니 말이에요."
    segment = Segment(
        id="seg_0010",
        start=77.57,
        end=85.61,
        duration=8.04,
        audio_for_gemma=str(segment_audio),
        audio_for_mix=str(segment_audio),
        source_script=SourceScript(
            text="でも、目だけはまだ、開けたままにしておいてください。ぼんやりと、",
            language="ja",
            backend="mock",
            start=77.57,
            end=85.61,
        ),
        script=JapaneseScript(
            ja_text="でも、目だけはまだ、開けたままにしておいてください。ぼんやりと、",
            tts_text=original_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            ref_style="whisper_close",
            expected_tts_duration_sec=4.5,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            requests.append(
                {
                    "text": request.text,
                    "ref_audio_path": request.ref_audio_path,
                    "prompt_text": request.prompt_text,
                    "seed": request.seed,
                    "repetition_penalty": request.repetition_penalty,
                }
            )
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            duration = 0.92 if "work/segments/audio" in request.ref_audio_path else 8.04
            _write_tone_wav(output_path, duration=duration)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.script is not None
    assert segment.script.tts_text == closed_text
    assert requests[0]["text"] == closed_text
    assert "work/segments/audio/seg_0010_mix.wav" in str(requests[0]["ref_audio_path"])
    assert requests[1]["text"] == closed_text
    assert requests[1]["ref_audio_path"] == str(static_ref_audio)
    assert requests[1]["repetition_penalty"] > requests[0]["repetition_penalty"]
    assert segment.tts is not None
    first_candidate = segment.tts.candidates[0]
    assert first_candidate.payload["omission_suspected"] is True
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["omission_retry"]["ref_fallback"] == "static_ref"
    assert selected.payload["omission_retry"]["text_normalization"] == "closed_open_sentence"
