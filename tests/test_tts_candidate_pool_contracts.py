from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
from conftest import write_tiny_wav

from asmr_dub_pipeline.schemas import JapaneseScript, ProjectConfig, Segment, SourceScript
from asmr_dub_pipeline.tts.candidate_store import CandidateStore
from asmr_dub_pipeline.tts.router import route_segment_tts
from asmr_dub_pipeline.tts.selector import select_tts_candidate
from asmr_dub_pipeline.tts.types import CandidateScore, TTSCandidate


def _segment(
    segment_id: str,
    *,
    duration: float = 3.0,
    source_text: str = "こんにちは",
    tts_text: str = "안녕하세요",
    analysis: dict[str, object] | None = None,
) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=duration,
        duration=duration,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        analysis=analysis or {},
        source_script=SourceScript(
            text=source_text,
            language="ja",
            backend="mock",
            start=0.0,
            end=duration,
        ),
        script=JapaneseScript(
            ja_text=source_text,
            tts_text=tts_text,
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=duration,
        ),
    )


def _candidate(
    tmp_path: Path,
    *,
    segment_id: str = "seg_0001",
    candidate_id: str,
    backend: str,
    duration_sec: float,
    wav_exists: bool = True,
    payload: dict[str, object] | None = None,
) -> TTSCandidate:
    wav_path = tmp_path / f"{candidate_id}.wav"
    if wav_exists:
        write_tiny_wav(wav_path, duration=duration_sec)
    return TTSCandidate(
        segment_id=segment_id,
        candidate_id=candidate_id,
        backend=backend,
        wav_path=str(wav_path),
        metadata_path="",
        duration_sec=duration_sec,
        input_hash="script-hash",
        backend_config_hash=f"{backend}-config",
        attempt=0,
        payload=payload or {},
        generation_id=f"tts-candidate:{candidate_id}",
    )


def test_router_routes_general_segment_to_gpt_sovits_only() -> None:
    route = route_segment_tts(_segment("seg_0001"), ProjectConfig(), requested_backend="auto")

    assert route.backends == ["gpt_sovits"]
    assert route.reason_codes == ["default_gpt_sovits"]


def test_router_adds_qwen_for_micro_numeric_and_previous_gsv_failure() -> None:
    cfg = ProjectConfig()

    micro = route_segment_tts(_segment("seg_micro", duration=1.0), cfg, requested_backend="auto")
    numeric = route_segment_tts(
        _segment("seg_numeric", source_text="10 9 8", tts_text="십, 구, 팔"), cfg, requested_backend="auto"
    )
    failed = route_segment_tts(
        _segment(
            "seg_failed",
            analysis={
                "ko_qc_repair_plan": {
                    "action": "regenerate_tts",
                    "root_cause": "gpt_sovits_pronunciation_qc_failed",
                }
            },
        ),
        cfg,
        requested_backend="auto",
    )

    assert micro.backends == ["gpt_sovits", "qwen_tts"]
    assert "micro_segment" in micro.reason_codes
    assert numeric.backends == ["gpt_sovits", "qwen_tts"]
    assert "numeric_sequence" in numeric.reason_codes
    assert failed.backends == ["gpt_sovits", "qwen_tts"]
    assert "previous_gsv_pronunciation_failure" in failed.reason_codes


def test_candidate_store_round_trips_candidate_and_selection(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path)
    candidate = _candidate(tmp_path, candidate_id="cand_001", backend="gpt_sovits", duration_sec=1.0)
    saved = store.save_candidate(candidate)

    loaded = store.load_segment_candidates("seg_0001")
    assert saved.exists()
    assert loaded == [candidate.model_copy(update={"metadata_path": str(saved)})]

    score = CandidateScore(
        candidate_id=candidate.candidate_id,
        backend=candidate.backend,
        blocked=False,
        hard_fail_reasons=[],
        score=0.95,
        score_parts={"duration_fit": 0.95},
    )
    selected_path = store.save_selected(
        "seg_0001",
        {
            "segment_id": "seg_0001",
            "selected_candidate_id": candidate.candidate_id,
            "backend": candidate.backend,
            "wav_path": candidate.wav_path,
            "selected_tts_generation_id": candidate.generation_id,
            "score": score.model_dump(mode="json"),
        },
    )

    assert json.loads(selected_path.read_text("utf-8"))["selected_candidate_id"] == "cand_001"
    assert store.load_selected("seg_0001")["backend"] == "gpt_sovits"


def test_selector_blocks_missing_wav_and_prefers_duration_fit(tmp_path: Path) -> None:
    segment = _segment("seg_0001", duration=1.0)
    missing = _candidate(
        tmp_path,
        candidate_id="missing",
        backend="gpt_sovits",
        duration_sec=1.0,
        wav_exists=False,
    )
    poor_fit = _candidate(
        tmp_path,
        candidate_id="poor",
        backend="gpt_sovits",
        duration_sec=1.7,
    )
    good_fit = _candidate(
        tmp_path,
        candidate_id="good",
        backend="qwen_tts",
        duration_sec=1.02,
    )

    result = select_tts_candidate(segment, [missing, poor_fit, good_fit], ProjectConfig())

    assert result.selected is not None
    assert result.selected.candidate_id == "good"
    rejected = {score.candidate_id: score for score in result.scores}
    assert rejected["missing"].blocked is True
    assert "missing_wav" in rejected["missing"].hard_fail_reasons


def test_selector_blocks_numeric_qc_failure_and_applies_qwen_bonus(tmp_path: Path) -> None:
    segment = _segment("seg_0001", duration=1.0, source_text="3 2 1", tts_text="삼, 이, 일")
    gsv_failed_numeric = _candidate(
        tmp_path,
        candidate_id="gsv_failed_numeric",
        backend="gpt_sovits",
        duration_sec=1.0,
        payload={"numeric_sequence_qc": {"gate": "fail"}},
    )
    qwen_numeric = _candidate(
        tmp_path,
        candidate_id="qwen_numeric",
        backend="qwen_tts",
        duration_sec=1.0,
        payload={"numeric_sequence_qc": {"gate": "pass", "exact_match": True}},
    )

    result = select_tts_candidate(segment, [gsv_failed_numeric, qwen_numeric], ProjectConfig())

    assert result.selected is not None
    assert result.selected.candidate_id == "qwen_numeric"
    scores = {score.candidate_id: score for score in result.scores}
    assert scores["gsv_failed_numeric"].blocked is True
    assert "numeric_sequence_qc_failed" in scores["gsv_failed_numeric"].hard_fail_reasons
    assert scores["qwen_numeric"].score_parts["backend_prior"] > 0


def test_selector_blocks_silent_audio(tmp_path: Path) -> None:
    segment = _segment("seg_0001", duration=1.0)
    silent_path = tmp_path / "silent.wav"
    silent_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(silent_path), [[0.0, 0.0]] * 48_000, 48_000)
    silent = TTSCandidate(
        segment_id="seg_0001",
        candidate_id="silent",
        backend="gpt_sovits",
        wav_path=str(silent_path),
        metadata_path="",
        duration_sec=1.0,
        input_hash="script-hash",
        backend_config_hash="gsv-config",
        attempt=0,
        payload={},
        generation_id="tts-candidate:silent",
    )

    result = select_tts_candidate(segment, [silent], ProjectConfig())

    assert result.selected is None
    assert result.scores[0].blocked is True
    assert "silent_or_too_quiet" in result.scores[0].hard_fail_reasons
