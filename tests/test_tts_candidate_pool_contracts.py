from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf
from conftest import write_tiny_wav

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.manifest_io import save_manifest
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
    TTSMetadata,
)
from asmr_dub_pipeline.schemas import TTSCandidate as LegacyTTSCandidate
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
    input_script_generation_id: str | None = None,
    input_script_hash: str | None = None,
    route_id: str | None = None,
    pool_generation_id: str | None = None,
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
        input_script_generation_id=input_script_generation_id,
        input_script_hash=input_script_hash,
        route_id=route_id,
        pool_generation_id=pool_generation_id,
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
    assert len(loaded) == 1
    assert loaded[0].model_copy(
        update={
            "source_wav_sha256": None,
            "wav_sha256": None,
            "source_wav_size_bytes": None,
            "wav_size_bytes": None,
            "source_wav_mtime_ns": None,
            "wav_mtime_ns": None,
        }
    ) == candidate.model_copy(update={"metadata_path": str(saved)})
    assert loaded[0].source_wav_sha256
    assert loaded[0].wav_sha256 == loaded[0].source_wav_sha256

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


def test_candidate_store_clear_segment_removes_only_target_metadata(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path)
    target = _candidate(tmp_path, candidate_id="cand_target", backend="gpt_sovits", duration_sec=1.0)
    other = _candidate(
        tmp_path,
        segment_id="seg_0002",
        candidate_id="cand_other",
        backend="qwen_tts",
        duration_sec=1.0,
    )
    target_path = store.save_candidate(target)
    other_path = store.save_candidate(other)
    selected_path = store.save_selected("seg_0001", {"selected_candidate_id": "cand_target"})
    other_selected_path = store.save_selected("seg_0002", {"selected_candidate_id": "cand_other"})

    result = store.clear_segment("seg_0001")

    assert result["cleared_candidate_metadata"] is True
    assert result["cleared_selected_metadata"] is True
    assert not target_path.exists()
    assert not selected_path.exists()
    assert other_path.exists()
    assert other_selected_path.exists()


def test_candidate_store_filters_stale_script_identity(tmp_path: Path) -> None:
    store = CandidateStore(tmp_path)
    stale = _candidate(
        tmp_path,
        candidate_id="stale",
        backend="gpt_sovits",
        duration_sec=1.0,
        input_script_generation_id="script:old",
        input_script_hash="hash-old",
        route_id="route-current",
    )
    current = _candidate(
        tmp_path,
        candidate_id="current",
        backend="qwen_tts",
        duration_sec=1.0,
        input_script_generation_id="script:current",
        input_script_hash="hash-current",
        route_id="route-current",
    )
    store.save_candidate(stale)
    store.save_candidate(current)

    filtered = store.load_segment_candidates(
        "seg_0001",
        expected_script_generation_id="script:current",
        expected_script_hash="hash-current",
        expected_route_id="route-current",
        discard_stale=True,
    )
    unfiltered = store.load_segment_candidates(
        "seg_0001",
        expected_script_generation_id="script:current",
        expected_script_hash="hash-current",
        expected_route_id="route-current",
        discard_stale=False,
    )

    assert [candidate.candidate_id for candidate in filtered] == ["current"]
    assert {candidate.candidate_id for candidate in unfiltered} == {"stale", "current"}
    stale_loaded = next(candidate for candidate in unfiltered if candidate.candidate_id == "stale")
    assert stale_loaded.payload["stale_filter"]["is_stale"] is True
    assert "input_script_generation_id_mismatch" in stale_loaded.payload["stale_filter"]["reasons"]


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


def test_tts_candidate_pool_clears_old_candidates_before_saving_new_pool(
    tmp_project_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from asmr_dub_pipeline.pipeline.stages import tts_candidates as stage

    segment = _segment("seg_0001")
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    store = CandidateStore(tmp_project_dir)
    old = _candidate(tmp_path, candidate_id="old", backend="gpt_sovits", duration_sec=1.0)
    old_path = store.save_candidate(old)
    selected_path = store.save_selected("seg_0001", {"selected_candidate_id": "old"})

    def fake_synth(ctx: PipelineContext, *args: object, **kwargs: object) -> None:
        manifest = ctx.reload_manifest()
        wav_path = write_tiny_wav(tmp_project_dir / "work" / "fake_synth" / "new.wav")
        manifest.segments[0].tts = TTSMetadata(
            backend="mock",
            selected_candidate_path=str(wav_path),
            candidates=[
                LegacyTTSCandidate(
                    candidate_index=0,
                    seed=1,
                    output_path=str(wav_path),
                    backend="mock",
                    candidate_id="new",
                    generation_id="tts-candidate:new",
                )
            ],
        )
        save_manifest(ctx.project_dir, manifest)

    monkeypatch.setattr(stage, "run_synth_stage", fake_synth)

    manifest = stage.run_tts_candidate_pool_stage(
        PipelineContext.load(tmp_project_dir),
        requested_backend="mock",
        mock=True,
    )

    assert not old_path.exists()
    assert not selected_path.exists()
    assert [candidate.candidate_id for candidate in store.load_segment_candidates("seg_0001")] == ["new"]
    pool_segment = manifest.segments[0].analysis["tts_candidate_pool"]
    assert pool_segment["cleared_candidate_metadata"] is True
    assert pool_segment["cleared_selected_metadata"] is True


def test_tts_select_rejects_only_stale_candidates(tmp_project_dir: Path, tmp_path: Path) -> None:
    from asmr_dub_pipeline.pipeline.artifacts import make_script_generation_id, stable_hash
    from asmr_dub_pipeline.pipeline.stages.tts_candidates import run_tts_select_stage

    segment = _segment(
        "seg_0001",
        analysis={"tts_route": {"segment_id": "seg_0001", "backends": ["gpt_sovits"], "route_id": "route-current"}},
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    store = CandidateStore(tmp_project_dir)
    store.save_candidate(
        _candidate(
            tmp_path,
            candidate_id="stale",
            backend="gpt_sovits",
            duration_sec=1.0,
            input_script_generation_id="script:old",
            input_script_hash="hash-old",
            route_id="route-current",
        )
    )

    manifest = run_tts_select_stage(PipelineContext.load(tmp_project_dir))

    selected = manifest.segments[0]
    assert selected.status == "needs_manual_review"
    assert selected.tts is None
    analysis = selected.analysis["tts_selection"]
    assert analysis["status"] == "manual_review"
    assert analysis["terminal_reason"] in {"all_candidates_stale", "stale_candidate_only"}
    assert analysis["stale_candidate_filtered_count"] == 1
    assert analysis["expected_script_generation_id"] == make_script_generation_id(segment.script)
    assert analysis["expected_script_hash"] == stable_hash(segment.script or {})


def test_selected_tts_generation_changes_when_wav_content_changes(
    tmp_project_dir: Path,
    tmp_path: Path,
) -> None:
    from asmr_dub_pipeline.pipeline.artifacts import make_script_generation_id, stable_hash
    from asmr_dub_pipeline.pipeline.stages.tts_candidates import run_tts_select_stage

    segment = _segment(
        "seg_0001",
        duration=1.0,
        analysis={
            "tts_route": {"segment_id": "seg_0001", "backends": ["gpt_sovits"], "route_id": "route-current"},
            "tts_candidate_pool": {"generation_id": "pool:current"},
        },
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    store = CandidateStore(tmp_project_dir)
    source_wav = write_tiny_wav(tmp_path / "candidate.wav", duration=1.0)
    script_generation_id = make_script_generation_id(segment.script)
    script_hash = stable_hash(segment.script or {})
    store.save_candidate(
        TTSCandidate(
            segment_id="seg_0001",
            candidate_id="same",
            backend="gpt_sovits",
            wav_path=str(source_wav),
            duration_sec=1.0,
            input_hash="script-hash",
            backend_config_hash="gsv-config",
            generation_id="tts-candidate:same",
            input_script_generation_id=script_generation_id,
            input_script_hash=script_hash,
            route_id="route-current",
            pool_generation_id="pool:current",
        )
    )

    first = run_tts_select_stage(PipelineContext.load(tmp_project_dir), force=True)
    first_id = first.segments[0].tts.selected_tts_generation_id
    first_selected = CandidateStore(tmp_project_dir).load_selected("seg_0001")

    write_tiny_wav(source_wav, sample_rate=44_100, duration=1.0)
    second = run_tts_select_stage(PipelineContext.load(tmp_project_dir), force=True)
    second_id = second.segments[0].tts.selected_tts_generation_id
    second_selected = CandidateStore(tmp_project_dir).load_selected("seg_0001")

    assert first_id != second_id
    assert first_selected["source_wav_sha256"] != second_selected["source_wav_sha256"]
    assert first_selected["final_wav_sha256"] != second_selected["final_wav_sha256"]
