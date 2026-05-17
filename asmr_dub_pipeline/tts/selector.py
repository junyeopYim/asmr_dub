from __future__ import annotations

from asmr_dub_pipeline.schemas import ProjectConfig, Segment
from asmr_dub_pipeline.tts.scoring import score_candidate
from asmr_dub_pipeline.tts.types import SelectionResult, TTSCandidate, TTSRoute


def select_tts_candidate(
    segment: Segment,
    candidates: list[TTSCandidate],
    cfg: ProjectConfig,
    *,
    route: TTSRoute | None = None,
) -> SelectionResult:
    """Select the highest-scoring non-blocked TTS candidate for one segment."""

    scores = [score_candidate(segment, candidate, cfg, route=route) for candidate in candidates]
    score_by_id = {score.candidate_id: score for score in scores}
    selectable = [
        candidate for candidate in candidates if not score_by_id[candidate.candidate_id].blocked
    ]
    selected = max(
        selectable,
        key=lambda candidate: (
            score_by_id[candidate.candidate_id].score,
            1 if candidate.backend == "gpt_sovits" else 0,
            candidate.candidate_id,
        ),
        default=None,
    )
    terminal_reason = None if selected is not None else "all_candidates_hard_failed"
    return SelectionResult(
        segment_id=segment.id,
        selected=selected,
        scores=scores,
        route_reason_codes=route.reason_codes if route else [],
        terminal_reason=terminal_reason,
    )
