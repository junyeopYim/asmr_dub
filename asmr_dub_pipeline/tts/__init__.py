"""Shared TTS candidate routing, storage, and selection helpers."""

from asmr_dub_pipeline.tts.candidate_store import CandidateStore
from asmr_dub_pipeline.tts.router import route_segment_tts
from asmr_dub_pipeline.tts.selector import select_tts_candidate
from asmr_dub_pipeline.tts.types import CandidateScore, SelectionResult, TTSCandidate, TTSRoute

__all__ = [
    "CandidateScore",
    "CandidateStore",
    "SelectionResult",
    "TTSCandidate",
    "TTSRoute",
    "route_segment_tts",
    "select_tts_candidate",
]
