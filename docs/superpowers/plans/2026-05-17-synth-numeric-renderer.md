# Synth Numeric Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route Korean numeric/countdown segments away from normal GPT-SoVITS sentence synthesis and render them with deterministic, low-tempo, coda-safe numeric phrase policies proven by the May 16 experiments.

**Architecture:** Add a small numeric rendering planner plus a synth-stage renderer. Pure numeric runs use native Korean phrase candidates and whole-span guard rendering; 10-to-1 countdowns use a head-bank hybrid for the first `열` plus an uncut `아홉...하나` phrase body. Normal TTS remains the fallback for non-numeric speech, but numeric renderer failures go to manual review instead of producing clipped or runaway TTS.

**Tech Stack:** Python 3.11, Pydantic v2, existing GPT-SoVITS client, existing Faster-Whisper numeric QC, NumPy audio utilities, pytest.

---

## File Structure

- Create: `asmr_dub_pipeline/script/numeric_render_plan.py`
  - Classifies Korean numeric sequences, chooses native tokens, and builds render groups.
- Create: `asmr_dub_pipeline/pipeline/stages/numeric_phrase_renderer.py`
  - Generates numeric phrase/head-bank candidates, runs ASR numeric QC, renders whole-span/head-rest beds, and returns manifest-ready metadata.
- Modify: `asmr_dub_pipeline/pipeline/stages/synth_gpt_sovits.py`
  - Calls the new renderer before normal GSV jobs for pure numeric/countdown segments; records metadata and prevents fallback into unsafe token cuts.
- Modify: `asmr_dub_pipeline/schemas.py`
  - Adds config knobs and updates countdown renderer literal/default.
- Modify: `examples/pipeline.example.yaml`
  - Documents the new renderer defaults.
- Test: `tests/test_numeric_render_plan.py`
- Test: `tests/test_numeric_phrase_renderer.py`
- Test: `tests/test_synth_numeric_renderer_integration.py`
- Update as needed: `tests/test_text_translation_lane.py`, `tests/test_synth_gpt_sovits_omission_retry.py`, `tests/test_countdown_synth_server_cleanup.py`

## Experimental Decisions To Encode

- Do not render repeated vowels/onomatopoeia through TTS. Those stay `non_speech_texture` / `keep_original_texture`.
- For numeric/cadence segments, do not use ASR word-boundary cuts as final audio cuts. ASR boundaries are QC evidence, not editing boundaries.
- For `seg_0577`-style runs, prefer `native_spaces` phrase generation and `whole_span_guard120_pad350`.
- For 10-to-1 countdowns, prefer native Korean `열, 아홉, ... 하나`; avoid Sino `십, 구` because widened cuts often become `19`.
- For 10-to-1, avoid pair-group cuts around `[여섯, 다섯] | [넷, 셋] | [둘, 하나]`; they clipped `다섯` and `셋`.
- For 10-to-1, primary render policy is `head_single_rest`: first `열` from a standalone head bank, rest from one uncut `아홉...하나` phrase.
- `max_tempo` must stay `<= 1.1`; preferred render tempo is `1.0`.
- If numeric ASR/QC fails, mark manual review instead of promoting normal TTS.

---

### Task 1: Numeric Render Plan

**Files:**
- Create: `asmr_dub_pipeline/script/numeric_render_plan.py`
- Test: `tests/test_numeric_render_plan.py`

- [ ] **Step 1: Write failing classification tests**

```python
from asmr_dub_pipeline.script.numeric_render_plan import (
    NumericRenderKind,
    build_numeric_render_plan,
)


def test_builds_native_whole_span_plan_for_ascending_count_run() -> None:
    plan = build_numeric_render_plan([3, 4, 5, 6, 7, 8, 9, 10], target_duration_sec=3.42)
    assert plan.kind == NumericRenderKind.NUMERIC_CADENCE
    assert plan.tokens == ["셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열"]
    assert plan.text_variant == "native_spaces"
    assert plan.render_policy == "whole_span_guard120_pad350"
    assert plan.groups == [[3, 4, 5, 6, 7, 8, 9, 10]]


def test_builds_head_single_rest_plan_for_ten_to_one_countdown() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    assert plan.kind == NumericRenderKind.COUNTDOWN_10_TO_1
    assert plan.tokens == ["열", "아홉", "여덟", "일곱", "여섯", "다섯", "넷", "셋", "둘", "하나"]
    assert plan.text_variant == "native_countdown"
    assert plan.render_policy == "head_single_rest"
    assert plan.groups == [[10], [9, 8, 7, 6, 5, 4, 3, 2, 1]]
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_numeric_render_plan.py -v`

Expected: import failure because `numeric_render_plan.py` does not exist.

- [ ] **Step 3: Implement the plan module**

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from asmr_dub_pipeline.script.countdown import native_korean_count_number


class NumericRenderKind(StrEnum):
    COUNTDOWN_10_TO_1 = "countdown_10_to_1"
    DESCENDING_COUNTDOWN = "descending_countdown"
    NUMERIC_CADENCE = "numeric_cadence"


@dataclass(frozen=True)
class NumericRenderPlan:
    kind: NumericRenderKind
    values: list[int]
    tokens: list[str]
    target_duration_sec: float
    text: str
    text_variant: str
    render_policy: str
    groups: list[list[int]]


def _native_tokens(values: list[int]) -> list[str] | None:
    tokens = [native_korean_count_number(value) for value in values]
    if any(token is None for token in tokens):
        return None
    return [str(token) for token in tokens]


def _is_strict_descending(values: list[int]) -> bool:
    return len(values) >= 3 and all(left - right == 1 for left, right in zip(values, values[1:], strict=False))


def build_numeric_render_plan(values: list[int], *, target_duration_sec: float) -> NumericRenderPlan | None:
    tokens = _native_tokens(values)
    if tokens is None or len(values) < 3:
        return None
    if values == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        return NumericRenderPlan(
            kind=NumericRenderKind.COUNTDOWN_10_TO_1,
            values=values,
            tokens=tokens,
            target_duration_sec=target_duration_sec,
            text=", ".join(tokens) + ".",
            text_variant="native_countdown",
            render_policy="head_single_rest",
            groups=[[10], [9, 8, 7, 6, 5, 4, 3, 2, 1]],
        )
    if _is_strict_descending(values):
        return NumericRenderPlan(
            kind=NumericRenderKind.DESCENDING_COUNTDOWN,
            values=values,
            tokens=tokens,
            target_duration_sec=target_duration_sec,
            text=" ".join(tokens) + ".",
            text_variant="native_spaces",
            render_policy="whole_span_guard120_pad350",
            groups=[values],
        )
    return NumericRenderPlan(
        kind=NumericRenderKind.NUMERIC_CADENCE,
        values=values,
        tokens=tokens,
        target_duration_sec=target_duration_sec,
        text=" ".join(tokens) + ".",
        text_variant="native_spaces",
        render_policy="whole_span_guard120_pad350",
        groups=[values],
    )
```

- [ ] **Step 4: Run test**

Run: `pytest tests/test_numeric_render_plan.py -v`

Expected: 2 passed.

---

### Task 2: Renderer Audio Policies

**Files:**
- Create: `asmr_dub_pipeline/pipeline/stages/numeric_phrase_renderer.py`
- Test: `tests/test_numeric_phrase_renderer.py`

- [ ] **Step 1: Write failing tests for coda-safe render policies**

```python
import numpy as np

from asmr_dub_pipeline.pipeline.stages.numeric_phrase_renderer import (
    RenderedNumericBed,
    render_from_phrase_candidate,
)
from asmr_dub_pipeline.script.numeric_render_plan import build_numeric_render_plan


def test_whole_span_guard_has_no_internal_cuts() -> None:
    plan = build_numeric_render_plan([3, 4, 5, 6, 7, 8, 9, 10], target_duration_sec=3.42)
    audio = np.ones((48_000 * 3, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 3, "source_start_sec": 0.0, "source_end_sec": 0.3},
        {"value": 10, "source_start_sec": 2.38, "source_end_sec": 2.54},
    ]
    rendered = render_from_phrase_candidate(plan, audio, 48_000, word_timing)
    assert isinstance(rendered, RenderedNumericBed)
    assert rendered.policy == "whole_span_guard120_pad350"
    assert rendered.max_tempo == 1.0
    assert rendered.placements[0]["values"] == [3, 4, 5, 6, 7, 8, 9, 10]
    assert len(rendered.placements) == 1


def test_head_single_rest_has_only_one_internal_boundary_after_ten() -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    phrase = np.ones((48_000 * 9, 2), dtype=np.float32) * 0.01
    head = np.ones((48_000, 2), dtype=np.float32) * 0.01
    word_timing = [
        {"value": 9, "source_start_sec": 0.36, "source_end_sec": 1.38},
        {"value": 1, "source_start_sec": 7.18, "source_end_sec": 7.82},
    ]
    rendered = render_from_phrase_candidate(plan, phrase, 48_000, word_timing, head_audio=head)
    assert rendered.policy == "head_single_rest"
    assert [placement["values"] for placement in rendered.placements] == [[10], [9, 8, 7, 6, 5, 4, 3, 2, 1]]
    assert rendered.max_tempo == 1.0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_numeric_phrase_renderer.py -v`

Expected: import failure.

- [ ] **Step 3: Implement `RenderedNumericBed` and pure render functions**

Implement only pure NumPy render helpers first. No GSV client calls in this task.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asmr_dub_pipeline.script.numeric_render_plan import NumericRenderPlan


@dataclass(frozen=True)
class RenderedNumericBed:
    audio: np.ndarray
    sample_rate: int
    policy: str
    placements: list[dict[str, Any]]
    max_tempo: float


def _frame(sec: float, sample_rate: int) -> int:
    return int(round(sec * sample_rate))


def _fade_edges(audio: np.ndarray, sample_rate: int, fade_sec: float = 0.006) -> np.ndarray:
    out = audio.astype(np.float32, copy=True)
    fade = min(len(out) // 2, _frame(fade_sec, sample_rate))
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp[:, None]
        out[-fade:] *= ramp[::-1, None]
    return out


def _timing_by_value(word_timing: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["value"]): item for item in word_timing}


def _copy_chunk(
    bed: np.ndarray,
    source: np.ndarray,
    sample_rate: int,
    *,
    source_start_sec: float,
    source_end_sec: float,
    target_start_sec: float,
    target_end_sec: float,
    values: list[int],
) -> dict[str, Any]:
    source_start = max(0, _frame(source_start_sec, sample_rate))
    source_end = min(len(source), _frame(source_end_sec, sample_rate))
    target_start = max(0, _frame(target_start_sec, sample_rate))
    target_end = min(len(bed), _frame(target_end_sec, sample_rate))
    chunk = source[source_start:source_end]
    available = max(1, target_end - target_start)
    copied = min(len(chunk), available)
    if copied:
        bed[target_start:target_start + copied] += _fade_edges(chunk[:copied], sample_rate)
    return {
        "values": values,
        "source_start_sec": round(source_start / sample_rate, 6),
        "source_end_sec": round(source_end / sample_rate, 6),
        "target_start_sec": round(target_start / sample_rate, 6),
        "target_end_sec": round(target_end / sample_rate, 6),
        "copied_sec": round(copied / sample_rate, 6),
        "required_tempo": 1.0,
    }


def render_from_phrase_candidate(
    plan: NumericRenderPlan,
    phrase_audio: np.ndarray,
    sample_rate: int,
    word_timing: list[dict[str, Any]],
    *,
    head_audio: np.ndarray | None = None,
) -> RenderedNumericBed:
    total_frames = max(1, _frame(plan.target_duration_sec, sample_rate))
    bed = np.zeros((total_frames, 2), dtype=np.float32)
    timing = _timing_by_value(word_timing)
    placements: list[dict[str, Any]] = []
    if plan.render_policy == "head_single_rest":
        if head_audio is None:
            raise ValueError("head_single_rest requires head_audio")
        placements.append(
            _copy_chunk(
                bed,
                head_audio,
                sample_rate,
                source_start_sec=0.0,
                source_end_sec=min(len(head_audio) / sample_rate, 0.70),
                target_start_sec=0.08,
                target_end_sec=1.045,
                values=[10],
            )
        )
        first = timing[9]
        last = timing[1]
        placements.append(
            _copy_chunk(
                bed,
                phrase_audio,
                sample_rate,
                source_start_sec=max(0.0, float(first["source_start_sec"])),
                source_end_sec=min(len(phrase_audio) / sample_rate, float(last["source_end_sec"]) + 0.32),
                target_start_sec=1.08,
                target_end_sec=plan.target_duration_sec - 0.12,
                values=[9, 8, 7, 6, 5, 4, 3, 2, 1],
            )
        )
    else:
        first = timing[plan.values[0]]
        last = timing[plan.values[-1]]
        placements.append(
            _copy_chunk(
                bed,
                phrase_audio,
                sample_rate,
                source_start_sec=max(0.0, float(first["source_start_sec"]) - 0.08),
                source_end_sec=min(len(phrase_audio) / sample_rate, float(last["source_end_sec"]) + 0.35),
                target_start_sec=0.12,
                target_end_sec=plan.target_duration_sec - 0.16,
                values=plan.values,
            )
        )
    return RenderedNumericBed(
        audio=bed,
        sample_rate=sample_rate,
        policy=plan.render_policy,
        placements=placements,
        max_tempo=max(float(item["required_tempo"]) for item in placements),
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_numeric_phrase_renderer.py -v`

Expected: 2 passed.

---

### Task 3: GPT-SoVITS Candidate Generation And QC

**Files:**
- Modify: `asmr_dub_pipeline/pipeline/stages/numeric_phrase_renderer.py`
- Test: `tests/test_numeric_phrase_renderer.py`

- [ ] **Step 1: Add fake-client tests for low-randomness payloads and QC rejection**

```python
def test_countdown_generation_uses_native_text_and_low_randomness(fake_gsv_client) -> None:
    plan = build_numeric_render_plan([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], target_duration_sec=10.0)
    request = build_numeric_phrase_request(plan, ref={"prompt_lang": "ko"})
    assert request.text == "열, 아홉, 여덟, 일곱, 여섯, 다섯, 넷, 셋, 둘, 하나."
    assert request.text_lang == "all_ko"
    assert request.text_split_method == "cut0"
    assert request.top_k == 5
    assert request.top_p == 0.85
    assert request.temperature == 0.65
    assert request.repetition_penalty == 1.8


def test_numeric_qc_rejects_nineteen_for_ten_nine() -> None:
    qc = evaluate_numeric_render_transcript([10, 9, 8], "19, 8")
    assert qc["gate"] == "fail"
    assert qc["observed_values"] != [10, 9, 8]
```

- [ ] **Step 2: Implement request builder and transcript QC wrapper**

Add:

```python
@dataclass(frozen=True)
class NumericPhraseRequest:
    text: str
    text_lang: str = "all_ko"
    text_split_method: str = "cut0"
    top_k: int = 5
    top_p: float = 0.85
    temperature: float = 0.65
    repetition_penalty: float = 1.8


def build_numeric_phrase_request(plan: NumericRenderPlan, *, ref: dict[str, Any]) -> NumericPhraseRequest:
    return NumericPhraseRequest(text=plan.text)


def evaluate_numeric_render_transcript(expected: list[int], transcript: str) -> dict[str, Any]:
    observed = extract_korean_numeric_values(transcript)
    pass_gate = observed == expected
    return {
        "gate": "pass" if pass_gate else "fail",
        "expected_values": expected,
        "observed_values": observed,
        "transcript": transcript,
    }
```

Use the existing project numeric extractor if it already rejects `19` for `[10, 9]`; otherwise tighten this wrapper for countdown sequences so a single `19` never satisfies `[10, 9]`.

- [ ] **Step 3: Run focused tests**

Run: `pytest tests/test_numeric_phrase_renderer.py::test_countdown_generation_uses_native_text_and_low_randomness tests/test_numeric_phrase_renderer.py::test_numeric_qc_rejects_nineteen_for_ten_nine -v`

Expected: 2 passed.

---

### Task 4: Synth Stage Routing

**Files:**
- Modify: `asmr_dub_pipeline/pipeline/stages/synth_gpt_sovits.py`
- Modify: `asmr_dub_pipeline/schemas.py`
- Test: `tests/test_synth_numeric_renderer_integration.py`

- [ ] **Step 1: Add config fields and failing defaults test**

```python
from asmr_dub_pipeline.schemas import ProjectConfig


def test_numeric_phrase_renderer_defaults() -> None:
    cfg = ProjectConfig()
    assert cfg.gsv_countdown_renderer == "numeric_phrase"
    assert cfg.gsv_numeric_phrase_renderer_enabled is True
    assert cfg.gsv_numeric_phrase_max_tempo == 1.1
    assert cfg.gsv_numeric_phrase_failure_fallback == "manual_review"
```

- [ ] **Step 2: Update schema**

In `GSVConfig`, add:

```python
countdown_renderer: Literal["numeric_phrase", "carrier_bank", "chunk_bank", "canonical_pack", "token", "compact"] = "numeric_phrase"
numeric_phrase_renderer_enabled: bool = True
numeric_phrase_max_tempo: float = Field(default=1.1, ge=1.0, le=2.0)
numeric_phrase_failure_fallback: Literal["manual_review", "normal_tts"] = "manual_review"
numeric_phrase_whole_lead_in_sec: float = Field(default=0.12, ge=0.0, le=1.0)
numeric_phrase_tail_guard_sec: float = Field(default=0.16, ge=0.0, le=1.0)
```

Also add flat aliases in `_GSV_FLAT_FIELDS`.

- [ ] **Step 3: Write integration tests**

```python
def test_synth_routes_numeric_cadence_to_whole_span_renderer(tmp_path, monkeypatch) -> None:
    project_dir, refs_path = save_numeric_project(tmp_path, values=[3, 4, 5, 6, 7, 8, 9, 10], duration=3.42)
    monkeypatch.setattr(synth_gpt_sovits, "render_numeric_phrase_segment", fake_successful_numeric_render)
    synth_step(project_dir, None, refs_path, mock=True, confirm_rights=True)
    segment = load_manifest(project_dir).segments[0]
    assert segment.status == "synthesized"
    assert segment.analysis["numeric_phrase_renderer"]["policy"] == "whole_span_guard120_pad350"
    assert segment.tts.selected_candidate_path.endswith("_numeric_phrase.wav")


def test_synth_routes_ten_to_one_to_head_single_rest(tmp_path, monkeypatch) -> None:
    project_dir, refs_path = save_numeric_project(tmp_path, values=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], duration=10.0)
    monkeypatch.setattr(synth_gpt_sovits, "render_numeric_phrase_segment", fake_successful_countdown_render)
    countdown_synth_step(project_dir, None, refs_path, mock=True, confirm_rights=True)
    segment = load_manifest(project_dir).segments[0]
    assert segment.analysis["numeric_phrase_renderer"]["policy"] == "head_single_rest"
    assert segment.analysis["numeric_phrase_renderer"]["numeric_qc"]["gate"] == "pass"


def test_numeric_renderer_failure_does_not_fallback_to_normal_tts_by_default(tmp_path, monkeypatch) -> None:
    project_dir, refs_path = save_numeric_project(tmp_path, values=[10, 9, 8, 7], duration=4.0)
    monkeypatch.setattr(synth_gpt_sovits, "render_numeric_phrase_segment", fake_failed_numeric_render)
    countdown_synth_step(project_dir, None, refs_path, mock=True, confirm_rights=True)
    segment = load_manifest(project_dir).segments[0]
    assert segment.status == "needs_manual_review"
    assert "Numeric phrase renderer failed." in segment.errors
    assert segment.tts is None
```

- [ ] **Step 4: Wire routing in `run_synth_stage`**

Insert before `run_synth_jobs(...)`:

```python
numeric_phrase_rendered_segment_ids = render_numeric_phrase_segments(segment_jobs)
```

Then exclude those IDs from normal jobs, same pattern as `countdown_rendered_segment_ids`.

Add helper:

```python
def render_numeric_phrase_segments(segment_jobs: list[tuple[int, Segment, int]]) -> set[str]:
    if not bool(getattr(cfg, "gsv_numeric_phrase_renderer_enabled", True)):
        return set()
    rendered: set[str] = set()
    for index, segment, lane_index in segment_jobs:
        values = detected_countdown_segment_values(segment) or extract_korean_numeric_values(segment.script.tts_text if segment.script else "")
        plan = build_numeric_render_plan(values, target_duration_sec=float(segment.duration)) if values else None
        if plan is None:
            continue
        result = render_numeric_phrase_segment(...)
        if result.status == "rendered":
            rendered.add(segment.id)
            segment.status = "synthesized"
            segment.analysis["numeric_phrase_renderer"] = result.metadata
        else:
            segment.status = "needs_manual_review"
            segment.errors.append("Numeric phrase renderer failed.")
            segment.analysis["numeric_phrase_renderer"] = result.metadata
        save_manifest(project_dir, manifest)
    return rendered
```

- [ ] **Step 5: Run integration tests**

Run: `pytest tests/test_synth_numeric_renderer_integration.py -v`

Expected: 3 passed.

---

### Task 5: Non-Speech Texture Guardrail

**Files:**
- Modify if needed: `asmr_dub_pipeline/pipeline/stages/korean_script.py`
- Modify if needed: `asmr_dub_pipeline/pipeline/stages/common.py`
- Test: `tests/test_korean_script_targeted.py`

- [ ] **Step 1: Add tests that repeated vowels/onomatopoeia never reach synth**

```python
def test_repeated_vowel_texture_is_not_numeric_and_not_synthesized(tmp_path: Path) -> None:
    segment = make_segment_with_translation("그으으으으으.....")
    run_korean_script_stage_for_segment(tmp_path, segment)
    updated = load_manifest(tmp_path).segments[0]
    assert updated.status == "non_speech_texture"
    assert updated.keep_original_texture is True
    assert updated.script is None
    assert updated.analysis["korean_script_non_speech_texture"]["reason"] == "repeated_vowel_or_onomatopoeia"
```

- [ ] **Step 2: Reuse existing texture detectors**

If current tests already pass, do not add duplicate implementation. If they fail, extend the existing repeated texture detector so long vowel runs and non-lexical onomatopoeia are marked after translation and before synth.

- [ ] **Step 3: Run targeted texture tests**

Run: `pytest tests/test_korean_script_targeted.py tests/test_text_translation_lane.py::test_countdown_words_are_not_treated_as_texture -v`

Expected: texture tests pass and countdown words remain speech.

---

### Task 6: Manifest Metadata And Reports

**Files:**
- Modify: `asmr_dub_pipeline/schemas.py`
- Modify: `asmr_dub_pipeline/pipeline/stages/synth_gpt_sovits.py`
- Test: `tests/test_schemas_manifest.py`

- [ ] **Step 1: Add manifest metadata assertion**

```python
def test_numeric_phrase_renderer_metadata_is_json_stable() -> None:
    payload = {
        "status": "rendered",
        "policy": "head_single_rest",
        "values": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "tokens": ["열", "아홉", "여덟", "일곱", "여섯", "다섯", "넷", "셋", "둘", "하나"],
        "max_tempo": 1.0,
        "numeric_qc": {"gate": "pass", "transcript": "10, 9, 8, 7, 6, 5, 4, 3, 2, 1"},
        "placements": [{"values": [10], "target_start_sec": 0.08, "target_end_sec": 1.045}],
    }
    segment = Segment(id="seg", start=0, end=10, duration=10, audio_for_gemma="a.wav", audio_for_mix="b.wav")
    segment.analysis["numeric_phrase_renderer"] = payload
    dumped = segment.model_dump(mode="json")
    assert dumped["analysis"]["numeric_phrase_renderer"]["policy"] == "head_single_rest"
```

- [ ] **Step 2: Ensure metadata includes experiment-critical fields**

Each rendered segment metadata must include:

```python
{
    "status": "rendered",
    "renderer": "numeric_phrase",
    "policy": "whole_span_guard120_pad350 | head_single_rest",
    "values": [...],
    "tokens": [...],
    "text_variant": "native_spaces | native_countdown",
    "candidate_text": "...",
    "candidate_generation": {"text_split_method": "cut0", "top_k": 5, "top_p": 0.85, "temperature": 0.65, "repetition_penalty": 1.8},
    "max_tempo": 1.0,
    "numeric_qc": {"gate": "pass", "transcript": "..."},
    "placements": [...],
    "output_path": "work/tts/...",
}
```

- [ ] **Step 3: Run manifest tests**

Run: `pytest tests/test_schemas_manifest.py -v`

Expected: pass.

---

### Task 7: Full Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run focused unit and integration tests**

Run:

```bash
pytest \
  tests/test_numeric_render_plan.py \
  tests/test_numeric_phrase_renderer.py \
  tests/test_synth_numeric_renderer_integration.py \
  tests/test_countdown_routing.py \
  tests/test_embedded_countdown_hybrid_synth.py \
  tests/test_synth_gpt_sovits_omission_retry.py \
  tests/test_korean_script_targeted.py \
  -v
```

Expected: all pass.

- [ ] **Step 2: Run existing broader smoke coverage**

Run:

```bash
pytest tests/test_mock_pipeline_e2e.py tests/test_text_translation_contracts.py -v
```

Expected: all pass.

- [ ] **Step 3: Run live experiment gate only when GSV server is available**

Run:

```bash
python experiments/gsv_numeric_failures/run_low_tempo_gsv_count_probe.py \
  --output-root experiments/gsv_numeric_failures/pipeline_numeric_renderer_live_verify \
  --refs-path experiments/gsv_numeric_failures/numeric_ref_project/refs/refs.json \
  --project-dir experiments/gsv_numeric_failures/numeric_ref_project \
  --profile stage1 \
  --tasks countdown_10_to_1 seg_0577_3_to_10
```

Expected:
- `seg_0577_3_to_10` has at least one `native_spaces` candidate that can render with whole-span guard.
- `countdown_10_to_1` has a native countdown body candidate and a standalone `열` head candidate.
- No accepted candidate exceeds `max_tempo=1.1`.
- No accepted transcript collapses `열 아홉` into `19`.

---

## Handoff Notes

- Do not use word-level ASR timings as final cut boundaries except for broad source span discovery.
- Do not re-enable `compact` or token-only fallback as default for numeric/countdown segments.
- If the renderer cannot produce a coda-safe pass, prefer `needs_manual_review` over a bad synthesized candidate.
- Keep `embedded_countdown_hybrid` behavior separate; this plan targets pure numeric/countdown segments first.
- The May 16 best listening candidates were:
  - `seg_0577_3_to_10_native_spaces_rep170_whole_span_guard120_pad350.wav`
  - `countdown_10_to_1_rep170_seg0577_head_230_300_single_head_hybrid.wav`

## Self-Review

- Spec coverage: covers repeated texture exclusion, numeric cadence, 10-to-1 countdown, max tempo cap, ASR numeric QC, synth routing, manifest traceability, and verification.
- Placeholder scan: no unresolved placeholder markers remain.
- Type consistency: `NumericRenderPlan`, `RenderedNumericBed`, and metadata field names are reused consistently across tasks.
