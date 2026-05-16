# GSV RVC Kana Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize Japanese transcripts used by GPT-SoVITS fine-tuning, GPT-SoVITS references, and RVC training audit/reference metadata into kana where possible.

**Architecture:** Add a small shared Japanese kana normalizer in `asmr_dub_pipeline/script/normalizer.py`, then call it only at model-training/reference boundaries. Store original and normalized text in manifests where text affects traceability, while sending normalized text to GPT-SoVITS list/ref payloads. If any kanji remains after normalization, keep the text and record a risk flag instead of blocking the pipeline.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, optional `pyopenjtalk` when available.

---

### Task 1: Shared Kana Normalizer

**Files:**
- Modify: `asmr_dub_pipeline/script/normalizer.py`
- Test: `tests/test_json_repair_normalizer.py`

- [ ] Write failing tests for kana-preserving text, fallback lexical conversion, and strict failure when kanji remains.
- [ ] Run `uv run pytest tests/test_json_repair_normalizer.py`.
- [ ] Implement a deterministic helper that converts Japanese text to hiragana/katakana where possible and records remaining kanji as a risk flag by default.
- [ ] Re-run `uv run pytest tests/test_json_repair_normalizer.py`.

### Task 2: GPT-SoVITS Dataset and References

**Files:**
- Modify: `asmr_dub_pipeline/gpt_sovits/few_shot.py`
- Modify: `asmr_dub_pipeline/pipeline/stages/common.py`
- Test: `tests/test_gpt_sovits_few_shot.py`

- [ ] Write failing tests proving `dataset.list`, training metadata, generated speaker refs, and segment-source refs use kana-normalized Japanese prompt text.
- [ ] Run focused tests and confirm the new expectations fail.
- [ ] Apply the shared normalizer at the dataset/ref boundary and record original text beside normalized text in metadata.
- [ ] Re-run focused GPT-SoVITS tests.

### Task 3: RVC Training and Voice Bank Metadata

**Files:**
- Modify: `asmr_dub_pipeline/pipeline/stages/common.py`
- Modify: `asmr_dub_pipeline/voice_bank/manager.py`
- Test: `tests/test_rvc_step.py`
- Test: `tests/test_voice_bank.py`

- [ ] Write failing tests proving RVC dataset manifests and voice bank generated refs record kana-normalized Japanese source text.
- [ ] Run focused RVC/voice-bank tests and confirm the new expectations fail.
- [ ] Apply the shared normalizer when writing RVC audit rows and voice-bank refs.
- [ ] Re-run focused RVC/voice-bank tests.

### Task 4: Final Verification

- [ ] Run the focused suite:

```bash
uv run pytest tests/test_json_repair_normalizer.py tests/test_gpt_sovits_few_shot.py tests/test_rvc_step.py tests/test_voice_bank.py
```

- [ ] Run nearby schema/config tests if config fields are touched.
- [ ] Summarize changed files, tests, and any remaining risk.
