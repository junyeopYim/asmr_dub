# RVC Training Quality Auto Epoch Implementation Plan

> **For Junyeop:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for implementation tasks. Keep changes scoped to RVC training config, dataset audit, epoch selection, and focused tests.

**Goal:** Add an optional RVC training quality policy that audits dataset quality, supports strict dataset filtering, and can choose an effective training epoch count from clean duration and quality grade while preserving the current fixed-epoch default.

**Architecture:** Extend `RVCConfig` with policy knobs, add deterministic dataset summary and epoch helpers in `pipeline/stages/common.py`, and apply the effective config in `pipeline/stages/rvc_train.py` before invoking the train client. Dataset manifests and stage state should record both configured and effective epoch decisions.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, existing local pipeline helpers.

## Task 1: Add Failing Tests First

**Files:**
- `tests/test_rvc_train_quality_policy.py`

**Steps:**
1. Add tests for dataset summary grading and recommended epochs.
2. Add tests for fixed vs auto effective epoch selection.
3. Add tests for strict policy reject reasons.
4. Run only the new test file and confirm it fails before production changes.

**Verification:**

```bash
uv run pytest tests/test_rvc_train_quality_policy.py
```

## Task 2: Extend RVC Config

**Files:**
- `asmr_dub_pipeline/schemas.py`

**Steps:**
1. Add `train_epoch_policy`, `train_quality_preset`, `train_max_clip_sec`, `train_min_snr_db`, `train_max_background_bleed_db`, `train_max_side_to_mid_db`, `train_target_clean_sec`, `train_auto_epoch_min`, and `train_auto_epoch_max`.
2. Add the same names to `_RVC_FLAT_FIELDS` so flat `rvc_train_*` config remains supported.
3. Add validation that `train_auto_epoch_max >= train_auto_epoch_min`.

**Verification:**

```bash
uv run pytest tests/test_project_config_nested.py tests/test_schemas_manifest.py
```

## Task 3: Add Dataset Audit and Strict Filtering

**Files:**
- `asmr_dub_pipeline/pipeline/stages/common.py`

**Steps:**
1. Add numeric stats helpers for quality, SNR, background bleed, side-to-mid ratio, CPS, and training rank scores.
2. Add deterministic `quality_grade`, `recommended_epoch_count`, dominant speaker ratio, missing speaker ratio, and reject reason counts to the RVC dataset summary.
3. Add strict policy reject reasons using the new thresholds.
4. Add optional target clean seconds trimming based on rank score.

**Verification:**

```bash
uv run pytest tests/test_rvc_train_quality_policy.py
```

## Task 4: Apply Effective Epochs at Train Time

**Files:**
- `asmr_dub_pipeline/pipeline/stages/rvc_train.py`
- `asmr_dub_pipeline/pipeline/stages/common.py`

**Steps:**
1. Add a helper that returns an effective train config and an epoch decision payload.
2. In single-speaker training, pass the effective config to command preview and train.
3. In multi-speaker training, compute the effective config after speaker-specific output paths are applied.
4. Record `epoch_policy`, configured epochs, effective epochs, quality preset, dataset grade, and recommendation in train manifests and stage state.

**Verification:**

```bash
uv run pytest tests/test_rvc_train_quality_policy.py tests/test_rvc_webui_train.py tests/test_rvc_step.py
```

## Task 5: Final Verification

**Steps:**
1. Run the focused test set.
2. If focused tests pass, run any nearby schema/config tests touched by this patch.
3. Summarize changed files and verification result.

**Verification:**

```bash
uv run pytest tests/test_rvc_train_quality_policy.py tests/test_rvc_webui_train.py tests/test_rvc_step.py tests/test_project_config_nested.py tests/test_schemas_manifest.py
```
