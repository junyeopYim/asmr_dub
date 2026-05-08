# RVC Training Quality Audit and Auto Epoch Design

## Goal

Improve RVC timbre transfer by making train-rvc choose cleaner, more uniform source voice data and by deriving the training epoch count from measurable dataset quality instead of relying only on a fixed default.

This change targets the training stage only. It does not change RVC inference candidate selection yet.

## Current Problem

The pipeline already filters RVC training clips with basic quality checks, source text speed checks, speaker checks, and clean-source metrics. The latest analyzed run had enough total duration, but the selected set still included low-scoring clips, long clips, slow/counting fragments, and possible mixed acoustic states. The RVC training command then used a fixed `train_epochs: 20`, which is appropriate for noisy or uncertain data but often too low for a clean, uniform ASMR voice model.

## Configuration

Add RVC training policy fields to `RVCConfig`:

- `train_epoch_policy`: `fixed` or `auto`, default `fixed` for backward compatibility.
- `train_quality_preset`: `balanced` or `strict`, default `balanced`.
- `train_max_clip_sec`: optional maximum accepted clip duration.
- `train_min_snr_db`: optional minimum estimated SNR when available.
- `train_max_background_bleed_db`: optional RVC-specific background bleed threshold.
- `train_max_side_to_mid_db`: optional RVC-specific stereo side threshold.
- `train_target_clean_sec`: preferred target duration before lower-ranked clips are no longer needed.
- `train_auto_epoch_min`, `train_auto_epoch_max`: bounds for auto-selected epochs.

Existing flat config aliases must keep working, following the current `rvc_*` field pattern.

## Dataset Audit

Extend `_rvc_train_dataset` so each accepted and rejected row records a compact audit payload:

- `quality_score`
- `duration_sec`
- `source_chars_per_sec`
- `background_bleed_db`
- `side_to_mid_db`
- `estimated_snr_db`
- `speaker_id`
- `analysis_speaker_count`
- `training_rank_score`
- `training_tier`
- `reject_reasons`

The ranking score should reward clean, speech-dense, low-noise, low-bleed clips and penalize long clips, fast text, stereo side energy, background bleed, and weak quality scores. It should remain deterministic and use only metrics already available locally.

## Selection Policy

Balanced mode preserves the current behavior as much as possible, but records the richer audit and uses ranking when trimming to a target duration.

Strict mode rejects clips that fail stricter thresholds:

- low `quality_score`
- excessive background bleed
- excessive side-to-mid energy
- too-long clip duration
- too-fast source text
- missing speaker id
- speaker count other than one
- disallowed effect or training tags

If strict mode produces insufficient clean duration, the stage should fail with the existing insufficient-training-data path instead of silently falling back to weaker clips.

## Dataset Grade

Add a dataset quality summary to `dataset_manifest.json.summary`:

- `quality_grade`: `excellent`, `good`, `mixed`, or `poor`
- `recommended_epoch_count`
- aggregate stats for duration, quality score, SNR, background bleed, side-to-mid, and source chars/sec
- count of strict and balanced rejects by reason

Initial grading rules:

- `excellent`: at least 600 clean seconds, median `quality_score >= 0.78`, p10 `quality_score >= 0.62`, median `background_bleed_db <= -30`, median `side_to_mid_db <= -12`, and at least 95% of accepted rows assigned to one speaker.
- `good`: at least 600 clean seconds, median `quality_score >= 0.70`, p10 `quality_score >= 0.50`, median `background_bleed_db <= -25`, median `side_to_mid_db <= -8`, and at least 90% of accepted rows assigned to one speaker.
- `mixed`: at least 300 clean seconds, but one or more `good` thresholds fail.
- `poor`: under 300 clean seconds, median `quality_score < 0.55`, or more than 25% of accepted rows missing a speaker id.

These thresholds are intentionally conservative and must be covered by tests.

## Auto Epoch Policy

When `train_epoch_policy` is `auto`, `run_rvc_train_stage` should build the dataset first, read the recommended epoch count from the dataset summary, and pass a temporary config copy into `RVCTrainCommandClient`.

Recommended ranges:

- `excellent`: 120-200 epochs
- `good`: 80-150 epochs
- `mixed`: 30-80 epochs
- `poor`: 20-30 epochs, or fail if insufficient

Clamp the selected value to `train_auto_epoch_min` and `train_auto_epoch_max`. Record both the configured policy and the effective epoch count in `rvc_train_manifest.json` and stage state.

## Manifest and Resumability

The dataset manifest remains deterministic and resumable. Existing fields stay intact. New audit fields are additive.

The training manifest should include:

- `configured_epochs`
- `effective_epochs`
- `epoch_policy`
- `quality_preset`
- `dataset_quality_grade`
- `dataset_quality_summary`

The actual rendered training command must show the effective epoch count.

## Tests

Add focused pytest coverage for:

- strict mode rejects low-quality, noisy, high-side, too-long, and missing-speaker clips.
- balanced mode remains compatible with existing accepted rows.
- dataset quality grades are deterministic.
- auto epoch policy selects higher epochs for clean/uniform data and lower epochs for mixed data.
- the rendered train command receives the effective epoch count.
- existing default fixed policy keeps `train_epochs` behavior unchanged.

## Out Of Scope

This design does not implement RVC inference candidate selection by timbre similarity. That should be a follow-up patch after the training dataset and epoch policy are reliable.
