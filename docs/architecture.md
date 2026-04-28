# Architecture

```text
input media
  |
  v
extract -> original_stereo_48k.wav + gemma_mono_16k.wav
  |
  v
segment -> segments_raw.json + per-segment clips
  |
  v
Gemma analyze -> segments_gemma.json
  |
  v
Gemma script + normalizer -> JapaneseScript metadata
  |
  v
GPT-SoVITS or mock synth -> candidates + selected final WAV
  |
  v
QC -> segment status + qc_manifest.json
  |
  v
mix -> dialogue_stem.wav + final_audio.wav + mix_manifest.json
  |
  v
export -> final WAV or muxed video + export_manifest.json
```

`PipelineManifest` is the central resumable state. Each command loads it, updates only its stage, and writes it atomically. The CLI stays thin; orchestration lives in `orchestrator.py` and `pipeline/steps.py`.

Mixing requires a completed QC stage and only includes segments with `status == "ok"` and a passing QC recommendation. Non-passing segments are recorded in the mix manifest and skipped.

Recommended ASMR mix defaults are encoded in `ProjectConfig` and copied into `work/mix/mix_manifest.json` plus `stage_state.mix` for traceability:

- `mix_profile: asmr_stereo`
- `mix_background_bed: preserve_original`, which keeps the extracted stereo 48 kHz bed when `original_stereo_48k` is available.
- `background_gain_db: -18.0`, low enough to retain texture without masking whispered dialogue.
- `mix_dialogue_gain_db: 0.0` and `mix_dialogue_fade_ms: null`, which keeps the deterministic per-style fade profiles.
- `mix_loudness_strategy: peak_guard_only` with `mix_peak_limit_dbfs: -1.0`; the pipeline does not apply target-LUFS or aggressive loudness normalization.

Per-segment mix records include inclusion status, selected candidate path, start time, estimated pan, applied spatial style, pan/gain/fade values, cue-derived pause padding, room-tone use, and QC recommendation.

All generated artifacts live under the project directory. Input media is never overwritten.
