# Troubleshooting

## ffmpeg or ffprobe missing

Install ffmpeg and ensure `ffmpeg` and `ffprobe` are on `PATH`. WAV-only mock flows can use Python fallbacks, but video and general media extraction/export require ffmpeg.

## Gemma HF backend unavailable

Install the optional HF dependencies and make sure the model is cached locally. The default config prevents hidden downloads with `local_files_only=True`.

## GPT-SoVITS returns JSON instead of WAV

The client treats JSON responses from `/tts` as errors. Check `ref_audio_path`, `prompt_lang`, `text_lang`, model weights, and server logs.

## GPT-SoVITS candidates keep running long

Candidate payloads include `retry.signals`, `speed_factor`, `seed`,
`repetition_penalty`, `target_duration_sec`, and `expected_tts_duration_sec`.
For long audio, the first retry raises `speed_factor`; the next retry requests
script shortening through the existing duration rewrite hook. If a prior QC pass
flagged repetition or omission, rerunning `synth` changes the seed and increases
`repetition_penalty` for that segment.

## Local smoke test does nothing

`python -m asmr_dub_pipeline.gpt_sovits.smoke` is intentionally skipped unless
`GSV_URL` is set. When `GSV_URL` is set, also set `GSV_CONFIRM_RIGHTS=1` and
`GSV_REF_AUDIO` to a local reference file you have rights/consent to use.

## Segment needs manual review

Silent or very short media can produce a manual-review segment. Provide `work/segments/manifests/segments_manual.json` to override the energy fallback.

## Export refuses a path

The pipeline refuses to overwrite input media or write generated outputs outside the project directory.
