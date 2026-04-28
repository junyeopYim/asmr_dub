# asmr-dub-pipeline

Local-first Japanese ASMR dubbing pipeline for user-provided media. It combines:

- Gemma-style audio and text analysis backends for ASR, translation, script generation, style tagging, and QC.
- GPT-SoVITS `api_v2` for Japanese TTS generation.
- ffmpeg and lightweight Python DSP utilities for extraction, segmentation, duration checks, stereo mixing, and export.

## Safety And Rights

You must own or have permission/consent for the source content, voice references, and distribution. Commands that process real media require `--confirm-rights`; the confirmation is written to the manifest audit metadata.

This project does not scrape, download from streaming sites, bypass DRM, or encourage cloning a real person's voice without consent. Processing is local-first unless you explicitly configure a remote Gemma or GPT-SoVITS endpoint.

## Install

```bash
python -m pip install -e ".[dev]"
```

For Hugging Face Gemma support:

```bash
python -m pip install -e ".[hf,dev]"
```

For local ASR transcription support:

```bash
python -m pip install -e ".[asr,dev]"
```

`ffmpeg` and `ffprobe` are runtime binaries. If they are missing, media extraction/export commands fail with a clear message; WAV-only mock tests can still use the Python fallback.

## Quickstart

```bash
python -m pip install -e ".[dev]"
asmr-dub init ./demo_project
asmr-dub run ./input.wav --project ./demo_project --confirm-rights --mock
```

`--mock` uses the deterministic local Gemma mock and synthetic TTS. It still requires `--confirm-rights` because the pipeline extracts and derives data from the input media.

For a one-command local smoke run against a file under `./audio`, use `full`.
It creates a run directory automatically unless `--project` is supplied, wires the
repo-local `.cache/huggingface` into `HF_HOME` when present, and defaults to mock
Gemma/TTS so it can finish without a running model server:

```bash
asmr-dub full ./audio/RJ01012948.mp4 --confirm-rights
```

Use `--real --gemma-backend hf` only after installing the optional HF dependencies.
Use `--real --gemma-backend llama_cpp` to run the repo-local GGUF HauhauCS Gemma
4 E4B Q4 cache through `llama-mtmd-cli`. Real GPT-SoVITS synthesis needs
project-local voice references you have rights/consent to use; `full --real`
now defaults to GPT-SoVITS few-shot training from source segments before
synthesis and can auto-start repo-local GPT-SoVITS installs such as
`.cache/third_party/GPT-SoVITS/api_v2.py`. Use `--zero-shot` to skip training
and use the reference-audio-only inference path.
The real llama.cpp Korean lane defaults to two `llama-server` instances on
incrementing ports, and real GPT-SoVITS synthesis defaults to three `api_v2`
instances on incrementing ports. Tune `gemma_text_concurrency` and
`gsv_concurrency` in `pipeline.yaml` to match available GPU memory. Real
faster-whisper transcription also defaults to rebuilding TTS segments from ASR
chunks (`asr_resegment_from_chunks: true`) so long ASR sentences are not copied
onto many tiny energy segments.

Expected mock outputs:

- `work/audio/original_stereo_48k.wav`
- `work/audio/gemma_mono_16k.wav`
- `work/segments/manifests/segments_raw.json`
- `work/segments/manifests/segments_gemma.json`
- `work/segments/manifests/segments_script.json`
- `work/transcribe/source_segments.jsonl` when `transcribe` is run
- `work/translate_ko/translation_bundles.jsonl` when `translate-ko` is run
- `work/tts/seg_0001_final.wav`
- `work/qc/qc_manifest.json`
- `work/mix/mix_manifest.json`
- `work/mix/dialogue_stem.wav`
- `work/mix/final_audio.wav`
- `work/export/export_manifest.json`
- `output/<input>_dub.wav`
- `work/manifest.json`

## Real Pipeline Outline

```bash
asmr-dub init ./project
asmr-dub extract ./owned_source.mp4 --project ./project --confirm-rights
asmr-dub segment --project ./project
asmr-dub transcribe --project ./project --asr-backend faster_whisper
asmr-dub translate-ko --project ./project --gemma-text-backend llama_server
asmr-dub analyze --project ./project --gemma-backend hf --model-id google/gemma-4-E4B-it
asmr-dub script --project ./project --gemma-backend hf
asmr-dub train-gsv --project ./project --confirm-rights
asmr-dub synth --project ./project --gsv-url http://127.0.0.1:9880 --refs refs/refs.json --confirm-rights
asmr-dub qc --project ./project --gemma-backend mock
asmr-dub mix --project ./project --confirm-rights
asmr-dub export ./owned_source.mp4 --project ./project --confirm-rights
```

For the local llama.cpp GGUF backend:

```bash
asmr-dub full ./audio/RJ01012948.mp4 --confirm-rights --real --gemma-backend llama_cpp
```

`full --real` runs `train-gsv` automatically unless both `--gpt-weights` and
`--sovits-weights` are supplied or `--zero-shot` is passed. Few-shot artifacts
are stored under `work/gpt_sovits/few_shot/`, including `dataset.list`, copied
training WAVs, generated train configs, logs, and selected `.ckpt`/`.pth`
weights. Training is GPU-heavy and may take a long time; reruns reuse matching
weights unless `--force-few-shot` or `train-gsv --force` is used.

If GPT-SoVITS is installed outside the known repo-local paths, add a server
command. When `gsv_concurrency` is greater than 1, use `{host}` and `{port}`
placeholders so each auto-started instance gets its own port:

```bash
asmr-dub full ./audio/RJ01012948.mp4 \
  --confirm-rights \
  --real \
  --gemma-backend llama_cpp \
  --gsv-server-command 'python /path/to/GPT-SoVITS/api_v2.py -a {host} -p {port} -c /path/to/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml'
```

After `extract`, derived-media stages require the existing confirmed audit. If you create or populate a project manually, pass `--confirm-rights` to the stage that first consumes the media. Non-mock `synth` always requires `--confirm-rights` for the current source and voice references.

The HF Gemma backend is lazy and defaults to `local_files_only=True` to avoid hidden model downloads. GPT-SoVITS is called only when you run non-mock `synth` with an explicit endpoint. The pipeline can auto-start a repo-local GPT-SoVITS `api_v2.py`, but it does not install models or voice references for you.

## Korean Translation Lane

`transcribe` creates segment-level source scripts from `work/audio/gemma_mono_16k.wav`
using local ASR. `translate-ko` then sends only text to Gemma4 through
`llama-server`; it never uses the unsupported `llama-mtmd-cli --audio` path.
By default it starts two text model lanes and assigns odd/even segment IDs to
separate ports such as `8080` and `8081`.
For real GPT-SoVITS Korean output, the client sends pure-language api_v2 codes
(`all_ko` text with `all_ja` reference prompts), preserves the Japanese
reference prompt text, and trims generated edge silence by default
(`gsv_trim_edge_silence: true`).
Korean translations are stored separately from Japanese GPT-SoVITS scripts:

```bash
asmr-dub transcribe --project ./project --confirm-rights --asr-backend faster_whisper
asmr-dub translate-ko --project ./project --confirm-rights --gemma-text-backend llama_server
```

For offline tests or pipeline checks, use `--asr-backend mock` and
`--gemma-text-backend mock`.

## ASMR Mixing Defaults

The default mix profile is `asmr_stereo`. It preserves the extracted stereo 48 kHz background bed when available, applies that bed at `background_gain_db: -18.0`, and places accepted dialogue with deterministic spatial profiles such as `left_close`, `right_close`, `center_close`, `center_far`, and `sleepy_center`.

Leave `mix_dialogue_fade_ms: null` to use each spatial style's ASMR-oriented fade profile. Set a number only when you intentionally want one global fade override. Natural pauses belong in `nonverbal_cues.pause_sec`; bracketed pause directions are removed from TTS text and carried as metadata for mixing.

The mixer uses `mix_loudness_strategy: peak_guard_only` with `mix_peak_limit_dbfs: -1.0`. It does not perform target-LUFS normalization, so quiet ASMR dynamics are preserved unless a peak guard is needed to prevent clipping. The effective mix config and background-bed decision are written to `work/mix/mix_manifest.json` and `stage_state.mix`.

## Tests

```bash
pytest -q
```

Tests use mock backends and tiny generated WAV files. They do not require real Gemma, GPT-SoVITS, GPU, model downloads, or network access.
