# asmr-dub-pipeline

Local-first Japanese ASMR dubbing pipeline for user-provided media. It combines:

- Gemma-style audio and text analysis backends for ASR, translation, script generation, style tagging, and QC.
- GPT-SoVITS `api_v2` for Japanese TTS generation.
- Required RVC post-processing for timbre correction after synthesis and before QC.
- ffmpeg and lightweight Python DSP utilities for extraction, segmentation, duration checks, stereo mixing, and export.

## Safety And Rights

You must own or have permission/consent for the source content, voice references, and distribution. Commands that process real media require `--confirm-rights`; the confirmation is written to the manifest audit metadata.

This project does not scrape, download from streaming sites, bypass DRM, or encourage cloning a real person's voice without consent. Processing is local-first unless you explicitly configure a remote Gemma or GPT-SoVITS endpoint or an external RVC command.

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

For optional Qwen ASR fallback checks during suspicious ASR repair:

```bash
python -m pip install -e ".[qwen-asr,dev]"
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

For split local files named like `RJ094817_1.mp4`, `RJ094817_2.mp4`, and
`RJ094817_3.mp4`, use the stable audio-only merge path:

```bash
asmr-dub full ./audio/RJ094817_1.mp4 --merge-parts --confirm-rights
```

`--merge-parts` detects consecutive sibling files with the same extension,
creates `work/input/<base>_merged_source.wav`, and records the part list,
hashes, durations, and selected input in the manifest. It does not concatenate
video streams; final export from a merged source is WAV-first.

Use `--real --gemma-backend hf` only after installing the optional HF dependencies.
Use `--real --gemma-backend llama_cpp` to run the configured local GGUF
Gemma model through `llama-mtmd-cli`. Real GPT-SoVITS
synthesis needs project-local voice references you have rights/consent to use; `full --real`
now defaults to GPT-SoVITS few-shot training from source segments before
synthesis and can auto-start repo-local GPT-SoVITS installs such as
`.cache/third_party/GPT-SoVITS/api_v2.py`. Use `--zero-shot` to skip training
and use the reference-audio-only inference path.
Production/full real mode also requires `train-rvc` and `rvc`. If the repo-local
RVC-WebUI checkout exists at
`.cache/third_party/Retrieval-based-Voice-Conversion-WebUI`, `full --real`
auto-populates command templates that first train a project-local RVC model and
then convert the selected GPT-SoVITS WAVs. Otherwise configure
`rvc.train_command` and `rvc.command` in `pipeline.yaml`; the pipeline fails
early if they are missing.
If GPT-SoVITS dependencies are installed in a different Python than this
project's `.venv`, set `ASMR_DUB_GSV_PYTHON=/path/to/python`.
The real llama.cpp Korean lane defaults to one `llama-server` and sends
concurrent text requests to its slots, while real GPT-SoVITS synthesis defaults
to three `api_v2` instances on incrementing ports. Tune `gemma.text_concurrency`
and `gsv.concurrency` in `pipeline.yaml` to match available GPU memory. Real
faster-whisper transcription also defaults to rebuilding TTS segments from ASR
chunks (`asr.resegment_from_chunks: true`) so long ASR sentences are not copied
onto many tiny energy segments.

## ASR Debugging

When ASR breaks on whispery Japanese ASMR, debug in this order:

1. Run only transcription with diagnostics:

```bash
asmr-dub transcribe --project ./project --asr-backend faster_whisper --asr-preset whisper --asr-diagnostics --confirm-rights
```

On a CUDA GPU with spare VRAM, keep the same model and decoding settings but
increase throughput with faster-whisper batched inference:

```bash
asmr-dub transcribe --project ./project --asr-backend faster_whisper --asr-preset whisper --asr-device cuda --asr-compute-type float16 --asr-batched --asr-batch-size 16 --asr-diagnostics --confirm-rights
```

2. Inspect `work/transcribe/asr_input_diagnostics.json`,
   `work/transcribe/asr_diagnostics_summary.json`, and
   `work/transcribe/asr_diagnostics.json`. Check which audio was selected,
   source-vocal RMS/peak/duration warnings, runtime device/compute/batch
   settings, raw vs repaired chunks, prompt leak rejections, sparse chunks, and
   manual-review reasons.

3. If VAD appears to cut whispered syllables, try:

```bash
asmr-dub transcribe --project ./project --asr-backend faster_whisper --asr-preset no_vad_repair --asr-vad-off --asr-diagnostics --confirm-rights
```

4. If source-separated vocals are too quiet or duration-mismatched, the pipeline
   falls back to `gemma_mono_16k` or an original-derived mono 16 kHz file and
   records the decision in `asr_input_diagnostics`.

5. To allow Qwen ASR as an optional suspicious-chunk repair verifier, install
   `.[qwen-asr]` and set `asr_qwen_repair_fallback_enabled: true` in
   `pipeline.yaml`. If `qwen-asr` is missing, transcription continues with a
   warning and faster-whisper repair candidates only.

Expected mock outputs:

- `work/audio/original_stereo_48k.wav`
- `work/audio/gemma_mono_16k.wav`
- `work/segments/manifests/segments_transcribe_seed.json` when `transcribe` starts without existing segments
- `work/segments/manifests/segments_raw.json`
- `work/segments/manifests/segments_final.json` when `segment` finalizes ASR-derived segments
- `work/segments/manifests/segments_gemma.json`
- `work/segments/manifests/segments_script.json`
- `work/transcribe/source_segments.jsonl` when `transcribe` is run
- `work/transcribe/asr_input_diagnostics.json` when `transcribe` is run
- `work/transcribe/asr_diagnostics_summary.json` when ASR diagnostics are enabled
- `work/transcribe/asr_diagnostics.json` when ASR diagnostics are enabled
- `work/translate_ko/translation_bundles.jsonl` when `translate-ko` is run
- `work/tts/seg_0001_final.wav`
- `work/rvc_train/rvc_train_manifest.json`
- `work/rvc/seg_0001_final.wav`
- `work/rvc/rvc_manifest.json`
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
asmr-dub separate-background --project ./project --confirm-rights
asmr-dub transcribe --project ./project --asr-backend faster_whisper
asmr-dub segment --project ./project
asmr-dub translate-ko --project ./project --gemma-text-backend llama_server
asmr-dub analyze --project ./project --gemma-backend hf --model-id google/gemma-4-E4B-it
asmr-dub script --project ./project --gemma-backend hf
asmr-dub train-gsv --project ./project --confirm-rights
asmr-dub synth --project ./project --gsv-url http://127.0.0.1:9880 --refs refs/refs.json --confirm-rights
asmr-dub train-rvc --project ./project --confirm-rights
asmr-dub rvc --project ./project --confirm-rights
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

When `runs/voice_bank_all` already contains source-separated stems for the same
input audio, `full --real` imports those stems before Demucs so the separation
step is reused. Use `--source-separation-cache /path/to/voice_bank_project` for
a different cache project, or `--no-source-separation-cache` to force normal
separation behavior.

If GPT-SoVITS is installed outside the known repo-local paths, add a server
command. When `gsv.concurrency` is greater than 1, use `{host}` and `{port}`
placeholders so each auto-started instance gets its own port:

```bash
asmr-dub full ./audio/RJ01012948.mp4 \
  --confirm-rights \
  --real \
  --gemma-backend llama_cpp \
  --gsv-server-command 'python /path/to/GPT-SoVITS/api_v2.py -a {host} -p {port} -c /path/to/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml'
```

After `extract`, derived-media stages require the existing confirmed audit. If you create or populate a project manually, pass `--confirm-rights` to the stage that first consumes the media. Non-mock `synth` always requires `--confirm-rights` for the current source and voice references. Real RVC also requires `--confirm-rights`.

The HF Gemma backend is lazy and defaults to `local_files_only=True` to avoid hidden model downloads. GPT-SoVITS is called only when you run non-mock `synth` with an explicit endpoint. RVC is invoked only through your configured external command template. The pipeline can auto-start a repo-local GPT-SoVITS `api_v2.py` and can call a repo-local RVC-WebUI checkout, but it does not install RVC, PyTorch, model dependencies, or voice references for you.

## Mandatory RVC

RVC is a post-processing stage, not a replacement TTS engine. GPT-SoVITS still
generates and selects the raw Korean TTS candidate. After synthesis, `train-rvc`
trains a project-local RVC voice model from source-derived segment audio under
`work/rvc_train/`; then `rvc` consumes the selected TTS WAV and writes the
timbre-corrected WAV under `work/rvc/`. Downstream QC, mix, and export consume
the RVC path. In production/full real mode, the pipeline does not fall back to
pre-RVC TTS by default.

`full --real` uses high-quality defaults for the local RVC-WebUI path when it is
present: 48 kHz v2 training, RMVPE F0 extraction, 200 training epochs, and the
default ASMR-safe conversion profile attempts. If RVC dependencies live in a
separate Python environment, point the pipeline at it with
`ASMR_DUB_RVC_PYTHON=/path/to/rvc/python`.

Validate RVC config before a long run:

```bash
asmr-dub rvc-validate --project ./project
```

Generic command-template example:

```yaml
rvc:
  required: true
  train_required: true
  train_backend: command
  train_batch_size: 0  # auto from GPU memory; set a positive value to pin it
  train_command:
    - "/path/to/rvc-train-wrapper"
    - "--dataset"
    - "{dataset}"
    - "--output-model"
    - "{output_model}"
    - "--output-index"
    - "{output_index}"
  backend: command
  working_dir: ".cache/third_party/Retrieval-based-Voice-Conversion"
  command:
    - ".cache/rvc_venv/bin/rvc"
    - "infer"
    - "-m"
    - "{model}"
    - "-i"
    - "{input}"
    - "-o"
    - "{output}"
    - "-fu"
    - "{f0_up_key}"
    - "-fm"
    - "{f0_method}"
    - "-if"
    - "{index}"
    - "-fr"
    - "{filter_radius}"
    - "-rsr"
    - "{resample_sr}"
    - "-rmr"
    - "{rms_mix_rate}"
    - "-p"
    - "{protect}"
  device: "cuda:0"
  failure_policy: "retry_then_error"
  allow_pre_rvc_fallback: false
  duration_tolerance: null
```

RVC-WebUI `infer_cli.py` style example:

```yaml
rvc:
  required: true
  train_required: true
  train_backend: command
  train_command:
    - ".cache/rvc_venv/bin/python"
    - "asmr_dub_pipeline/rvc/webui_train.py"
    - "--rvc-root"
    - ".cache/third_party/Retrieval-based-Voice-Conversion-WebUI"
    - "--dataset"
    - "{dataset}"
    - "--experiment-name"
    - "{experiment_name}"
    - "--output-model"
    - "{output_model}"
    - "--output-index"
    - "{output_index}"
    - "--sample-rate"
    - "48k"
    - "--device"
    - "{device}"
    - "--epochs"
    - "200"
    - "--batch-size"
    - "{batch_size}"
  backend: command
  command:
    - ".cache/rvc_venv/bin/python"
    - "asmr_dub_pipeline/rvc/webui_infer.py"
    - "--rvc-root"
    - ".cache/third_party/Retrieval-based-Voice-Conversion-WebUI"
    - "--input"
    - "{input}"
    - "--output"
    - "{output}"
    - "--model"
    - "{model}"
    - "--index"
    - "{index}"
    - "--f0-method"
    - "{f0_method}"
    - "--f0-up-key"
    - "{f0_up_key}"
    - "--index-rate"
    - "{index_rate}"
    - "--filter-radius"
    - "{filter_radius}"
    - "--resample-sr"
    - "{resample_sr}"
    - "--rms-mix-rate"
    - "{rms_mix_rate}"
    - "--protect"
    - "{protect}"
    - "--device"
    - "{device}"
  device: "cuda:0"
  failure_policy: "retry_then_error"
  allow_pre_rvc_fallback: false
```

RVC implementations differ, so verify exact flags against your installed RVC
tool. Real RVC support assets and dependencies must be supplied externally; this
project does not download or install them. The project-local voice model is
created by `train-rvc`. To strengthen timbre, tune `rvc.index_rate` or profile
`index_rate` upward. To reduce artifacts, try lower `index_rate`. For
ASMR/whisper-like material, compare the default `rmvpe` and `crepe` profile
attempts.

## Korean Translation Lane

`transcribe` runs local ASR over the source vocal audio. If no segments exist
yet, it creates a single temporary transcription seed, then `segment` finalizes
the ASR timestamps into TTS-ready segment clips. `translate-ko` then sends only text to Gemma4 through
`llama-server`; it never uses the unsupported `llama-mtmd-cli --audio` path.
By default it starts one text model server and uses `gemma.text_concurrency`
slot workers against the same endpoint instead of duplicating the model across
ports.
For real GPT-SoVITS Korean output, the client sends pure-language api_v2 codes
(`all_ko` text with `all_ja` reference prompts), preserves the Japanese
reference prompt text, and trims generated edge silence by default
(`gsv.trim_edge_silence: true`).
Korean translations are stored separately from Japanese GPT-SoVITS scripts:

```bash
asmr-dub transcribe --project ./project --confirm-rights --asr-backend faster_whisper
asmr-dub translate-ko --project ./project --confirm-rights --gemma-text-backend llama_server
```

`translate-ko` writes `work/translate_ko/diagnostics.json` alongside
`summary.json` and `translation_bundles.jsonl`. The diagnostics artifact records
raw, repaired, and final translation bundles, split/single retry attempts,
quality counters, severe ASR backcheck hits, deterministic digit repairs, and
the accepted or rejected reason for each segment. Segments with severe domain
smells such as `媚薬` becoming `변비약`, raw numeric leftovers after repair,
Japanese/CJK/Latin residue, or other unsafe Korean TTS text are left in
`needs_manual_review` instead of being scripted for synthesis.

To revisit only failed/manual-review translation work, use:

```bash
asmr-dub translate-ko --project ./project --confirm-rights --gemma-text-backend llama_server --retry-failed
asmr-dub translate-ko --project ./project --repair-only
asmr-dub translate-ko --project ./project --retry-failed --force-retranslate-failed
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

Tests use mock backends and tiny generated WAV files. They do not require real Gemma, GPT-SoVITS, RVC, GPU, model downloads, or network access.
