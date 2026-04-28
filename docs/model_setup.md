# Model Setup

## Gemma

Backends:

- `mock`: deterministic local test backend.
- `hf`: Hugging Face Transformers backend for Gemma 4 E2B/E4B audio-capable models.
- `http`: explicit user-configured JSON endpoint.
- `llama_cpp`: local llama.cpp `llama-mtmd-cli` backend for GGUF Gemma 4 models.

The HF backend lazy-imports `transformers` and `torch` only when used. It defaults to `local_files_only=True`, so it will not silently download model weights. Cache the model yourself or adjust `pipeline.yaml` if you intentionally want downloads.

The `full` command checks the repository-local cache and, when `.cache/huggingface`
exists, sets `HF_HOME` and `TRANSFORMERS_CACHE` defaults for that process before
running. This makes cached models such as `.cache/huggingface/hub/models--google--gemma-4-E4B-it`
visible to HF tooling, but the `hf` backend still requires the optional
`transformers`, `torch`, and `accelerate` packages to be installed.

Gemma clips are prepared as mono 16 kHz WAV files for audio understanding. See
`docs/gemma_audio_preprocessing.md` for the full clip contract, including the
30 second maximum segment length.

Gemma responses must be JSON objects that validate against the task schema. The
real HF and HTTP backends strip markdown fences, extract the largest JSON object,
validate with Pydantic, and attempt one strict JSON repair prompt on parse or
schema failure. Mock mode remains deterministic and does not use repair prompts.

The `http` backend is a project-specific JSON endpoint at `/gemma`. It is not a vLLM or OpenAI-compatible `/v1/chat/completions` adapter.

The `llama_cpp` backend invokes `llama-mtmd-cli` locally for each Gemma task and
expects a GGUF model plus the matching multimodal projector. The default paths
target the repo-local HauhauCS Gemma 4 E4B Q4 cache:

```yaml
gemma_llama_cpp_cli_path: .cache/llama_cpp/src/llama.cpp/build/bin/llama-mtmd-cli
gemma_llama_cpp_model_path: .cache/llama_cpp/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
gemma_llama_cpp_mmproj_path: .cache/llama_cpp/models/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf
```

Run it with:

```bash
asmr-dub full ./audio/RJ01012948.mp4 --confirm-rights --real --gemma-backend llama_cpp
```

This backend is local, but slower than a resident server because it launches the
llama.cpp process per segment. Its output still must be strict JSON matching the
pipeline task schemas.

## GPT-SoVITS

You can start GPT-SoVITS `api_v2.py` manually after installing GPT-SoVITS and
loading only models and references you have rights/consent to use, for example:

```bash
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

Then run:

```bash
asmr-dub synth --project ./project --gsv-url http://127.0.0.1:9880 --refs refs/refs.json --confirm-rights
```

For one-command real runs, `full --real` can manage a local GPT-SoVITS server
for the duration of the pipeline. It first checks whether `gsv_url` is already
listening; if so, it reuses that process and does not shut it down. If not, it
starts the configured command and terminates only the process it started when
the run finishes. Real runs now default to few-shot GPT-SoVITS: the pipeline
collects about 60 seconds of transcribed source segments, writes a project-local
training set, runs GPT-SoVITS prepare/train scripts, loads the resulting
`.ckpt`/`.pth` weights through `api_v2`, and then calls `/tts`.

```bash
asmr-dub full ./audio/RJ01012948.mp4 \
  --confirm-rights \
  --real \
  --gemma-backend llama_cpp
```

Use `--zero-shot` to skip fine-tuning and synthesize with only `refs/refs.json`
reference audio. Use `--force-few-shot` to retrain even when the source segment
fingerprint and generated weights already match. You can also run only the
training stage:

```bash
asmr-dub train-gsv --project ./project --confirm-rights
```

By default, auto-start looks for repo-local installs including
`.cache/third_party/GPT-SoVITS/api_v2.py`,
`.cache/gpt_sovits/GPT_SoVITS/api_v2.py`, and sibling `GPT-SoVITS` folders.
When it finds `api_v2.py`, it uses that directory as the server working
directory and passes the adjacent `GPT_SoVITS/configs/tts_infer.yaml` when it
exists.

If GPT-SoVITS is installed somewhere else, store the command in `pipeline.yaml`:

```yaml
gsv_auto_start: true
gsv_server_command:
  - python
  - /path/to/GPT-SoVITS/api_v2.py
  - -a
  - 127.0.0.1
  - -p
  - "9880"
  - -c
  - /path/to/GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml
gsv_server_cwd: /path/to/GPT-SoVITS
gsv_server_startup_timeout_sec: 120.0
gsv_server_shutdown_timeout_sec: 10.0
gsv_few_shot_enabled: true
gsv_few_shot_target_sec: 60.0
gsv_few_shot_min_clip_sec: 1.0
gsv_few_shot_max_clip_sec: 10.0
gsv_few_shot_force: false
gsv_few_shot_version: auto
```

If no command is supplied and no known `api_v2.py` exists, it fails before
synthesis with a message telling you to install GPT-SoVITS or set
`gsv_server_command`.

Cached GPT-SoVITS weights under `.cache/gpt_sovits` are not an in-process
backend. They are available for a GPT-SoVITS `api_v2` server that you start
yourself or let the pipeline auto-start with an explicit command; the pipeline
still connects to that server over HTTP.

`refs/refs.json` must live inside the project and all `ref_audio_path` and `aux_ref_audio_paths` entries must also resolve inside the project. The CLI sends normalized absolute paths to GPT-SoVITS, so the server process must be able to read those files. Non-mock synthesis requires a fresh `--confirm-rights` because it uses the current voice-reference file.

Optional model switching can be configured in `pipeline.yaml` or passed on the CLI:

```yaml
gsv_gpt_weights_path: null
gsv_sovits_weights_path: null
```

```bash
asmr-dub synth --project ./project --confirm-rights --gpt-weights /path/to/model.ckpt --sovits-weights /path/to/model.pth
```

When `train-gsv` has completed, standalone `synth` automatically uses
`work/manifest.json` artifacts `gsv_few_shot_gpt_weights` and
`gsv_few_shot_sovits_weights` unless explicit weight paths are passed.
Few-shot data, configs, logs, and weights stay under
`work/gpt_sovits/few_shot/`. The pipeline never edits the GPT-SoVITS checkout
or source media in place.

Remote endpoints may receive text/audio metadata and must be explicitly configured by you. This project does not install Gemma, GPT-SoVITS, model weights, CUDA, or server processes.

### `/tts` payload contract

The client posts JSON to `POST {gsv_url}/tts` and expects WAV bytes back. JSON
responses, empty bodies, and non-audio bodies are treated as errors and are not
saved as WAV files.

Required text/reference fields:

- `text`: normalized Japanese TTS text for the segment.
- `text_lang`: `ja`.
- `ref_audio_path`: absolute path to a project-local voice reference readable by the GPT-SoVITS server.
- `prompt_text`: transcript for the reference audio.
- `prompt_lang`: usually `ja`.
- `aux_ref_audio_paths`: zero or more additional project-local reference paths.

Generation fields sent explicitly for deterministic records:

- `top_k`, `top_p`, `temperature`
- `text_split_method`
- `batch_size`, `batch_threshold`, `split_bucket`
- `speed_factor`
- `fragment_interval`
- `seed`
- `media_type: wav`
- `streaming_mode: false`
- `parallel_infer`
- `repetition_penalty`
- `sample_steps`
- `super_sampling`
- `overlap_length`
- `min_chunk_length`

Each manifest candidate records the exact payload plus retry metadata. If a
candidate is too long for the source segment, synthesis first retries with a
higher `speed_factor`; if it is still too long, it retries after requesting the
existing script-duration rewrite hook. If a previous QC pass flagged repetition
or omission and the segment is synthesized again, the next GPT-SoVITS attempt
uses a deterministic new seed and a higher `repetition_penalty`.

### Optional smoke test

Routine tests never require a running GPT-SoVITS server. For a local manual
HTTP smoke test, set the environment variables below; when `GSV_URL` is unset,
the hook exits successfully without contacting anything. The pytest smoke test
uses `GSV_REF_AUDIO_PATH`; the standalone module uses `GSV_REF_AUDIO` and
requires `GSV_CONFIRM_RIGHTS=1` so voice-reference consent is explicit.

```bash
GSV_URL=http://127.0.0.1:9880 \
GSV_CONFIRM_RIGHTS=1 \
GSV_REF_AUDIO=/absolute/path/inside/project/refs/whisper_close.wav \
GSV_PROMPT_TEXT='耳元で、ゆっくり囁いていきますね。' \
python -m asmr_dub_pipeline.gpt_sovits.smoke
```

Optional variables: `GSV_TEXT`, `GSV_OUT`, `GSV_PROMPT_LANG`, and `GSV_SEED`.
