# asmr-dub-pipeline Codex Guidance

This project is a local-first Japanese ASMR dubbing automation pipeline. It should accept user-provided media, preserve high-quality source audio for mixing, create analysis clips for model backends, generate and synthesize Japanese ASMR dubbing candidates, perform QC, and mux final outputs only after explicit user rights confirmation.

## Project Defaults

- Use Python 3.11+.
- Prefer `uv` or standard `pip` compatible project setup.
- Use `pyproject.toml`.
- Use Pydantic v2 for schemas.
- Use Typer for CLI.
- Use Rich for logs and progress.
- Use pytest.
- Prefer small, composable modules over one monolithic script.
- Add docstrings and type hints.

## Model and Backend Boundaries

- External heavyweight models must be behind interfaces and support mock backends.
- Never require downloading huge model weights during tests.
- Gemma 4 E2B/E4B usage should support audio+text analysis, ASR, Japanese ASMR script generation, style tagging, and QC through deterministic structured outputs.
- GPT-SoVITS integration should target the api_v2 `/tts` endpoint through a configurable client and record all candidate generation details.

## Rights and Safety

- Require an explicit `--confirm-rights` flag before processing real input.
- Require user confirmation that they own or have rights/consent for the source content and any voice references.
- Do not implement scraping or downloading copyrighted content.
- Do not encourage unauthorized voice cloning or unauthorized content reuse.
- Avoid storing unnecessary private data, credentials, or source-derived artifacts outside the configured project work directory.

## Media and Artifacts

- Never modify input media in place.
- Keep generated artifacts under a project work directory.
- Preserve stereo 48 kHz audio for mixing.
- Create mono 16 kHz segment clips for Gemma audio+text analysis.
- Record all generated audio candidates and QC decisions.
- All JSON manifests must be deterministic and resumable.
- Manifest records should make each stage traceable from source media to segment analysis, script variants, TTS candidates, QC decisions, selected audio, mix decisions, and final muxed output.

## Testing Expectations

- Use pytest for unit, integration, and CLI tests.
- Prefer tiny fixtures, generated synthetic media, and mock backends for Gemma and GPT-SoVITS.
- Tests must not require live model servers, remote services, huge downloads, or large copyrighted assets.
- Include edge cases for missing rights confirmation, corrupt media metadata, empty or silent segments, failed model responses, JSON parse recovery, retry exhaustion, duration mismatch, and resumable manifests.
