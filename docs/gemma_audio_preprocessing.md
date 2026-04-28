# Gemma Audio Preprocessing

Gemma audio-understanding backends consume short analysis clips, not the preserved mix source.

Expected clip contract:

- Mono WAV.
- 16 kHz sample rate.
- 30 seconds maximum per segment.
- No in-place mutation of user input media.
- Segment paths must remain under the project work directory and be traceable through manifests.

The pipeline preserves stereo 48 kHz audio separately for mixing. The mono 16 kHz files are only for Gemma ASR, translation, style tagging, script generation, and QC prompts. Current energy segmentation defaults to 20 second chunks, which keeps clips below the 30 second Gemma upper bound.

Real HF smoke checks are opt-in. Tests and normal runs must not download model weights; use cached weights and run with:

```bash
RUN_GEMMA_SMOKE=1 python -m pytest -q -k "gemma and smoke"
```

The HF backend defaults to `local_files_only=True`, so the smoke path also expects the selected Gemma model to be cached unless a caller explicitly opts into downloads.
