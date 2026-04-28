# Safety And Rights

Use this tool only with media, scripts, and voice references that you own or have permission/consent to use and distribute.

Commands that first touch real or source-derived media require either a confirmed manifest audit or a fresh `--confirm-rights`:

- `inspect`
- `extract`
- `segment`
- `analyze`
- `script`
- `qc`
- `synth`
- `mix`
- `export`
- `run`

`extract`, `inspect`, `mix`, `export`, and `run` require `--confirm-rights` directly. Derived stages such as `segment`, `analyze`, `script`, and `qc` can rely on the audit recorded by `extract`; pass `--confirm-rights` if you manually populated the project. Non-mock `synth` always requires `--confirm-rights` for the current source and voice references.

The manifest records confirmation time, command, source path, source SHA-256, voice reference notice, distribution notice, and local processing notice.

This project does not implement scraping, streaming-site downloads, DRM bypass, cookie/token acquisition, or unauthorized voice cloning workflows. Prompts and docs refer to authorized style references only.
