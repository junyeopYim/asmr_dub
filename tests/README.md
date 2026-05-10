# Test Suite Policy

The default pytest run is intentionally small. It should protect stable safety,
manifest, backend-contract, and mock-pipeline behavior without becoming a
scenario archive.

## Default Suite

- Put small, durable behavior checks in the default suite.
- Mark default-suite tests with `contract` when they protect a stable API,
  safety rule, manifest shape, or pipeline stage contract.
- Keep default tests tiny: generated fixtures, mock backends, and no model
  downloads or live servers.

## Regression Suites

- Mark bulky historical cases with `regression`.
- Add a new historical ASR/translation/TTS example to a table or fixture before
  adding a new test function.
- Prefer one policy test that loops over cases instead of one test function per
  observed media artifact.
- Run regression suites explicitly when changing nearby behavior:

```bash
uv run pytest -m regression
```

## Real Model Smoke

- Mark checks that need local models, heavyweight runtimes, or live servers with
  `real_model`.
- Keep smoke tests opt-in and documented with the required environment variables.

## Growth Guard

`tests/test_suite_policy.py` caps test file size. If it fails, reduce duplication,
move examples into fixtures, or split out a small contract test instead of
raising the limit.
