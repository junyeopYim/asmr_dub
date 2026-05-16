from __future__ import annotations

import json
from pathlib import Path

from asmr_dub_pipeline.script.normalizer import normalize_japanese_kana_text

from .schemas import GPTSoVITSRef


class GPTSoVITSRefsError(ValueError):
    pass


def resolve_refs_json_path(path: Path, project_dir: Path | None = None) -> Path:
    actual = path.expanduser()
    if not actual.is_absolute() and project_dir:
        actual = project_dir / actual
    actual = actual.resolve()
    if project_dir:
        root = project_dir.expanduser().resolve()
        try:
            actual.relative_to(root)
        except ValueError as exc:
            raise GPTSoVITSRefsError(f"refs JSON must be inside the project directory: {actual}") from exc
    return actual


def _resolve_ref_audio_path(project_root: Path, style: str, raw_path: str, field_name: str) -> str:
    ref_path = Path(raw_path).expanduser()
    resolved = (project_root / ref_path).resolve() if not ref_path.is_absolute() else ref_path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise GPTSoVITSRefsError(
            f"{field_name} for style {style!r} must be inside the project directory: {resolved}"
        ) from exc
    return str(resolved)


def _canonical_language(value: str | None) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"ja", "jp", "jpn", "japanese"}:
        return "ja"
    if normalized in {"ko", "kr", "kor", "korean"}:
        return "ko"
    return normalized


def _normalize_ref_prompt_text(_style: str, ref: GPTSoVITSRef) -> GPTSoVITSRef:
    if _canonical_language(ref.prompt_lang) != "ja":
        return ref
    normalized = normalize_japanese_kana_text(ref.prompt_text)
    return ref.model_copy(
        update={
            "prompt_text": normalized.text,
            "prompt_text_original": ref.prompt_text_original or normalized.original_text,
            "text_normalization": {
                "policy": "ja_hiragana",
                "risk_flags": normalized.risk_flags,
            },
        }
    )


def load_refs(path: Path, project_dir: Path | None = None) -> dict[str, GPTSoVITSRef]:
    actual = resolve_refs_json_path(path, project_dir)
    data = json.loads(actual.read_text("utf-8"))
    if not isinstance(data, dict):
        raise GPTSoVITSRefsError("refs JSON must be an object keyed by style name")
    refs = {key: GPTSoVITSRef.model_validate(value) for key, value in data.items()}
    if project_dir:
        root = project_dir.expanduser().resolve()
        for key, ref in refs.items():
            refs[key] = ref.model_copy(
                update={
                    "ref_audio_path": _resolve_ref_audio_path(root, key, ref.ref_audio_path, "ref_audio_path"),
                    "aux_ref_audio_paths": [
                        _resolve_ref_audio_path(root, key, aux_path, "aux_ref_audio_paths")
                        for aux_path in ref.aux_ref_audio_paths
                    ],
                }
            )
    refs = {key: _normalize_ref_prompt_text(key, ref) for key, ref in refs.items()}
    return refs


def resolve_ref(
    refs: dict[str, GPTSoVITSRef],
    style: str,
    fallback_style: str = "whisper_close",
) -> GPTSoVITSRef:
    if style in refs:
        return refs[style]
    if fallback_style in refs:
        return refs[fallback_style]
    raise GPTSoVITSRefsError(
        f"Missing ref style {style!r}. Available styles: {', '.join(sorted(refs))}"
    )
