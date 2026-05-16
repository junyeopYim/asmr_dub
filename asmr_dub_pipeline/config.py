from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import (
    BUILTIN_ASR_CORRECTION_PROFILE,
    ProjectConfig,
    load_asr_correction_profile,
)

STANDARD_DIRS = [
    "input",
    "work/audio",
    "work/source_separation",
    "work/transcribe",
    "work/translate_ko",
    "work/gpt_sovits",
    "work/gpt_sovits/few_shot",
    "work/gpt_sovits/few_shot/wavs",
    "work/gpt_sovits/few_shot/configs",
    "work/gpt_sovits/few_shot/logs",
    "work/gpt_sovits/few_shot/weights/gpt",
    "work/gpt_sovits/few_shot/weights/sovits",
    "work/segments/audio",
    "work/segments/manifests",
    "work/tts/candidates",
    "work/tts/qwen",
    "work/tts/qwen/candidates",
    "work/tts/fish",
    "work/tts/fish/candidates",
    "work/tts/cosyvoice",
    "work/tts/cosyvoice/candidates",
    "work/rvc_train/dataset",
    "work/rvc_train/model",
    "work/rvc_train/logs",
    "work/rvc/candidates",
    "work/rvc/logs",
    "voice_bank",
    "voice_bank/sources",
    "voice_bank/speakers",
    "work/qc",
    "work/mix",
    "work/export",
    "refs",
    "output",
]

DEFAULT_REFS = {
    "whisper_close": {
        "ref_audio_path": "refs/whisper_close.wav",
        "prompt_text": "それじゃー……みみもとで、ゆっくりささやいていきますね。",
        "prompt_text_original": "それじゃあ……耳元で、ゆっくり囁いていきますね。",
        "prompt_lang": "ja",
    },
    "sleepy": {
        "ref_audio_path": "refs/sleepy.wav",
        "prompt_text": "もーすこしだけ、ちからをぬいてくださいね。",
        "prompt_text_original": "もう少しだけ、力を抜いてくださいね。",
        "prompt_lang": "ja",
    },
}
PROJECT_ASR_PROFILE_PATH = "profiles/asr/project.yaml"


def project_path(project_dir: Path | str, *parts: str) -> Path:
    return Path(project_dir).expanduser().resolve().joinpath(*parts)


def create_project_structure(project_dir: Path | str) -> None:
    root = Path(project_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for rel in STANDARD_DIRS:
        (root / rel).mkdir(parents=True, exist_ok=True)
    config_path = root / "pipeline.yaml"
    if not config_path.exists():
        save_project_config(ProjectConfig(project_name=root.name), config_path)
    refs_path = root / "refs" / "refs.json"
    if not refs_path.exists():
        import json

        refs_path.write_text(json.dumps(DEFAULT_REFS, ensure_ascii=False, indent=2) + "\n", "utf-8")


def load_project_config(project_dir: Path | str) -> ProjectConfig:
    path = Path(project_dir).expanduser().resolve() / "pipeline.yaml"
    if not path.exists():
        config = ProjectConfig(project_name=Path(project_dir).name or "asmr-dub-project")
        return _apply_asr_profile_guidance_defaults(config)
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project config must be a mapping: {path}")
    config = ProjectConfig.model_validate(data)
    config.asr.correction_profile = load_asr_correction_profile(
        config.asr.correction_profile_path,
        base_dir=path.parent,
    )
    return _apply_asr_profile_guidance_defaults(config)


def _apply_asr_profile_guidance_defaults(config: ProjectConfig) -> ProjectConfig:
    profile = config.asr.correction_profile
    if not config.asr.initial_prompt.strip() and profile.initial_prompt.strip():
        config.asr.initial_prompt = profile.initial_prompt.strip()
    if not config.asr.review_initial_prompt.strip() and profile.review_initial_prompt.strip():
        config.asr.review_initial_prompt = profile.review_initial_prompt.strip()
    if not config.asr.qwen_context.strip() and profile.qwen_context.strip():
        config.asr.qwen_context = profile.qwen_context.strip()
    return config


def save_project_config(config: ProjectConfig, path: Path) -> None:
    dump_config = config.model_copy(deep=True)
    profile_payload = dump_config.asr.correction_profile.model_dump(mode="json")
    default_profile_payload = load_asr_correction_profile(
        BUILTIN_ASR_CORRECTION_PROFILE
    ).model_dump(mode="json")
    if profile_payload != default_profile_payload:
        raw_profile_path = dump_config.asr.correction_profile_path
        if raw_profile_path and not str(raw_profile_path).startswith("builtin:"):
            rel_profile_path = str(raw_profile_path)
        else:
            rel_profile_path = PROJECT_ASR_PROFILE_PATH
        profile_path = Path(rel_profile_path).expanduser()
        if not profile_path.is_absolute():
            profile_path = path.parent / profile_path
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            yaml.safe_dump(profile_payload, allow_unicode=True, sort_keys=True),
            "utf-8",
        )
        dump_config.asr.correction_profile_path = rel_profile_path
    payload: dict[str, Any] = dump_config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), "utf-8")
