from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .schemas import ProjectConfig

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
        "prompt_text": "それじゃあ……耳元で、ゆっくり囁いていきますね。",
        "prompt_lang": "ja",
    },
    "sleepy": {
        "ref_audio_path": "refs/sleepy.wav",
        "prompt_text": "もう少しだけ、力を抜いてくださいね。",
        "prompt_lang": "ja",
    },
}


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
        return ProjectConfig(project_name=Path(project_dir).name or "asmr-dub-project")
    data = yaml.safe_load(path.read_text("utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project config must be a mapping: {path}")
    return ProjectConfig.model_validate(data)


def save_project_config(config: ProjectConfig, path: Path) -> None:
    payload: dict[str, Any] = config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=True), "utf-8")
