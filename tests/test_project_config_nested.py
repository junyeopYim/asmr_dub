from __future__ import annotations

from pathlib import Path

import yaml

from asmr_dub_pipeline.config import load_project_config, save_project_config
from asmr_dub_pipeline.schemas import ProjectConfig


def test_project_config_serializes_nested_sections_without_asr_profile_payload() -> None:
    cfg = ProjectConfig(
        asr_backend="mock",
        gemma_llama_cpp_model_path="models/gemma.gguf",
        gsv_url="http://127.0.0.1:9881",
        rvc_backend="mock",
        mix_peak_limit_dbfs=-2.0,
        voice_bank_path="voice_bank/custom.json",
        hf_local_files_only=False,
    )

    payload = cfg.model_dump(mode="json")

    assert payload["asr"]["backend"] == "mock"
    assert payload["gemma"]["llama_cpp_model_path"] == "models/gemma.gguf"
    assert payload["gsv"]["url"] == "http://127.0.0.1:9881"
    assert payload["rvc"]["backend"] == "mock"
    assert payload["mix"]["peak_limit_dbfs"] == -2.0
    assert payload["voice_bank"]["path"] == "voice_bank/custom.json"
    assert payload["safety"]["hf_local_files_only"] is False
    assert payload["asr"]["correction_profile_path"] == "builtin:asmr_ja"
    assert "correction_profile" not in payload["asr"]
    assert "asr_backend" not in payload
    assert "asr_text_replacements" not in payload
    assert "gemma_llama_cpp_model_path" not in payload

    assert cfg.asr_backend == "mock"
    assert cfg.gemma_llama_cpp_model_path == "models/gemma.gguf"
    assert cfg.gsv_url == "http://127.0.0.1:9881"
    assert cfg.rvc_backend == "mock"
    assert cfg.mix_peak_limit_dbfs == -2.0
    assert cfg.voice_bank_path == "voice_bank/custom.json"
    assert cfg.hf_local_files_only is False


def test_legacy_flat_project_config_migrates_to_nested_sections() -> None:
    cfg = ProjectConfig.model_validate(
        {
            "project_name": "legacy",
            "asr_backend": "mock",
            "asr_review_enabled": True,
            "gemma_text_batch_size": 2,
            "gsv_tts_max_speed_factor": 1.05,
            "rvc_backend": "mock",
            "mix_sample_rate": 44_100,
            "voice_bank_path": "voice_bank/custom.json",
            "speaker_assignment_backend": "pyannote",
            "hf_local_files_only": False,
        }
    )

    assert cfg.asr.backend == "mock"
    assert cfg.asr.review_enabled is True
    assert cfg.gemma.text_batch_size == 2
    assert cfg.gsv.tts_max_speed_factor == 1.05
    assert cfg.rvc.backend == "mock"
    assert cfg.mix.sample_rate == 44_100
    assert cfg.voice_bank.path == "voice_bank/custom.json"
    assert cfg.voice_bank.speaker_assignment_backend == "pyannote"
    assert cfg.safety.hf_local_files_only is False


def test_load_project_config_applies_external_asr_correction_profile(tmp_path: Path) -> None:
    project = tmp_path / "project"
    profile = project / "profiles" / "asr" / "custom.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text(
        yaml.safe_dump(
            {
                "hotwords": "耳舐め カスタム",
                "repair_suspicious_text_patterns": ["聞き間違い"],
                "text_replacements": {"聞き間違い": "聞き直し"},
                "review_suspicious_text_patterns": ["レビュー対象"],
                "review_candidate_replacements": {"レビュー対象": "修正後"},
                "translation_backcheck_source_patterns": ["原文確認"],
                "translation_backcheck_ko_patterns": ["번역확인"],
            },
            allow_unicode=True,
            sort_keys=True,
        ),
        "utf-8",
    )
    (project / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "project_name": "project",
                "asr": {
                    "backend": "mock",
                    "correction_profile_path": "profiles/asr/custom.yaml",
                },
            },
            allow_unicode=True,
            sort_keys=True,
        ),
        "utf-8",
    )

    cfg = load_project_config(project)

    assert cfg.asr_backend == "mock"
    assert cfg.asr_hotwords == "耳舐め カスタム"
    assert cfg.asr_repair_suspicious_text_patterns == ["聞き間違い"]
    assert cfg.asr_text_replacements == {"聞き間違い": "聞き直し"}
    assert cfg.asr_review_suspicious_text_patterns == ["レビュー対象"]
    assert cfg.asr_review_candidate_replacements == {"レビュー対象": "修正後"}
    assert cfg.asr_translation_backcheck_source_patterns == ["原文確認"]
    assert cfg.asr_translation_backcheck_ko_patterns == ["번역확인"]


def test_save_project_config_writes_nested_yaml_without_loaded_asr_profile(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    cfg = ProjectConfig()

    save_project_config(cfg, path)

    payload = yaml.safe_load(path.read_text("utf-8"))
    assert payload["asr"]["correction_profile_path"] == "builtin:asmr_ja"
    assert "correction_profile" not in payload["asr"]
    assert "asr_text_replacements" not in payload


def test_save_project_config_writes_custom_asr_profile_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "pipeline.yaml"
    cfg = ProjectConfig(asr_hotwords="耳舐め", asr_text_replacements={"誤": "正"})

    save_project_config(cfg, path)

    payload = yaml.safe_load(path.read_text("utf-8"))
    assert payload["asr"]["correction_profile_path"] == "profiles/asr/project.yaml"
    assert "correction_profile" not in payload["asr"]
    profile = yaml.safe_load((tmp_path / "profiles" / "asr" / "project.yaml").read_text("utf-8"))
    assert profile["hotwords"] == "耳舐め"
    assert profile["text_replacements"] == {"誤": "正"}
