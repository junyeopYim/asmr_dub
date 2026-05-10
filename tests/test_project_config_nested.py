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


def test_gsv_few_shot_target_sec_is_not_supported() -> None:
    payloads = [
        {"project_name": "legacy", "gsv_few_shot_target_sec": 120.0},
        {"project_name": "legacy", "gsv": {"few_shot_target_sec": 120.0}},
    ]
    for payload in payloads:
        try:
            ProjectConfig.model_validate(payload)
        except ValueError as exc:
            assert "few_shot_target_sec" in str(exc)
        else:
            raise AssertionError(f"few_shot_target_sec should be rejected for {payload}")


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


def test_load_project_config_applies_builtin_asr_prompt_guidance_when_blank(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "project_name": "project",
                "asr": {
                    "backend": "faster_whisper",
                    "correction_profile_path": "builtin:asmr_ja",
                    "initial_prompt": "",
                    "review_initial_prompt": "",
                    "qwen_context": "",
                },
            },
            allow_unicode=True,
            sort_keys=True,
        ),
        "utf-8",
    )

    cfg = load_project_config(project)

    assert cfg.asr_initial_prompt == ""
    assert "気持ちいい" not in cfg.asr_initial_prompt
    assert "カウントダウン" not in cfg.asr_initial_prompt
    assert "アクメ" not in cfg.asr_initial_prompt
    assert "愛撫" not in cfg.asr_initial_prompt
    assert "快感蓄積" not in cfg.asr_initial_prompt
    assert "レーザー" not in cfg.asr_initial_prompt
    assert "子宮" not in cfg.asr_initial_prompt
    assert "悪夢ノイド" not in cfg.asr_initial_prompt
    assert cfg.asr_review_initial_prompt == ""
    assert "Prefer audio evidence over domain assumptions" in cfg.qwen_asr_context


def test_builtin_asr_profile_guards_kaikan_misrecognitions() -> None:
    cfg = ProjectConfig()

    assert "女体化" in cfg.asr_hotwords
    assert "採集マシーン" in cfg.asr_hotwords
    assert "スキーン腺" in cfg.asr_hotwords
    assert "陰核" in cfg.asr_hotwords
    assert "陰核基部" in cfg.asr_hotwords
    assert "ポルチオ" in cfg.asr_hotwords
    assert "電マ" in cfg.asr_hotwords
    assert "オナニー" not in cfg.asr_hotwords
    assert "ゆっくり" not in cfg.asr_hotwords
    assert cfg.asr.correction_profile.review_initial_prompt == ""
    assert "悪夢します" in cfg.asr_review_suspicious_text_patterns
    assert "悪夢の前兆" in cfg.asr_review_suspicious_text_patterns
    assert "悪夢し" in cfg.asr_review_suspicious_text_patterns
    assert "悪夢、させ" in cfg.asr_review_suspicious_text_patterns
    assert "女の子悪夢" in cfg.asr_review_suspicious_text_patterns
    assert "ような悪夢" in cfg.asr_review_suspicious_text_patterns
    assert "尾薬" in cfg.asr_review_suspicious_text_patterns
    assert "体感を自分の体" in cfg.asr_repair_suspicious_text_patterns
    assert "体感が蓄積" in cfg.asr_repair_suspicious_text_patterns
    assert "体感を発し" in cfg.asr_review_suspicious_text_patterns
    assert cfg.asr_text_replacements["体感がさらに隠れ上が"] == "快感がさらに膨れ上が"
    assert cfg.asr_text_replacements["体感を生み出"] == "快感を生み出"
    assert cfg.asr_text_replacements["女の子悪夢"] == "女の子アクメ"
    assert cfg.asr_text_replacements["ような悪夢"] == "ようなアクメ"
    assert cfg.asr_text_replacements["尾薬"] == "媚薬"
    assert cfg.asr_review_candidate_replacements["悪夢します"] == "アクメします"
    assert cfg.asr_review_candidate_replacements["悪夢を確認"] == "アクメを確認"
    assert cfg.asr_review_candidate_replacements["悪夢し"] == "アクメし"
    assert cfg.asr_review_candidate_replacements["悪夢、させ"] == "アクメさせ"
    assert cfg.asr_review_candidate_replacements["女の子悪夢"] == "女の子アクメ"
    assert cfg.asr_review_candidate_replacements["ような悪夢"] == "ようなアクメ"
    assert cfg.asr_review_candidate_replacements["会館"] == "快感"
    assert cfg.asr_review_candidate_replacements["開館"] == "快感"
    assert cfg.asr_review_candidate_replacements["受精悪夢する"] == "受精アクメする"
    assert cfg.asr_review_candidate_replacements["出産悪夢する"] == "出産アクメする"
    assert cfg.asr_review_candidate_replacements["触手帳"] == "触手ちゃん"
    assert "(?:悪夢|悪目|明け目|アカメ)顔" in cfg.asr_repair_suspicious_text_patterns
    assert (
        "電話(?:を挟|邪魔|を(?:当て|押し当て|こす|擦)|で(?:クリ|乳首|おまんこ|陰核|刺激))"
        in cfg.asr_repair_suspicious_text_patterns
    )
    assert "(?:悪夢|悪目|明け目|アカメ)顔" in cfg.asr_review_suspicious_text_patterns
    assert cfg.asr_review_candidate_replacements["電話"] == "電マ"
    assert cfg.asr_review_candidate_replacements["尾薬"] == "媚薬"
    assert cfg.asr_review_candidate_replacements["体感蓄積"] == "快感蓄積"
    assert cfg.asr_text_replacements["男性機"] == "男性器"
    assert cfg.asr_text_replacements["女性機"] == "女性器"
    assert cfg.asr_text_replacements["断水器"] == "男性器"
    assert cfg.asr_text_replacements["愛婦"] == "愛撫"
    assert cfg.asr_text_replacements["見栄えなく"] == "見境なく"
    assert cfg.asr_text_replacements["ブルブル指揮されて"] == "ブルブル刺激されて"
    assert "気筒" in cfg.asr_repair_suspicious_text_patterns
    assert cfg.asr_review_candidate_replacements["男性機"] == "男性器"
    assert cfg.asr_review_candidate_replacements["女性機"] == "女性器"
    assert cfg.asr_review_candidate_replacements["断水器"] == "男性器"
    assert cfg.asr_review_candidate_replacements["愛婦"] == "愛撫"
    assert cfg.asr_review_candidate_replacements["見栄えなく"] == "見境なく"
    assert cfg.asr_review_candidate_replacements["ブルブル指揮されて"] == "ブルブル刺激されて"
    assert "腫瘍位" in cfg.asr_review_suspicious_text_patterns
    assert "情人" in cfg.asr_review_suspicious_text_patterns
    assert "ジェジオ" in cfg.asr_review_suspicious_text_patterns
    assert "逸落" in cfg.asr_review_suspicious_text_patterns
    assert "キチュウ" in cfg.asr_review_suspicious_text_patterns
    assert "溶かされるのは" in cfg.asr_review_suspicious_text_patterns
    assert "悪夢まで" in cfg.asr_translation_backcheck_source_patterns
    assert "悪夢し" in cfg.asr_translation_backcheck_source_patterns
    assert "女の子悪夢" in cfg.asr_translation_backcheck_source_patterns
    assert "尾薬" in cfg.asr_translation_backcheck_source_patterns
    assert "体感蓄積" in cfg.asr_translation_backcheck_source_patterns


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
