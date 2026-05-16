from __future__ import annotations

from pathlib import Path

import pytest

from asmr_dub_pipeline.asr.base import ASRChunk
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.schemas import ProjectConfig


def test_asr_review_guarded_auto_resolves_conservative_manual_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_review_enabled=True,
        asr_review_backend="mock",
        asr_review_suspicious_text_patterns=["高校に放出"],
        asr_review_candidate_replacements={"高校に放出": "口腔に放出"},
        asr_review_auto_resolution_rules=[
            {
                "id": "koukou_to_oral_release",
                "source": "高校に放出",
                "target": "口腔に放出",
                "required_all": ["生体コア", "ザーメン"],
                "negative_any": ["高校生", "学校", "授業", "校舎"],
            }
        ],
    )
    chunks = [
        ASRChunk(
            start=8373.446,
            end=8376.546,
            text="疑似ザーメン 生体コアの高校に放出完了",
            language="ja",
            confidence=0.97,
        )
    ]

    class ConservativeReviewClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def review_asr_candidates_for_mock(
            self,
            items: list[dict[str, object]],
            batch_id: str,
        ) -> dict[str, dict[str, object]]:
            assert batch_id == "asr_review_0001"
            assert items[0]["chunk_id"] == "chunk_0001"
            return {
                "chunk_0001": {
                    "chunk_id": "chunk_0001",
                    "decision": "manual_review",
                    "selected_candidate_id": "original",
                    "confidence": 0.97,
                    "heard_text": "疑似ザーメン 生体コアの高校に放出完了",
                    "reason": "conservative model kept the suspicious original",
                    "risk_terms": [],
                }
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", ConservativeReviewClient)

    reviewed, summary = pipeline_steps._review_asr_chunks_with_model(
        chunks,
        backend=object(),
        project_dir=tmp_project_dir,
        review_audio_path=tmp_project_dir / "missing.wav",
        audio_duration_sec=8400.0,
        cfg=cfg,
    )

    assert reviewed[0].text == "疑似ザーメン 生体コアの口腔に放出完了"
    assert summary["replaced"] == 1
    assert summary["manual_review"] == 0
    assert summary["guarded_auto_replaced"] == 1
    assert summary["items"][0]["accepted"] is True
    assert summary["items"][0]["resolution"] == "guarded_auto_replace"
    assert summary["items"][0]["auto_resolution_rule_id"] == "koukou_to_oral_release"
    assert summary["items"][0]["model_decision"] == "manual_review"
    assert summary["items"][0]["model_selected_candidate_id"] == "original"
    assert summary["items"][0]["selected_candidate_id"] == "domain_replacement"
    assert summary["items"][0]["selected_text"] == "疑似ザーメン 生体コアの口腔に放出完了"


def test_asr_review_guarded_auto_keeps_negative_context_manual_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_review_enabled=True,
        asr_review_backend="mock",
        asr_review_suspicious_text_patterns=["高校に放出"],
        asr_review_candidate_replacements={"高校に放出": "口腔に放出"},
        asr_review_auto_resolution_rules=[
            {
                "id": "koukou_to_oral_release",
                "source": "高校に放出",
                "target": "口腔に放出",
                "required_any": ["放出"],
                "negative_any": ["高校生", "学校", "授業", "校舎"],
            }
        ],
    )
    chunks = [
        ASRChunk(
            start=1.0,
            end=4.0,
            text="高校に放出された授業資料を確認します",
            language="ja",
            confidence=0.97,
        )
    ]

    class ConservativeReviewClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def review_asr_candidates_for_mock(
            self,
            items: list[dict[str, object]],
            _batch_id: str,
        ) -> dict[str, dict[str, object]]:
            return {
                str(items[0]["chunk_id"]): {
                    "chunk_id": str(items[0]["chunk_id"]),
                    "decision": "manual_review",
                    "selected_candidate_id": "original",
                    "confidence": 0.97,
                    "reason": "school context should not be auto replaced",
                    "risk_terms": [],
                }
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", ConservativeReviewClient)

    reviewed, summary = pipeline_steps._review_asr_chunks_with_model(
        chunks,
        backend=object(),
        project_dir=tmp_project_dir,
        review_audio_path=tmp_project_dir / "missing.wav",
        audio_duration_sec=10.0,
        cfg=cfg,
    )

    assert reviewed[0].text == "高校に放出された授業資料を確認します"
    assert summary["replaced"] == 0
    assert summary["manual_review"] == 1
    assert summary["guarded_auto_replaced"] == 0
    assert summary["items"][0]["accepted"] is False
    assert summary["items"][0].get("resolution") is None


def test_default_asr_profile_includes_koukou_auto_resolution_rule() -> None:
    cfg = ProjectConfig(project_name="test-project")

    assert any(
        rule.id == "koukou_to_oral_release"
        and rule.source == "高校に放出"
        and rule.target == "口腔に放出"
        for rule in cfg.asr_review_auto_resolution_rules
    )
