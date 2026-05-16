from __future__ import annotations

from collections import Counter

import pytest

from asmr_dub_pipeline.asr.base import ASRChunk
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.schemas import KoreanTranslation, ProjectConfig, Segment, SourceScript
from asmr_dub_pipeline.script.duration_rewrite import korean_tts_speech_char_count
from asmr_dub_pipeline.script.korean_tts_fit import (
    fit_korean_tts_budget,
    salvage_korean_translation,
    sanitize_korean_tts_text,
)

pytestmark = pytest.mark.contract


def _segment_with_translation(
    *,
    source: str,
    ko_literal: str,
    ko_natural: str,
) -> Segment:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
    )
    segment.source_script = SourceScript(
        text=source,
        language="ja",
        confidence=0.99,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal=ko_literal,
        ko_natural=ko_natural,
        notes=[],
        confidence=0.9,
        model="mock",
        batch_id="batch_0001",
    )
    return segment


def test_korean_tts_sanitizer_normalizes_digits_acronyms_and_symbols() -> None:
    sanitized, notes = sanitize_korean_tts_text("ASMR 5/OK—TTS")

    assert sanitized == "에이에스엠알 오 오케이 티티에스"
    assert notes == ["korean_tts_sanitized"]


def test_korean_tts_budget_fitter_reduces_over_budget_text() -> None:
    fitted, notes = fit_korean_tts_budget("좋아요 반드시 그렇게 돼요", max_speech_chars=10)

    assert korean_tts_speech_char_count(fitted) <= 10
    assert fitted
    assert notes == ["korean_tts_budget_fit"]


def test_salvage_korean_translation_prefers_valid_shorter_tts_text() -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=2.98,
        duration=2.98,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
        source_script=SourceScript(
            text="いいね 必ずそうなる",
            language="ja",
            confidence=0.99,
            backend="mock",
            start=0.0,
            end=2.98,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="좋아요 반드시 그렇게 돼요",
            ko_natural="좋아요 반드시 그렇게 돼요",
            notes=[],
            confidence=0.9,
            model="mock",
            batch_id="batch_0001",
        ),
    )

    result = salvage_korean_translation(segment, segment.translation_ko)

    assert result is not None
    salvaged, notes = result
    assert korean_tts_speech_char_count(salvaged.ko_natural) <= 10
    assert "korean_tts_budget_fit" in notes
    assert "korean_tts_budget_fit" in salvaged.notes


def test_asr_prompt_leak_contract_rejects_prompts_but_keeps_dialogue() -> None:
    cfg = ProjectConfig(
        project_name="test-project",
        asr_initial_prompt=(
            "Japanese ASMR domain terms: 快感 快感蓄積 快感増幅 快感の波 "
            "気持ちいい レーザー 子宮 悪夢ノイド"
        ),
    )

    assert pipeline_steps._asr_candidate_looks_prompt_leaked(cfg.asr.correction_profile.qwen_context, cfg)
    assert pipeline_steps._asr_candidate_looks_prompt_leaked("気持ちいい レーザー 子宮 悪夢ノイド", cfg)
    assert not pipeline_steps._asr_candidate_looks_prompt_leaked(
        "ありがとうございましたよしお前たち明日は今日よりもっと可愛くて",
        cfg,
    )


def test_asr_text_replacement_contract_repairs_domain_hits_and_keeps_safe_context() -> None:
    cfg = ProjectConfig()
    chunks, summary = pipeline_steps._apply_asr_text_replacements_to_chunks_with_summary(
        [
            ASRChunk(
                start=0.0,
                end=4.0,
                text="あっという間に釣りが来ちゃう",
                language="ja",
                confidence=0.92,
            ),
            ASRChunk(
                start=4.0,
                end=8.0,
                text="市民会館のホールでイベントを見た",
                language="ja",
                confidence=0.95,
            ),
            ASRChunk(
                start=8.0,
                end=12.0,
                text="怖い女の悪夢を見て眠れない",
                language="ja",
                confidence=0.96,
            ),
        ],
        cfg.asr_text_replacements,
        contextual_replacements=cfg.asr_review_candidate_replacements,
    )

    assert summary["chunks_changed"] == 1
    assert [chunk.text for chunk in chunks] == [
        "あっという間に絶頂が来ちゃう",
        "市民会館のホールでイベントを見た",
        "怖い女の悪夢を見て眠れない",
    ]


def test_korean_postprocess_contract_repairs_high_risk_asr_translation_artifacts() -> None:
    akume = _segment_with_translation(
        source="メスイキ悪夢が止まらない",
        ko_literal="암컷 절정 악몽이 멈추지 않습니다.",
        ko_natural="암컷 절정 악몽이 멈추지 않아요.",
    )
    guriguri = _segment_with_translation(
        source="やわらかくねじるように グリグリしてあげる",
        ko_literal="부드럽게 비틀듯이 그리그리 해줄게",
        ko_natural="부드럽게 비틀듯이, 그리그리 해줄게",
    )

    homophone_count = pipeline_steps._apply_korean_asr_homophone_postprocess(
        [akume],
        [],
        Counter(),
    )
    onomatopoeia_count = pipeline_steps._apply_korean_onomatopoeia_postprocess(
        [guriguri],
        [],
        Counter(),
    )

    assert homophone_count == 1
    assert akume.translation_ko is not None
    assert akume.translation_ko.ko_natural == "암컷 절정이 멈추지 않아요."
    assert onomatopoeia_count == 1
    assert guriguri.translation_ko is not None
    assert guriguri.translation_ko.ko_natural == "부드럽게 비틀듯이, 문질문질 해줄게"
