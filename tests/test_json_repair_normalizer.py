from __future__ import annotations

import pytest

from asmr_dub_pipeline.gemma.json_repair import JSONRepairError, loads_json_dict
from asmr_dub_pipeline.gemma.prompts import analysis_prompt, audio_style_prompt
from asmr_dub_pipeline.gemma.schemas import validate_gemma_task_response
from asmr_dub_pipeline.gemma.text_translate import LlamaServerTranslationClient
from asmr_dub_pipeline.schemas import JapaneseScript, Segment
from asmr_dub_pipeline.script.normalizer import normalize_script_payload, normalize_tts_text
from asmr_dub_pipeline.script.text_qc import preflight_tts_text


def test_json_repair_handles_fence_and_trailing_comma() -> None:
    assert loads_json_dict('```json\n{"a": 1,}\n```') == {"a": 1}
    assert loads_json_dict('Here: {"a": 2, "b": [1,2,], } done') == {"a": 2, "b": [1, 2]}


def test_json_repair_rejects_non_object() -> None:
    with pytest.raises(JSONRepairError):
        loads_json_dict("[1, 2, 3]")


def test_normalizer_moves_cues_and_preserves_pacing() -> None:
    result = normalize_tts_text("(小声で)ASMR OK 3分、10秒……ね♡")
    assert "小声" not in result.text
    assert "エーエスエムアール" in result.text
    assert "オーケー" in result.text
    assert "さんぷん" in result.text
    assert "じゅうびょう" in result.text
    assert "……" in result.text
    assert any(cue.kind == "style" for cue in result.cues)
    assert any(cue.kind == "soft_affect" for cue in result.cues)


def test_script_payload_normalizes_ja_text_too() -> None:
    script = normalize_script_payload(
        {
            "ja_text": "[耳元で]OKです。",
            "tts_text": "(小声で)OKです。",
            "expected_tts_duration_sec": 1.0,
        }
    )
    assert "[" not in script.ja_text
    assert "(" not in script.tts_text
    assert "オーケー" in script.ja_text
    assert len(script.nonverbal_cues) >= 2


def test_korean_normalizer_preserves_cues_and_text_qc_blocks_kana() -> None:
    result = normalize_tts_text("(귓가에)ASMR OK... 괜찮아요♡", language="ko")
    assert "(" not in result.text
    assert "에이에스엠알" in result.text
    assert "오케이" in result.text
    assert any(cue.kind == "soft_affect" for cue in result.cues)

    script = normalize_script_payload(
        {
            "ja_text": "少し近づきますね",
            "tts_text": "少し近づきますね",
            "tts_language": "ko",
        },
        language="ko",
    )
    qc = preflight_tts_text(script, target_language="ko", source_text="少し近づきますね")
    assert qc.blocked
    assert "korean_tts_contains_kana" in qc.issues


def test_korean_normalizer_strips_leading_japanese_sentence_fragment() -> None:
    result = normalize_tts_text("。징, 징…", language="ko")

    assert result.text == "징, 징…"
    assert "normalized_tts_text" in result.risk_flags


def test_korean_normalizer_spells_risky_tokens_and_splits_long_clauses() -> None:
    result = normalize_tts_text("GPT-SoVITS 3% OK & RVC", language="ko")
    assert result.text == "지피티 소비츠 삼 퍼센트 오케이 그리고 알브이씨"
    assert "normalized_latin_token" in result.risk_flags
    assert "normalized_numeric_token" in result.risk_flags
    assert "normalized_symbol_token" in result.risk_flags
    assert not preflight_tts_text(
        JapaneseScript(ja_text=result.text, tts_text=result.text, tts_language="ko"),
        target_language="ko",
    ).blocked

    speech_safe = normalize_tts_text("지금은요— '에?' 2014 BGM", language="ko")
    assert speech_safe.text == "지금은요… 에? 이천십사 비지엠"
    assert "normalized_latin_token" in speech_safe.risk_flags
    assert "normalized_numeric_token" in speech_safe.risk_flags
    assert "normalized_symbol_token" in speech_safe.risk_flags
    assert not preflight_tts_text(
        JapaneseScript(
            ja_text=speech_safe.text,
            tts_text=speech_safe.text,
            tts_language="ko",
        ),
        target_language="ko",
    ).blocked

    raw_risky = JapaneseScript(
        ja_text="테스트",
        tts_text="오늘은 GPT-SoVITS 3%로 갈게요",
        tts_language="ko",
    )
    raw_qc = preflight_tts_text(raw_risky, target_language="ko")
    assert raw_qc.blocked
    assert "korean_tts_contains_latin" in raw_qc.issues
    assert "korean_tts_contains_digit" in raw_qc.issues
    assert "korean_tts_contains_pronunciation_symbol" in raw_qc.issues

    long_result = normalize_tts_text(
        "오늘은 아주 조용하게 숨을 천천히 고르고 조금 더 가까이 다가가서 "
        "편안하게 들리도록 말해드릴게요 계속 긴장을 풀어주세요",
        language="ko",
    )
    assert "," in long_result.text
    assert "split_long_korean_clause" in long_result.risk_flags
    assert not preflight_tts_text(
        JapaneseScript(ja_text=long_result.text, tts_text=long_result.text, tts_language="ko"),
        target_language="ko",
    ).blocked


def test_pause_cue_gets_timing_metadata_not_tts_text() -> None:
    script = normalize_script_payload(
        {
            "ja_text": "ここで待ってね。",
            "tts_text": "ここで(一拍待つ)待ってね。",
            "spatial_style": "sleepy_center",
            "expected_tts_duration_sec": 1.0,
        }
    )

    assert script.spatial_style == "sleepy_center"
    assert "(" not in script.tts_text
    pause_cues = [cue for cue in script.nonverbal_cues if cue.kind == "pause"]
    assert pause_cues
    assert pause_cues[0].pause_sec == 0.25


def test_gemma_analysis_contract_marks_effected_voice_without_extra_speaker() -> None:
    payload = {
        "source_language": "ja",
        "transcript_original": "もしもし、聞こえますか。",
        "literal_ja": "もしもし、聞こえますか。",
        "speech_style": "telephone filtered whisper",
        "speaker_count": 1,
        "emotion": "gentle",
        "pace": "slow",
        "volume": "soft",
        "nonverbal_cues": [],
        "spatial_style": "center",
        "style_tags": ["telephone", "soft_whisper"],
        "estimated_pan": 0.0,
        "keep_original_texture": True,
        "risk_flags": [],
        "confidence": 0.92,
        "voice_training": {
            "clean_voice": False,
            "eligible": False,
            "reason": "telephone filter is an effect on the same speaker",
            "effect_tags": ["telephone"],
            "same_speaker_under_effect": True,
        },
    }

    result = validate_gemma_task_response("analyze", payload)

    assert result["speaker_count"] == 1
    assert result["voice_training"]["same_speaker_under_effect"] is True
    assert result["voice_training"]["eligible"] is False
    assert result["voice_training"]["effect_tags"] == ["telephone"]

    prompt = analysis_prompt(
        Segment(
            id="seg_0001",
            start=0.0,
            end=1.0,
            duration=1.0,
            audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
            audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        )
    )
    assert "distinct human speakers" in prompt
    assert "voice_training" in prompt


def test_audio_style_prompt_is_compact_and_effect_focused() -> None:
    prompt = audio_style_prompt(
        Segment(
            id="seg_0001",
            start=10.0,
            end=12.0,
            duration=2.0,
            audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
            audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        )
    )

    assert "Required keys: style_tags, nonverbal_cues, spatial_style, estimated_pan" in prompt
    assert "effect_tags" in prompt
    assert "none" in prompt
    assert "effect_events" in prompt
    assert "Allowed spatial_style:" in prompt
    assert "left_close" in prompt
    assert "source_info" not in prompt
    assert "Context:" not in prompt
    assert "translation" not in prompt.lower()


def test_audio_style_schema_normalizes_spatial_style_aliases() -> None:
    payload = {
        "style_tags": [],
        "nonverbal_cues": [],
        "spatial_style": "center_left",
        "estimated_pan": -0.35,
        "keep_original_texture": True,
        "risk_flags": [],
        "confidence": 0.9,
        "effect_events": [],
        "voice_training": {
            "clean_voice": True,
            "eligible": True,
            "reason": "plain voice",
            "effect_tags": [],
            "same_speaker_under_effect": False,
        },
    }

    result = validate_gemma_task_response("audio_style", payload)

    assert result["spatial_style"] == "left_close"


def test_audio_style_schema_defaults_ambiguous_spatial_style_to_center() -> None:
    payload = {
        "style_tags": [],
        "nonverbal_cues": [],
        "spatial_style": "here",
        "estimated_pan": 0.0,
        "keep_original_texture": True,
        "risk_flags": [],
        "confidence": 0.9,
        "effect_events": [],
        "voice_training": {
            "clean_voice": True,
            "eligible": True,
            "reason": "plain voice",
            "effect_tags": [],
            "same_speaker_under_effect": False,
        },
    }

    result = validate_gemma_task_response("audio_style", payload)

    assert result["spatial_style"] == "center"


def test_audio_style_schema_locally_repairs_loose_effect_events() -> None:
    payload = {
        "effect_events": [
            {
                "name": "Echo",
                "target": "Voice Effect",
                "start": "0.2",
                "end": "1.4",
                "intensity": "high",
                "confidence": "medium",
                "params": ["wet"],
            },
            {"tag": "none"},
            "not an event object",
        ],
        "voice_training": {"effect_tags": "echo"},
    }

    result = validate_gemma_task_response("audio_style", payload)

    assert result["voice_training"]["effect_tags"] == ["echo"]
    assert result["effect_events"] == [
        {
            "tag": "echo",
            "target": "voice",
            "start_sec": 0.2,
            "end_sec": 1.4,
            "intensity": 1.0,
            "confidence": 0.6,
            "params": {},
        }
    ]


def test_audio_style_client_uses_local_repair_before_model_repair() -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=2.0,
        duration=2.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
    )

    class FakeClient(LlamaServerTranslationClient):
        def __init__(self) -> None:
            self.retries = 1
            self.n_predict = 384

        def _complete_with_input_audio(self, *_args: object, **_kwargs: object) -> str:
            return (
                '{"effect_events":[{"name":"Echo","target":"Voice Effect",'
                '"start":"0.2","end":"1.4","intensity":"high",'
                '"confidence":"medium","params":["wet"]}],'
                '"voice_training":{"effect_tags":"echo"}}'
            )

        def _complete(self, *_args: object, **_kwargs: object) -> str:
            pytest.fail("audio-style should normalize loose effect_events before model JSON repair")

    result = FakeClient().analyze_audio_style("unused.wav", segment)

    assert result["effect_events"][0]["tag"] == "echo"
    assert result["effect_events"][0]["confidence"] == pytest.approx(0.6)
