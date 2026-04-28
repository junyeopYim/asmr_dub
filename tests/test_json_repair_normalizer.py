from __future__ import annotations

import pytest

from asmr_dub_pipeline.gemma.json_repair import JSONRepairError, loads_json_dict
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
