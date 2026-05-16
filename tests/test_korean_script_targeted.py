from __future__ import annotations

from pathlib import Path

from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import korean_script_step
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    RVCMetadata,
    Segment,
    SourceScript,
    TTSMetadata,
)


def _translated_segment(segment_id: str, text: str, translation: str, status: str) -> Segment:
    return Segment(
        id=segment_id,
        start=0.0,
        end=3.0,
        duration=3.0,
        audio_for_gemma=f"work/segments/audio/{segment_id}_gemma.wav",
        audio_for_mix=f"work/segments/audio/{segment_id}_mix.wav",
        status=status,
        source_script=SourceScript(
            text=text,
            language="ja",
            backend="mock",
            start=0.0,
            end=3.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal=translation,
            ko_natural=translation,
            model="test",
            batch_id=f"batch_{segment_id}",
        ),
    )


def test_korean_script_only_segments_preserves_existing_rvc_segment(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    keep = _translated_segment("seg_keep", "残す", "기존 문장입니다.", "rvc_converted")
    keep.script = JapaneseScript(
        literal_ja="残す",
        ja_text="残す",
        tts_text="기존 문장입니다.",
        tts_language="ko",
        source_language="ja",
        target_language="ko",
    )
    keep.tts = TTSMetadata(selected_candidate_path="work/tts/seg_keep_final.wav")
    keep.rvc = RVCMetadata(
        backend="mock",
        input_path="work/tts/seg_keep_final.wav",
        output_path="work/rvc/seg_keep_final.wav",
        accepted=True,
    )
    target = _translated_segment("seg_target", "やり直す", "새로 만들 문장입니다.", "transcribed")
    manifest = PipelineManifest(segments=[keep, target])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True, only_segment_ids={"seg_target"})

    updated = {segment.id: segment for segment in load_manifest(tmp_path).segments}
    assert updated["seg_keep"].status == "rvc_converted"
    assert updated["seg_keep"].script is not None
    assert updated["seg_keep"].script.tts_text == "기존 문장입니다."
    assert updated["seg_keep"].tts is not None
    assert updated["seg_keep"].tts.selected_candidate_path == "work/tts/seg_keep_final.wav"
    assert updated["seg_keep"].rvc is not None
    assert updated["seg_keep"].rvc.accepted is True
    assert updated["seg_target"].status == "scripted"
    assert updated["seg_target"].script is not None
    assert updated["seg_target"].script.tts_text == "새로 만들 문장이에요."


def test_korean_script_periodizes_numeric_cadence_segments(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    segment = _translated_segment(
        "seg_counting",
        "二、三、四、五、六、七、八",
        "둘, 셋, 넷. 제자리. 다섯, 여섯, 일곱, 여덟",
        "transcribed",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True)

    updated = load_manifest(tmp_path).segments[0]
    assert updated.status == "scripted"
    assert updated.script is not None
    assert updated.script.tts_text == "둘. 셋. 넷. 제자리. 다섯. 여섯. 일곱. 여덟."
    assert "korean_numeric_cadence_periodized" in updated.script.risk_flags
    assert updated.analysis["korean_numeric_cadence_periodization"]["variant"] == (
        "native_periods_no_compact"
    )


def test_korean_script_marks_repeated_texture_after_translation(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    segment = _translated_segment(
        "seg_texture",
        "グーッ、グーッと力を込めて。",
        "으으음, 으으음 하고 힘을 꽉 주세요.",
        "transcribed",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True)

    updated = load_manifest(tmp_path)
    texture = updated.segments[0]
    assert texture.status == "non_speech_texture"
    assert texture.keep_original_texture is True
    assert texture.script is None
    assert texture.translation_ko is not None
    assert texture.translation_ko.ko_natural == "으으음, 으으음 하고 힘을 꽉 주세요."
    assert texture.errors == ["korean_script_repeated_texture_keep_original"]
    assert texture.analysis["korean_script_non_speech_texture"] == {
        "action": "keep_original_texture",
        "reason": "repeated_vowel_or_onomatopoeia_after_translation",
        "source_text": "グーッ、グーッと力を込めて。",
        "tts_text": "으으음, 으으음 하고 힘을 꽉 주세요.",
    }
    assert updated.stage_state["korean-script"]["non_speech_texture"] == 1


def test_korean_script_keeps_tapping_instruction_scriptable(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    segment = _translated_segment(
        "seg_tapping",
        "トントン叩きますね。",
        "톡톡 두드려 드릴게요.",
        "transcribed",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True)

    updated = load_manifest(tmp_path)
    tapping = updated.segments[0]
    assert tapping.status == "scripted"
    assert tapping.script is not None
    assert tapping.script.tts_text == "톡톡 두드려 드릴게요."
    assert "korean_script_non_speech_texture" not in tapping.analysis
    assert updated.stage_state["korean-script"]["non_speech_texture"] == 0


def test_korean_script_keeps_vowel_like_speech_scriptable(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    segment = _translated_segment(
        "seg_vowel_speech",
        "ああ、わかりました。",
        "아아아아, 이제 알겠어요.",
        "transcribed",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True)

    updated = load_manifest(tmp_path)
    vowel_speech = updated.segments[0]
    assert vowel_speech.status == "scripted"
    assert vowel_speech.script is not None
    assert vowel_speech.script.tts_text == "아아아아, 이제 알겠어요."
    assert "korean_script_non_speech_texture" not in vowel_speech.analysis
    assert updated.stage_state["korean-script"]["non_speech_texture"] == 0


def test_korean_script_marks_repeated_texture_with_incidental_numbers(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    segment = _translated_segment(
        "seg_texture_number",
        "グーッ、グーッ、3回。",
        "으으음, 으으음, 세 번.",
        "transcribed",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True)

    texture = load_manifest(tmp_path).segments[0]
    assert texture.status == "non_speech_texture"
    assert texture.keep_original_texture is True
    assert texture.script is None
    assert texture.analysis["korean_script_non_speech_texture"] == {
        "action": "keep_original_texture",
        "reason": "repeated_vowel_or_onomatopoeia_after_translation",
        "source_text": "グーッ、グーッ、3回。",
        "tts_text": "으으음, 으으음, 세 번.",
    }


def test_korean_script_marks_long_vowel_after_translation_as_texture(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    segment = _translated_segment(
        "seg_long_vowel",
        "ぐーっと。",
        "그으으으으으.....",
        "transcribed",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True)

    updated = load_manifest(tmp_path)
    texture = updated.segments[0]
    assert texture.status == "non_speech_texture"
    assert texture.keep_original_texture is True
    assert texture.script is None
    assert texture.analysis["korean_script_non_speech_texture"] == {
        "action": "keep_original_texture",
        "reason": "repeated_vowel_or_onomatopoeia_after_translation",
        "source_text": "ぐーっと。",
        "tts_text": "그으으으으으…",
    }
    assert updated.stage_state["korean-script"]["non_speech_texture"] == 1


def test_korean_script_keeps_korean_countdown_words_scriptable(tmp_path: Path) -> None:
    save_project_config(ProjectConfig(project_name="targeted-script"), tmp_path / "pipeline.yaml")
    segment = _translated_segment(
        "seg_countdown",
        "一、二、三。",
        "하나, 둘, 셋.",
        "transcribed",
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_path, manifest)

    korean_script_step(tmp_path, confirm_rights=True)

    countdown = load_manifest(tmp_path).segments[0]
    assert countdown.status == "scripted"
    assert countdown.script is not None
    assert countdown.script.tts_text == "하나. 둘. 셋."
    assert "korean_script_non_speech_texture" not in countdown.analysis
