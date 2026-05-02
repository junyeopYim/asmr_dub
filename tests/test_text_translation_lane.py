from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from asmr_dub_pipeline import orchestrator
from asmr_dub_pipeline.asr.base import ASRChunk, map_chunks_to_segments
from asmr_dub_pipeline.audio.features import duration_sec, write_audio
from asmr_dub_pipeline.cli import app
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gemma.text_translate import (
    LlamaServerTranslationClient,
    parse_translation_response,
)
from asmr_dub_pipeline.pipeline import steps as pipeline_steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import (
    extract_step,
    korean_script_step,
    prepare_source_voice_refs_step,
    segment_step,
    transcribe_step,
    translate_ko_step,
)
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    KoreanTranslation,
    PipelineManifest,
    ProjectConfig,
    QCMetadata,
    Segment,
    SourceScript,
    TTSCandidate,
    TTSMetadata,
)
from asmr_dub_pipeline.script.korean_colloquial import (
    COLLOQUIAL_REWRITE_NOTE,
    colloquialize_korean_text,
)


def sample_segment(
    segment_id: str = "seg_0001",
    *,
    start: float = 0.0,
    end: float = 1.0,
) -> Segment:
    return Segment(
        id=segment_id,
        start=start,
        end=end,
        duration=end - start,
        audio_for_gemma="gemma.wav",
        audio_for_mix="mix.wav",
    )


def test_asr_chunks_map_to_segments_by_overlap() -> None:
    segments = [
        sample_segment("seg_0001", start=0.0, end=1.0),
        sample_segment("seg_0002", start=1.0, end=2.0),
        sample_segment("seg_0003", start=2.0, end=3.0),
    ]
    chunks = [
        ASRChunk(start=0.0, end=0.5, text="おはよう", language="ja", confidence=0.8),
        ASRChunk(start=0.5, end=1.0, text="ございます", language="ja", confidence=0.4),
        ASRChunk(start=1.2, end=1.8, text="耳元です", language="ja", confidence=0.9),
    ]

    mapped = map_chunks_to_segments(segments, chunks, backend="mock")

    assert mapped["seg_0001"] is not None
    assert mapped["seg_0001"].text == "おはよう ございます"
    assert mapped["seg_0001"].confidence == pytest.approx(0.6)
    assert mapped["seg_0001"].language == "ja"
    assert mapped["seg_0002"] is not None
    assert mapped["seg_0002"].text == "耳元です"
    assert mapped["seg_0003"] is None


def test_asr_chunks_do_not_duplicate_long_chunk_across_micro_segments() -> None:
    segments = [
        sample_segment("seg_0001", start=0.0, end=1.0),
        sample_segment("seg_0002", start=1.0, end=2.0),
        sample_segment("seg_0003", start=2.0, end=3.0),
    ]
    chunks = [
        ASRChunk(
            start=0.2,
            end=2.8,
            text="初めて催眠音声に出演させていただきました",
            language="ja",
            confidence=0.7,
        )
    ]

    mapped = map_chunks_to_segments(segments, chunks, backend="mock")

    assert mapped["seg_0001"] is None
    assert mapped["seg_0002"] is not None
    assert mapped["seg_0002"].text == "初めて催眠音声に出演させていただきました"
    assert mapped["seg_0003"] is None


def test_asr_resegment_groups_short_adjacent_chunks_for_tts(tmp_project_dir: Path) -> None:
    segments = pipeline_steps._segments_from_asr_chunks(
        [
            ASRChunk(start=0.0, end=0.3, text="お疲れ様でした", language="ja", confidence=0.8),
            ASRChunk(start=0.5, end=1.1, text="初めまして", language="ja", confidence=0.6),
            ASRChunk(start=2.0, end=3.1, text="ゆっくり聞いてください", language="ja", confidence=0.9),
        ],
        project_dir=tmp_project_dir,
        backend="faster_whisper",
        fallback_language="ja",
        audio_duration_sec=4.0,
        min_segment_sec=0.8,
        merge_gap_sec=0.45,
    )

    assert [segment.duration for segment in segments] == [1.1, 1.1]
    assert segments[0].source_script is not None
    assert segments[0].source_script.text == "お疲れ様でした 初めまして"
    assert segments[1].source_script is not None
    assert segments[1].source_script.text == "ゆっくり聞いてください"


def test_source_script_and_korean_translation_schema_round_trip() -> None:
    segment = sample_segment()
    segment.source_script = SourceScript(
        text="少し近づきますね",
        language="ja",
        confidence=0.91,
        backend="faster_whisper",
        start=segment.start,
        end=segment.end,
    )
    segment.translation_ko = KoreanTranslation(
        ko_literal="조금 가까이 갈게요.",
        ko_natural="조금 더 가까이 갈게요.",
        notes=["gentle ASMR tone"],
        confidence=0.87,
        model="gemma4",
        batch_id="batch_0001",
    )

    restored = Segment.model_validate(segment.model_dump(mode="json"))

    assert restored.source_script is not None
    assert restored.source_script.text == "少し近づきますね"
    assert restored.translation_ko is not None
    assert restored.translation_ko.ko_natural == "조금 더 가까이 갈게요."


def test_project_config_defaults_translate_ko_uses_single_server_slots() -> None:
    assert ProjectConfig().gemma_text_batch_size == 12
    assert ProjectConfig().gemma_text_context_radius == 4
    assert ProjectConfig().gemma_text_two_pass is True
    assert ProjectConfig().gemma_text_concurrency == 4
    assert ProjectConfig().gsv_concurrency == 3
    assert ProjectConfig().source_language == "ja"
    assert ProjectConfig().target_language == "ko"
    assert ProjectConfig(target_language="kr").target_language == "ko"
    assert ProjectConfig().asr_resegment_from_chunks is True
    assert ProjectConfig().asr_resegment_min_sec == pytest.approx(0.8)
    assert ProjectConfig().gsv_trim_edge_silence is True
    assert ProjectConfig().gsv_ref_min_sec == pytest.approx(3.0)
    assert ProjectConfig().gsv_ref_max_sec == pytest.approx(10.0)
    assert ProjectConfig().gsv_tts_min_speed_factor == pytest.approx(0.85)
    assert ProjectConfig().gsv_tts_max_speed_factor == pytest.approx(1.12)
    assert ProjectConfig().gsv_top_k == 15
    assert ProjectConfig().gsv_top_p == pytest.approx(1.0)
    assert ProjectConfig().gsv_temperature == pytest.approx(1.0)
    assert ProjectConfig().gsv_text_split_method == "cut5"
    assert ProjectConfig().gsv_parallel_infer is True
    assert ProjectConfig().gsv_repetition_penalty == pytest.approx(1.35)
    assert ProjectConfig().gsv_sample_steps == 32
    assert ProjectConfig().gsv_super_sampling is False
    assert ProjectConfig().gsv_overlap_length == 2
    assert ProjectConfig().gsv_min_chunk_length == 16
    assert ProjectConfig().gsv_fragment_interval == pytest.approx(0.3)


def test_source_voice_ref_selection_extends_short_candidate_to_duration_window(
    tmp_project_dir: Path,
) -> None:
    manifest = PipelineManifest(
        project_config=ProjectConfig(project_name=tmp_project_dir.name),
        segments=[
            Segment(
                id="seg_short",
                start=0.0,
                end=1.0,
                duration=1.0,
                audio_for_gemma="work/segments/audio/seg_short_gemma.wav",
                audio_for_mix="work/segments/audio/seg_short_mix.wav",
                source_script=SourceScript(
                    text="短すぎます。",
                    language="ja",
                    backend="mock",
                    start=0.0,
                    end=1.0,
                ),
            ),
            Segment(
                id="seg_valid",
                start=1.0,
                end=5.0,
                duration=4.0,
                audio_for_gemma="work/segments/audio/seg_valid_gemma.wav",
                audio_for_mix="work/segments/audio/seg_valid_mix.wav",
                source_script=SourceScript(
                    text="参照音声に使える長さです。",
                    language="ja",
                    backend="mock",
                    start=1.0,
                    end=5.0,
                ),
            ),
        ],
    )

    selected = pipeline_steps._select_voice_ref_spans(tmp_project_dir, manifest, manifest.project_config)

    assert [[segment.id for segment in span.segments] for span in selected] == [
        ["seg_short", "seg_valid"]
    ]
    assert selected[0].duration == pytest.approx(5.0)


def test_prepare_source_voice_refs_writes_combined_short_reference_span(
    tmp_project_dir: Path,
) -> None:
    save_project_config(ProjectConfig(project_name=tmp_project_dir.name), tmp_project_dir / "pipeline.yaml")
    sample_rate = 48_000
    segments = []
    for segment_id, start, end, text, frequency in [
        ("seg_short", 0.0, 1.0, "短いです。", 440.0),
        ("seg_next", 1.0, 3.2, "続きです。", 660.0),
    ]:
        t = np.arange(int(sample_rate * (end - start)), dtype=np.float32) / sample_rate
        tone = 0.1 * np.sin(2 * np.pi * frequency * t)
        audio = np.stack([tone, tone * 0.8], axis=1)
        mix_path = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        gemma_path = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_gemma.wav"
        write_audio(mix_path, audio, sample_rate)
        write_audio(gemma_path, audio[:, :1], sample_rate)
        segments.append(
            Segment(
                id=segment_id,
                start=start,
                end=end,
                duration=end - start,
                audio_for_gemma=str(gemma_path),
                audio_for_mix=str(mix_path),
                source_script=SourceScript(
                    text=text,
                    language="ja",
                    backend="mock",
                    start=start,
                    end=end,
                ),
            )
        )
    save_manifest(
        tmp_project_dir,
        PipelineManifest(
            project_config=ProjectConfig(project_name=tmp_project_dir.name),
            segments=segments,
        ),
    )

    manifest = prepare_source_voice_refs_step(tmp_project_dir, confirm_rights=True)

    refs = json.loads((tmp_project_dir / "refs" / "refs.json").read_text("utf-8"))
    ref_path = tmp_project_dir / refs["whisper_close"]["ref_audio_path"]
    assert duration_sec(ref_path) == pytest.approx(3.2)
    assert refs["whisper_close"]["prompt_text"] == "短いです。 続きです。"
    ref_qc = json.loads(Path(manifest.artifacts["source_voice_ref_qc"]).read_text("utf-8"))
    assert ref_qc["refs"][0]["selected_segment_ids"] == ["seg_short", "seg_next"]


def _force_single_translation_lane(project_dir: Path) -> None:
    save_project_config(
        ProjectConfig(
            project_name=project_dir.name,
            gemma_text_batch_size=40,
            gemma_text_concurrency=1,
        ),
        project_dir / "pipeline.yaml",
    )


def test_translate_ko_passes_neighbor_context_to_context_aware_clients(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_batch_size=1,
            gemma_text_concurrency=1,
            gemma_text_context_radius=1,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    for index in range(2, 4):
        start = base.end + (index - 2) * base.duration
        manifest.segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text=f"前後の台詞です {index}",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=start,
                    end=start + base.duration,
                ),
            )
        )
    save_manifest(tmp_project_dir, manifest)
    contexts: list[tuple[str, list[str], list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
            context_segments: list[Segment] | None = None,
        ) -> dict[str, KoreanTranslation]:
            contexts.append(
                (
                    batch_id,
                    [segment.id for segment in segments],
                    [segment.id for segment in (context_segments or [])],
                )
            )
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    assert contexts == [
        ("batch_0001", ["seg_0001"], ["seg_0001", "seg_0002"]),
        ("batch_0002", ["seg_0002"], ["seg_0001", "seg_0002", "seg_0003"]),
        ("batch_0003", ["seg_0003"], ["seg_0002", "seg_0003"]),
    ]
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["translate-ko"]["context_radius"] == 1
    assert manifest.stage_state["translate-ko"]["two_pass"] is True


def test_translate_ko_force_retranslate_ignores_resumed_translations(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    calls: list[str] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append(batch_id)
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {len(calls)}",
                    ko_natural=f"자연 {len(calls)}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")
    first_manifest = load_manifest(tmp_project_dir)
    first_translation = first_manifest.segments[0].translation_ko
    assert first_translation is not None
    assert first_translation.ko_natural == "자연 1"

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")
    assert calls == ["batch_0001"]

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock", force_retranslate=True)

    manifest = load_manifest(tmp_project_dir)
    translation = manifest.segments[0].translation_ko
    assert translation is not None
    assert translation.ko_natural == "자연 2"
    assert manifest.stage_state["translate-ko"]["force_retranslate"] is True


def test_korean_draft_mix_allows_timing_only_qc_regeneration() -> None:
    segment = sample_segment()
    segment.status = "needs_regeneration"
    segment.script = JapaneseScript(ja_text="안녕하세요.", tts_text="안녕하세요.", tts_language="ko")
    segment.tts = TTSMetadata(selected_candidate_path="work/tts/seg_0001_final.wav")
    segment.qc = QCMetadata(
        recommendation="regenerate",
        status="needs_regeneration",
        issues=["duration_ratio_out_of_range"],
    )

    assert pipeline_steps._include_segment_in_mix(segment, allow_korean_timing_draft=True)
    segment.qc.issues.append("clipping_detected")
    assert not pipeline_steps._include_segment_in_mix(segment, allow_korean_timing_draft=True)


def test_mix_blocks_selected_candidate_with_failed_duration_gate() -> None:
    segment = sample_segment()
    segment.status = "ok"
    segment.script = JapaneseScript(ja_text="こんにちは", tts_text="안녕하세요.", tts_language="ko")
    segment.tts = TTSMetadata(
        selected_candidate_path="work/tts/seg_0001_final.wav",
        candidates=[
            TTSCandidate(
                candidate_index=0,
                seed=1,
                output_path="work/tts/seg_0001_final.wav",
                duration_sec=0.2,
                selected=True,
                duration_ratio=0.2,
                duration_gate="too_short",
                acceptable_for_mix=False,
            )
        ],
    )
    segment.qc = QCMetadata(recommendation="pass", status="ok")

    assert not pipeline_steps._include_segment_in_mix(segment, allow_korean_timing_draft=False)


def test_llama_server_translation_client_repairs_invalid_json() -> None:
    requests: list[dict[str, object]] = []
    repaired = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "직역",
            "ko_natural": "자연",
            "notes": [],
            "confidence": 0.9,
            "model": "gemma4",
            "batch_id": "batch_0001",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = "not json" if len(requests) == 1 else json.dumps(repaired, ensure_ascii=False)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="こんにちは",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "자연"
    assert len(requests) == 3
    second_messages = requests[1]["messages"]
    assert isinstance(second_messages, list)
    assert "Repair the previous response" in second_messages[0]["content"]


def test_translation_parser_accepts_label_confidence_and_string_notes() -> None:
    parsed = parse_translation_response(
        json.dumps(
            [
                {
                    "segment_id": "seg_0001",
                    "ko_literal": "직역",
                    "ko_natural": "자연",
                    "notes": "tone preserved",
                    "confidence": "High",
                },
                {
                    "segment_id": "seg_0002",
                    "ko_literal": "직역2",
                    "ko_natural": "자연2",
                    "notes": "none",
                    "confidence": "87%",
                },
            ]
        ),
        batch_id="batch_0001",
        model="gemma4",
    )

    assert parsed["seg_0001"].confidence == pytest.approx(0.9)
    assert parsed["seg_0001"].notes == ["tone preserved"]
    assert parsed["seg_0002"].confidence == pytest.approx(0.87)
    assert parsed["seg_0002"].notes == []


def test_translation_parser_accepts_minimal_model_output() -> None:
    parsed = parse_translation_response(
        json.dumps([{"segment_id": "seg_0001", "ko_natural": "안녕하세요."}]),
        batch_id="batch_0001",
        model="gemma4",
    )

    translation = parsed["seg_0001"]
    assert translation.ko_natural == "안녕하세요."
    assert translation.ko_literal == "안녕하세요."
    assert translation.notes == []
    assert translation.confidence is None
    assert translation.model == "gemma4"
    assert translation.batch_id == "batch_0001"


def test_llama_server_translation_client_uses_literal_then_natural_pass() -> None:
    requests: list[dict[str, object]] = []
    responses = [
        [{"segment_id": "seg_0001", "ko_literal": "어서 오세요, 오빠."}],
        [{"segment_id": "seg_0001", "ko_natural": "어서 와요, 오빠."}],
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(responses[len(requests) - 1], ensure_ascii=False)}}
                ]
            },
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="いらっしゃいませお兄さん",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_literal == "어서 오세요, 오빠."
    assert translations["seg_0001"].ko_natural == "어서 와요, 오빠."
    assert len(requests) == 2
    first_prompt = requests[0]["messages"][0]["content"]
    second_prompt = requests[1]["messages"][0]["content"]
    assert "First pass" in first_prompt
    assert "Second pass" in second_prompt
    assert "ko_literal" in second_prompt


def test_llama_server_translation_client_repairs_low_quality_translation() -> None:
    requests: list[dict[str, object]] = []
    bad = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "だって",
            "ko_natural": "だって",
            "notes": [],
            "confidence": 0.98,
            "model": "gemma4",
            "batch_id": "batch_0001",
        }
    ]
    repaired = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "왜냐하면.",
            "ko_natural": "왜냐하면요.",
            "notes": [],
            "confidence": 0.98,
            "model": "gemma4",
            "batch_id": "batch_0001",
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        content = bad if len(requests) == 1 else repaired
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}]},
        )

    segment = sample_segment()
    segment.source_script = SourceScript(
        text="だって",
        language="ja",
        confidence=0.9,
        backend="mock",
        start=0.0,
        end=1.0,
    )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=1,
        n_predict=128,
    )

    translations = client.translate_batch([segment], "batch_0001")

    assert translations["seg_0001"].ko_natural == "왜냐하면요."
    assert len(requests) == 3
    second_prompt = requests[1]["messages"][0]["content"]
    assert "Original input" in second_prompt
    assert "だって" in second_prompt


def test_llama_server_translation_client_keeps_valid_items_from_mixed_batch() -> None:
    mixed = [
        {
            "segment_id": "seg_0001",
            "ko_literal": "안녕하세요.",
            "ko_natural": "안녕하세요.",
            "notes": [],
            "confidence": 0.95,
            "model": "gemma4",
            "batch_id": "batch_0001",
        },
        {
            "segment_id": "seg_0002",
            "ko_literal": "だって",
            "ko_natural": "だって",
            "notes": [],
            "confidence": 0.95,
            "model": "gemma4",
            "batch_id": "batch_0001",
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(mixed, ensure_ascii=False)}}]},
        )

    segments = [sample_segment("seg_0001"), sample_segment("seg_0002")]
    for segment, text in zip(segments, ["こんにちは", "だって"], strict=True):
        segment.source_script = SourceScript(
            text=text,
            language="ja",
            confidence=0.9,
            backend="mock",
            start=segment.start,
            end=segment.end,
        )
    client = LlamaServerTranslationClient(
        "http://gemma.local",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        model="gemma4",
        retries=0,
        n_predict=128,
    )

    translations = client.translate_batch(segments, "batch_0001")

    assert set(translations) == {"seg_0001"}
    assert translations["seg_0001"].ko_natural == "안녕하세요."


def test_korean_colloquializer_rewrites_stiff_polite_forms() -> None:
    text = "저는 괜찮습니다. 이것은 좋은 것입니다. 천천히 하겠습니다."

    assert colloquialize_korean_text(text) == "전 괜찮아요. 이건 좋은 거예요. 천천히 할게요."


def test_translate_ko_colloquializes_finished_translations(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            return {
                segment.id: KoreanTranslation(
                    ko_literal="저는 괜찮습니다.",
                    ko_natural="저는 괜찮습니다. 이것은 좋은 것입니다. 천천히 하겠습니다.",
                    notes=["model_note"],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "MockTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")

    manifest = load_manifest(tmp_project_dir)
    translation = manifest.segments[0].translation_ko
    assert translation is not None
    assert translation.ko_natural == "전 괜찮아요. 이건 좋은 거예요. 천천히 할게요."
    assert translation.notes == ["model_note", COLLOQUIAL_REWRITE_NOTE]
    assert manifest.stage_state["translate-ko"]["colloquialized"] == 1

    rows = [
        json.loads(line)
        for line in Path(manifest.artifacts["translation_bundles"]).read_text().splitlines()
    ]
    assert rows[0]["colloquialized"] is True
    assert rows[0]["translation_ko"]["ko_natural"] == translation.ko_natural


def test_transcribe_and_translate_mock_steps_write_artifacts(
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)

    transcribe_step(tmp_project_dir, asr_backend="mock")
    translate_ko_step(tmp_project_dir, gemma_text_backend="mock")
    korean_script_step(tmp_project_dir, confirm_rights=True)
    save_project_config(
        ProjectConfig(project_name=tmp_project_dir.name, gsv_ref_min_sec=0.1),
        tmp_project_dir / "pipeline.yaml",
    )
    prepare_source_voice_refs_step(tmp_project_dir)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["transcribe"]["status"] == "completed"
    assert manifest.stage_state["translate-ko"]["status"] == "completed"
    assert manifest.stage_state["korean-script"]["status"] == "completed"
    assert manifest.stage_state["prepare-refs"]["status"] == "completed"
    assert manifest.segments
    assert manifest.segments[0].source_script is not None
    assert manifest.segments[0].translation_ko is not None
    assert manifest.segments[0].script is not None
    assert manifest.segments[0].script.tts_language == "ko"
    assert manifest.segments[0].script.source_language == "ja"
    assert manifest.segments[0].script.target_language == "ko"
    assert manifest.segments[0].script.ja_text.startswith("mock source script")
    assert manifest.segments[0].script.tts_text.startswith("자연 번역,")
    assert Path(manifest.artifacts["source_segments"]).exists()
    assert Path(manifest.artifacts["segments_transcribed"]).exists()
    assert Path(manifest.artifacts["translation_bundles"]).exists()
    assert Path(manifest.artifacts["translation_summary"]).exists()
    assert Path(manifest.artifacts["segments_ko_script"]).exists()
    refs = json.loads((tmp_project_dir / "refs" / "refs.json").read_text("utf-8"))
    assert refs["whisper_close"]["prompt_text"].startswith("mock source script")
    assert refs["whisper_close"]["prompt_lang"] == "ja"
    assert refs["whisper_close"]["target_language"] == "ko"
    assert (tmp_project_dir / refs["whisper_close"]["ref_audio_path"]).exists()
    ref_qc = json.loads(Path(manifest.artifacts["source_voice_ref_qc"]).read_text("utf-8"))
    assert ref_qc["refs"][0]["source_language"] == "ja"
    assert ref_qc["refs"][0]["target_language"] == "ko"
    summary = json.loads(Path(manifest.artifacts["translation_summary"]).read_text("utf-8"))
    assert summary["backend"] == "mock"
    assert summary["translated"] == len(manifest.segments)


def test_korean_script_blocks_japanese_fallback_text(tmp_project_dir: Path) -> None:
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.0,
        duration=1.0,
        audio_for_gemma="work/segments/audio/seg_0001_gemma.wav",
        audio_for_mix="work/segments/audio/seg_0001_mix.wav",
        source_script=SourceScript(
            text="少し近づきますね",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.0,
        ),
        translation_ko=KoreanTranslation(
            ko_literal="少し近づきますね",
            ko_natural="少し近づきますね",
            model="mock",
            batch_id="batch_0001",
        ),
    )
    manifest = PipelineManifest(segments=[segment])
    manifest.stage_state["translate-ko"] = {"status": "completed"}
    save_manifest(tmp_project_dir, manifest)

    korean_script_step(tmp_project_dir, confirm_rights=True)

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].status == "needs_manual_review"
    assert "korean_tts_contains_kana" in manifest.segments[0].errors[-1]


def test_transcribe_and_translate_mock_cli(
    cli_runner,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)

    transcribe_result = cli_runner.invoke(
        app,
        ["transcribe", "-p", str(tmp_project_dir), "--asr-backend", "mock"],
    )
    translate_result = cli_runner.invoke(
        app,
        ["translate-ko", "-p", str(tmp_project_dir), "--gemma-text-backend", "mock"],
    )

    assert transcribe_result.exit_code == 0, transcribe_result.output
    assert translate_result.exit_code == 0, translate_result.output
    assert "- 원문:" in translate_result.output
    assert "- 번역문:" in translate_result.output
    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].source_script is not None
    assert manifest.segments[0].translation_ko is not None


def test_translate_ko_requires_transcribe_stage(tiny_wav_path: Path, tmp_project_dir: Path) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)

    with pytest.raises(ValueError, match="transcribe"):
        translate_ko_step(tmp_project_dir, gemma_text_backend="mock")


def test_translate_ko_retries_missing_model_items(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    if len(manifest.segments) == 1:
        base = manifest.segments[0]
        manifest.segments.append(
            Segment(
                id="seg_0002",
                start=base.end,
                end=base.end + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text="追加の台詞です",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=base.end,
                    end=base.end + base.duration,
                ),
            )
        )
        save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, [segment.id for segment in segments]))
            selected = segments if len(segments) == 1 else segments[:1]
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in selected
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert all(segment.translation_ko is not None for segment in manifest.segments)
    assert any(batch_id.startswith("batch_0001_single_") for batch_id, _ in calls)


def test_translate_ko_falls_back_to_single_segments_when_batch_fails(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    manifest.segments.append(
        Segment(
            id="seg_0002",
            start=base.end,
            end=base.end + base.duration,
            duration=base.duration,
            audio_for_gemma=base.audio_for_gemma,
            audio_for_mix=base.audio_for_mix,
            source_script=SourceScript(
                text="追加の台詞です",
                language="ja",
                confidence=0.99,
                backend="mock",
                start=base.end,
                end=base.end + base.duration,
            ),
        )
    )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, [segment.id for segment in segments]))
            if len(segments) > 1:
                raise RuntimeError("bad batch json")
            segment = segments[0]
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert all(segment.translation_ko is not None for segment in manifest.segments)
    assert all(
        not any(error.startswith("Korean translation batch failed") for error in segment.errors)
        for segment in manifest.segments
    )
    assert calls[0] == ("batch_0001", ["seg_0001", "seg_0002"])
    assert {tuple(ids) for _, ids in calls[1:]} == {("seg_0001",), ("seg_0002",)}


def test_translate_ko_fails_fast_when_llama_server_connection_is_refused(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    state = {"stopped": False}

    class FakeServer:
        started = True
        reused_existing = False
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            state["stopped"] = True

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            raise RuntimeError("Gemma text translation failed: [Errno 111] Connection refused")

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    with pytest.raises(RuntimeError, match="Connection refused"):
        translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert manifest.segments[0].translation_ko is None
    assert state["stopped"] is True
    assert not (tmp_project_dir / "work" / "translate_ko" / "translation_bundles.jsonl").exists()


def test_translate_ko_splits_failed_batches_before_single_retries(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    _force_single_translation_lane(tmp_project_dir)
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    for index in range(2, 5):
        start = base.end + (index - 2) * base.duration
        manifest.segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text=f"追加の台詞です {index}",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=start,
                    end=start + base.duration,
                ),
            )
        )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, [segment.id for segment in segments]))
            if len(segments) > 2:
                raise RuntimeError("batch too large")
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", lambda **kwargs: FakeServer())
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    manifest = load_manifest(tmp_project_dir)
    assert all(segment.translation_ko is not None for segment in manifest.segments)
    assert calls[0] == ("batch_0001", ["seg_0001", "seg_0002", "seg_0003", "seg_0004"])
    assert [len(ids) for _, ids in calls[1:]] == [2, 2]


def test_translate_ko_uses_single_llama_server_slot_workers(
    monkeypatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    extract_step(tiny_wav_path, tmp_project_dir, confirm_rights=True)
    segment_step(tmp_project_dir)
    transcribe_step(tmp_project_dir, asr_backend="mock")
    save_project_config(
        ProjectConfig(
            project_name=tmp_project_dir.name,
            gemma_text_batch_size=1,
            gemma_text_concurrency=2,
        ),
        tmp_project_dir / "pipeline.yaml",
    )
    manifest = load_manifest(tmp_project_dir)
    base = manifest.segments[0]
    for index in range(2, 5):
        start = base.end + (index - 2) * base.duration
        manifest.segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=start,
                end=start + base.duration,
                duration=base.duration,
                audio_for_gemma=base.audio_for_gemma,
                audio_for_mix=base.audio_for_mix,
                source_script=SourceScript(
                    text=f"追加の台詞です {index}",
                    language="ja",
                    confidence=0.99,
                    backend="mock",
                    start=start,
                    end=start + base.duration,
                ),
            )
        )
    save_manifest(tmp_project_dir, manifest)
    server_base_urls: list[str] = []
    calls: list[tuple[str, str, list[str]]] = []

    class FakeServer:
        started = False
        reused_existing = True
        log_path = None

        def __init__(self, **kwargs: object) -> None:
            self.base_url = str(kwargs["base_url"])
            self.log_path = kwargs.get("log_path")
            server_base_urls.append(self.base_url)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class FakeClient:
        def __init__(self, base_url: str, *args: object, **kwargs: object) -> None:
            self.base_url = base_url

        def translate_batch(
            self,
            segments: list[Segment],
            batch_id: str,
        ) -> dict[str, KoreanTranslation]:
            calls.append((batch_id, self.base_url, [segment.id for segment in segments]))
            return {
                segment.id: KoreanTranslation(
                    ko_literal=f"직역 {segment.id}",
                    ko_natural=f"자연 {segment.id}",
                    notes=[],
                    confidence=0.9,
                    model="fake",
                    batch_id=batch_id,
                )
                for segment in segments
            }

    monkeypatch.setattr(pipeline_steps, "ManagedGemmaTextServer", FakeServer)
    monkeypatch.setattr(pipeline_steps, "LlamaServerTranslationClient", FakeClient)

    translate_ko_step(tmp_project_dir, gemma_text_backend="llama_server")

    assert server_base_urls == ["http://127.0.0.1:8080"]
    assert {base_url for _, base_url, _ in calls} == {"http://127.0.0.1:8080"}
    calls_by_batch = {batch_id: ids for batch_id, _, ids in calls}
    assert calls_by_batch == {
        "batch_0001": ["seg_0001"],
        "batch_0002": ["seg_0002"],
        "batch_0003": ["seg_0003"],
        "batch_0004": ["seg_0004"],
    }
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["translate-ko"]["concurrency"] == 2
    assert manifest.stage_state["translate-ko"]["server"]["server_count"] == 1
    assert manifest.stage_state["translate-ko"]["server"]["mode"] == "single_server_slots"


def test_target_language_ko_full_uses_text_only_korean_lane(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def step(name: str):
        def inner(*args: object, **kwargs: object) -> PipelineManifest:
            calls.append((name, args, kwargs))
            return PipelineManifest()

        return inner

    monkeypatch.setattr(orchestrator, "init_project", step("init"))
    monkeypatch.setattr(orchestrator, "extract_step", step("extract"))
    monkeypatch.setattr(orchestrator, "source_separation_step", step("source-separation"))
    monkeypatch.setattr(orchestrator, "segment_step", step("segment"))
    monkeypatch.setattr(orchestrator, "analyze_step", step("analyze"))
    monkeypatch.setattr(orchestrator, "script_step", step("script"))
    monkeypatch.setattr(orchestrator, "transcribe_step", step("transcribe"))
    monkeypatch.setattr(orchestrator, "translate_ko_step", step("translate-ko"))
    monkeypatch.setattr(orchestrator, "korean_script_step", step("korean-script"))
    monkeypatch.setattr(orchestrator, "prepare_source_voice_refs_step", step("prepare-refs"))
    monkeypatch.setattr(orchestrator, "gsv_few_shot_step", step("gsv-few-shot"))
    monkeypatch.setattr(orchestrator, "synth_step", step("synth"))
    monkeypatch.setattr(orchestrator, "rvc_train_step", step("train-rvc"))
    monkeypatch.setattr(orchestrator, "rvc_step", step("rvc"))
    monkeypatch.setattr(orchestrator, "qc_step", step("qc"))
    monkeypatch.setattr(orchestrator, "mix_step", step("mix"))
    monkeypatch.setattr(orchestrator, "export_step", step("export"))
    monkeypatch.setattr(orchestrator, "validate_rvc_config", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator, "validate_rvc_training_config", lambda *args, **kwargs: None)

    orchestrator.run_pipeline(
        tmp_path / "input.wav",
        tmp_path / "project",
        confirm_rights=True,
        mock=False,
        gemma_backend="hf",
        target_language="kr",
        asr_backend="qwen_asr",
    )

    assert [name for name, _, _ in calls] == [
        "init",
        "extract",
        "source-separation",
        "segment",
        "transcribe",
        "translate-ko",
        "korean-script",
        "prepare-refs",
        "gsv-few-shot",
        "synth",
        "train-rvc",
        "rvc",
        "qc",
        "mix",
        "export",
    ]
    assert calls[4][2]["asr_backend"] == "qwen_asr"
    assert calls[5][1][1] == "llama_server"
    assert calls[9][2]["mock"] is False
    assert calls[10][2]["mock"] is False
    assert calls[11][2]["mock"] is False
    assert calls[12][1][1] == "mock"
