from __future__ import annotations

from pathlib import Path

import pytest

from asmr_dub_pipeline.asr.base import ASRChunk, ASRWord
from asmr_dub_pipeline.audio.features import duration_sec
from asmr_dub_pipeline.pipeline.stages import common as pipeline_common
from asmr_dub_pipeline.schemas import PipelineManifest, ProjectConfig, Segment, SourceScript


def _source_script(text: str, *, end: float = 10.0) -> SourceScript:
    return SourceScript(
        text=text,
        language="ja",
        confidence=0.98,
        backend="faster_whisper",
        start=0.0,
        end=end,
    )


def _segment(
    segment_id: str,
    text: str,
    *,
    status: str = "transcribed",
    errors: list[str] | None = None,
    start: float = 0.0,
    end: float = 5.0,
) -> Segment:
    return Segment(
        id=segment_id,
        start=start,
        end=end,
        duration=end - start,
        audio_for_gemma=f"{segment_id}_gemma.wav",
        audio_for_mix=f"{segment_id}_mix.wav",
        status=status,
        errors=errors or [],
        source_script=_source_script(text, end=end),
    )


def test_sparse_numeric_asr_chunk_is_sent_to_llama_audio_review(
    monkeypatch: pytest.MonkeyPatch,
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_review_enabled=True,
        asr_review_backend="llama_server_audio",
        asr_review_audio_padding_sec=0.05,
        asr_review_generate_candidates=False,
        gemma_text_server_auto_start=False,
    )
    chunks = [
        ASRChunk(
            start=0.0,
            end=13.5,
            text="5 4 3",
            language="ja",
            confidence=0.93,
        )
    ]
    review_calls: list[tuple[str, Path, list[dict[str, object]]]] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def review_asr_candidates_with_audio(
            self,
            items: list[dict[str, object]],
            batch_id: str,
            audio_path: Path,
        ) -> dict[str, dict[str, object]]:
            assert audio_path.exists()
            review_calls.append((batch_id, audio_path, items))
            return {
                "chunk_0001": {
                    "chunk_id": "chunk_0001",
                    "heard_text": "5 4 3",
                    "decision": "keep",
                    "selected_candidate_id": "original",
                    "confidence": 0.96,
                    "reason": "audio supports the sparse countdown.",
                    "risk_terms": [],
                }
            }

    monkeypatch.setattr(pipeline_common, "LlamaServerTranslationClient", FakeClient)

    reviewed, summary = pipeline_common._review_asr_chunks_with_model(
        chunks,
        backend=object(),
        project_dir=tmp_project_dir,
        review_audio_path=tiny_wav_path,
        audio_duration_sec=duration_sec(tiny_wav_path),
        cfg=cfg,
    )

    assert review_calls
    assert summary["attempted"] == 1
    assert summary["reviewed"] == 1
    assert summary["audio_input"]["enabled"] is True
    assert summary["audio_input"]["created"] == 1
    assert summary["items"][0]["suspicious_patterns"] == ["asr_sparse_text_density"]
    assert reviewed[0].text == "5 4 3"


def test_asr_review_adds_qwen_fallback_candidate_to_reduce_manual_review(
    tiny_wav_path: Path,
    tmp_project_dir: Path,
) -> None:
    cfg = ProjectConfig(
        project_name=tmp_project_dir.name,
        asr_review_enabled=True,
        asr_review_backend="mock",
        asr_review_generate_candidates=True,
        asr_review_suspicious_text_patterns=["誤認語"],
        asr_review_candidate_padding_sec=[0.1],
    )
    chunks = [
        ASRChunk(
            start=0.0,
            end=1.0,
            text="君は全身を誤認語に変えられ",
            language="ja",
            confidence=0.96,
        )
    ]

    class EmptyLocalBackend:
        name = "faster_whisper"

        def transcribe_with_options(
            self,
            audio_path: Path,
            segments: list[Segment],
            **kwargs: object,
        ) -> list[ASRChunk]:
            _ = audio_path, segments, kwargs
            return []

    class QwenFallbackBackend:
        name = "qwen_asr"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path: Path, segments: list[Segment]) -> list[ASRChunk]:
            assert audio_path.exists()
            _ = segments
            self.calls += 1
            return [
                ASRChunk(
                    start=0.0,
                    end=1.0,
                    text="君は全身を正しい原文に変えられ",
                    language="ja",
                    confidence=None,
                )
            ]

    qwen = QwenFallbackBackend()

    reviewed, summary = pipeline_common._review_asr_chunks_with_model(
        chunks,
        backend=EmptyLocalBackend(),
        project_dir=tmp_project_dir,
        review_audio_path=tiny_wav_path,
        audio_duration_sec=duration_sec(tiny_wav_path),
        cfg=cfg,
        qwen_fallback_backend=qwen,
    )

    assert qwen.calls == 1
    assert reviewed[0].text == "君は全身を正しい原文に変えられ"
    assert summary["generated_qwen_candidates"] == 1
    assert summary["replaced"] == 1
    assert summary["manual_review"] == 0
    assert summary["items"][0]["selected_candidate_id"] == "qwen_asr_pad_1"


def test_asr_review_treats_dominant_ngram_repetition_as_texture() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="チン" * 36,
        language="ja",
        backend="faster_whisper",
        start=1398.27,
        end=1406.277,
        confidence=0.98,
    )

    assert (
        pipeline_common._source_script_non_speech_texture_reason(source_script)
        == "asr_non_speech_texture"
    )
    assert pipeline_common._source_script_asr_review_reasons(source_script, cfg) == []


def test_asr_keep_original_long_sparse_fragment_is_texture_override() -> None:
    cfg = ProjectConfig(project_name="test-project")
    segment = Segment(
        id="seg_0177",
        start=1384.41,
        end=1393.24,
        duration=8.83,
        audio_for_gemma="seg_0177_gemma.wav",
        audio_for_mix="seg_0177_mix.wav",
        keep_original_texture=True,
    )
    source_script = SourceScript(
        text="れ",
        language="ja",
        backend="faster_whisper",
        start=1384.41,
        end=1393.24,
        confidence=0.89,
    )

    assert (
        pipeline_common._source_script_keep_original_texture_override_reason(
            segment,
            source_script,
            cfg,
        )
        == "asr_non_speech_texture"
    )


def test_asr_review_blocks_mixed_moan_with_speech_cue_before_tts() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="はぁはぁ気持ちいいはぁはぁ",
        language="ja",
        backend="faster_whisper",
        start=540.0,
        end=548.0,
        confidence=0.94,
    )

    assert pipeline_common._source_script_non_speech_texture_reason(source_script) is None
    assert pipeline_common._source_script_asr_review_reasons(source_script, cfg) == [
        "asr_mixed_texture_speech"
    ]


@pytest.mark.parametrize(
    ("source_text", "source_pattern", "candidate_text"),
    [
        ("私は俗に言うサキバスってやつね", "サキバス", "私は俗に言うサキュバスってやつね"),
        (
            "クリトリスとスペンスニューセンに押し当てられます",
            "スペンスニューセン",
            "クリトリスとスキーン腺に押し当てられます",
        ),
        ("この肉が男性気を刺激して", "男性気", "この肉が男性器を刺激して"),
        ("男性機も遠目にはそうでもなくても", "男性機", "男性器も遠目にはそうでもなくても"),
        ("快感に震える女性気", "女性気", "快感に震える女性器"),
        ("とってもはしたない女性機だから", "女性機", "とってもはしたない女性器だから"),
        ("作り物の断水器なのに", "断水器", "作り物の男性器なのに"),
        ("年下の女に愛婦されて", "愛婦", "年下の女に愛撫されて"),
        ("あなたの女性器は見栄えなく", "見栄えなく", "あなたの女性器は見境なく"),
        (
            "ポルチオがブルブル指揮されて膨らむ",
            "ブルブル指揮されて",
            "ポルチオがブルブル刺激されて膨らむ",
        ),
        ("サルグツアーを外されました", "サルグツアー", "猿ぐつわを外されました"),
        ("今から銃数え下してゼロになったら", "銃数え", "今から10数え下してゼロになったら"),
        ("お腹がなめ打ち無意識で突き上げて", "なめ打ち", "お腹が波打ち無意識で突き上げて"),
        ("清掃ぶってるのは興奮してくると", "清掃ぶって", "清楚ぶってるのは興奮してくると"),
    ],
)
def test_domain_asr_misrecognitions_generate_review_candidate(
    source_text: str,
    source_pattern: str,
    candidate_text: str,
) -> None:
    cfg = ProjectConfig(project_name="project")
    chunk = ASRChunk(
        start=0.0,
        end=5.0,
        text=source_text,
        language="ja",
        confidence=0.98,
    )

    item = pipeline_common._asr_review_item([chunk], 0, cfg=cfg)

    assert item is not None
    assert source_pattern in item["suspicious_patterns"]
    assert {"candidate_id": "domain_replacement", "text": candidate_text} in item["candidates"]


def test_contextual_akume_and_kaikan_variants_generate_review_candidates() -> None:
    cfg = ProjectConfig(project_name="project")
    chunks = [
        ASRChunk(
            start=0.0,
            end=5.0,
            text="ピストンするたびに悪夢でお出迎えしてください",
            language="ja",
            confidence=0.98,
        ),
        ASRChunk(
            start=5.0,
            end=10.0,
            text="喘ぎ声出さないと外観を発散できないよ",
            language="ja",
            confidence=0.98,
        ),
    ]

    akume_item = pipeline_common._asr_review_item(chunks, 0, cfg=cfg)
    kaikan_item = pipeline_common._asr_review_item(chunks, 1, cfg=cfg)

    assert akume_item is not None
    assert {
        "candidate_id": "domain_replacement",
        "text": "ピストンするたびにアクメでお出迎えしてください",
    } in akume_item["candidates"]
    assert kaikan_item is not None
    assert {
        "candidate_id": "domain_replacement",
        "text": "喘ぎ声出さないと快感を発散できないよ",
    } in kaikan_item["candidates"]


@pytest.mark.parametrize(
    "text",
    [
        "さあ死ぬほど怖い悪夢を体験しなさい",
        "その手のひらを回転させて",
        "性欲の強さを体感してもらうわ",
    ],
)
def test_legitimate_contexts_are_not_sent_to_asr_review(text: str) -> None:
    cfg = ProjectConfig(project_name="project")
    chunk = ASRChunk(
        start=0.0,
        end=5.0,
        text=text,
        language="ja",
        confidence=0.98,
    )

    assert pipeline_common._asr_review_item([chunk], 0, cfg=cfg) is None


def test_short_sparse_one_character_asr_is_manual_review() -> None:
    cfg = ProjectConfig(project_name="project")

    assert pipeline_common._source_script_asr_review_reasons(
        _source_script("子", end=10.0),
        cfg,
    ) == ["asr_sparse_text_density"]
    assert pipeline_common._source_script_asr_review_reasons(
        _source_script("吸って", end=10.0),
        cfg,
    ) == []


def test_asr_postprocess_review_report_splits_auto_candidate_manual_and_keep() -> None:
    cfg = ProjectConfig(project_name="project")
    manifest = PipelineManifest(
        project_config=cfg,
        segments=[
            _segment("seg_0001", "私は俗に言うサキュバスってやつね"),
            _segment(
                "seg_0002",
                "ピストンするたびに悪夢でお出迎えしてください",
                status="needs_manual_review",
                errors=["asr_suspicious_pattern:悪夢でお出迎え"],
            ),
            _segment(
                "seg_0003",
                "子",
                status="needs_manual_review",
                errors=["asr_sparse_text_density"],
                end=10.0,
            ),
            _segment("seg_0004", "その手のひらを回転させて"),
        ],
    )
    replacements_summary = {
        "items": [
            {
                "start": 0.0,
                "end": 5.0,
                "original_text": "私は俗に言うサキバスってやつね",
                "replaced_text": "私は俗に言うサキュバスってやつね",
                "hits": [{"source": "サキバス", "target": "サキュバス", "count": 1}],
            }
        ]
    }

    report = pipeline_common._build_asr_postprocess_review_report(
        manifest,
        cfg=cfg,
        replacements_summary=replacements_summary,
    )

    assert report["summary"] == {
        "segment_count": 4,
        "item_count": 3,
        "auto_replace": 1,
        "candidate_review": 1,
        "manual_review": 1,
        "keep": 1,
    }
    assert [(item["segment_id"], item["action"]) for item in report["items"]] == [
        ("seg_0001", "auto_replace"),
        ("seg_0002", "candidate_review"),
        ("seg_0003", "manual_review"),
    ]
    assert report["items"][0]["candidate_text"] == "私は俗に言うサキュバスってやつね"
    assert report["items"][1]["candidate_text"] == "ピストンするたびにアクメでお出迎えしてください"
    assert "candidate_text" not in report["items"][2]


def test_asr_postprocess_report_flags_candidate_replacement_even_without_prior_error() -> None:
    cfg = ProjectConfig(
        project_name="project",
        asr_review_candidate_replacements={"男性機": "男性器"},
    )
    manifest = PipelineManifest(
        project_config=cfg,
        segments=[_segment("seg_0001", "男性機も遠目にはそうでもなくても")],
    )

    report = pipeline_common._build_asr_postprocess_review_report(
        manifest,
        cfg=cfg,
        replacements_summary={"items": []},
    )

    assert report["summary"] == {
        "segment_count": 1,
        "item_count": 1,
        "auto_replace": 0,
        "candidate_review": 1,
        "manual_review": 0,
        "keep": 0,
    }
    assert report["items"][0]["candidate_text"] == "男性器も遠目にはそうでもなくても"


def test_asr_postprocess_report_includes_normalized_candidate_equivalence() -> None:
    cfg = ProjectConfig(
        project_name="project",
        asr_review_candidate_replacements={"外観という名": "快感という名"},
    )
    manifest = PipelineManifest(
        project_config=cfg,
        segments=[
            _segment(
                "seg_0001",
                "初めては無理だったけど 外観という名の首輪がはめられている",
            )
        ],
    )

    report = pipeline_common._build_asr_postprocess_review_report(
        manifest,
        cfg=cfg,
        replacements_summary={"items": []},
    )

    item = report["items"][0]
    assert item["candidate_text"] == "初めては無理だったけど 快感という名の首輪がはめられている"
    assert item["normalized_candidate_text"] == "初めては無理だったけど快感という名の首輪がはめられている"
    assert item["equivalent_to_final_source"] is True


def test_repair_chunk_split_uses_large_word_timestamp_gap() -> None:
    chunk = ASRChunk(
        start=100.0,
        end=107.0,
        text="満たされるんだ若い女の体に負けて",
        language="ja",
        confidence=0.92,
        words=[
            ASRWord(start=100.1, end=100.5, text="満たされる"),
            ASRWord(start=100.55, end=100.8, text="んだ"),
            ASRWord(start=102.1, end=102.4, text="若い"),
            ASRWord(start=102.45, end=102.8, text="女の"),
            ASRWord(start=102.85, end=103.2, text="体に"),
            ASRWord(start=103.25, end=103.7, text="負けて"),
        ],
    )

    split = pipeline_common._split_asr_chunks_for_repair(
        [chunk],
        audio_duration_sec=120.0,
        max_chunk_sec=14.0,
    )

    assert [item.text for item in split] == ["満たされるんだ", "若い女の体に負けて"]
    assert split[0].end == pytest.approx(100.8)
    assert split[1].start == pytest.approx(102.1)


def test_repair_candidate_vote_prefers_repeated_domain_equivalent_text() -> None:
    cfg = ProjectConfig(
        project_name="project",
        asr_review_candidate_replacements={"外観という名": "快感という名"},
    )
    preferred = [
        ASRChunk(
            start=0.0,
            end=6.0,
            text="初めては無理だったけど 外観という名の首輪がはめられている",
            language="ja",
            confidence=0.94,
        )
    ]
    same_after_replacement = [
        ASRChunk(
            start=0.0,
            end=6.0,
            text="初めては無理だったけど、外観という名の首輪がはめられている。",
            language="ja",
            confidence=0.92,
        )
    ]
    high_score_outlier = [
        ASRChunk(
            start=0.0,
            end=6.0,
            text="別の候補テキストが混ざっている",
            language="ja",
            confidence=0.99,
        )
    ]

    voted = pipeline_common._select_voted_asr_repair_candidate(
        [
            ("no_vad_clean", preferred, 0.1),
            ("vad_no_hotwords", same_after_replacement, 0.0),
            ("wide_no_vad_clean", high_score_outlier, 3.0),
        ],
        cfg=cfg,
    )

    assert voted is not None
    assert voted.candidate_id == "no_vad_clean"
    assert voted.vote_count == 2
    assert voted.normalized_text == "初めては無理だったけど快感という名の首輪がはめられている"


def test_repair_candidate_with_domain_replacement_available_is_accepted() -> None:
    cfg = ProjectConfig(
        project_name="project",
        asr_repair_suspicious_text_patterns=["外観という名"],
        asr_review_suspicious_text_patterns=["外観という名"],
        asr_review_candidate_replacements={"外観という名": "快感という名"},
    )
    original = ASRChunk(
        start=0.0,
        end=6.0,
        text="初めては無理だったけど 外観という名の首輪がはめられている",
        language="ja",
        confidence=0.96,
    )
    candidate = ASRChunk(
        start=0.0,
        end=6.0,
        text="初めては無理だったけど、外観という名の首輪がはめられている。",
        language="ja",
        confidence=0.95,
    )

    accepted, _score, reason = pipeline_common._asr_repair_candidate_score(
        original,
        [candidate],
        cfg=cfg,
        prompt_leaked=False,
        candidate_id="no_vad_clean",
    )

    assert accepted is True
    assert reason == "accepted"


def test_rejected_repair_reason_is_ignored_when_candidate_matches_domain_replaced_source() -> None:
    cfg = ProjectConfig(
        project_name="project",
        asr_review_candidate_replacements={"外観という名": "快感という名"},
    )
    source_script = SourceScript(
        text="初めては無理だったけど 快感という名の首輪がはめられている",
        language="ja",
        confidence=0.97,
        backend="faster_whisper",
        start=10.0,
        end=16.0,
    )
    repair_summary = {
        "items": [
            {
                "start": 10.0,
                "end": 16.0,
                "accepted": False,
                "original_text": "初めては無理だったけど 外観という名の首輪がはめられている",
                "candidate_text": "",
                "attempts": [
                    {
                        "candidate_text": "初めては無理だったけど、外観という名の首輪がはめられている。",
                        "reason": "candidate_review_blocked:asr_candidate_replacement_available",
                    }
                ],
            }
        ]
    }

    assert pipeline_common._source_script_rejected_repair_reasons(
        source_script,
        repair_summary,
        cfg=cfg,
    ) == []


def test_qwen_repair_candidate_is_rejected_when_domain_replacement_still_leaves_suspicion() -> None:
    cfg = ProjectConfig(
        project_name="project",
        asr_repair_suspicious_text_patterns=["悪夢し"],
        asr_review_candidate_replacements={"未補正語": "補正語"},
    )
    original = ASRChunk(
        start=0.0,
        end=4.0,
        text="悪夢していいですよ",
        language="ja",
        confidence=0.91,
    )
    candidate = ASRChunk(
        start=0.0,
        end=4.0,
        text="悪夢していいですよ 幸せな未補正語",
        language="ja",
        confidence=None,
    )

    accepted, score, reason = pipeline_common._asr_repair_candidate_score(
        original,
        [candidate],
        cfg=cfg,
        prompt_leaked=False,
        candidate_id="qwen_asr_fallback",
    )

    assert accepted is False
    assert score == -70.0
    assert reason == "qwen_fallback_still_suspicious"


def test_rejected_repair_prompt_suffix_requires_final_source_prompt_leak() -> None:
    cfg = ProjectConfig(project_name="project")
    source_script = SourceScript(
        text="若い女の体に負けて",
        language="ja",
        confidence=0.97,
        backend="faster_whisper",
        start=20.0,
        end=26.0,
    )
    repair_summary = {
        "items": [
            {
                "start": 20.0,
                "end": 26.0,
                "accepted": False,
                "original_text": "満たされるんだ 若い女の体に負けて",
                "candidate_text": "",
                "attempts": [
                    {
                        "candidate_text": "これはASR修復候補です 若い女の体に負けて",
                        "reason": "prompt_or_hallucination_leak",
                    },
                    {
                        "candidate_text": "満たされるんだ 若い女の体に負けて",
                        "reason": "not_confidently_better_than_original",
                    },
                ],
            }
        ]
    }

    assert pipeline_common._source_script_rejected_repair_reasons(
        source_script,
        repair_summary,
        cfg=cfg,
    ) == ["asr_repair_rejected"]


def test_asr_repair_rejection_is_suppressed_for_clean_segment_retry_source() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="でもそんな中にさらに追加で暗示を入れます。",
        language="ja",
        backend="faster_whisper:segment_retry:vad_no_hotwords",
        start=5657.453,
        end=5662.773,
        confidence=0.93,
    )

    assert pipeline_common._filter_asr_repair_review_reasons(
        source_script,
        cfg,
        review_reasons=[],
        repair_review_reasons=["asr_repair_rejected"],
    ) == []


def test_asr_repair_rejected_short_clean_fragment_becomes_warning() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="これ",
        language="ja",
        backend="faster_whisper",
        start=8735.08,
        end=8735.64,
        confidence=0.897,
    )

    assert pipeline_common._source_script_short_clean_fragment_warning_reason(
        source_script,
        cfg,
    ) == "asr_short_clean_fragment_auto_accepted"
    assert pipeline_common._filter_asr_repair_review_reasons(
        source_script,
        cfg,
        review_reasons=[],
        repair_review_reasons=["asr_repair_rejected"],
    ) == []
    assert pipeline_common._source_script_asr_warning_reasons(source_script, cfg) == [
        "asr_short_clean_fragment_auto_accepted"
    ]


def test_asr_repair_rejection_is_kept_for_numeric_segment_retry_source() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="3 2 1 10 0 9 心地よさが頭の中に溢れていく 7 埋め尽くしていく 6",
        language="ja",
        backend="faster_whisper:segment_retry:vad_no_hotwords",
        start=506.03,
        end=515.43,
        confidence=0.97,
    )

    assert pipeline_common._filter_asr_repair_review_reasons(
        source_script,
        cfg,
        review_reasons=[],
        repair_review_reasons=["asr_repair_rejected"],
    ) == ["asr_repair_rejected"]


def test_rejected_asr_repair_ignores_partial_candidate_equivalent_to_final_source() -> None:
    cfg = ProjectConfig(project_name="test-project")
    source_script = SourceScript(
        text="でもそんなあなたに さらに追加で暗示を入れます",
        language="ja",
        confidence=0.95,
        backend="faster_whisper",
        start=8343.146,
        end=8348.386,
    )
    repair_summary = {
        "items": [
            {
                "start": 8343.146,
                "end": 8344.866,
                "accepted": False,
                "reject_reason": "candidate_vote_required",
                "attempts": [
                    {
                        "candidate_id": "no_vad_clean",
                        "reason": "candidate_review_blocked:asr_suspicious_pattern:そんなお腹",
                        "candidate_text": "でもそんなお腹にさらに",
                    }
                ],
            }
        ]
    }

    assert pipeline_common._source_script_rejected_repair_reasons(
        source_script,
        repair_summary,
        cfg=cfg,
    ) == []
