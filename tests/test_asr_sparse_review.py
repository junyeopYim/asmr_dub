from __future__ import annotations

from pathlib import Path

import pytest

from asmr_dub_pipeline.asr.base import ASRChunk
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


def test_qwen_repair_candidate_is_rejected_when_candidate_replacement_is_available() -> None:
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
        text="アクメしていいですよ 幸せな未補正語",
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
    assert score == -75.0
    assert reason == "candidate_review_blocked:asr_candidate_replacement_available"
