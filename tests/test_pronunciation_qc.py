from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import write_tiny_wav

from asmr_dub_pipeline.asr.base import ASRChunk
from asmr_dub_pipeline.audio.features import write_audio
from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gpt_sovits.client import build_tts_request
from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import init_project, synth_step
from asmr_dub_pipeline.qc.pronunciation_qc import (
    evaluate_numeric_sequence_text,
    evaluate_pronunciation_text,
)
from asmr_dub_pipeline.schemas import (
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
)


def _write_tone_wav(path: Path, duration: float, sample_rate: int = 48_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    tone = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    write_audio(path, np.stack([tone, tone], axis=1), sample_rate)
    return path


def test_korean_pronunciation_qc_scores_hangul_coverage() -> None:
    passed = evaluate_pronunciation_text(
        "천천히 숨을 쉬어 주세요.",
        "천천히 숨을 쉬어 주세요",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )
    failed = evaluate_pronunciation_text(
        "천천히 숨을 쉬어 주세요.",
        "천천히",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )

    assert passed.gate == "pass"
    assert passed.coverage == pytest.approx(1.0)
    assert failed.gate == "fail"
    assert failed.coverage < 0.62


def test_numeric_sequence_qc_requires_ordered_contiguous_counting() -> None:
    passed = evaluate_numeric_sequence_text(
        "둘. 셋. 넷. 제자리. 다섯. 여섯. 일곱. 여덟.",
        "둘 셋 넷 제자리 다섯 여섯 일곱 여덟",
    )
    failed = evaluate_numeric_sequence_text(
        "둘. 셋. 넷. 제자리. 다섯. 여섯. 일곱. 여덟.",
        "둘 셋 넷 제자리 다섯 여섯 일곱 십",
    )

    assert passed.gate == "pass"
    assert passed.expected_values == [2, 3, 4, 5, 6, 7, 8]
    assert passed.observed_values == [2, 3, 4, 5, 6, 7, 8]
    assert passed.ordered_pass is True
    assert passed.contiguous_pass is True
    assert failed.gate == "fail"
    assert failed.ordered_pass is False
    assert failed.contiguous_pass is False
    assert failed.missing_values == [8]
    assert "numeric_sequence_contiguous_mismatch" in failed.issues


def test_korean_pronunciation_qc_fails_repeated_single_syllable_overrun() -> None:
    result = evaluate_pronunciation_text(
        "오",
        "오오오",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )

    assert result.coverage == pytest.approx(1.0)
    assert result.observed_units == 3
    assert result.extra_units == 2
    assert result.gate == "fail"
    assert "observed_pronunciation_too_long" in result.issues
    assert "repetition_overrun" in result.issues


def test_korean_pronunciation_qc_normalizes_digits_and_rejects_suffix_contamination() -> None:
    digit_result = evaluate_pronunciation_text(
        "이",
        "2",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )
    suffix_result = evaluate_pronunciation_text(
        "이",
        "2번",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )

    assert digit_result.gate == "pass"
    assert digit_result.coverage == pytest.approx(1.0)
    assert suffix_result.gate == "fail"
    assert suffix_result.coverage == pytest.approx(1.0)
    assert "suffix_contamination" in suffix_result.issues


def test_korean_pronunciation_qc_accepts_exact_short_il_asr_alias_only() -> None:
    alias_result = evaluate_pronunciation_text(
        "일",
        "예",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )
    contaminated_result = evaluate_pronunciation_text(
        "일",
        "예요",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )

    assert alias_result.gate == "pass"
    assert alias_result.coverage == pytest.approx(1.0)
    assert contaminated_result.gate == "fail"


@pytest.mark.parametrize("observed", ["유", "류"])
def test_korean_pronunciation_qc_accepts_exact_short_yuk_asr_alias(observed: str) -> None:
    alias_result = evaluate_pronunciation_text(
        "육",
        observed,
        pass_coverage=0.82,
        warn_coverage=0.62,
    )
    contaminated_result = evaluate_pronunciation_text(
        "육",
        f"{observed}다시",
        pass_coverage=0.82,
        warn_coverage=0.62,
    )

    assert alias_result.gate == "pass"
    assert alias_result.coverage == pytest.approx(1.0)
    assert contaminated_result.gate == "fail"


def test_korean_pronunciation_qc_failure_triggers_clarity_retry(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_max_attempts_per_candidate=1,
        gsv_retry_candidate_count=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_pronunciation_qc_backend="mock",
        gsv_korean_clarity_retry_candidate_count=1,
        gsv_korean_clarity_temperature=0.61,
        gsv_korean_clarity_top_k=4,
        gsv_korean_clarity_top_p=0.7,
        gsv_korean_clarity_parallel_infer=False,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.15,
        duration=1.15,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="こんにちは",
            language="ja",
            backend="mock",
            start=0.0,
            end=1.15,
        ),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.15,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            requests.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path)
            return output_path

    class FakeASRBackend:
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            self.calls += 1
            text = "테스트" if self.calls == 1 else "안녕하세요"
            return [ASRChunk(start=0.0, end=1.15, text=text, language="ko")]

    fake_asr = FakeASRBackend()

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> FakeASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return fake_asr

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert len(requests) == 2
    assert requests[0]["temperature"] == pytest.approx(1.0)
    assert requests[1]["temperature"] == pytest.approx(0.61)
    assert requests[1]["top_k"] == 4
    assert requests[1]["top_p"] == pytest.approx(0.7)
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["synth_pass"] == "korean_clarity_retry"
    assert selected.payload["pronunciation_qc"]["gate"] == "pass"
    assert manifest.stage_state["synth"]["korean_clarity_retry"]["attempted_segments"] == [
        "seg_0001"
    ]


def test_synth_runs_pronunciation_qc_only_for_selected_candidate(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=3,
        gsv_concurrency=1,
        gsv_ref_mode="static",
        gsv_max_attempts_per_candidate=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_korean_clarity_retry_enabled=False,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.2)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="こんにちは",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=1.2,
        ),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.2,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    class FakeASRBackend:
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            self.calls += 1
            return [ASRChunk(start=0.0, end=1.2, text="안녕하세요", language="ko")]

    fake_asr = FakeASRBackend()

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> FakeASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return fake_asr

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.tts is not None
    assert fake_asr.calls == 1
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["pronunciation_qc"]["gate"] == "pass"
    assert sum(
        1
        for candidate in segment.tts.candidates
        if isinstance(candidate.payload.get("pronunciation_qc"), dict)
    ) == 1


def test_strict_numeric_sequence_qc_rejects_selected_candidate_before_mix(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=2,
        gsv_concurrency=1,
        gsv_ref_mode="static",
        gsv_max_attempts_per_candidate=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_korean_clarity_retry_enabled=False,
        gsv_pronunciation_qc_backend="mock",
        gsv_numeric_sequence_qc_enabled=True,
        gsv_numeric_sequence_qc_require_contiguous=True,
        gsv_numeric_sequence_qc_failure_blocks_mix=True,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=2.4)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=2.4,
        duration=2.4,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="二、三、四、五、六、七、八",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=2.4,
        ),
        script=JapaneseScript(
            literal_ja="二、三、四、五、六、七、八",
            ja_text="二、三、四、五、六、七、八",
            tts_text="둘. 셋. 넷. 제자리. 다섯. 여섯. 일곱. 여덟.",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=2.4,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            return _write_tone_wav(output_path, duration=2.4)

    class FakeASRBackend:
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            self.calls += 1
            text = (
                "둘 셋 넷 제자리 다섯 여섯 일곱 십"
                if self.calls == 1
                else "둘 셋 넷 제자리 다섯 여섯 일곱 여덟"
            )
            return [ASRChunk(start=0.0, end=2.4, text=text, language="ko")]

    fake_asr = FakeASRBackend()

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> FakeASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return fake_asr

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert segment.status == "synthesized"
    assert segment.tts is not None
    first, second = segment.tts.candidates[:2]
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert fake_asr.calls == 2
    assert selected is second
    assert first.acceptable_for_mix is False
    assert first.selection_reason == "numeric_sequence_qc_failed"
    assert first.payload["pronunciation_qc"]["gate"] == "pass"
    assert first.payload["numeric_sequence_qc"]["gate"] == "fail"
    assert second.payload["numeric_sequence_qc"]["gate"] == "pass"


def test_selected_pronunciation_qc_disables_batched_faster_whisper(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        asr_batched_inference=True,
        gsv_ref_mode="static",
        gsv_max_attempts_per_candidate=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_korean_clarity_retry_enabled=False,
        gsv_pronunciation_qc_backend="faster_whisper",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio, duration=1.2)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(
            text="こんにちは",
            language="ja",
            backend="faster_whisper",
            start=0.0,
            end=1.2,
        ),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=1.2,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    created_configs: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    class FakeASRBackend:
        name = "fake"

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            return [ASRChunk(start=0.0, end=1.2, text="안녕하세요", language="ko")]

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> FakeASRBackend:
        assert kind == "faster_whisper"
        created_configs.append(config)
        return FakeASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    assert created_configs
    assert created_configs[0]["language"] == "ko"
    assert created_configs[0]["batched_inference"] is False
    assert created_configs[0]["vad_filter"] is False


def test_synth_splits_selected_pronunciation_qc_across_two_workers(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=2,
        gsv_pronunciation_qc_workers=2,
        gsv_ref_mode="static",
        gsv_max_attempts_per_candidate=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_korean_clarity_retry_enabled=False,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)

    segments: list[Segment] = []
    for offset, segment_id in enumerate(["seg_0001", "seg_0002"]):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
        write_tiny_wav(audio, duration=1.2)
        segments.append(
            Segment(
                id=segment_id,
                start=float(offset) * 1.2,
                end=(float(offset) + 1.0) * 1.2,
                duration=1.2,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text="こんにちは",
                    language="ja",
                    backend="faster_whisper",
                    start=float(offset) * 1.2,
                    end=(float(offset) + 1.0) * 1.2,
                ),
                script=JapaneseScript(
                    literal_ja="こんにちは",
                    ja_text="こんにちは",
                    tts_text="안녕하세요",
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                    expected_tts_duration_sec=1.2,
                ),
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2)
            return output_path

    class FakeASRBackend:
        instances: list[FakeASRBackend] = []
        active = 0
        max_active = 0
        lock = threading.Lock()
        barrier = threading.Barrier(2, timeout=3.0)

        name = "fake"

        def __init__(self) -> None:
            self.calls = 0
            FakeASRBackend.instances.append(self)

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            self.calls += 1
            with FakeASRBackend.lock:
                FakeASRBackend.active += 1
                FakeASRBackend.max_active = max(
                    FakeASRBackend.max_active,
                    FakeASRBackend.active,
                )
            try:
                FakeASRBackend.barrier.wait()
                return [ASRChunk(start=0.0, end=1.2, text="안녕하세요", language="ko")]
            finally:
                with FakeASRBackend.lock:
                    FakeASRBackend.active -= 1

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> FakeASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return FakeASRBackend()

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    assert [segment.status for segment in manifest.segments] == ["synthesized", "synthesized"]
    assert len(FakeASRBackend.instances) == 2
    assert FakeASRBackend.max_active == 2
    assert sorted(instance.calls for instance in FakeASRBackend.instances) == [1, 1]


def test_segment_ref_warn_pronunciation_falls_back_to_static_ref(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(
        project_name="test",
        candidate_count=1,
        gsv_concurrency=1,
        gsv_ref_mode="segment",
        gsv_ref_min_sec=1.0,
        gsv_ref_max_sec=4.0,
        gsv_ref_min_quality_score=0.0,
        gsv_korean_segment_ref_enabled=True,
        gsv_max_attempts_per_candidate=1,
        gsv_retry_candidate_count=1,
        gsv_duration_rewrite_backend="none",
        gsv_low_temperature_retry_enabled=False,
        gsv_korean_clarity_retry_enabled=False,
        gsv_pronunciation_qc_backend="mock",
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    _write_tone_wav(ref_audio, duration=3.2)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    _write_tone_wav(audio, duration=3.2)
    segment = Segment(
        id="seg_0001",
        speaker_id="speaker_0001",
        start=0.0,
        end=3.2,
        duration=3.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(tmp_project_dir)),
        analysis={"speaker_count": 1},
        source_script=SourceScript(
            text="こんにちは",
            language="ja",
            backend="mock",
            start=0.0,
            end=3.2,
        ),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
            expected_tts_duration_sec=3.2,
        ),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    requests: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            requests.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            _write_tone_wav(output_path, duration=3.2)
            return output_path

    class FakeASRBackend:
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_path: Path, segments) -> list[ASRChunk]:
            self.calls += 1
            text = "안녕하세" if self.calls == 1 else "안녕하세요"
            return [ASRChunk(start=0.0, end=3.2, text=text, language="ko")]

    fake_asr = FakeASRBackend()

    def fake_create_asr_backend(kind: str, config: dict[str, object]) -> FakeASRBackend:
        assert kind == "mock"
        assert config["language"] == "ko"
        return fake_asr

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)
    monkeypatch.setattr(steps, "create_asr_backend", fake_create_asr_backend)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    segment = manifest.segments[0]
    assert len(requests) == 2
    assert "work/segments/audio/seg_0001_mix.wav" in str(requests[0]["ref_audio_path"])
    assert requests[1]["ref_audio_path"] == str(ref_audio)
    assert segment.status == "synthesized"
    assert segment.tts is not None
    selected = next(candidate for candidate in segment.tts.candidates if candidate.selected)
    assert selected.payload["synth_pass"] == "static_ref_retry"
    assert selected.payload["pronunciation_qc"]["gate"] == "pass"
    assert manifest.stage_state["synth"]["static_ref_retry"]["attempted_segments"] == [
        "seg_0001"
    ]
