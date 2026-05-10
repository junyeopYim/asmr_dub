from __future__ import annotations

import os
from pathlib import Path

import pytest

from asmr_dub_pipeline.gemma.base import GemmaUnavailableError
from asmr_dub_pipeline.gemma.hf_client import HFGemmaBackend
from asmr_dub_pipeline.gemma.schemas import TASK_REQUIRED_KEYS
from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSClient
from asmr_dub_pipeline.gpt_sovits.schemas import GPTSoVITSRef, GPTSoVITSTTSOptions
from asmr_dub_pipeline.schemas import Segment

pytestmark = [pytest.mark.real_model, pytest.mark.smoke]


def sample_segment() -> Segment:
    return Segment(
        id="seg_0001",
        start=0,
        end=1,
        duration=1,
        audio_for_gemma="a.wav",
        audio_for_mix="b.wav",
    )


@pytest.mark.skipif(not os.environ.get("GSV_URL"), reason="set GSV_URL for local GPT-SoVITS smoke")
def test_local_gpt_sovits_tts_smoke(tmp_path: Path) -> None:
    ref_audio_path = os.environ.get("GSV_REF_AUDIO_PATH")
    if not ref_audio_path:
        pytest.skip("set GSV_REF_AUDIO_PATH to a server-visible reference WAV path")

    client = GPTSoVITSClient(
        os.environ["GSV_URL"],
        timeout_sec=float(os.environ.get("GSV_TIMEOUT_SEC", "120")),
        retries=int(os.environ.get("GSV_RETRIES", "0")),
    )
    ref = GPTSoVITSRef(
        ref_audio_path=ref_audio_path,
        prompt_text=os.environ.get("GSV_PROMPT_TEXT", "こんにちは。"),
        prompt_lang=os.environ.get("GSV_PROMPT_LANG", "ja"),
    )
    request = client.build_payload(
        os.environ.get("GSV_TEXT", "ゆっくり……テストです。"),
        ref,
        GPTSoVITSTTSOptions(seed=int(os.environ.get("GSV_SEED", "1234"))),
    )

    output_path = client.synthesize_to_file(request, tmp_path / "gsv_smoke.wav")

    assert output_path.stat().st_size > 44
    with output_path.open("rb") as handle:
        assert handle.read(4) == b"RIFF"


@pytest.mark.skipif(
    os.environ.get("RUN_GEMMA_SMOKE") != "1",
    reason="set RUN_GEMMA_SMOKE=1 for local cached HF Gemma smoke",
)
def test_hf_gemma_analyze_segment_smoke(tiny_wav_path: Path) -> None:
    backend = HFGemmaBackend(
        model_id=os.environ.get("GEMMA_MODEL_ID", "google/gemma-4-E4B-it"),
        local_files_only=True,
    )
    try:
        result = backend.analyze_segment(tiny_wav_path, sample_segment(), {"smoke": True})
    except GemmaUnavailableError as exc:
        pytest.skip(f"HF Gemma smoke unavailable without a local cached model: {exc}")

    assert TASK_REQUIRED_KEYS["analyze"] <= set(result)
