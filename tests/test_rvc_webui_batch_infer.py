from __future__ import annotations

import json
import sys
from pathlib import Path

from asmr_dub_pipeline.rvc import webui_batch_infer


def test_batch_infer_reuses_loaded_model_for_job_file(monkeypatch, tmp_path: Path) -> None:
    rvc_root = tmp_path / "rvc"
    rvc_root.mkdir()
    model_path = tmp_path / "model" / "speaker.pth"
    index_path = tmp_path / "model" / "added_speaker.index"
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    index_path.write_bytes(b"index")
    input_one = tmp_path / "input_one.wav"
    input_two = tmp_path / "input_two.wav"
    input_one.write_bytes(b"input")
    input_two.write_bytes(b"input")
    jobs_path = tmp_path / "jobs.jsonl"
    results_path = tmp_path / "results.jsonl"
    output_one = tmp_path / "out_one.wav"
    output_two = tmp_path / "out_two.wav"
    jobs_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"segment_id": "seg_0001", "input_path": str(input_one), "output_path": str(output_one)},
                {"segment_id": "seg_0002", "input_path": str(input_two), "output_path": str(output_two)},
            ]
        )
        + "\n",
        "utf-8",
    )
    loaded_models: list[str] = []
    original_argv = [
        "webui_batch_infer.py",
        "--rvc-root",
        str(rvc_root),
        "--jobs",
        str(jobs_path),
        "--results",
        str(results_path),
        "--model",
        str(model_path),
        "--index",
        str(index_path),
    ]

    class FakeConfig:
        device = "cpu"
        is_half = False

        def __init__(self) -> None:
            assert sys.argv == ["webui_batch_infer.py"]

    class FakeVC:
        def __init__(self, config: FakeConfig) -> None:
            self.config = config

        def get_vc(self, model_name: str) -> None:
            assert sys.argv == ["webui_batch_infer.py"]
            loaded_models.append(model_name)

        def vc_single(self, *args: object) -> tuple[str, tuple[int, list[int]]]:
            return "Success.", (48_000, [0, 0, 0])

    monkeypatch.setattr(webui_batch_infer, "_link_or_copy_missing_assets", lambda _rvc_root: None)
    monkeypatch.setattr(webui_batch_infer, "_load_rvc_components", lambda _rvc_root: (FakeConfig, FakeVC))
    monkeypatch.setattr(webui_batch_infer, "_write_wav", lambda path, _sample_rate, _audio: Path(path).write_bytes(b"wav"))
    monkeypatch.setattr(
        sys,
        "argv",
        original_argv[:],
    )

    webui_batch_infer.main()

    results = [json.loads(line) for line in results_path.read_text("utf-8").splitlines()]
    assert sys.argv == original_argv
    assert loaded_models == ["speaker.pth"]
    assert [row["segment_id"] for row in results] == ["seg_0001", "seg_0002"]
    assert all(row["returncode"] == 0 for row in results)
    assert output_one.exists()
    assert output_two.exists()
