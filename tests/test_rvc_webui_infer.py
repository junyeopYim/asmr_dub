from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from asmr_dub_pipeline.rvc import webui_infer


def test_infer_registers_hubert_safe_globals_without_forcing_all_torch_loads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rvc_root = tmp_path / "rvc"
    rvc_root.mkdir()
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    model_path = tmp_path / "model" / "speaker.pth"
    index_path = tmp_path / "model" / "added_speaker.index"
    input_path.write_bytes(b"input")
    model_path.parent.mkdir()
    model_path.write_bytes(b"model")
    index_path.write_bytes(b"index")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        output_path.write_bytes(b"wav")
        return SimpleNamespace(returncode=0)

    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setattr(webui_infer, "_link_or_copy_missing_assets", lambda _rvc_root: None)
    monkeypatch.setattr(webui_infer.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "webui_infer.py",
            "--rvc-root",
            str(rvc_root),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            str(model_path),
            "--index",
            str(index_path),
        ],
    )

    webui_infer.main()

    env = captured["env"]
    assert isinstance(env, dict)
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in env
    assert env["weight_root"] == str(model_path.parent.resolve())
    assert env["index_root"] == str(index_path.parent.resolve())
    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [sys.executable, "-c", webui_infer._INFER_CLI_COMPAT]
    assert captured["cwd"] == str(rvc_root.resolve())
    assert captured["check"] is False
    assert captured["text"] is True
