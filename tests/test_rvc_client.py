from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from conftest import write_tiny_wav

from asmr_dub_pipeline.rvc import (
    RVCCommandClient,
    RVCCommandError,
    render_rvc_command,
    render_rvc_train_command,
)
from asmr_dub_pipeline.rvc import client as rvc_client
from asmr_dub_pipeline.schemas import ProjectConfig


def test_command_template_placeholder_replacement() -> None:
    command = render_rvc_command(
        ["rvc", "infer", "-i", "{input}", "-o", "{output}", "-m", "{model}", "--profile", "{profile_name}"],
        {
            "input": "/tmp/in with spaces.wav",
            "output": "/tmp/out.wav",
            "model": "/tmp/model.pth",
            "profile_name": "rmvpe_index045",
        },
    )

    assert command == [
        "rvc",
        "infer",
        "-i",
        "/tmp/in with spaces.wav",
        "-o",
        "/tmp/out.wav",
        "-m",
        "/tmp/model.pth",
        "--profile",
        "rmvpe_index045",
    ]


def test_command_template_rejects_unknown_placeholder() -> None:
    with pytest.raises(RVCCommandError, match="Unsupported"):
        render_rvc_command(["rvc", "{bad}"], {"bad": "x"})


def test_train_command_template_renders_speed_placeholders(tmp_path: Path) -> None:
    cfg = ProjectConfig(
        rvc_train_batch_size=6,
        rvc_train_preprocess_processes=0,
        rvc_train_f0_workers=2,
        rvc_train_feature_workers=2,
        rvc_train_save_every_epoch=50,
        rvc_train_reuse_intermediate_cache=True,
    )

    command = render_rvc_train_command(
        [
            "train",
            "{sample_rate}",
            "{batch_size}",
            "{preprocess_processes}",
            "{f0_workers}",
            "{feature_workers}",
            "{save_every_epoch}",
            "{reuse_intermediate_cache}",
        ],
        project_dir=tmp_path,
        dataset_dir=tmp_path / "dataset",
        work_dir=tmp_path / "work",
        model_path=tmp_path / "model.pth",
        index_path=tmp_path / "model.index",
        cfg=cfg,
    )

    assert command == ["train", "48k", "6", "0", "2", "2", "50", "true"]


def test_train_command_template_auto_batch_size_uses_gpu_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rvc_client, "_gpu_total_memory_mib", lambda _device: 16_311)
    cfg = ProjectConfig(rvc_train_batch_size=0, rvc_device="cuda:0")

    command = render_rvc_train_command(
        ["train", "{batch_size}"],
        project_dir=tmp_path,
        dataset_dir=tmp_path / "dataset",
        work_dir=tmp_path / "work",
        model_path=tmp_path / "model.pth",
        index_path=tmp_path / "model.index",
        cfg=cfg,
    )

    assert command == ["train", "22"]


def test_command_client_does_not_use_shell(tmp_path: Path) -> None:
    seen: dict[str, object] = {}
    input_path = write_tiny_wav(tmp_path / "input.wav")
    output_path = tmp_path / "out.wav"

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["command"] = command
        seen["kwargs"] = kwargs
        output_path.write_bytes(input_path.read_bytes())
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    cfg = ProjectConfig(rvc_backend="command", rvc_command=["mock", "{input}", "{output}"])
    client = RVCCommandClient(cfg.rvc_command, runner=runner)
    client.convert(
        input_path,
        output_path,
        model_path=tmp_path / "model.pth",
        index_path=None,
        cfg=cfg,
        profile=cfg.rvc_auto_profiles[0],
        segment_id="seg_0001",
    )

    assert isinstance(seen["command"], list)
    assert "shell" not in seen["kwargs"]


def test_successful_command_creates_output(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    output_path = tmp_path / "out.wav"
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=[
            sys.executable,
            "-c",
            "import shutil,sys; shutil.copyfile(sys.argv[1], sys.argv[2])",
            "{input}",
            "{output}",
        ],
    )
    client = RVCCommandClient(cfg.rvc_command, timeout_sec=5)

    result = client.convert(
        input_path,
        output_path,
        model_path=tmp_path / "model.pth",
        index_path=None,
        cfg=cfg,
        profile=cfg.rvc_auto_profiles[0],
        segment_id="seg_0001",
    )

    assert result.returncode == 0
    assert output_path.exists()


def test_command_client_streams_subprocess_output(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    output_path = tmp_path / "out.wav"
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=[
            sys.executable,
            "-u",
            "-c",
            "import shutil,sys; print('hello'); print('warn', file=sys.stderr); shutil.copyfile(sys.argv[1], sys.argv[2])",
            "{input}",
            "{output}",
        ],
    )
    client = RVCCommandClient(cfg.rvc_command, timeout_sec=5, stream_output=True, log_prefix="rvc-test")

    result = client.convert(
        input_path,
        output_path,
        model_path=tmp_path / "model.pth",
        index_path=None,
        cfg=cfg,
        profile=cfg.rvc_auto_profiles[0],
        segment_id="seg_0001",
    )

    captured = capfd.readouterr()
    assert "rvc-test: hello" in captured.out
    assert "rvc-test: warn" in captured.out
    assert "hello" in result.stdout
    assert "warn" in result.stdout
    assert output_path.exists()


def test_non_zero_command_raises_with_stdout_stderr(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=[sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"],
    )
    client = RVCCommandClient(cfg.rvc_command, timeout_sec=5)

    with pytest.raises(RVCCommandError, match="exit code 3.*out.*err"):
        client.convert(
            input_path,
            tmp_path / "out.wav",
            model_path=tmp_path / "model.pth",
            index_path=None,
            cfg=cfg,
            profile=cfg.rvc_auto_profiles[0],
            segment_id="seg_0001",
        )


def test_missing_output_raises_clear_error(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    cfg = ProjectConfig(rvc_backend="command", rvc_command=[sys.executable, "-c", "print('done')"])
    client = RVCCommandClient(cfg.rvc_command, timeout_sec=5)

    with pytest.raises(RVCCommandError, match="did not create output"):
        client.convert(
            input_path,
            tmp_path / "out.wav",
            model_path=tmp_path / "model.pth",
            index_path=None,
            cfg=cfg,
            profile=cfg.rvc_auto_profiles[0],
            segment_id="seg_0001",
        )


def test_timeout_handling(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=[sys.executable, "-c", "import time; time.sleep(5)"],
    )
    client = RVCCommandClient(cfg.rvc_command, timeout_sec=0.01)

    with pytest.raises(RVCCommandError, match="timed out"):
        client.convert(
            input_path,
            tmp_path / "out.wav",
            model_path=tmp_path / "model.pth",
            index_path=None,
            cfg=cfg,
            profile=cfg.rvc_auto_profiles[0],
            segment_id="seg_0001",
        )


def test_existing_output_reuse_when_force_false(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    output_path = write_tiny_wav(tmp_path / "out.wav")
    cfg = ProjectConfig(rvc_backend="command", rvc_command=[sys.executable, "-c", "raise SystemExit(99)"])
    client = RVCCommandClient(cfg.rvc_command, timeout_sec=5)

    result = client.convert(
        input_path,
        output_path,
        model_path=tmp_path / "model.pth",
        index_path=None,
        cfg=cfg,
        profile=cfg.rvc_auto_profiles[0],
        segment_id="seg_0001",
    )

    assert result.reused_existing is True
