from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import write_tiny_wav

from asmr_dub_pipeline.rvc import (
    RVCBatchCommandClient,
    RVCBatchJob,
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
        rvc_train_epochs=20,
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
            "{epochs}",
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

    assert command == ["train", "48k", "6", "0", "2", "2", "20", "50", "true"]


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


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX process groups")
def test_stream_timeout_kills_child_process_group(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    marker_path = tmp_path / "child_survived.txt"
    child_code = (
        "import pathlib, sys, time; "
        "time.sleep(1); "
        "pathlib.Path(sys.argv[1]).write_text('alive', encoding='utf-8')"
    )
    parent_code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); "
        "time.sleep(10)"
    )
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_command=[sys.executable, "-c", parent_code, str(marker_path), child_code],
    )
    client = RVCCommandClient(cfg.rvc_command, timeout_sec=0.2, stream_output=True)

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

    time.sleep(1.2)
    assert not marker_path.exists()


def test_existing_output_reuse_when_force_false(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    output_path = write_tiny_wav(tmp_path / "out.wav")
    future = time.time() + 10
    os.utime(output_path, (future, future))
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


def test_existing_output_is_not_reused_when_input_is_newer(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    output_path = write_tiny_wav(tmp_path / "out.wav")
    past = time.time() - 20
    future = time.time() + 20
    os.utime(output_path, (past, past))
    os.utime(input_path, (future, future))
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        calls.append(command)
        output_path.write_bytes(input_path.read_bytes())
        return subprocess.CompletedProcess(command, 0, stdout="fresh", stderr="")

    cfg = ProjectConfig(rvc_backend="command", rvc_command=["mock", "{input}", "{output}"])
    client = RVCCommandClient(cfg.rvc_command, runner=runner, timeout_sec=5)

    result = client.convert(
        input_path,
        output_path,
        model_path=tmp_path / "model.pth",
        index_path=None,
        cfg=cfg,
        profile=cfg.rvc_auto_profiles[0],
        segment_id="seg_0001",
    )

    assert result.reused_existing is False
    assert result.stdout == "fresh"
    assert len(calls) == 1


def test_batch_command_client_writes_jobs_and_reads_results(tmp_path: Path) -> None:
    input_path = write_tiny_wav(tmp_path / "input.wav")
    output_path = tmp_path / "out.wav"
    model_path = tmp_path / "model.pth"
    index_path = tmp_path / "added.index"
    model_path.write_bytes(b"model")
    index_path.write_bytes(b"index")
    cfg = ProjectConfig(
        rvc_backend="command",
        rvc_batch_command=[
            "batch",
            "--jobs",
            "{jobs}",
            "--results",
            "{results}",
            "--model",
            "{model}",
            "--index",
            "{index}",
        ],
    )

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        jobs_path = Path(command[command.index("--jobs") + 1])
        results_path = Path(command[command.index("--results") + 1])
        rows = [json.loads(line) for line in jobs_path.read_text("utf-8").splitlines()]
        with results_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                Path(row["output_path"]).write_bytes(input_path.read_bytes())
                handle.write(
                    json.dumps(
                        {
                            "segment_id": row["segment_id"],
                            "output_path": row["output_path"],
                            "returncode": 0,
                            "elapsed_sec": 1.25,
                            "stdout": "ok",
                            "stderr": "",
                        }
                    )
                    + "\n"
                )
        return subprocess.CompletedProcess(command, 0, stdout="batch ok", stderr="")

    client = RVCBatchCommandClient(cfg.rvc_batch_command, runner=runner)
    results = client.convert_many(
        [
            RVCBatchJob(
                segment_id="seg_0001",
                input_path=input_path,
                output_path=output_path,
                model_path=model_path,
                index_path=index_path,
                profile=cfg.rvc_auto_profiles[0],
            )
        ],
        jobs_path=tmp_path / "jobs.jsonl",
        results_path=tmp_path / "results.jsonl",
        model_path=model_path,
        index_path=index_path,
        cfg=cfg,
        profile=cfg.rvc_auto_profiles[0],
    )

    assert results["seg_0001"].returncode == 0
    assert results["seg_0001"].elapsed_sec == 1.25
    assert output_path.exists()


def test_batch_command_client_only_reuses_outputs_fresh_for_inputs(tmp_path: Path) -> None:
    fresh_input = write_tiny_wav(tmp_path / "fresh_input.wav")
    fresh_output = write_tiny_wav(tmp_path / "fresh_output.wav")
    stale_input = write_tiny_wav(tmp_path / "stale_input.wav")
    stale_output = write_tiny_wav(tmp_path / "stale_output.wav")
    now = time.time()
    os.utime(fresh_input, (now - 20, now - 20))
    os.utime(fresh_output, (now + 20, now + 20))
    os.utime(stale_input, (now + 30, now + 30))
    os.utime(stale_output, (now - 30, now - 30))

    model_path = tmp_path / "model.pth"
    index_path = tmp_path / "added.index"
    model_path.write_bytes(b"model")
    index_path.write_bytes(b"index")
    cfg = ProjectConfig(rvc_backend="command", rvc_batch_command=["batch", "{jobs}", "{results}"])

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        _ = kwargs
        jobs_path = Path(command[1])
        results_path = Path(command[2])
        rows = [json.loads(line) for line in jobs_path.read_text("utf-8").splitlines()]
        assert [row["segment_id"] for row in rows] == ["seg_stale"]
        with results_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                Path(row["output_path"]).write_bytes(Path(row["input_path"]).read_bytes())
                handle.write(
                    json.dumps(
                        {
                            "segment_id": row["segment_id"],
                            "output_path": row["output_path"],
                            "returncode": 0,
                            "elapsed_sec": 2.5,
                        }
                    )
                    + "\n"
                )
        return subprocess.CompletedProcess(command, 0, stdout="batch ok", stderr="")

    client = RVCBatchCommandClient(cfg.rvc_batch_command, runner=runner)
    results = client.convert_many(
        [
            RVCBatchJob(
                segment_id="seg_fresh",
                input_path=fresh_input,
                output_path=fresh_output,
                model_path=model_path,
                index_path=index_path,
                profile=cfg.rvc_auto_profiles[0],
            ),
            RVCBatchJob(
                segment_id="seg_stale",
                input_path=stale_input,
                output_path=stale_output,
                model_path=model_path,
                index_path=index_path,
                profile=cfg.rvc_auto_profiles[0],
            ),
        ],
        jobs_path=tmp_path / "jobs.jsonl",
        results_path=tmp_path / "results.jsonl",
        model_path=model_path,
        index_path=index_path,
        cfg=cfg,
        profile=cfg.rvc_auto_profiles[0],
    )

    assert results["seg_fresh"].reused_existing is True
    assert results["seg_stale"].reused_existing is False
    assert results["seg_stale"].elapsed_sec == 2.5
