from __future__ import annotations

import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from asmr_dub_pipeline.rvc import webui_train


def _write_preprocess_outputs(exp_dir: Path, names: tuple[str, ...] = ("seg_0001", "seg_0002")) -> None:
    for relative in ("0_gt_wavs", "1_16k_wavs"):
        (exp_dir / relative).mkdir(parents=True, exist_ok=True)
    for name in names:
        (exp_dir / "0_gt_wavs" / f"{name}.wav").write_bytes(b"gt")
        (exp_dir / "1_16k_wavs" / f"{name}.wav").write_bytes(b"wav")
    (exp_dir / "preprocess.log").write_text("end preprocess\n", "utf-8")


def _write_f0_outputs(exp_dir: Path) -> None:
    for relative in ("2a_f0", "2b-f0nsf"):
        (exp_dir / relative).mkdir(parents=True, exist_ok=True)
    for wav in sorted((exp_dir / "1_16k_wavs").glob("*.wav")):
        (exp_dir / "2a_f0" / f"{wav.name}.npy").write_bytes(b"f0")
        (exp_dir / "2b-f0nsf" / f"{wav.name}.npy").write_bytes(b"f0nsf")


def _write_feature_outputs(exp_dir: Path, version: str = "v2") -> None:
    feature_dir = exp_dir / ("3_feature256" if version == "v1" else "3_feature768")
    feature_dir.mkdir(parents=True, exist_ok=True)
    for wav in sorted((exp_dir / "1_16k_wavs").glob("*.wav")):
        (feature_dir / wav.with_suffix(".npy").name).write_bytes(b"feature")


def test_run_scopes_torch_load_compat_env_to_trusted_checkpoints(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    monkeypatch.setattr(webui_train.subprocess, "run", fake_run)

    webui_train._run(["python", "script.py"], cwd=tmp_path)

    env = captured["env"]
    assert isinstance(env, dict)
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in env
    assert captured["cwd"] == str(tmp_path)
    assert captured["check"] is False
    assert captured["text"] is True


def test_run_enables_torch_load_compat_for_trusted_checkpoints(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)
    monkeypatch.setattr(webui_train.subprocess, "run", fake_run)

    webui_train._run(["python", "script.py"], cwd=tmp_path, trusted_checkpoint=True)

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"


def test_reset_experiment_dir_removes_stale_training_artifacts(tmp_path: Path) -> None:
    exp_dir = tmp_path / "logs" / "asmr-speaker"
    stale_feature_dir = exp_dir / "3_feature768"
    stale_feature_dir.mkdir(parents=True)
    (stale_feature_dir / "old.npy").write_bytes(b"stale")
    (exp_dir / "G_2333333.pth").write_bytes(b"old checkpoint")

    webui_train._reset_experiment_dir(exp_dir)

    assert exp_dir.exists()
    assert list(exp_dir.iterdir()) == []


def test_worker_auto_defaults() -> None:
    assert webui_train._resolve_worker_count(0, auto=8) == 8
    assert webui_train._resolve_worker_count(3, auto=8) == 3
    assert webui_train._auto_gpu_workers("cuda:0") == 2
    assert webui_train._auto_gpu_workers("cpu") == 1


def test_run_parallel_preserves_failed_worker_output(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        _ = kwargs
        if command[-1] == "bad":
            return SimpleNamespace(returncode=7, stdout="worker stdout", stderr="worker stderr")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(webui_train.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="exit code 7.*worker stdout.*worker stderr"):
        webui_train._run_parallel([["cmd", "ok"], ["cmd", "bad"]], cwd=tmp_path)


def test_cache_manifest_match_and_complete_outputs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "seg_0001.wav").write_bytes(b"one")
    (dataset / "seg_0002.wav").write_bytes(b"two")
    exp_dir = tmp_path / "logs" / "speaker"

    fingerprint = webui_train._dataset_fingerprint(
        dataset,
        sample_rate="48k",
        version="v2",
        f0_method="rmvpe",
    )
    webui_train._write_cache_manifest(exp_dir, fingerprint)

    assert webui_train._matching_cache(exp_dir, fingerprint) is True
    assert webui_train._matching_cache(exp_dir, {**fingerprint, "sample_rate": "40k"}) is False

    _write_preprocess_outputs(exp_dir)
    _write_f0_outputs(exp_dir)
    _write_feature_outputs(exp_dir, "v2")
    webui_train._mark_stage_done(exp_dir, fingerprint, "preprocess")

    assert webui_train._stage_done(exp_dir, "preprocess") is True
    assert webui_train._preprocess_outputs_complete(exp_dir) is True
    assert webui_train._f0_outputs_complete(exp_dir) is True
    assert webui_train._feature_outputs_complete(exp_dir, "v2") is True


def test_write_filelist_creates_missing_mute_training_assets(tmp_path: Path) -> None:
    rvc_root = tmp_path / "rvc"
    exp_dir = rvc_root / "logs" / "speaker"
    _write_preprocess_outputs(exp_dir, ("seg_0001",))
    _write_f0_outputs(exp_dir)
    _write_feature_outputs(exp_dir, "v2")

    webui_train._write_filelist(rvc_root, exp_dir, "48k", "v2", if_f0=True)

    mute_root = rvc_root / "logs" / "mute"
    mute_wav = mute_root / "0_gt_wavs" / "mute48k.wav"
    with wave.open(str(mute_wav), "rb") as wav:
        assert wav.getframerate() == 48_000
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0
    assert np.load(mute_root / "3_feature768" / "mute.npy").shape[1] == 768
    assert np.load(mute_root / "2a_f0" / "mute.wav.npy").ndim == 1
    assert np.load(mute_root / "2b-f0nsf" / "mute.wav.npy").ndim == 1
    filelist = (exp_dir / "filelist.txt").read_text("utf-8")
    assert str(mute_wav) in filelist


def test_train_index_caps_large_feature_sets(monkeypatch, tmp_path: Path) -> None:
    exp_dir = tmp_path / "logs" / "speaker"
    feature_dir = exp_dir / "3_feature768"
    feature_dir.mkdir(parents=True)
    np.save(feature_dir / "a.npy", np.ones((5, 768), dtype=np.float32))
    np.save(feature_dir / "b.npy", np.ones((5, 768), dtype=np.float32))
    output_index = tmp_path / "out" / "added.index"
    trained_shapes: list[tuple[int, ...]] = []
    index_specs: list[str] = []

    class FakeIndex:
        def train(self, values: np.ndarray) -> None:
            trained_shapes.append(values.shape)

        def add(self, values: np.ndarray) -> None:
            _ = values

    def fake_index_factory(dimension: int, spec: str) -> FakeIndex:
        assert dimension == 768
        index_specs.append(spec)
        return FakeIndex()

    fake_faiss = SimpleNamespace(
        index_factory=fake_index_factory,
        extract_index_ivf=lambda index: SimpleNamespace(nprobe=0),
        write_index=lambda index, path: Path(path).write_bytes(b"index"),
    )
    monkeypatch.setitem(sys.modules, "faiss", fake_faiss)
    monkeypatch.setattr(webui_train, "INDEX_MAX_TRAIN_FRAMES", 3, raising=False)
    monkeypatch.setattr(webui_train, "INDEX_MAX_IVF", 1, raising=False)

    webui_train._train_index(exp_dir, "speaker", "v2", output_index)

    assert trained_shapes == [(3, 768)]
    assert index_specs == ["IVF1,Flat"]
    assert output_index.read_bytes() == b"index"


def test_main_partitions_f0_and_feature_workers(monkeypatch, tmp_path: Path) -> None:
    rvc_root = tmp_path / "rvc"
    dataset = tmp_path / "dataset"
    output_model = tmp_path / "out" / "model.pth"
    output_index = tmp_path / "out" / "model.index"
    experiment_name = "speaker"
    exp_dir = rvc_root / "logs" / experiment_name
    exported = tmp_path / "exported.pth"
    dataset.mkdir()
    (dataset / "seg_0001.wav").write_bytes(b"wav")
    (rvc_root / "configs" / "v2").mkdir(parents=True)
    (rvc_root / "configs" / "v2" / "40k.json").write_text("{}", "utf-8")
    exported.write_bytes(b"model")

    run_parallel_commands: list[list[list[str]]] = []
    train_commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path) -> None:
        _ = cwd
        if any("preprocess.py" in arg for arg in command):
            _write_preprocess_outputs(exp_dir, ("seg_0001",))
        if any("train.py" in arg for arg in command):
            train_commands.append(command)

    def fake_run_parallel(
        commands: list[list[str]],
        *,
        cwd: Path,
        trusted_checkpoint: bool = False,
    ) -> None:
        _ = cwd, trusted_checkpoint
        run_parallel_commands.append(commands)
        if commands and "extract_f0_rmvpe.py" in commands[0][1]:
            _write_f0_outputs(exp_dir)
        if commands and "extract_feature_print.py" in commands[0][1]:
            _write_feature_outputs(exp_dir, "v2")

    monkeypatch.setattr(webui_train, "_run", fake_run)
    monkeypatch.setattr(webui_train, "_run_parallel", fake_run_parallel)
    def fake_train_index(exp: Path, name: str, version: str, output: Path) -> None:
        _ = exp, name, version
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"index")

    monkeypatch.setattr(webui_train, "_train_index", fake_train_index)
    monkeypatch.setattr(webui_train, "_latest_exported_weight", lambda root, name, min_mtime: exported)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "webui_train.py",
            "--rvc-root",
            str(rvc_root),
            "--dataset",
            str(dataset),
            "--experiment-name",
            experiment_name,
            "--output-model",
            str(output_model),
            "--output-index",
            str(output_index),
            "--f0-workers",
            "3",
            "--feature-workers",
            "2",
        ],
    )

    webui_train.main()

    assert [[command[2], command[3]] for command in run_parallel_commands[0]] == [["3", "0"], ["3", "1"], ["3", "2"]]
    assert [[command[3], command[4]] for command in run_parallel_commands[1]] == [["2", "0"], ["2", "1"]]
    assert "-se" in train_commands[0]
    assert train_commands[0][train_commands[0].index("-se") + 1] == "5"
    assert output_model.read_bytes() == b"model"
