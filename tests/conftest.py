from __future__ import annotations

import hashlib
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def tmp_project_dir(tmp_path: Path) -> Path:
    return tmp_path / "project"


def write_tiny_wav(path: Path, sample_rate: int = 48_000, duration: float = 1.2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 440.0 * t)
    silence = np.zeros(int(sample_rate * 0.25), dtype=np.float32)
    signal = np.concatenate([tone[: int(sample_rate * 0.45)], silence, tone[: int(sample_rate * 0.45)]])
    stereo = np.stack([signal, signal * 0.8], axis=1)
    sf.write(str(path), stereo, sample_rate)
    return path


@pytest.fixture
def tiny_wav_path(tmp_path: Path) -> Path:
    return write_tiny_wav(tmp_path / "input.wav")


@pytest.fixture
def tiny_wav_bytes() -> bytes:
    buffer = BytesIO()
    t = np.arange(4800, dtype=np.float32) / 48_000
    data = np.stack([0.05 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1)
    sf.write(buffer, data, 48_000, format="WAV")
    return buffer.getvalue()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
