from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.schemas import Segment

from .base import GemmaBackend, GemmaResponseParseError, GemmaUnavailableError
from .json_repair import JSONRepairError
from .parser import parse_gemma_task_response
from .prompts import analysis_prompt, json_repair_prompt, qc_prompt, script_prompt
from .schemas import TASK_REQUIRED_KEYS, TaskName

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_CPP_DIR = (
    Path(".cache")
    / "llama_cpp"
    / "models"
    / "HauhauCS"
    / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"
)
DEFAULT_LLAMA_CPP_MODEL = (
    DEFAULT_LLAMA_CPP_DIR / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf"
)
DEFAULT_LLAMA_CPP_MMPROJ = (
    DEFAULT_LLAMA_CPP_DIR / "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
)
DEFAULT_LLAMA_CPP_CLI = (
    Path(".cache") / "llama_cpp" / "src" / "llama.cpp" / "build" / "bin" / "llama-mtmd-cli"
)


def _resolve_existing_path(value: str | Path, label: str) -> Path:
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, REPO_ROOT / raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    locations = ", ".join(str(candidate) for candidate in candidates)
    raise GemmaUnavailableError(f"llama.cpp {label} not found: {value} (tried {locations})")


def _json_schema_for_task(task: TaskName) -> str:
    return json.dumps(
        {
            "type": "object",
            "required": sorted(TASK_REQUIRED_KEYS[task]),
            "additionalProperties": True,
        },
        ensure_ascii=False,
    )


class LlamaCppGemmaBackend(GemmaBackend):
    supports_repair_prompt = True

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_LLAMA_CPP_MODEL,
        mmproj_path: str | Path = DEFAULT_LLAMA_CPP_MMPROJ,
        cli_path: str | Path = DEFAULT_LLAMA_CPP_CLI,
        timeout_sec: float = 600.0,
        ctx_size: int = 4096,
        n_predict: int = 1024,
        gpu_layers: int = 999,
        temperature: float = 0.0,
        seed: int = 12345,
        extra_args: Sequence[str] | str | None = None,
    ) -> None:
        self.model_path = _resolve_existing_path(model_path, "model")
        self.mmproj_path = _resolve_existing_path(mmproj_path, "mmproj")
        self.cli_path = _resolve_existing_path(cli_path, "CLI")
        self.timeout_sec = timeout_sec
        self.ctx_size = ctx_size
        self.n_predict = n_predict
        self.gpu_layers = gpu_layers
        self.temperature = temperature
        self.seed = seed
        self.extra_args = (
            shlex.split(extra_args)
            if isinstance(extra_args, str)
            else [str(arg) for arg in extra_args or []]
        )

    def _command(self, prompt: str, task: TaskName, audio_path: Path | None) -> list[str]:
        command = [
            str(self.cli_path),
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "-p",
            prompt,
            "-c",
            str(self.ctx_size),
            "-n",
            str(self.n_predict),
            "-ngl",
            str(self.gpu_layers),
            "--temp",
            str(self.temperature),
            "--seed",
            str(self.seed),
            "--json-schema",
            _json_schema_for_task(task),
            "--no-warmup",
        ]
        if audio_path is not None:
            command.extend(["--audio", str(audio_path)])
        command.extend(self.extra_args)
        return command

    def _run(self, prompt: str, task: TaskName, audio_path: Path | None) -> str:
        try:
            completed = subprocess.run(
                self._command(prompt, task, audio_path),
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
        except FileNotFoundError as exc:
            raise GemmaUnavailableError(f"llama.cpp CLI not executable: {self.cli_path}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GemmaUnavailableError(
                f"llama.cpp {task} generation timed out after {self.timeout_sec:g}s."
            ) from exc

        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()[:1200]
            raise GemmaUnavailableError(
                f"llama.cpp {task} generation failed with exit code {completed.returncode}: {details}"
            )
        return completed.stdout

    def _generate(self, audio_path: Path, prompt: str, task: TaskName) -> dict[str, Any]:
        audio = _resolve_existing_path(audio_path, "audio")
        raw_response = ""
        try:
            raw_response = self._run(prompt, task, audio)
            return parse_gemma_task_response(task, raw_response)
        except JSONRepairError as exc:
            try:
                repaired = self._run(json_repair_prompt(task, raw_response, str(exc)), task, None)
                return parse_gemma_task_response(task, repaired)
            except JSONRepairError as repair_exc:
                raise GemmaResponseParseError(
                    f"llama.cpp {task} response did not match the JSON contract after repair: {repair_exc}"
                ) from repair_exc
        except GemmaUnavailableError:
            raise
        except Exception as exc:
            raise GemmaUnavailableError(f"llama.cpp {task} generation failed: {exc}") from exc

    def analyze_segment(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._generate(audio_path, analysis_prompt(segment, context), "analyze")

    def generate_script(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._generate(audio_path, script_prompt(segment, segment.analysis, context), "script")

    def qc_audio(
        self,
        audio_path: Path,
        target_text: str,
        segment: Segment,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._generate(audio_path, qc_prompt(segment, target_text, context), "qc")
