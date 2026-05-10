from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asmr_dub_pipeline.audio.features import load_audio, resample_linear, to_mono
from asmr_dub_pipeline.schemas import Segment

from .base import GemmaBackend, GemmaResponseParseError, GemmaUnavailableError
from .json_repair import JSONRepairError
from .parser import parse_gemma_task_response
from .prompts import analysis_prompt, audio_style_prompt, json_repair_prompt, qc_prompt, script_prompt
from .schemas import TaskName


class HFGemmaBackend(GemmaBackend):
    supports_repair_prompt = True

    def __init__(self, model_id: str = "google/gemma-4-E4B-it", local_files_only: bool = True) -> None:
        self.model_id = model_id
        self.local_files_only = local_files_only
        self._processor = None
        self._model = None

    def _load(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except Exception as exc:
            raise GemmaUnavailableError(
                "transformers with Gemma4 multimodal support is required for the HF backend."
            ) from exc
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                local_files_only=self.local_files_only,
            )
            self._model = AutoModelForMultimodalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype="auto",
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise GemmaUnavailableError(
                f"Could not load {self.model_id}. Cache the model locally or enable downloads explicitly."
            ) from exc
        return self._processor, self._model

    def _decode(self, messages: list[dict[str, Any]], audio: Any | None = None) -> str:
        processor, model = self._load()
        if not hasattr(processor, "apply_chat_template"):
            raise GemmaUnavailableError("Installed processor does not support Gemma4 audio chat templates.")
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kwargs: dict[str, Any] = {"text": [text], "return_tensors": "pt"}
        if audio is not None:
            kwargs.update({"audio": [audio], "sampling_rate": 16_000})
        inputs = processor(**kwargs)
        if hasattr(inputs, "to") and hasattr(model, "device"):
            dtype = getattr(model, "dtype", None)
            inputs = inputs.to(model.device, dtype=dtype) if dtype is not None else inputs.to(model.device)
        input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        outputs = model.generate(**inputs, max_new_tokens=1024)
        generated = outputs[:, input_len:] if input_len else outputs
        return processor.batch_decode(generated, skip_special_tokens=True)[0]

    def _repair(self, task: TaskName, raw_response: str, error: Exception) -> dict[str, Any]:
        prompt = json_repair_prompt(task, raw_response, str(error))
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        repaired = self._decode(messages)
        return parse_gemma_task_response(task, repaired)

    def _generate(self, audio_path: Path, prompt: str, task: TaskName) -> dict[str, Any]:
        data, sr = load_audio(audio_path)
        mono = to_mono(data)
        mono_16k = resample_linear(mono, sr, 16_000)
        mono_16k_audio = mono_16k[None, :]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": mono_16k_audio},
                ],
            }
        ]
        decoded = ""
        try:
            decoded = self._decode(messages, audio=mono_16k_audio)
            return parse_gemma_task_response(task, decoded)
        except JSONRepairError as exc:
            try:
                return self._repair(task, decoded, exc)
            except JSONRepairError as repair_exc:
                raise GemmaResponseParseError(
                    f"HF Gemma {task} response did not match the JSON contract after repair: {repair_exc}"
                ) from repair_exc
        except Exception as exc:
            raise GemmaUnavailableError(f"HF Gemma audio generation failed: {exc}") from exc

    def analyze_segment(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._generate(audio_path, analysis_prompt(segment, context), "analyze")

    def analyze_audio_style(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._generate(audio_path, audio_style_prompt(segment), "audio_style")

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
