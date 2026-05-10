from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from asmr_dub_pipeline.schemas import Segment

from .base import GemmaBackend, GemmaHTTPError, GemmaResponseParseError
from .json_repair import JSONRepairError
from .parser import parse_gemma_task_response
from .prompts import analysis_prompt, audio_style_prompt, qc_prompt, repair_prompt, script_prompt
from .schemas import TaskName


class HTTPGemmaBackend(GemmaBackend):
    supports_repair_prompt = True

    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        retries: int = 2,
        model_id: str = "google/gemma-4-E4B-it",
        send_audio: bool = False,
        repair_on_parse_error: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.model_id = model_id
        self.send_audio = send_audio
        self.repair_on_parse_error = repair_on_parse_error
        self.client = client or httpx.Client(timeout=timeout)

    def _parse_response(self, task: TaskName, response: httpx.Response) -> dict[str, Any]:
        ctype = response.headers.get("content-type", "")
        try:
            if "json" in ctype:
                data = response.json()
                return parse_gemma_task_response(task, data)
            else:
                return parse_gemma_task_response(task, response.text)
        except (ValueError, JSONRepairError) as exc:
            raise GemmaResponseParseError(f"Gemma {task} response did not match the JSON contract: {exc}") from exc

    def _send_payload(self, task: TaskName, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(f"{self.base_url}/gemma", json=payload)
        if response.status_code >= 400:
            raise GemmaHTTPError(f"Gemma HTTP error {response.status_code}: {response.text[:500]}")
        return self._parse_response(task, response)

    def _post(
        self,
        task: TaskName,
        prompt: str,
        audio_path: Path,
        segment: Segment,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": task,
            "model_id": self.model_id,
            "prompt": prompt,
            "segment": segment.model_dump(mode="json"),
            "context": dict(context),
            "audio_path": str(audio_path),
        }
        if self.send_audio:
            payload["audio_base64"] = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        last_error: Exception | None = None
        repair_attempted = False
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(f"{self.base_url}/gemma", json=payload)
                if response.status_code >= 500 and attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                if response.status_code >= 400:
                    raise GemmaHTTPError(f"Gemma HTTP error {response.status_code}: {response.text[:500]}")
                return self._parse_response(task, response)
            except GemmaResponseParseError as exc:
                last_error = exc
                if self.repair_on_parse_error and not repair_attempted:
                    repair_attempted = True
                    repair_payload = dict(payload)
                    repair_payload["prompt"] = repair_prompt(task, prompt, response.text, str(exc))
                    repair_payload["repair_of"] = task
                    repair_payload["previous_response"] = response.text[:4000]
                    try:
                        return self._send_payload(task, repair_payload)
                    except GemmaResponseParseError as repair_exc:
                        last_error = repair_exc
                        raise repair_exc from exc
                raise
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break
        raise GemmaHTTPError(f"Gemma HTTP request failed for {segment.id}: {last_error}")

    def analyze_segment(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._post("analyze", analysis_prompt(segment, context), audio_path, segment, context)

    def analyze_audio_style(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._post("audio_style", audio_style_prompt(segment), audio_path, segment, context)

    def generate_script(self, audio_path: Path, segment: Segment, context: Mapping[str, Any]) -> dict[str, Any]:
        return self._post(
            "script",
            script_prompt(segment, segment.analysis, context),
            audio_path,
            segment,
            context,
        )

    def qc_audio(
        self,
        audio_path: Path,
        target_text: str,
        segment: Segment,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._post("qc", qc_prompt(segment, target_text, context), audio_path, segment, context)
