from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .client import GPTSoVITSClient, GPTSoVITSError
from .schemas import GPTSoVITSRef, GPTSoVITSTTSOptions


@dataclass(frozen=True)
class GPTSoVITSSmokeConfig:
    base_url: str
    ref_audio_path: Path
    prompt_text: str
    text: str
    output_path: Path
    prompt_lang: str = "ja"
    seed: int = 10101


def smoke_config_from_env(env: Mapping[str, str] | None = None) -> GPTSoVITSSmokeConfig | None:
    env = os.environ if env is None else env
    base_url = env.get("GSV_URL")
    if not base_url:
        return None
    if env.get("GSV_CONFIRM_RIGHTS") != "1":
        raise GPTSoVITSError(
            "Set GSV_CONFIRM_RIGHTS=1 to confirm rights/consent for the smoke-test voice reference."
        )
    ref_audio_path = env.get("GSV_REF_AUDIO")
    if not ref_audio_path:
        raise GPTSoVITSError("Set GSV_REF_AUDIO to a local reference WAV readable by api_v2.")
    resolved_ref_audio_path = Path(ref_audio_path).expanduser().resolve()
    if not resolved_ref_audio_path.exists():
        raise GPTSoVITSError(f"GSV_REF_AUDIO does not exist: {resolved_ref_audio_path}")
    return GPTSoVITSSmokeConfig(
        base_url=base_url,
        ref_audio_path=resolved_ref_audio_path,
        prompt_text=env.get("GSV_PROMPT_TEXT", "耳元で、ゆっくり囁いていきますね。"),
        text=env.get("GSV_TEXT", "これは接続確認の短い音声です。"),
        output_path=Path(env.get("GSV_OUT", "work/tts/gsv_smoke.wav")).expanduser().resolve(),
        prompt_lang=env.get("GSV_PROMPT_LANG", "ja"),
        seed=int(env.get("GSV_SEED", "10101")),
    )


def run_smoke(config: GPTSoVITSSmokeConfig) -> Path:
    ref = GPTSoVITSRef(
        ref_audio_path=str(config.ref_audio_path),
        prompt_text=config.prompt_text,
        prompt_lang=config.prompt_lang,
    )
    client = GPTSoVITSClient(config.base_url)
    request = client.build_payload(
        config.text,
        ref,
        GPTSoVITSTTSOptions(seed=config.seed),
    )
    return client.synthesize_to_file(request, config.output_path)


def main() -> int:
    config = smoke_config_from_env()
    if config is None:
        print("Skipped GPT-SoVITS smoke test: GSV_URL is not set.")
        return 0
    output = run_smoke(config)
    print(f"GPT-SoVITS smoke test wrote: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
