from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asmr_dub_pipeline.audio.features import (  # noqa: E402
    duration_sec,
    ensure_stereo,
    load_audio,
    peak_dbfs,
    resample_linear,
    rms_dbfs,
    trim_edge_silence,
    write_audio,
)
from asmr_dub_pipeline.gpt_sovits.client import GPTSoVITSClient  # noqa: E402
from asmr_dub_pipeline.gpt_sovits.refs import load_refs, resolve_ref  # noqa: E402
from asmr_dub_pipeline.gpt_sovits.schemas import GPTSoVITSTTSOptions  # noqa: E402
from asmr_dub_pipeline.gpt_sovits.server import ManagedGPTSoVITSServer  # noqa: E402

DEFAULT_TOKENS = ("다섯", "넷", "셋", "둘", "하나", "영")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one GPT-SoVITS wav per Korean countdown token and assemble a fixed-slot timeline.",
    )
    parser.add_argument("--confirm-rights", action="store_true", help="Confirm rights/consent for refs.")
    parser.add_argument("--url", default="http://127.0.0.1:9880", help="GPT-SoVITS api_v2 URL.")
    parser.add_argument("--auto-start", action="store_true", help="Start local GPT-SoVITS if needed.")
    parser.add_argument("--startup-timeout-sec", type=float, default=300.0)
    parser.add_argument("--refs", default="refs/refs.json", help="Project-local refs JSON.")
    parser.add_argument("--ref-style", default="whisper_close")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--tokens", nargs="+", default=list(DEFAULT_TOKENS))
    parser.add_argument("--slot-sec", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--seed", type=int, default=82345)
    parser.add_argument("--speed-factor", type=float, default=1.2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--timeout-sec", type=float, default=240.0)
    parser.add_argument("--retries", type=int, default=1)
    return parser.parse_args()


def safe_label(text: str) -> str:
    label = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text).strip("_")
    return label or "token"


def infer_refs_project_dir(refs_path: Path) -> Path:
    resolved = (PROJECT_ROOT / refs_path).resolve() if not refs_path.is_absolute() else refs_path.resolve()
    if resolved.parent.name == "refs":
        return resolved.parent.parent
    return PROJECT_ROOT


def normalize_audio_for_timeline(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    data, source_rate = load_audio(path)
    data = ensure_stereo(data)
    if source_rate != sample_rate:
        data = resample_linear(data, source_rate, sample_rate)
    return data.astype(np.float32, copy=False), sample_rate


def synthesize_tokens(args: argparse.Namespace) -> dict[str, Any]:
    if not args.confirm_rights:
        raise SystemExit(
            "Refusing to synthesize without --confirm-rights. "
            "Only use refs you have rights/consent to use."
        )
    if args.slot_sec <= 0:
        raise SystemExit("--slot-sec must be greater than zero.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    token_dir = out_dir / "tokens"
    raw_dir = out_dir / "raw"
    token_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    refs_path = Path(args.refs)
    refs_actual = (PROJECT_ROOT / refs_path).resolve() if not refs_path.is_absolute() else refs_path.resolve()
    refs_project_dir = infer_refs_project_dir(refs_actual)
    refs = load_refs(refs_actual, refs_project_dir)
    ref = resolve_ref(refs, args.ref_style)
    client = GPTSoVITSClient(args.url, timeout_sec=args.timeout_sec, retries=args.retries)
    options = GPTSoVITSTTSOptions(
        text_lang="ko",
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        text_split_method="cut0",
        speed_factor=args.speed_factor,
        fragment_interval=0.0,
        seed=args.seed,
        parallel_infer=False,
        repetition_penalty=1.25,
        sample_steps=32,
        super_sampling=True,
        overlap_length=2,
        min_chunk_length=8,
    )

    token_reports: list[dict[str, Any]] = []
    for index, token in enumerate(args.tokens):
        raw_path = raw_dir / f"{index:02d}_{safe_label(token)}.wav"
        trimmed_path = token_dir / f"{index:02d}_{safe_label(token)}.wav"
        request = client.build_payload(
            token,
            ref,
            options.model_copy(update={"seed": args.seed + index}),
        )
        client.synthesize_to_file(request, raw_path)
        shutil.copyfile(raw_path, trimmed_path)
        trim = trim_edge_silence(trimmed_path, threshold_db=-50.0, keep_sec=0.04)
        raw_duration = duration_sec(raw_path)
        trimmed_duration = duration_sec(trimmed_path)
        token_reports.append(
            {
                "index": index,
                "text": token,
                "raw_path": str(raw_path),
                "trimmed_path": str(trimmed_path),
                "raw_duration_sec": round(raw_duration, 6),
                "trimmed_duration_sec": round(trimmed_duration, 6),
                "slot_sec": round(args.slot_sec, 6),
                "slot_overflow_sec": round(max(0.0, trimmed_duration - args.slot_sec), 6),
                "trim": trim,
                "peak_dbfs": round(peak_dbfs(trimmed_path), 3),
                "rms_dbfs": round(rms_dbfs(trimmed_path), 3),
                "payload": request.as_payload(),
            }
        )
        print(
            f"[{index:02d}] {token}: raw={raw_duration:.3f}s "
            f"trimmed={trimmed_duration:.3f}s overflow={max(0.0, trimmed_duration - args.slot_sec):.3f}s"
        )

    timeline_path = out_dir / f"timeline_{args.slot_sec:g}s.wav"
    timeline_report = write_timeline(token_reports, timeline_path, args.sample_rate, args.slot_sec)
    report = {
        "url": args.url,
        "ref_style": args.ref_style,
        "tokens": token_reports,
        "timeline": timeline_report,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), "utf-8")
    print(f"timeline: {timeline_path}")
    print(f"report: {report_path}")
    return report


def write_timeline(
    token_reports: list[dict[str, Any]],
    output_path: Path,
    sample_rate: int,
    slot_sec: float,
) -> dict[str, Any]:
    slot_frames = max(1, int(round(slot_sec * sample_rate)))
    total_frames = max(1, slot_frames * len(token_reports))
    timeline = np.zeros((total_frames, 2), dtype=np.float32)
    placements: list[dict[str, Any]] = []
    previous_end = 0

    for item in token_reports:
        data, _ = normalize_audio_for_timeline(Path(item["trimmed_path"]), sample_rate)
        slot_start = int(item["index"]) * slot_frames
        slot_end = slot_start + slot_frames
        start = slot_start + max(0, (slot_frames - len(data)) // 2)
        end = min(total_frames, start + len(data))
        source_frames = max(0, end - start)
        if source_frames:
            timeline[start:end] += data[:source_frames]
        placements.append(
            {
                "index": item["index"],
                "text": item["text"],
                "slot_start_sec": round(slot_start / sample_rate, 6),
                "slot_end_sec": round(slot_end / sample_rate, 6),
                "placed_start_sec": round(start / sample_rate, 6),
                "placed_end_sec": round(end / sample_rate, 6),
                "overlaps_previous": start < previous_end,
                "clipped_at_timeline_end": source_frames < len(data),
            }
        )
        previous_end = max(previous_end, end)

    peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if peak > 0.98:
        timeline *= 0.98 / peak
    write_audio(output_path, timeline, sample_rate)
    return {
        "path": str(output_path),
        "duration_sec": round(duration_sec(output_path), 6),
        "sample_rate": sample_rate,
        "slot_sec": round(slot_sec, 6),
        "peak_dbfs": round(peak_dbfs(output_path), 3),
        "rms_dbfs": round(rms_dbfs(output_path), 3),
        "placements": placements,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    log_path = out_dir / "gpt_sovits_server.log"
    manager = ManagedGPTSoVITSServer(
        enabled=bool(args.auto_start),
        base_url=args.url,
        log_path=log_path,
        startup_timeout_sec=args.startup_timeout_sec,
    )
    with manager:
        synthesize_tokens(args)


if __name__ == "__main__":
    main()
