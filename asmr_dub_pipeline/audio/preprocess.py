from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg
from .features import ensure_stereo, load_audio, resample_linear, to_mono, write_audio

_NUMBERED_PART_RE = re.compile(r"^(?P<base>.+)_(?P<part>[1-9][0-9]*)$")


@dataclass(frozen=True)
class NumberedPartPlan:
    requested_path: Path
    base_stem: str | None
    parts: tuple[Path, ...]
    status: str
    reason: str

    @property
    def should_merge(self) -> bool:
        return self.status == "merge"


def numbered_part_base_stem(input_path: Path) -> str | None:
    match = _NUMBERED_PART_RE.match(input_path.stem)
    return match.group("base") if match else None


def plan_numbered_part_merge(input_path: Path) -> NumberedPartPlan:
    input_path = input_path.expanduser().resolve()
    base_stem = numbered_part_base_stem(input_path)
    if base_stem is None:
        return NumberedPartPlan(
            requested_path=input_path,
            base_stem=None,
            parts=(input_path,),
            status="not_numbered_part",
            reason="Input filename does not end with _<number>.",
        )
    base_path = input_path.with_name(base_stem + input_path.suffix)
    if base_path.exists():
        return NumberedPartPlan(
            requested_path=input_path,
            base_stem=base_stem,
            parts=(input_path,),
            status="ambiguous_base_file_exists",
            reason=f"Refusing automatic merge because base file also exists: {base_path}",
        )
    indexed: dict[int, Path] = {}
    for sibling in input_path.parent.iterdir():
        if not sibling.is_file() or sibling.suffix.lower() != input_path.suffix.lower():
            continue
        match = _NUMBERED_PART_RE.match(sibling.stem)
        if not match or match.group("base") != base_stem:
            continue
        indexed[int(match.group("part"))] = sibling.resolve()
    if 1 not in indexed:
        return NumberedPartPlan(
            requested_path=input_path,
            base_stem=base_stem,
            parts=(input_path,),
            status="missing_first_part",
            reason=f"Cannot merge numbered parts for {base_stem}: missing _1 file.",
        )
    ordered_indexes = sorted(indexed)
    expected_indexes = list(range(1, ordered_indexes[-1] + 1))
    if ordered_indexes != expected_indexes:
        missing = sorted(set(expected_indexes) - set(ordered_indexes))
        return NumberedPartPlan(
            requested_path=input_path,
            base_stem=base_stem,
            parts=tuple(indexed[index] for index in ordered_indexes),
            status="missing_numbered_part",
            reason=f"Cannot merge numbered parts for {base_stem}: missing part(s) {missing}.",
        )
    parts = tuple(indexed[index] for index in ordered_indexes)
    if len(parts) < 2:
        return NumberedPartPlan(
            requested_path=input_path,
            base_stem=base_stem,
            parts=parts,
            status="single_numbered_part",
            reason="Only one numbered part was found.",
        )
    return NumberedPartPlan(
        requested_path=input_path,
        base_stem=base_stem,
        parts=parts,
        status="merge",
        reason=f"Found {len(parts)} consecutive numbered parts.",
    )


def merge_numbered_parts_to_audio(
    plan: NumberedPartPlan,
    project_dir: Path,
    *,
    sample_rate: int = 48_000,
    channels: int = 2,
) -> Path:
    if not plan.should_merge or plan.base_stem is None:
        raise ValueError(f"Numbered part plan is not mergeable: {plan.status}")
    output = project_dir / "work" / "input" / f"{plan.base_stem}_merged_source.wav"
    return ffmpeg.concat_audio_to_wav(
        list(plan.parts),
        output,
        sample_rate=sample_rate,
        channels=channels,
    )


def extract_project_audio(input_path: Path, project_dir: Path) -> tuple[Path, Path]:
    stereo = project_dir / "work" / "audio" / "original_stereo_48k.wav"
    mono = project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    try:
        ffmpeg.extract_stereo_48k(input_path, stereo)
        ffmpeg.extract_mono_16k(input_path, mono)
    except ffmpeg.FFmpegError:
        data, sr = load_audio(input_path)
        stereo_data = ensure_stereo(resample_linear(data, sr, 48_000))
        mono_data = resample_linear(to_mono(data), sr, 16_000)
        write_audio(stereo, stereo_data, 48_000)
        write_audio(mono, mono_data[:, None] if mono_data.ndim == 1 else mono_data, 16_000)
    return stereo, mono


def probe_with_fallback(input_path: Path):
    try:
        return ffmpeg.probe_media(input_path)
    except ffmpeg.FFmpegError:
        from asmr_dub_pipeline.schemas import SourceInfo

        data, sr = load_audio(input_path)
        return SourceInfo(
            path=str(input_path),
            duration_sec=len(data) / sr,
            sample_rate=sr,
            channels=data.shape[1],
            codec="wav",
            format_name=input_path.suffix.lstrip(".") or "audio",
            has_video=False,
            raw={},
        )
