from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import ffmpeg
from .features import ensure_stereo, load_audio, resample_linear, to_mono, write_audio

_NUMBERED_PART_RE = re.compile(r"^(?P<base>.+)_(?P<part>[1-9][0-9]*)$")
_NATURAL_SORT_RE = re.compile(r"(\d+)")
_MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
_FOLDER_ASR_CLEAN_TOKENS = (
    "効果音無し",
    "効果音なし",
    "効果音無",
    "効果音ぬき",
    "効果音抜き",
    "se無し",
    "seなし",
    "se無",
    "音無し",
    "音なし",
    "音無",
    "no_se",
    "nose",
    "without_se",
    "without se",
)
_FOLDER_EXCLUDE_TOKENS = (
    "readme",
    "sample",
    "trial",
    "体験版",
    "サンプル",
    "samplevoice",
    "pv",
    "予告",
    "注意事項",
)
_FOLDER_MAIN_TOKENS = (
    "本編",
    "メイン",
    "main",
    "voice",
    "音声",
    "wav",
    "mp3",
)
_FOLDER_SIDE_TOKENS = (
    "おまけ",
    "オマケ",
    "bonus",
    "特典",
    "フリートーク",
    "キャストトーク",
)
_FILENAME_TRANSLATIONS_KO = (
    ("プロローグ", "프롤로그"),
    ("エピローグ", "에필로그"),
    ("オープニング", "오프닝"),
    ("エンディング", "엔딩"),
    ("催眠誘導", "최면유도"),
    ("催眠深化", "최면심화"),
    ("催眠解除", "최면해제"),
    ("女体化", "여체화"),
    ("射精管理", "사정관리"),
    ("連続絶頂", "연속절정"),
    ("絶頂", "절정"),
    ("耳舐め", "귀핥기"),
    ("耳かき", "귀청소"),
    ("耳掻き", "귀청소"),
    ("耳責め", "귀자극"),
    ("添い寝", "같이잠"),
    ("安眠", "숙면"),
    ("寝かしつけ", "재우기"),
    ("ささやき", "속삭임"),
    ("囁き", "속삭임"),
    ("吐息", "숨소리"),
    ("効果音無し", "효과음없음"),
    ("効果音なし", "효과음없음"),
    ("効果音無", "효과음없음"),
    ("本編", "본편"),
    ("音声", "음성"),
    ("誘導", "유도"),
    ("深化", "심화"),
    ("解除", "해제"),
    ("注意事項", "주의사항"),
    ("おまけ", "보너스"),
    ("特典", "특전"),
    ("対男性", "대남성"),
    ("対ふたなり", "대후타나리"),
)


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


@dataclass(frozen=True)
class FolderInputPlan:
    requested_path: Path
    mix_parts: tuple[Path, ...]
    asr_parts: tuple[Path, ...]
    status: str
    reason: str
    asr_source_status: str

    @property
    def should_prepare(self) -> bool:
        return self.status == "planned"


def _natural_path_key(path: Path) -> tuple[object, ...]:
    normalized = unicodedata.normalize("NFKC", str(path).lower())
    pieces: list[object] = []
    for piece in _NATURAL_SORT_RE.split(normalized):
        pieces.append(int(piece) if piece.isdigit() else piece)
    return tuple(pieces)


def _path_text(path: Path) -> str:
    return unicodedata.normalize("NFKC", str(path).replace("\\", "/").lower())


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    normalized = unicodedata.normalize("NFKC", text.lower())
    return any(token.lower() in normalized for token in tokens)


def _is_media_file(path: Path) -> bool:
    name = path.name
    if name.endswith(":Zone.Identifier"):
        return False
    return path.suffix.lower() in _MEDIA_EXTENSIONS


def _collect_folder_media(input_path: Path) -> tuple[Path, ...]:
    files = []
    for path in input_path.rglob("*"):
        if not path.is_file() or not _is_media_file(path):
            continue
        relative_path = path.relative_to(input_path)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        files.append(path.resolve())
    files = [
        path
        for path in files
        if not _contains_any(_path_text(path.relative_to(input_path)), _FOLDER_EXCLUDE_TOKENS)
    ]
    return tuple(sorted(files, key=_natural_path_key))


def _folder_group_key(input_path: Path, media_path: Path) -> Path:
    try:
        return media_path.parent.resolve().relative_to(input_path.resolve())
    except ValueError:
        return media_path.parent.resolve()


def _probe_duration_or_zero(path: Path) -> float:
    try:
        return float(probe_with_fallback(path).duration_sec)
    except Exception:
        return 0.0


def _score_mix_group(group_key: Path, parts: tuple[Path, ...]) -> tuple[float, int, float, str]:
    text = _path_text(group_key)
    score = 0.0
    if _contains_any(text, _FOLDER_MAIN_TOKENS):
        score += 40.0
    if _contains_any(text, _FOLDER_ASR_CLEAN_TOKENS):
        score -= 80.0
    if _contains_any(text, _FOLDER_SIDE_TOKENS):
        score -= 30.0
    total_duration = sum(_probe_duration_or_zero(path) for path in parts)
    score += min(len(parts), 20) * 2.0
    score += min(total_duration, 600.0) / 600.0
    return (score, len(parts), total_duration, str(group_key))


def _score_asr_group(
    group_key: Path,
    parts: tuple[Path, ...],
    *,
    mix_count: int,
    mix_duration: float,
) -> tuple[float, int, float, str]:
    text = _path_text(group_key)
    total_duration = sum(_probe_duration_or_zero(path) for path in parts)
    score = 0.0
    if _contains_any(text, _FOLDER_ASR_CLEAN_TOKENS):
        score += 80.0
    if len(parts) == mix_count:
        score += 20.0
    if mix_duration > 0:
        ratio = total_duration / mix_duration
        if 0.85 <= ratio <= 1.15:
            score += 20.0
        else:
            score -= min(abs(1.0 - ratio) * 50.0, 40.0)
    if _contains_any(text, _FOLDER_SIDE_TOKENS):
        score -= 20.0
    return (score, len(parts), total_duration, str(group_key))


def _group_media_by_parent(input_path: Path, files: tuple[Path, ...]) -> dict[Path, tuple[Path, ...]]:
    grouped: dict[Path, list[Path]] = {}
    for path in files:
        grouped.setdefault(_folder_group_key(input_path, path), []).append(path)
    return {
        key: tuple(sorted(paths, key=_natural_path_key))
        for key, paths in grouped.items()
    }


def plan_folder_input(input_path: Path) -> FolderInputPlan:
    input_path = input_path.expanduser().resolve()
    if not input_path.is_dir():
        return FolderInputPlan(
            requested_path=input_path,
            mix_parts=(),
            asr_parts=(),
            status="not_folder",
            reason="Input path is not a directory.",
            asr_source_status="not_applicable",
        )
    files = _collect_folder_media(input_path)
    if not files:
        return FolderInputPlan(
            requested_path=input_path,
            mix_parts=(),
            asr_parts=(),
            status="no_media_files",
            reason="No supported media files were found in the input directory.",
            asr_source_status="not_applicable",
        )
    groups = _group_media_by_parent(input_path, files)
    mix_group_key, mix_parts = max(
        groups.items(),
        key=lambda item: _score_mix_group(item[0], item[1]),
    )
    mix_duration = sum(_probe_duration_or_zero(path) for path in mix_parts)
    clean_groups = {
        key: parts
        for key, parts in groups.items()
        if key != mix_group_key and _contains_any(_path_text(key), _FOLDER_ASR_CLEAN_TOKENS)
    }
    if clean_groups:
        asr_group_key, asr_parts = max(
            clean_groups.items(),
            key=lambda item: _score_asr_group(
                item[0],
                item[1],
                mix_count=len(mix_parts),
                mix_duration=mix_duration,
            ),
        )
        asr_duration = sum(_probe_duration_or_zero(path) for path in asr_parts)
        if len(asr_parts) == len(mix_parts) or not mix_duration or 0.75 <= asr_duration / mix_duration <= 1.25:
            return FolderInputPlan(
                requested_path=input_path,
                mix_parts=tuple(mix_parts),
                asr_parts=tuple(asr_parts),
                status="planned",
                reason=f"Selected mix group '{mix_group_key}' and clean ASR group '{asr_group_key}'.",
                asr_source_status="separate_asr_parts",
            )
    return FolderInputPlan(
        requested_path=input_path,
        mix_parts=tuple(mix_parts),
        asr_parts=tuple(mix_parts),
        status="planned",
        reason=f"Selected mix group '{mix_group_key}' for both mix and ASR.",
        asr_source_status="mix_parts",
    )


def translate_media_stem_to_korean(stem: str) -> str:
    translated = unicodedata.normalize("NFKC", stem).strip()
    for source, target in _FILENAME_TRANSLATIONS_KO:
        translated = translated.replace(source, target)
    translated = re.sub(r"[\\/:*?\"<>|]+", "_", translated)
    translated = re.sub(r"\s+", "_", translated)
    translated = re.sub(r"_+", "_", translated)
    translated = translated.strip(" ._")
    return translated or "part"


def _part_metadata(parts: tuple[Path, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cursor = 0.0
    for index, path in enumerate(parts, start=1):
        info = probe_with_fallback(path)
        duration = float(info.duration_sec)
        end = cursor + duration
        records.append(
            {
                "part_index": index,
                "path": str(path),
                "duration_sec": duration,
                "start_sec": cursor,
                "end_sec": end,
                "stem": path.stem,
                "translated_stem_ko": translate_media_stem_to_korean(path.stem),
                "has_video": info.has_video,
                "sample_rate": info.sample_rate,
                "channels": info.channels,
                "codec": info.codec,
            }
        )
        cursor = end
    return records


def folder_input_metadata(
    plan: FolderInputPlan,
    *,
    mix_audio_path: Path,
    asr_audio_path: Path,
) -> dict[str, Any]:
    mix_parts = _part_metadata(plan.mix_parts)
    asr_parts = _part_metadata(plan.asr_parts)
    total_duration = mix_parts[-1]["end_sec"] if mix_parts else 0.0
    return {
        "requested": True,
        "input_kind": "folder",
        "status": plan.status,
        "reason": plan.reason,
        "requested_path": str(plan.requested_path),
        "selected_input_path": str(mix_audio_path),
        "selected_asr_input_path": str(asr_audio_path),
        "asr_source_status": plan.asr_source_status,
        "part_count": len(mix_parts),
        "mix_parts": mix_parts,
        "asr_parts": asr_parts,
        "total_part_duration_sec": total_duration,
    }


def _convert_audio_file(
    input_path: Path,
    output_path: Path,
    *,
    sample_rate: int,
    channels: int,
) -> Path:
    try:
        if sample_rate == 48_000 and channels == 2:
            return ffmpeg.extract_stereo_48k(input_path, output_path)
        if sample_rate == 16_000 and channels == 1:
            return ffmpeg.extract_mono_16k(input_path, output_path)
    except ffmpeg.FFmpegError:
        pass
    data, sr = load_audio(input_path)
    if channels == 1:
        converted = resample_linear(to_mono(data), sr, sample_rate)
        if converted.ndim == 1:
            converted = converted[:, None]
    else:
        converted = ensure_stereo(resample_linear(data, sr, sample_rate))
    write_audio(output_path, converted, sample_rate)
    return output_path


def _concat_or_convert_parts(
    parts: tuple[Path, ...],
    output_path: Path,
    *,
    sample_rate: int,
    channels: int,
) -> Path:
    if not parts:
        raise ValueError("At least one media file is required.")
    if len(parts) == 1:
        return _convert_audio_file(parts[0], output_path, sample_rate=sample_rate, channels=channels)
    try:
        return ffmpeg.concat_audio_to_wav(
            list(parts),
            output_path,
            sample_rate=sample_rate,
            channels=channels,
        )
    except ffmpeg.FFmpegError:
        clips = []
        for part in parts:
            data, sr = load_audio(part)
            if channels == 1:
                converted = resample_linear(to_mono(data), sr, sample_rate)
                if converted.ndim == 1:
                    converted = converted[:, None]
            else:
                converted = ensure_stereo(resample_linear(data, sr, sample_rate))
            clips.append(converted)
        import numpy as np

        write_audio(output_path, np.concatenate(clips, axis=0), sample_rate)
        return output_path


def prepare_folder_input_audio(plan: FolderInputPlan, project_dir: Path) -> tuple[Path, Path]:
    if not plan.should_prepare:
        raise ValueError(plan.reason)
    stereo = project_dir / "work" / "audio" / "original_stereo_48k.wav"
    mono = project_dir / "work" / "audio" / "gemma_mono_16k.wav"
    _concat_or_convert_parts(plan.mix_parts, stereo, sample_rate=48_000, channels=2)
    _concat_or_convert_parts(plan.asr_parts, mono, sample_rate=16_000, channels=1)
    return stereo, mono


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
