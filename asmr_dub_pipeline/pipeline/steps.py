from __future__ import annotations

# ruff: noqa: F401,F403,F405,I001

import sys
import types

from asmr_dub_pipeline.pipeline.context import PipelineContext
from asmr_dub_pipeline.pipeline.stages.common import *
from asmr_dub_pipeline.pipeline.stages.project import init_project
from asmr_dub_pipeline.pipeline.stages.project import inspect_input
from asmr_dub_pipeline.pipeline.stages.extract import run_extract_stage
from asmr_dub_pipeline.pipeline.stages.source_separation import run_source_separation_stage
from asmr_dub_pipeline.pipeline.stages.voice_bank_cache import run_import_voice_bank_source_separation_cache_stage
from asmr_dub_pipeline.pipeline.stages.segment import run_segment_stage
from asmr_dub_pipeline.pipeline.stages.transcribe import run_transcribe_stage
from asmr_dub_pipeline.pipeline.stages.analyze import run_analyze_stage
from asmr_dub_pipeline.pipeline.stages.audio_style import run_audio_style_stage
from asmr_dub_pipeline.pipeline.stages.script import run_script_stage
from asmr_dub_pipeline.pipeline.stages.translate_ko import run_translate_ko_stage
from asmr_dub_pipeline.pipeline.stages.korean_script import run_korean_script_stage
from asmr_dub_pipeline.pipeline.stages.source_speakers import run_source_speakers_stage
from asmr_dub_pipeline.pipeline.stages.speaker_assignment import run_assign_speakers_stage
from asmr_dub_pipeline.pipeline.stages.voice_refs import run_prepare_source_voice_refs_stage
from asmr_dub_pipeline.pipeline.stages.gsv_few_shot import run_gsv_few_shot_stage
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import run_countdown_synth_stage
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import run_synth_stage
from asmr_dub_pipeline.pipeline.stages.synth_qwen import run_synth_qwen_stage
from asmr_dub_pipeline.pipeline.stages.tts_candidates import run_tts_candidate_pool_stage
from asmr_dub_pipeline.pipeline.stages.tts_candidates import run_tts_select_stage
from asmr_dub_pipeline.pipeline.stages.rvc_train import run_rvc_train_stage
from asmr_dub_pipeline.pipeline.stages.rvc_train import run_skip_rvc_train_for_voice_bank_stage
from asmr_dub_pipeline.pipeline.stages.rvc import run_rvc_stage
from asmr_dub_pipeline.pipeline.stages.qc import run_qc_stage
from asmr_dub_pipeline.pipeline.stages.experimental_tts import run_synth_experimental_tts_stage
from asmr_dub_pipeline.pipeline.stages.regenerate import run_regenerate_needs_stage
from asmr_dub_pipeline.pipeline.stages.auto_repair import run_auto_repair_stage
from asmr_dub_pipeline.pipeline.stages.mix import run_mix_stage
from asmr_dub_pipeline.pipeline.stages.export import run_export_stage

from asmr_dub_pipeline.pipeline.stages import analyze as _analyze_stage
from asmr_dub_pipeline.pipeline.stages import audio_style as _audio_style_stage
from asmr_dub_pipeline.pipeline.stages import auto_repair as _auto_repair_stage
from asmr_dub_pipeline.pipeline.stages import experimental_tts as _experimental_tts_stage
from asmr_dub_pipeline.pipeline.stages import export as _export_stage
from asmr_dub_pipeline.pipeline.stages import extract as _extract_stage
from asmr_dub_pipeline.pipeline.stages import gsv_few_shot as _gsv_few_shot_stage
from asmr_dub_pipeline.pipeline.stages import korean_script as _korean_script_stage
from asmr_dub_pipeline.pipeline.stages import mix as _mix_stage
from asmr_dub_pipeline.pipeline.stages import project as _project_stage
from asmr_dub_pipeline.pipeline.stages import qc as _qc_stage
from asmr_dub_pipeline.pipeline.stages import regenerate as _regenerate_stage
from asmr_dub_pipeline.pipeline.stages import rvc as _rvc_stage
from asmr_dub_pipeline.pipeline.stages import rvc_train as _rvc_train_stage
from asmr_dub_pipeline.pipeline.stages import script as _script_stage
from asmr_dub_pipeline.pipeline.stages import segment as _segment_stage
from asmr_dub_pipeline.pipeline.stages import source_separation as _source_separation_stage
from asmr_dub_pipeline.pipeline.stages import source_speakers as _source_speakers_stage
from asmr_dub_pipeline.pipeline.stages import speaker_assignment as _speaker_assignment_stage
from asmr_dub_pipeline.pipeline.stages import synth_gpt_sovits as _synth_gpt_sovits_stage
from asmr_dub_pipeline.pipeline.stages import synth_qwen as _synth_qwen_stage
from asmr_dub_pipeline.pipeline.stages import tts_candidates as _tts_candidates_stage
from asmr_dub_pipeline.pipeline.stages import transcribe as _transcribe_stage
from asmr_dub_pipeline.pipeline.stages import translate_ko as _translate_ko_stage
from asmr_dub_pipeline.pipeline.stages import voice_bank_cache as _voice_bank_cache_stage
from asmr_dub_pipeline.pipeline.stages import voice_refs as _voice_refs_stage
from asmr_dub_pipeline.pipeline.stages import common as _common_stage

def extract_step(input_path: Path, project_dir: Path, confirm_rights: bool, merge_parts: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_extract_stage(ctx, input_path, confirm_rights, merge_parts)

def source_separation_step(project_dir: Path, confirm_rights: bool = False, force: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_source_separation_stage(ctx, confirm_rights, force)

def import_voice_bank_source_separation_cache_step(project_dir: Path, input_path: Path, cache_project_dir: Path) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_import_voice_bank_source_separation_cache_stage(ctx, input_path, cache_project_dir)

def segment_step(project_dir: Path, confirm_rights: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_segment_stage(ctx, confirm_rights)

def transcribe_step(project_dir: Path, asr_backend: str | None = None, confirm_rights: bool = False, asr_review: bool | None = None, asr_preset: str | None = None, asr_vad_off: bool | None = None, asr_diagnostics: bool | None = None, asr_device: str | None = None, asr_compute_type: str | None = None, asr_batched_inference: bool | None = None, asr_batch_size: int | None = None, asr_repair_enabled: bool | None = None, asr_backend_factory: Any | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_transcribe_stage(ctx, asr_backend, confirm_rights, asr_review, asr_preset, asr_vad_off, asr_diagnostics, asr_device, asr_compute_type, asr_batched_inference, asr_batch_size, asr_repair_enabled, asr_backend_factory)

def analyze_step(project_dir: Path, backend_kind: str, model_id: str | None = None, confirm_rights: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_analyze_stage(ctx, backend_kind, model_id, confirm_rights)

def audio_style_step(project_dir: Path, backend_kind: str, model_id: str | None = None, confirm_rights: bool = False, force: bool = False, scope: str = "all") -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_audio_style_stage(ctx, backend_kind, model_id, confirm_rights, force, scope)

def script_step(project_dir: Path, backend_kind: str, confirm_rights: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_script_stage(ctx, backend_kind, confirm_rights)

def translate_ko_step(
    project_dir: Path,
    gemma_text_backend: str | None = None,
    confirm_rights: bool = False,
    force_retranslate: bool = False,
    retry_failed: bool = False,
    repair_only: bool = False,
    force_retranslate_failed: bool = False,
    *,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_translate_ko_stage(
        ctx,
        gemma_text_backend,
        confirm_rights,
        force_retranslate,
        retry_failed,
        repair_only,
        force_retranslate_failed,
        only_segment_ids=only_segment_ids,
    )

def korean_script_step(
    project_dir: Path,
    confirm_rights: bool = False,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_korean_script_stage(ctx, confirm_rights, only_segment_ids)

def source_speakers_step(project_dir: Path, backend_kind: str | None = None, confirm_rights: bool = False, jobs: int = 4) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_source_speakers_stage(ctx, backend_kind, confirm_rights, jobs)

def assign_speakers_step(project_dir: Path, voice_bank_path: Path | None = None, backend_kind: str | None = None, require_all: bool = True) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_assign_speakers_stage(ctx, voice_bank_path, backend_kind, require_all)

def prepare_source_voice_refs_step(project_dir: Path, refs_path: Path | None = None, confirm_rights: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_prepare_source_voice_refs_stage(ctx, refs_path, confirm_rights)

def gsv_few_shot_step(project_dir: Path, confirm_rights: bool = False, force: bool | None = None, gsv_url: str | None = None, gsv_server_command: list[str] | str | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_gsv_few_shot_stage(ctx, confirm_rights, force, gsv_url, gsv_server_command)

def synth_step(project_dir: Path, gsv_url: str | None, refs_path: Path, mock: bool = False, confirm_rights: bool = False, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, use_trained_gpt: bool = False, only_segment_ids: set[str] | None = None, retry_failed: bool = False, force: bool = False, render_countdowns: bool = True) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_synth_stage(ctx, gsv_url, refs_path, mock, confirm_rights, gpt_weights_path, sovits_weights_path, auto_gsv_server, gsv_server_command, use_trained_gpt, only_segment_ids, retry_failed, force, render_countdowns=render_countdowns)

def countdown_synth_step(project_dir: Path, gsv_url: str | None, refs_path: Path, mock: bool = False, confirm_rights: bool = False, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, use_trained_gpt: bool = False, only_segment_ids: set[str] | None = None, retry_failed: bool = False, force: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_countdown_synth_stage(ctx, gsv_url, refs_path, mock, confirm_rights, gpt_weights_path, sovits_weights_path, auto_gsv_server, gsv_server_command, use_trained_gpt, only_segment_ids, retry_failed, force)

def synth_qwen_step(project_dir: Path, refs_path: Path, confirm_rights: bool = False, *, model_id: str | None = None, candidate_count: int | None = None, candidate_batch_size: int | None = None, segment_batch_size: int | None = None, target_vram_gb: float | None = None, promote: bool = False, local_files_only: bool | None = None, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_synth_qwen_stage(ctx, refs_path, confirm_rights, model_id=model_id, candidate_count=candidate_count, candidate_batch_size=candidate_batch_size, segment_batch_size=segment_batch_size, target_vram_gb=target_vram_gb, promote=promote, local_files_only=local_files_only, only_segment_ids=only_segment_ids)

def tts_candidate_pool_step(project_dir: Path, *, refs_path: Path = Path("refs/refs.json"), confirm_rights: bool = False, requested_backend: str = "auto", gsv_url: str | None = None, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, use_trained_gpt: bool = False, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, qwen_model_id: str | None = None, qwen_candidate_count: int | None = None, qwen_local_files_only: bool | None = None, only_segment_ids: set[str] | None = None, mock: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_tts_candidate_pool_stage(ctx, refs_path=refs_path, confirm_rights=confirm_rights, requested_backend=requested_backend, gsv_url=gsv_url, gpt_weights_path=gpt_weights_path, sovits_weights_path=sovits_weights_path, use_trained_gpt=use_trained_gpt, auto_gsv_server=auto_gsv_server, gsv_server_command=gsv_server_command, qwen_model_id=qwen_model_id, qwen_candidate_count=qwen_candidate_count, qwen_local_files_only=qwen_local_files_only, only_segment_ids=only_segment_ids, mock=mock)

def tts_select_step(project_dir: Path, *, only_segment_ids: set[str] | None = None, force: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_tts_select_stage(ctx, only_segment_ids=only_segment_ids, force=force)

def rvc_train_step(project_dir: Path, confirm_rights: bool = False, force: bool = False, mock: bool | None = None, runner: Any | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_rvc_train_stage(ctx, confirm_rights, force, mock, runner)

def skip_rvc_train_for_voice_bank_step(project_dir: Path) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_skip_rvc_train_for_voice_bank_stage(ctx)

def rvc_step(project_dir: Path, confirm_rights: bool = False, force: bool = False, mock: bool | None = None, runner: Any | None = None, only_segment_ids: set[str] | None = None, retry_failed: bool = False) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_rvc_stage(ctx, confirm_rights, force, mock, runner, only_segment_ids, retry_failed)

def qc_step(project_dir: Path, backend_kind: str, confirm_rights: bool = False, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_qc_stage(ctx, backend_kind, confirm_rights, only_segment_ids)

def synth_experimental_tts_step(project_dir: Path, refs_path: Path, *, backend: str, confirm_rights: bool = False, base_url: str | None = None, candidate_count: int | None = None, promote: bool = False, only_segment_ids: set[str] | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_synth_experimental_tts_stage(ctx, refs_path, backend=backend, confirm_rights=confirm_rights, base_url=base_url, candidate_count=candidate_count, promote=promote, only_segment_ids=only_segment_ids)

def regenerate_needs_step(project_dir: Path, *, refs_path: Path = Path('refs/refs.json'), confirm_rights: bool = False, gemma_backend: str = 'mock', tts_backend: str = 'gpt-sovits', gsv_url: str | None = None, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, use_trained_gpt: bool = False, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None, qwen_model_id: str | None = None, qwen_candidate_count: int | None = None, qwen_local_files_only: bool | None = None, experimental_tts_base_url: str | None = None, experimental_tts_candidate_count: int | None = None, only_segment_ids: set[str] | None = None, target_segment_ids: set[str] | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_regenerate_needs_stage(ctx, refs_path=refs_path, confirm_rights=confirm_rights, gemma_backend=gemma_backend, tts_backend=tts_backend, gsv_url=gsv_url, gpt_weights_path=gpt_weights_path, sovits_weights_path=sovits_weights_path, use_trained_gpt=use_trained_gpt, auto_gsv_server=auto_gsv_server, gsv_server_command=gsv_server_command, qwen_model_id=qwen_model_id, qwen_candidate_count=qwen_candidate_count, qwen_local_files_only=qwen_local_files_only, experimental_tts_base_url=experimental_tts_base_url, experimental_tts_candidate_count=experimental_tts_candidate_count, only_segment_ids=only_segment_ids, target_segment_ids=target_segment_ids)

def auto_repair_step(project_dir: Path, *, refs_path: Path = Path("refs/refs.json"), confirm_rights: bool = False, max_attempts: int | None = None, plan_only: bool = False, only_segment_ids: set[str] | None = None, gemma_backend: str = "mock", tts_backend: str = "gpt-sovits", gsv_url: str | None = None, gpt_weights_path: str | None = None, sovits_weights_path: str | None = None, use_trained_gpt: bool = False, auto_gsv_server: bool | None = None, gsv_server_command: list[str] | str | None = None) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_auto_repair_stage(ctx, refs_path=refs_path, confirm_rights=confirm_rights, max_attempts=max_attempts, plan_only=plan_only, only_segment_ids=only_segment_ids, gemma_backend=gemma_backend, tts_backend=tts_backend, gsv_url=gsv_url, gpt_weights_path=gpt_weights_path, sovits_weights_path=sovits_weights_path, use_trained_gpt=use_trained_gpt, auto_gsv_server=auto_gsv_server, gsv_server_command=gsv_server_command)

def mix_step(project_dir: Path, confirm_rights: bool) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_mix_stage(ctx, confirm_rights)

def export_step(input_path: Path, project_dir: Path, confirm_rights: bool) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_export_stage(ctx, input_path, confirm_rights)


_COMPAT_MODULES = (_common_stage, _analyze_stage, _audio_style_stage, _auto_repair_stage, _experimental_tts_stage, _export_stage, _extract_stage, _gsv_few_shot_stage, _korean_script_stage, _mix_stage, _project_stage, _qc_stage, _regenerate_stage, _rvc_stage, _rvc_train_stage, _script_stage, _segment_stage, _source_separation_stage, _source_speakers_stage, _speaker_assignment_stage, _synth_gpt_sovits_stage, _synth_qwen_stage, _tts_candidates_stage, _transcribe_stage, _translate_ko_stage, _voice_bank_cache_stage, _voice_refs_stage,)

class _StepsCompatModule(types.ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _COMPAT_MODULES:
            if hasattr(module, name):
                setattr(module, name, value)

sys.modules[__name__].__class__ = _StepsCompatModule
