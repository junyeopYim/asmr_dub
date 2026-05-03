"""Stage entrypoints for the ASMR dubbing pipeline."""

# ruff: noqa: I001

from asmr_dub_pipeline.pipeline.stages.project import init_project
from asmr_dub_pipeline.pipeline.stages.project import inspect_input
from asmr_dub_pipeline.pipeline.stages.extract import run_extract_stage
from asmr_dub_pipeline.pipeline.stages.source_separation import run_source_separation_stage
from asmr_dub_pipeline.pipeline.stages.voice_bank_cache import run_import_voice_bank_source_separation_cache_stage
from asmr_dub_pipeline.pipeline.stages.segment import run_segment_stage
from asmr_dub_pipeline.pipeline.stages.transcribe import run_transcribe_stage
from asmr_dub_pipeline.pipeline.stages.analyze import run_analyze_stage
from asmr_dub_pipeline.pipeline.stages.script import run_script_stage
from asmr_dub_pipeline.pipeline.stages.translate_ko import run_translate_ko_stage
from asmr_dub_pipeline.pipeline.stages.korean_script import run_korean_script_stage
from asmr_dub_pipeline.pipeline.stages.speaker_assignment import run_assign_speakers_stage
from asmr_dub_pipeline.pipeline.stages.voice_refs import run_prepare_source_voice_refs_stage
from asmr_dub_pipeline.pipeline.stages.gsv_few_shot import run_gsv_few_shot_stage
from asmr_dub_pipeline.pipeline.stages.synth_gpt_sovits import run_synth_stage
from asmr_dub_pipeline.pipeline.stages.synth_qwen import run_synth_qwen_stage
from asmr_dub_pipeline.pipeline.stages.rvc_train import run_rvc_train_stage
from asmr_dub_pipeline.pipeline.stages.rvc_train import run_skip_rvc_train_for_voice_bank_stage
from asmr_dub_pipeline.pipeline.stages.rvc import run_rvc_stage
from asmr_dub_pipeline.pipeline.stages.qc import run_qc_stage
from asmr_dub_pipeline.pipeline.stages.experimental_tts import run_synth_experimental_tts_stage
from asmr_dub_pipeline.pipeline.stages.regenerate import run_regenerate_needs_stage
from asmr_dub_pipeline.pipeline.stages.mix import run_mix_stage
from asmr_dub_pipeline.pipeline.stages.export import run_export_stage

__all__ = [
    "init_project",
    "inspect_input",
    "run_extract_stage",
    "run_source_separation_stage",
    "run_import_voice_bank_source_separation_cache_stage",
    "run_segment_stage",
    "run_transcribe_stage",
    "run_analyze_stage",
    "run_script_stage",
    "run_translate_ko_stage",
    "run_korean_script_stage",
    "run_assign_speakers_stage",
    "run_prepare_source_voice_refs_stage",
    "run_gsv_few_shot_stage",
    "run_synth_stage",
    "run_synth_qwen_stage",
    "run_rvc_train_stage",
    "run_skip_rvc_train_for_voice_bank_stage",
    "run_rvc_stage",
    "run_qc_stage",
    "run_synth_experimental_tts_stage",
    "run_regenerate_needs_stage",
    "run_mix_stage",
    "run_export_stage",
]
