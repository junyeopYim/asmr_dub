from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from conftest import write_tiny_wav

from asmr_dub_pipeline.config import save_project_config
from asmr_dub_pipeline.gpt_sovits import few_shot
from asmr_dub_pipeline.gpt_sovits.client import build_tts_request
from asmr_dub_pipeline.gpt_sovits.few_shot import (
    build_training_dataset,
    discover_install,
    train_few_shot,
)
from asmr_dub_pipeline.pipeline import steps
from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest
from asmr_dub_pipeline.pipeline.steps import init_project, synth_step
from asmr_dub_pipeline.schemas import (
    GSVSpeakerConfig,
    JapaneseScript,
    PipelineManifest,
    ProjectConfig,
    Segment,
    SourceScript,
)


def _segment(project_dir: Path, segment_id: str, start: float, duration: float, text: str) -> Segment:
    audio = project_dir / "work" / "segments" / "audio" / f"{segment_id}_mix.wav"
    write_tiny_wav(audio, duration=duration)
    return Segment(
        id=segment_id,
        start=start,
        end=start + duration,
        duration=duration,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio.relative_to(project_dir)),
        source_script=SourceScript(
            text=text,
            language="ja",
            backend="mock",
            start=start,
            end=start + duration,
        ),
    )


def _manifest_with_segments(project_dir: Path, durations: list[float]) -> PipelineManifest:
    segments = [
        _segment(project_dir, f"seg_{index:04d}", index * 10.0, duration, f"台詞 {index}")
        for index, duration in enumerate(durations, start=1)
    ]
    return PipelineManifest(segments=segments)


def _fake_gsv_install(root: Path) -> Path:
    api = root / "api_v2.py"
    configs = root / "GPT_SoVITS" / "configs"
    pretrained = root / "GPT_SoVITS" / "pretrained_models"
    configs.mkdir(parents=True)
    pretrained.mkdir(parents=True)
    api.write_text("", "utf-8")
    (pretrained / "s1v3.ckpt").write_bytes(b"gpt-base")
    (pretrained / "s2Gv4.pth").write_bytes(b"sovits-base")
    (pretrained / "bert").mkdir()
    (pretrained / "hubert").mkdir()
    (configs / "tts_infer.yaml").write_text(
        yaml.safe_dump(
            {
                "custom": {
                    "version": "v4",
                    "bert_base_path": "GPT_SoVITS/pretrained_models/bert",
                    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/hubert",
                    "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
                    "vits_weights_path": "GPT_SoVITS/pretrained_models/s2Gv4.pth",
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    (configs / "s1longer-v2.yaml").write_text(
        yaml.safe_dump({"train": {}, "data": {}, "model": {}}, sort_keys=True),
        "utf-8",
    )
    (configs / "s2.json").write_text(
        json.dumps({"train": {}, "data": {}, "model": {}}, sort_keys=True),
        "utf-8",
    )
    return api


def test_discover_install_uses_tts_config_from_server_command(tmp_project_dir: Path) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    default_config = api.parent / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    default_payload = yaml.safe_load(default_config.read_text("utf-8"))
    default_payload["custom"]["vits_weights_path"] = "GPT_SoVITS/pretrained_models/missing.pth"
    default_config.write_text(yaml.safe_dump(default_payload, sort_keys=True), "utf-8")
    command_config = tmp_project_dir / "tts_infer.local.yaml"
    command_config.write_text(
        yaml.safe_dump(
            {
                "custom": {
                    "version": "v4",
                    "bert_base_path": "GPT_SoVITS/pretrained_models/bert",
                    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/hubert",
                    "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
                    "vits_weights_path": "GPT_SoVITS/pretrained_models/s2Gv4.pth",
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    command = ["python", str(api), "-c", str(command_config)]

    install = discover_install(
        ProjectConfig(project_name="test", gsv_server_command=command),
        command=command,
    )

    assert install.tts_config_path == command_config.resolve()
    assert install.pretrained_sovits_path.name == "s2Gv4.pth"


def test_few_shot_training_python_skips_project_venv_missing_deps(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(project_name="test", gsv_server_command=["python", str(api)])
    install = discover_install(cfg)
    base_python = tmp_project_dir / "conda" / "bin" / "python"
    venv_python = tmp_project_dir / ".venv" / "bin" / "python"
    monkeypatch.setenv("PATH", str(venv_python.parent))
    monkeypatch.setattr(few_shot.sys, "executable", str(venv_python))
    monkeypatch.setattr(few_shot.sys, "base_prefix", str(base_python.parent.parent))

    def fake_missing_imports(python: str, modules) -> list[str] | None:
        if python == str(base_python):
            return []
        if python in {"python", str(venv_python)}:
            return ["transformers", "librosa"]
        return None

    monkeypatch.setattr(few_shot, "_python_missing_imports", fake_missing_imports)

    selected = few_shot._select_training_python(cfg, install, None, require_modules=True)

    assert selected == str(base_python)


def test_few_shot_training_python_env_override(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(project_name="test", gsv_server_command=["python", str(api)])
    install = discover_install(cfg)
    override = tmp_project_dir / "gsv_env" / "bin" / "python"
    monkeypatch.setenv("ASMR_DUB_GSV_PYTHON", str(override))
    monkeypatch.setattr(few_shot, "_python_missing_imports", lambda python, modules: [])

    selected = few_shot._select_training_python(cfg, install, None, require_modules=True)

    assert selected == str(override)


def test_few_shot_dataset_selects_source_segments(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 0.5, 1.4])
    cfg = ProjectConfig(
        project_name="test",
        gsv_few_shot_target_sec=2.0,
        gsv_few_shot_min_clip_sec=1.0,
        gsv_few_shot_max_clip_sec=10.0,
    )

    dataset = build_training_dataset(tmp_project_dir, manifest, cfg)

    assert [item.segment_id for item in dataset.items] == ["seg_0001", "seg_0003"]
    assert dataset.total_duration_sec == pytest.approx(2.4)
    lines = dataset.list_path.read_text("utf-8").splitlines()
    assert lines == [
        "seg_0001.wav|source_voice|ja|台詞 1",
        "seg_0003.wav|source_voice|ja|台詞 3",
    ]
    assert (dataset.wav_dir / "seg_0001.wav").exists()
    qc = json.loads((tmp_project_dir / "work/gpt_sovits/few_shot/source_clip_qc.json").read_text("utf-8"))
    assert qc["clips"][0]["source_language"] == "ja"
    assert qc["clips"][0]["target_language"] == "ko"
    assert qc["clips"][0]["quality_score"] > 0


def test_few_shot_dataset_rejects_too_little_source_voice(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    manifest = _manifest_with_segments(tmp_project_dir, [1.0])
    cfg = ProjectConfig(project_name="test", gsv_few_shot_target_sec=2.0)

    with pytest.raises(Exception, match="Not enough source voice data"):
        build_training_dataset(tmp_project_dir, manifest, cfg)


def test_few_shot_training_runs_commands_and_reuses_matching_weights(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(
        project_name="test",
        gsv_url="http://127.0.0.1:9880",
        gsv_server_command=["python", str(api)],
        gsv_few_shot_target_sec=2.0,
    )
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.2])
    commands: list[list[str]] = []
    progress_events: list[tuple[str, str, int, int, str | None]] = []

    def runner(command, **kwargs):
        commands.append(list(command))
        env_paths = kwargs["env"]["PYTHONPATH"].split(":")
        assert str(api.parent) in env_paths
        assert str(api.parent / "GPT_SoVITS") in env_paths
        if any("s2_train_v3_lora.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/sovits/final.pth"
            weights.write_bytes(b"sovits-trained")
        if any("s1_train.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/gpt/final.ckpt"
            weights.write_bytes(b"gpt-trained")
        return subprocess.CompletedProcess(command, 0)

    def record_progress(event) -> None:
        progress_events.append((event.phase, event.status, event.index, event.total, event.detail))

    first = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner, progress_callback=record_progress)
    second = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner, progress_callback=record_progress)
    forced = train_few_shot(
        tmp_project_dir,
        manifest,
        cfg,
        force=True,
        runner=runner,
        progress_callback=record_progress,
    )

    assert first.status == "completed"
    assert second.status == "skipped"
    assert second.reused_existing is True
    assert forced.status == "completed"
    assert [Path(command[2]).name for command in commands[:4]] == [
        "1-get-text.py",
        "2-get-hubert-wav32k.py",
        "3-get-semantic.py",
        "s2_train_v3_lora.py",
    ]
    assert sum(any("s1_train.py" in part for part in command) for command in commands) == 2
    assert ("dataset", "completed", 0, 5, "selected=2 duration=2.20s") in progress_events
    assert any(event[:4] == ("fine-tune-sovits", "started", 4, 5) for event in progress_events)
    assert any(event[:4] == ("fine-tune-gpt", "completed", 5, 5) for event in progress_events)
    assert any(event[:2] == ("reuse", "skipped") for event in progress_events)


def test_run_logged_streams_training_output(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    events = []
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('epoch 1\\r'); sys.stdout.flush(); "
        "sys.stdout.write('saving final\\n'); sys.stdout.flush()",
    ]

    few_shot._run_logged(
        subprocess.run,
        command,
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=log_path,
        phase="fine-tune-gpt",
        index=1,
        total=1,
        progress_callback=events.append,
    )

    output_events = [event.detail for event in events if event.status == "output"]
    assert "epoch 1" in output_events
    assert "saving final" in output_events
    assert "saving final" in log_path.read_text("utf-8")


def test_few_shot_fingerprint_changes_when_training_audio_changes(tmp_project_dir: Path) -> None:
    init_project(tmp_project_dir)
    api = _fake_gsv_install(tmp_project_dir / "gsv")
    cfg = ProjectConfig(
        project_name="test",
        gsv_url="http://127.0.0.1:9880",
        gsv_server_command=["python", str(api)],
        gsv_few_shot_target_sec=2.0,
    )
    manifest = _manifest_with_segments(tmp_project_dir, [1.0, 1.2])

    def runner(command, **_kwargs):
        if any("s2_train_v3_lora.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/sovits/final.pth"
            weights.write_bytes(b"sovits-trained")
        if any("s1_train.py" in part for part in command):
            weights = tmp_project_dir / "work/gpt_sovits/few_shot/weights/gpt/final.ckpt"
            weights.write_bytes(b"gpt-trained")
        return subprocess.CompletedProcess(command, 0)

    first = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner)
    changed_audio = tmp_project_dir / manifest.segments[0].audio_for_mix
    write_tiny_wav(changed_audio, duration=1.0, sample_rate=44_100)
    second = train_few_shot(tmp_project_dir, manifest, cfg, runner=runner)

    assert first.fingerprint != second.fingerprint
    assert second.reused_existing is False


def test_synth_loads_few_shot_weights_from_manifest(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", gsv_gpt_weights_policy="few_shot")
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    gpt = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "gpt" / "final.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    gpt.parent.mkdir(parents=True, exist_ok=True)
    sovits.parent.mkdir(parents=True, exist_ok=True)
    gpt.write_bytes(b"gpt")
    sovits.write_bytes(b"sovits")
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(text="テスト", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="テスト",
            ja_text="テスト",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    manifest = PipelineManifest(
        segments=[segment],
        artifacts={
            "gsv_few_shot_gpt_weights": str(gpt),
            "gsv_few_shot_sovits_weights": str(sovits),
        },
    )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, str]] = []
    prompt_texts: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            prompt_texts.append(ref.prompt_text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    assert calls[:2] == [("gpt", str(gpt)), ("sovits", str(sovits))]
    assert prompt_texts and prompt_texts[0]


@pytest.mark.parametrize("use_trained_gpt", [False, True])
def test_synth_gpt_selection_for_korean_few_shot_voice(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    use_trained_gpt: bool,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test")
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    gpt = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "gpt" / "final.ckpt"
    sovits = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "weights" / "sovits" / "final.pth"
    training_manifest = tmp_project_dir / "work" / "gpt_sovits" / "few_shot" / "training_manifest.json"
    for path, payload in (
        (base_gpt, b"base-gpt"),
        (gpt, b"few-shot-gpt"),
        (sovits, b"few-shot-sovits"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    training_manifest.write_text(
        json.dumps(
            {
                "fingerprint_payload": {
                    "gpt_sovits": {
                        "pretrained_gpt_path": str(base_gpt),
                    }
                }
            },
            sort_keys=True,
        ),
        "utf-8",
    )
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=1.2,
        duration=1.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
        script=JapaneseScript(
            literal_ja="こんにちは",
            ja_text="こんにちは",
            tts_text="안녕하세요",
            tts_language="ko",
            source_language="ja",
            target_language="ko",
        ),
    )
    manifest = PipelineManifest(
        segments=[segment],
        artifacts={
            "gsv_few_shot_gpt_weights": str(gpt),
            "gsv_few_shot_sovits_weights": str(sovits),
            "gsv_few_shot_manifest": str(training_manifest),
        },
    )
    save_manifest(tmp_project_dir, manifest)
    calls: list[tuple[str, str]] = []
    payloads: list[dict[str, object]] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payloads.append(request.as_payload())
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
        use_trained_gpt=use_trained_gpt,
    )

    expected_gpt = gpt if use_trained_gpt else base_gpt
    assert calls[:2] == [("gpt", str(expected_gpt)), ("sovits", str(sovits))]
    assert payloads[0]["text"] == "안녕하세요"
    assert payloads[0]["text_lang"] == "all_ko"
    assert payloads[0]["prompt_lang"] == "all_ja"
    assert payloads[0]["prompt_text"]


def test_synth_switches_gpt_and_sovits_weights_per_voice_bank_speaker(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    base_gpt = tmp_project_dir / "gsv" / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt"
    base_gpt.parent.mkdir(parents=True, exist_ok=True)
    base_gpt.write_bytes(b"base-gpt")
    speaker_models: dict[str, GSVSpeakerConfig] = {}
    for speaker_id, prompt in (("speaker_0001", "こんにちは"), ("speaker_0002", "おやすみ")):
        speaker_dir = tmp_project_dir / "voice_bank" / "speakers" / speaker_id
        gpt = speaker_dir / "gsv" / "v001" / "gpt.ckpt"
        sovits = speaker_dir / "gsv" / "v001" / "final.pth"
        ref_audio = speaker_dir / "refs" / "whisper_close.wav"
        refs_json = speaker_dir / "refs" / "refs.json"
        sovits.parent.mkdir(parents=True, exist_ok=True)
        gpt.write_bytes(f"{speaker_id}-gpt".encode())
        sovits.write_bytes(speaker_id.encode())
        write_tiny_wav(ref_audio)
        refs_json.write_text(
            json.dumps(
                {
                    "whisper_close": {
                        "ref_audio_path": str(ref_audio.relative_to(tmp_project_dir)),
                        "prompt_text": prompt,
                        "prompt_lang": "ja",
                    }
                }
            ),
            "utf-8",
        )
        speaker_models[speaker_id] = GSVSpeakerConfig(
            gpt_weights_path=str(gpt.relative_to(tmp_project_dir)),
            sovits_weights_path=str(sovits.relative_to(tmp_project_dir)),
            refs_path=str(refs_json.relative_to(tmp_project_dir)),
        )
    cfg = ProjectConfig(
        project_name="test",
        gsv_concurrency=1,
        gsv_gpt_weights_path=str(base_gpt),
        gsv_speaker_models=speaker_models,
    )
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    segments: list[Segment] = []
    for index, speaker_id in enumerate(("speaker_0001", "speaker_0002", "speaker_0001"), start=1):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"seg_{index:04d}_mix.wav"
        write_tiny_wav(audio)
        segments.append(
            Segment(
                id=f"seg_{index:04d}",
                speaker_id=speaker_id,
                start=(index - 1) * 1.2,
                end=index * 1.2,
                duration=1.2,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(text="こんにちは", language="ja", backend="mock", start=0.0, end=1.2),
                script=JapaneseScript(
                    literal_ja="こんにちは",
                    ja_text="こんにちは",
                    tts_text=f"안녕하세요 {index}",
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                ),
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))
    calls: list[tuple[str, str]] = []
    prompt_texts: list[str] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            calls.append(("gpt", path))
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            calls.append(("sovits", path))
            return "success"

        def build_payload(self, text, ref, options=None):
            prompt_texts.append(ref.prompt_text)
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=Path("refs/refs.json"),
        mock=False,
        confirm_rights=True,
    )

    assert calls == [
        ("gpt", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "gpt.ckpt")),
        ("sovits", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "final.pth")),
        ("gpt", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0002" / "gsv" / "v001" / "gpt.ckpt")),
        ("sovits", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0002" / "gsv" / "v001" / "final.pth")),
        ("gpt", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "gpt.ckpt")),
        ("sovits", str(tmp_project_dir / "voice_bank" / "speakers" / "speaker_0001" / "gsv" / "v001" / "final.pth")),
    ]
    assert prompt_texts == ["こんにちは", "おやすみ", "こんにちは"]
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["synth"]["model_switch"]["gpt_weights_mode"] == "speaker_voice_bank"
    assert manifest.stage_state["synth"]["model_switch"]["sovits_weights_mode"] == "speaker_voice_bank"


def test_synth_rewrites_korean_duration_before_unbounded_speedup(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", gsv_tts_max_speed_factor=1.05)
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    audio = tmp_project_dir / "work" / "segments" / "audio" / "seg_0001_mix.wav"
    write_tiny_wav(audio)
    long_text = "수고하셨습니다 오늘도 정말 잘하셨어요"
    segment = Segment(
        id="seg_0001",
        start=0.0,
        end=0.2,
        duration=0.2,
        audio_for_gemma=str(audio),
        audio_for_mix=str(audio),
        script=JapaneseScript(ja_text="原文", tts_text=long_text, tts_language="ko"),
    )
    save_manifest(tmp_project_dir, PipelineManifest(segments=[segment]))
    payload_texts: list[str] = []
    payload_speeds: list[float] = []

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            request = build_tts_request(text, ref, options)
            payload_texts.append(request.text)
            payload_speeds.append(request.speed_factor)
            return request

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            write_tiny_wav(output_path, duration=1.2 if request.text == long_text else 0.2)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://gsv.local",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    manifest = load_manifest(tmp_project_dir)
    scripted = manifest.segments[0].script
    assert scripted is not None
    assert scripted.tts_text != long_text
    assert scripted.tts_text.endswith(".")
    assert "。" not in scripted.tts_text
    assert "duration_rewrite_shortened" in scripted.risk_flags
    assert payload_texts[:2] == [long_text, long_text]
    assert payload_texts[-1] == scripted.tts_text
    assert all(speed <= cfg.gsv_tts_max_speed_factor for speed in payload_speeds)
    selected = next(candidate for candidate in manifest.segments[0].tts.candidates if candidate.selected)
    assert selected.payload["text"] == scripted.tts_text


def test_synth_uses_three_gsv_lanes_by_segment_id(
    tmp_project_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_project(tmp_project_dir)
    cfg = ProjectConfig(project_name="test", gsv_concurrency=3)
    save_project_config(cfg, tmp_project_dir / "pipeline.yaml")
    ref_audio = tmp_project_dir / "refs" / "whisper_close.wav"
    write_tiny_wav(ref_audio)
    segments: list[Segment] = []
    for index in range(1, 7):
        audio = tmp_project_dir / "work" / "segments" / "audio" / f"seg_{index:04d}_mix.wav"
        write_tiny_wav(audio)
        source_text = f"テスト {index}"
        tts_text = f"테스트 {index}"
        segments.append(
            Segment(
                id=f"seg_{index:04d}",
                start=(index - 1) * 1.2,
                end=index * 1.2,
                duration=1.2,
                audio_for_gemma=str(audio),
                audio_for_mix=str(audio),
                source_script=SourceScript(
                    text=source_text,
                    language="ja",
                    backend="mock",
                    start=(index - 1) * 1.2,
                    end=index * 1.2,
                ),
                script=JapaneseScript(
                    literal_ja=source_text,
                    ja_text=source_text,
                    tts_text=tts_text,
                    tts_language="ko",
                    source_language="ja",
                    target_language="ko",
                ),
            )
        )
    save_manifest(tmp_project_dir, PipelineManifest(segments=segments))
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, base_url: str, *args: object, **kwargs: object) -> None:
            self.base_url = base_url

        def set_gpt_weights(self, path: str) -> str:
            return "success"

        def set_sovits_weights(self, path: str) -> str:
            return "success"

        def build_payload(self, text, ref, options=None):
            return build_tts_request(text, ref, options)

        def synthesize_to_file(self, request, output_path: Path) -> Path:
            calls.append((request.text, self.base_url))
            write_tiny_wav(output_path)
            return output_path

    monkeypatch.setattr(steps, "GPTSoVITSClient", FakeClient)

    synth_step(
        tmp_project_dir,
        gsv_url="http://127.0.0.1:9880",
        refs_path=tmp_project_dir / "refs" / "refs.json",
        mock=False,
        confirm_rights=True,
    )

    by_text = dict(calls)
    assert by_text["테스트 1"] == "http://127.0.0.1:9880"
    assert by_text["테스트 2"] == "http://127.0.0.1:9881"
    assert by_text["테스트 3"] == "http://127.0.0.1:9882"
    assert by_text["테스트 4"] == "http://127.0.0.1:9880"
    assert by_text["테스트 5"] == "http://127.0.0.1:9881"
    assert by_text["테스트 6"] == "http://127.0.0.1:9882"
    manifest = load_manifest(tmp_project_dir)
    assert manifest.stage_state["synth"]["concurrency"] == 3


def test_semantic_parts_merge_with_upstream_header(tmp_project_dir: Path) -> None:
    dataset_dir = tmp_project_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "6-name2semantic-0.tsv").write_text("seg_0001\t1 2 3\n", "utf-8")
    (dataset_dir / "6-name2semantic-1.tsv").write_text("seg_0002\t4 5 6\n", "utf-8")

    from asmr_dub_pipeline.gpt_sovits import few_shot

    few_shot._merge_dataset_part_outputs(dataset_dir)

    assert (dataset_dir / "6-name2semantic.tsv").read_text("utf-8").splitlines() == [
        "item_name\tsemantic_audio",
        "seg_0001\t1 2 3",
        "seg_0002\t4 5 6",
    ]


def test_pandas_shim_default_header_matches_gpt_sovits_semantics(tmp_project_dir: Path) -> None:
    from asmr_dub_pipeline.gpt_sovits.shims import pandas

    tmp_project_dir.mkdir(parents=True)
    csv_path = tmp_project_dir / "semantic.tsv"
    csv_path.write_text("item_name\tsemantic_audio\nseg_0001\t1 2 3\n", "utf-8")

    default_frame = pandas.read_csv(csv_path, delimiter="\t")
    raw_frame = pandas.read_csv(csv_path, delimiter="\t", header=None)

    assert len(default_frame) == 1
    assert default_frame.iloc[0, 0] == "seg_0001"
    assert len(raw_frame) == 2
