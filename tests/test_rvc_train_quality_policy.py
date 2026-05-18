from __future__ import annotations

from pathlib import Path

import pytest

from asmr_dub_pipeline.pipeline.stages import common
from asmr_dub_pipeline.schemas import ProjectConfig


def _dataset_row(
    tmp_path: Path,
    index: int,
    *,
    duration_sec: float = 120.0,
    speaker_id: str = "speaker_0001",
    quality_score: float = 0.82,
    estimated_snr_db: float = 28.0,
    background_bleed_db: float = -34.0,
    side_to_mid_db: float = -16.0,
    source_chars_per_sec: float = 3.4,
) -> dict[str, object]:
    segment_id = f"seg_{index:04d}"
    return {
        "segment_id": segment_id,
        "speaker_id": speaker_id,
        "source_path": str(tmp_path / f"{segment_id}.wav"),
        "dataset_path": str(tmp_path / "dataset" / f"{segment_id}.wav"),
        "duration_sec": duration_sec,
        "quality_score": quality_score,
        "estimated_snr_db": estimated_snr_db,
        "background_bleed_db": background_bleed_db,
        "side_to_mid_db": side_to_mid_db,
        "source_chars_per_sec": source_chars_per_sec,
        "training_rank_score": quality_score,
    }


def _patch_duration_from_rows(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    durations = {Path(str(row["source_path"])).name: float(row["duration_sec"]) for row in rows}
    monkeypatch.setattr(common, "duration_sec", lambda path: durations[Path(path).name])


def test_rvc_dataset_summary_grades_excellent_and_recommends_high_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_dataset_row(tmp_path, index) for index in range(1, 7)]
    _patch_duration_from_rows(monkeypatch, rows)

    summary = common._rvc_training_dataset_summary(rows, ProjectConfig())

    assert summary["quality_grade"] == "excellent"
    assert summary["recommended_epoch_count"] == 160
    assert summary["clean_duration_sec"] == pytest.approx(720.0)
    assert summary["quality_score_stats"]["median"] == pytest.approx(0.82)
    assert summary["background_bleed_db_stats"]["median"] == pytest.approx(-34.0)
    assert summary["side_to_mid_db_stats"]["median"] == pytest.approx(-16.0)
    assert summary["dominant_speaker_id"] == "speaker_0001"
    assert summary["dominant_speaker_ratio"] == pytest.approx(1.0)


def test_rvc_dataset_summary_grades_mixed_when_quality_shape_is_not_good(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _dataset_row(
            tmp_path,
            index,
            duration_sec=120.0,
            background_bleed_db=-24.0,
            side_to_mid_db=-5.0,
        )
        for index in range(1, 7)
    ]
    _patch_duration_from_rows(monkeypatch, rows)

    summary = common._rvc_training_dataset_summary(rows, ProjectConfig())

    assert summary["quality_grade"] == "mixed"
    assert summary["recommended_epoch_count"] == 60


def test_rvc_effective_epoch_config_keeps_fixed_policy() -> None:
    cfg = ProjectConfig(rvc_train_epoch_policy="fixed", rvc_train_epochs=20)

    effective_cfg, decision = common._rvc_train_effective_epoch_config(
        cfg,
        {"quality_grade": "excellent", "recommended_epoch_count": 160},
    )

    assert effective_cfg.rvc_train_epochs == 20
    assert decision["policy"] == "fixed"
    assert decision["configured_epochs"] == 20
    assert decision["effective_epochs"] == 20
    assert decision["recommended_epoch_count"] == 160


def test_rvc_effective_epoch_config_uses_auto_recommendation_with_clamps() -> None:
    cfg = ProjectConfig(
        rvc_train_epoch_policy="auto",
        rvc_train_epochs=20,
        rvc_train_auto_epoch_min=80,
        rvc_train_auto_epoch_max=140,
    )

    effective_cfg, decision = common._rvc_train_effective_epoch_config(
        cfg,
        {"quality_grade": "excellent", "recommended_epoch_count": 160},
    )

    assert effective_cfg.rvc_train_epochs == 140
    assert decision["policy"] == "auto"
    assert decision["configured_epochs"] == 20
    assert decision["effective_epochs"] == 140
    assert decision["recommended_epoch_count"] == 160


def test_rvc_strict_policy_rejects_low_quality_and_bleed() -> None:
    cfg = ProjectConfig(
        rvc_train_quality_preset="strict",
        rvc_train_max_clip_sec=8.0,
        rvc_train_min_snr_db=22.0,
        rvc_train_max_background_bleed_db=-30.0,
        rvc_train_max_side_to_mid_db=-12.0,
    )
    row = {
        "duration_sec": 12.0,
        "quality_score": 0.59,
        "estimated_snr_db": 18.5,
        "background_bleed_db": -20.0,
        "side_to_mid_db": -6.0,
    }

    assert common._rvc_train_policy_reject_reasons(row, cfg) == (
        "rvc_train_duration_sec_above_max:12.000>8.000",
        "rvc_train_quality_score_below_strict_min:0.590<0.600",
        "rvc_train_estimated_snr_db_below_min:18.500<22.000",
        "rvc_train_background_bleed_db_above_max:-20.000>-30.000",
        "rvc_train_side_to_mid_db_above_max:-6.000>-12.000",
    )


def test_rvc_quality_policy_flat_config_fields_migrate_to_nested() -> None:
    cfg = ProjectConfig.model_validate(
        {
            "project_name": "legacy",
            "rvc_train_epoch_policy": "auto",
            "rvc_train_quality_preset": "strict",
            "rvc_train_target_clean_sec": 900.0,
            "rvc_train_auto_epoch_min": 80,
            "rvc_train_auto_epoch_max": 180,
        }
    )

    assert cfg.rvc.train_epoch_policy == "auto"
    assert cfg.rvc.train_quality_preset == "strict"
    assert cfg.rvc_train_epoch_policy == "auto"
    assert cfg.rvc_train_target_clean_sec == pytest.approx(900.0)


def test_rvc_low_data_above_absolute_minimum_is_trainable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration = 393.16 / 91
    rows = [_dataset_row(tmp_path, index, duration_sec=duration) for index in range(1, 92)]
    _patch_duration_from_rows(monkeypatch, rows)
    cfg = ProjectConfig(
        rvc_train_backend="command",
        rvc_train_target_clean_sec=600.0,
        rvc_train_absolute_min_clean_sec=180.0,
        rvc_train_min_clean_segments=1,
    )

    summary = common._rvc_training_dataset_summary(rows, cfg)

    assert summary["insufficient"] is False
    assert summary["trainable"] is True
    assert summary["low_data_mode"] is True
    assert summary["clean_duration_sec"] == pytest.approx(393.16)
    assert summary["target_clean_sec"] == pytest.approx(600.0)
    assert summary["absolute_min_clean_sec"] == pytest.approx(180.0)
    assert summary["official_recommended_min_sec"] == 600
    assert "below_recommended_clean_duration" in summary["low_data_warning"]
    assert summary["trainability_reason"] == "low_data_mode"


def test_rvc_below_absolute_minimum_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [_dataset_row(tmp_path, index, duration_sec=50.0) for index in range(1, 4)]
    _patch_duration_from_rows(monkeypatch, rows)
    cfg = ProjectConfig(
        rvc_train_backend="command",
        rvc_train_target_clean_sec=600.0,
        rvc_train_absolute_min_clean_sec=180.0,
        rvc_train_min_clean_segments=1,
    )

    summary = common._rvc_training_dataset_summary(rows, cfg)

    assert summary["insufficient"] is True
    assert summary["trainable"] is False
    assert summary["low_data_mode"] is False
    assert summary["trainability_reason"] == "clean_duration_below_absolute_min"
    assert summary["insufficient_reasons"] == ["clean_duration_sec_below_absolute_min:150<180"]


def test_rvc_low_data_epoch_scaling_increases_recommendation_under_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration = 393.16 / 91
    rows = [_dataset_row(tmp_path, index, duration_sec=duration) for index in range(1, 92)]
    _patch_duration_from_rows(monkeypatch, rows)
    cfg = ProjectConfig(
        rvc_train_backend="command",
        rvc_train_epoch_policy="fixed",
        rvc_train_epochs=60,
        rvc_train_auto_epoch_min=60,
        rvc_train_auto_epoch_max=300,
        rvc_train_low_data_max_epochs=90,
        rvc_train_target_clean_sec=600.0,
        rvc_train_absolute_min_clean_sec=180.0,
        rvc_train_min_clean_segments=1,
    )
    summary = common._rvc_training_dataset_summary(rows, cfg)

    effective_cfg, decision = common._rvc_train_effective_epoch_config(cfg, summary)

    assert decision["low_data_mode"] is True
    assert decision["low_data_scale"] == pytest.approx((600.0 / 393.16) ** 0.5)
    assert decision["recommended_epoch_count_low_data"] > summary["recommended_epoch_count"]
    assert decision["effective_epochs"] == decision["recommended_epoch_count_low_data"]
    assert decision["effective_epochs"] <= 90
    assert decision["effective_epoch_reason"] == "low_data_scaled"
    assert decision["configured_train_epochs"] == 60
    assert decision["final_train_epochs"] == decision["effective_epochs"]
    assert decision["base_recommended_epoch_count"] == summary["recommended_epoch_count"]
    assert decision["recommended_epoch_count_low_data"] == summary["recommended_epoch_count_low_data"]
    assert summary["configured_train_epochs"] == 60
    assert summary["final_train_epochs"] == decision["effective_epochs"]
    assert summary["official_recommended_min_sec"] == 600
    assert effective_cfg.rvc_train_epochs == decision["effective_epochs"]
