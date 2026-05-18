from __future__ import annotations

from pathlib import Path

import pytest

from asmr_dub_pipeline.pipeline.stages import common
from asmr_dub_pipeline.schemas import ProjectConfig


def test_train_rvc_low_data_epoch_decision_records_effective_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for index in range(1, 92):
        segment_id = f"seg_{index:04d}"
        rows.append(
            {
                "segment_id": segment_id,
                "speaker_id": "speaker_0001",
                "source_path": str(tmp_path / f"{segment_id}.wav"),
                "dataset_path": str(tmp_path / "dataset" / f"{segment_id}.wav"),
                "duration_sec": 393.16 / 91,
                "quality_score": 0.82,
                "estimated_snr_db": 28.0,
                "background_bleed_db": -34.0,
                "side_to_mid_db": -16.0,
                "source_chars_per_sec": 3.4,
                "training_rank_score": 0.82,
            }
        )
    durations = {Path(str(row["source_path"])).name: float(row["duration_sec"]) for row in rows}
    monkeypatch.setattr(common, "duration_sec", lambda path: durations[Path(path).name])
    cfg = ProjectConfig(
        rvc_train_backend="command",
        rvc_train_epoch_policy="auto",
        rvc_train_auto_epoch_min=80,
        rvc_train_auto_epoch_max=300,
        rvc_train_target_clean_sec=600.0,
        rvc_train_absolute_min_clean_sec=180.0,
    )

    summary = common._rvc_training_dataset_summary(rows, cfg)
    effective_cfg, decision = common._rvc_train_effective_epoch_config(cfg, summary)

    assert summary["low_data_mode"] is True
    assert summary["effective_train_epochs"] == decision["effective_epochs"]
    assert summary["effective_epoch_reason"] == "low_data_scaled"
    assert effective_cfg.rvc_train_epochs == decision["effective_epochs"]
