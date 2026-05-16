# Countdown Recovery And Hybrid Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `runs/20260510T174455Z_RJ372458`에서 mock 번역으로 오염된 52개 임베디드 카운트다운 세그먼트만 복구하고, 일반 synth와 hybrid countdown renderer 중 어떤 경로가 실제 발화 타이밍을 맞추는지 실험으로 판정합니다.

**Architecture:** 복구는 전체 2083개를 재처리하지 않고 오염 마커가 있는 52개만 reset, real translate, targeted Korean script, targeted synth/RVC로 진행합니다. 실험은 복구된 번역을 기준으로 plain synth, forced-number script, hybrid splice 세 경로를 같은 샘플에 적용하고 ASR 발음/숫자별 타이밍 지표로 비교합니다.

**Tech Stack:** Python 3.11, Typer CLI, Pydantic manifest, pytest, jq, GPT-SoVITS, Gemma llama_server, existing countdown source-anchor metadata.

---

## 현재 증거

- 프로젝트: `runs/20260510T174455Z_RJ372458`
- 오염 대상: `analysis.embedded_countdown_translation_repair` 마커가 있는 52개
- 현재 오염 상태:
  - `mock_translation_count`: 52
  - `polluted_tts_count`: 5
  - `no_tts_count`: 47
  - 52개 모두 `translation_ko.model == "mock"`
  - mock 문장: `자연 번역: 부드럽게 속삭여 드릴게요.`
- 추가 손상:
  - 기존 정상 RVC 2031개의 `rvc` metadata와 output은 남아 있지만, 전체 `korean-script` 재실행 때문에 status가 `scripted`로 내려와 있음
  - 이 2031개는 `rvc.accepted == true`와 `rvc.output_path` 존재 여부를 확인한 뒤 `rvc_converted`로 status만 복원해야 함
- 서버 상태:
  - GPT-SoVITS `9880/9881/9882` 서버는 종료 완료
  - `8080`, `9880-9885`에 ASMR pipeline 서버 없음

## 복구 원칙

- 전체 `--force-retranslate` 금지: 2083개 전체 번역 오염 위험이 큼
- 전체 `korean-script` CLI 재실행 금지: 기존 정상 2031개의 status를 다시 내릴 수 있음
- 전체 `rvc` CLI 재실행 금지: 기존 정상 RVC 산출물을 불필요하게 다시 건드릴 수 있음
- 52개만 targeted reset/translate/script/synth/RVC
- 복구 중 생성되는 잘못된 TTS 파일은 manifest에서 분리하고, 필요하면 별도 snapshot에만 보존

---

## Task 1: 상태 스냅샷과 오염 대상 고정

**Files:**
- Create: `runs/20260510T174455Z_RJ372458/work/recovery_snapshots/<timestamp>/manifest.before_recovery.json`
- Create: `runs/20260510T174455Z_RJ372458/work/recovery_snapshots/<timestamp>/polluted_segments.json`
- Create: `runs/20260510T174455Z_RJ372458/work/recovery_snapshots/<timestamp>/polluted_segments.tsv`

- [ ] **Step 1: 서버가 내려갔는지 확인**

Run:

```bash
ss -ltnp | rg ':(8080|9880|9881|9882|9883|9884|9885)\b|api_v2|llama' || true
```

Expected:

```text
# no ASMR pipeline server rows
```

- [ ] **Step 2: recovery snapshot 생성**

Run:

```bash
PROJECT=runs/20260510T174455Z_RJ372458
TS=$(date -u +%Y%m%dT%H%M%SZ)
SNAP="$PROJECT/work/recovery_snapshots/$TS"
mkdir -p "$SNAP"
cp "$PROJECT/work/manifest.json" "$SNAP/manifest.before_recovery.json"
cp "$PROJECT/pipeline.yaml" "$SNAP/pipeline.yaml"
jq '[.segments[] | select(.analysis.embedded_countdown_translation_repair)]' \
  "$PROJECT/work/manifest.json" > "$SNAP/polluted_segments.json"
jq -r '.segments[] | select(.analysis.embedded_countdown_translation_repair) |
  [.id, .status, (.source_script.text // ""), (.translation_ko.model // ""), (.translation_ko.ko_natural // ""), (.script.tts_text // ""), ((.tts.selected_candidate_path? // "") | tostring), ((.rvc.output_path? // "") | tostring)] | @tsv' \
  "$PROJECT/work/manifest.json" > "$SNAP/polluted_segments.tsv"
echo "$SNAP"
```

Expected:

```text
runs/20260510T174455Z_RJ372458/work/recovery_snapshots/<timestamp>
```

- [ ] **Step 3: 오염 범위 검증**

Run:

```bash
PROJECT=runs/20260510T174455Z_RJ372458
jq '{
  repair_marker_count: ([.segments[] | select(.analysis.embedded_countdown_translation_repair)] | length),
  mock_translation_count: ([.segments[] | select(.analysis.embedded_countdown_translation_repair and ((.translation_ko.model // "") == "mock"))] | length),
  polluted_tts_count: ([.segments[] | select(.analysis.embedded_countdown_translation_repair and ((.tts.selected_candidate_path? // "") != ""))] | length),
  accepted_rvc_count: ([.segments[] | select(.analysis.embedded_countdown_translation_repair and (.rvc.accepted == true))] | length)
}' "$PROJECT/work/manifest.json"
```

Expected:

```json
{
  "repair_marker_count": 52,
  "mock_translation_count": 52,
  "polluted_tts_count": 5,
  "accepted_rvc_count": 0
}
```

---

## Task 2: targeted recovery reset 스크립트 작성

**Files:**
- Create: `experiments/countdown_recovery/run_targeted_recovery_reset.py`

- [ ] **Step 1: 스크립트 생성**

Create `experiments/countdown_recovery/run_targeted_recovery_reset.py`:

```python
from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from asmr_dub_pipeline.pipeline.manifest_io import load_manifest, save_manifest

MOCK_TEXT = "자연 번역: 부드럽게 속삭여 드릴게요."


def path_exists(value: str | None) -> bool:
    return bool(value and Path(value).exists())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    manifest = load_manifest(project)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = project / "work" / "recovery_snapshots" / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = project / "work" / "manifest.json"
    shutil.copy2(manifest_path, snapshot_dir / "manifest.before_targeted_reset.json")

    polluted_ids: list[str] = []
    restored_rvc_status_ids: list[str] = []
    preserved_normal_ids: list[str] = []

    for segment in manifest.segments:
        is_polluted = (
            bool(segment.analysis.get("embedded_countdown_translation_repair"))
            and segment.translation_ko is not None
            and segment.translation_ko.model == "mock"
            and segment.translation_ko.ko_natural == MOCK_TEXT
        )
        if is_polluted:
            polluted_ids.append(segment.id)
            segment.status = "needs_manual_review"
            segment.translation_ko = None
            segment.script = None
            segment.tts = None
            segment.rvc = None
            segment.qc = None
            segment.mix = {}
            segment.errors = [
                error
                for error in segment.errors
                if "자연 번역" not in error
                and not error.startswith("GPT-SoVITS synthesis failed")
                and error != "RVC requires segment.tts.selected_candidate_path from synth."
            ]
            marker = "recovery: mock translate-ko contamination; targeted real retranslation required"
            if marker not in segment.errors:
                segment.errors.append(marker)
            segment.analysis["mock_translation_contamination_recovery"] = {
                "detected_at": timestamp,
                "action": "cleared_translation_script_tts_rvc_qc_mix",
            }
            continue

        if (
            segment.status == "scripted"
            and segment.rvc is not None
            and segment.rvc.accepted
            and path_exists(segment.rvc.output_path)
        ):
            segment.status = "rvc_converted"
            restored_rvc_status_ids.append(segment.id)
        else:
            preserved_normal_ids.append(segment.id)

    if len(polluted_ids) != 52:
        raise SystemExit(f"Expected 52 polluted segments, got {len(polluted_ids)}")

    summary = {
        "project": str(project),
        "snapshot_dir": str(snapshot_dir),
        "apply": args.apply,
        "polluted_ids": polluted_ids,
        "polluted_count": len(polluted_ids),
        "restored_rvc_status_count": len(restored_rvc_status_ids),
        "restored_rvc_status_ids_sample": restored_rvc_status_ids[:20],
    }
    (snapshot_dir / "targeted_reset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.apply:
        save_manifest(project, manifest)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: dry-run 실행**

Run:

```bash
uv run python experiments/countdown_recovery/run_targeted_recovery_reset.py \
  --project runs/20260510T174455Z_RJ372458
```

Expected:

```json
{
  "polluted_count": 52,
  "restored_rvc_status_count": 2031
}
```

- [ ] **Step 3: apply 실행**

Run:

```bash
uv run python experiments/countdown_recovery/run_targeted_recovery_reset.py \
  --project runs/20260510T174455Z_RJ372458 \
  --apply
```

Expected:

```json
{
  "polluted_count": 52,
  "restored_rvc_status_count": 2031
}
```

- [ ] **Step 4: reset 결과 검증**

Run:

```bash
PROJECT=runs/20260510T174455Z_RJ372458
jq '{
  counts: ([.segments[].status] | group_by(.) | map({key:.[0], value:length}) | from_entries),
  polluted_needs_review: ([.segments[] | select(.analysis.mock_translation_contamination_recovery and .status=="needs_manual_review")] | length),
  restored_rvc_converted: ([.segments[] | select(.status=="rvc_converted" and .rvc.accepted==true)] | length),
  remaining_mock_polluted: ([.segments[] | select(.analysis.embedded_countdown_translation_repair and ((.translation_ko.model // "") == "mock"))] | length)
}' "$PROJECT/work/manifest.json"
```

Expected:

```json
{
  "polluted_needs_review": 52,
  "restored_rvc_converted": 2031,
  "remaining_mock_polluted": 0
}
```

---

## Task 3: real translate-ko로 52개만 재번역

**Files:**
- Modify by pipeline: `runs/20260510T174455Z_RJ372458/work/manifest.json`
- Modify by pipeline: `runs/20260510T174455Z_RJ372458/work/translate_ko/*`

- [ ] **Step 1: real translation 실행**

Run:

```bash
uv run asmr-dub translate-ko \
  --project runs/20260510T174455Z_RJ372458 \
  --gemma-text-backend llama_server \
  --retry-failed \
  --force-retranslate-failed \
  --confirm-rights
```

Expected:

```text
translate-ko complete - backend=llama_server
```

- [ ] **Step 2: mock 문장 제거 검증**

Run:

```bash
PROJECT=runs/20260510T174455Z_RJ372458
jq '{
  polluted_transcribed: ([.segments[] | select(.analysis.mock_translation_contamination_recovery and .status=="transcribed")] | length),
  remaining_mock_text: ([.segments[] | select(.analysis.mock_translation_contamination_recovery and ((.translation_ko.ko_natural // "") == "자연 번역: 부드럽게 속삭여 드릴게요."))] | length),
  remaining_mock_model: ([.segments[] | select(.analysis.mock_translation_contamination_recovery and ((.translation_ko.model // "") == "mock"))] | length),
  missing_translation: ([.segments[] | select(.analysis.mock_translation_contamination_recovery and (.translation_ko == null))] | length)
}' "$PROJECT/work/manifest.json"
```

Expected:

```json
{
  "polluted_transcribed": 52,
  "remaining_mock_text": 0,
  "remaining_mock_model": 0,
  "missing_translation": 0
}
```

- [ ] **Step 3: 사람이 읽을 검토 TSV 생성**

Run:

```bash
PROJECT=runs/20260510T174455Z_RJ372458
mkdir -p "$PROJECT/work/recovery_review"
jq -r '.segments[] | select(.analysis.mock_translation_contamination_recovery) |
  [.id, (.source_script.text // ""), (.translation_ko.ko_natural // "")] | @tsv' \
  "$PROJECT/work/manifest.json" > "$PROJECT/work/recovery_review/retranslated_52.tsv"
```

Expected:

```text
runs/20260510T174455Z_RJ372458/work/recovery_review/retranslated_52.tsv exists
```

---

## Task 4: targeted Korean script 지원 추가

**Files:**
- Modify: `asmr_dub_pipeline/pipeline/stages/korean_script.py`
- Modify: `asmr_dub_pipeline/pipeline/steps.py`
- Modify: `asmr_dub_pipeline/cli.py`
- Test: `tests/test_text_translation_lane.py` or new `tests/test_korean_script_targeted.py`

- [ ] **Step 1: failing test 작성**

Test intent:

```python
def test_korean_script_only_segments_preserves_existing_rvc_segment(tmp_path: Path) -> None:
    # seg_keep: status rvc_converted, existing script/tts/rvc
    # seg_target: status transcribed, new translation
    # run korean_script_step(..., only_segment_ids={"seg_target"})
    # assert seg_keep.status == "rvc_converted"
    # assert seg_keep.script unchanged
    # assert seg_target.status == "scripted"
```

Run:

```bash
uv run pytest tests/test_korean_script_targeted.py::test_korean_script_only_segments_preserves_existing_rvc_segment
```

Expected:

```text
FAIL because korean_script_step has no only_segment_ids argument
```

- [ ] **Step 2: implementation**

Required behavior:

```python
def run_korean_script_stage(
    ctx: PipelineContext,
    confirm_rights: bool = False,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    ...
    for index, segment in enumerate(manifest.segments, start=1):
        if only_segment_ids is not None and segment.id not in only_segment_ids:
            continue
        ...
```

Update `steps.py`:

```python
def korean_script_step(
    project_dir: Path,
    confirm_rights: bool = False,
    only_segment_ids: set[str] | None = None,
) -> PipelineManifest:
    ctx = PipelineContext.load(project_dir)
    return run_korean_script_stage(ctx, confirm_rights, only_segment_ids)
```

Update `cli.py`:

```python
only_segments: str | None = typer.Option(
    None,
    "--only-segments",
    help="Comma- or whitespace-separated segment IDs to script.",
)
...
korean_script_step(
    project.expanduser().resolve(),
    confirm_rights=confirm_rights,
    only_segment_ids=_parse_only_segment_ids(only_segments),
)
```

- [ ] **Step 3: test 통과 확인**

Run:

```bash
uv run pytest tests/test_korean_script_targeted.py::test_korean_script_only_segments_preserves_existing_rvc_segment
uv run ruff check asmr_dub_pipeline/pipeline/stages/korean_script.py asmr_dub_pipeline/pipeline/steps.py asmr_dub_pipeline/cli.py tests/test_korean_script_targeted.py
```

Expected:

```text
PASS
All checks passed
```

---

## Task 5: 52개만 Korean script 재생성

**Files:**
- Modify by pipeline: `runs/20260510T174455Z_RJ372458/work/manifest.json`
- Modify by pipeline: `runs/20260510T174455Z_RJ372458/work/segments/manifests/segments_ko_script.json`

- [ ] **Step 1: 오염 ID 목록 생성**

Run:

```bash
PROJECT=runs/20260510T174455Z_RJ372458
IDS=$(jq -r '[.segments[] | select(.analysis.mock_translation_contamination_recovery) | .id] | join(",")' "$PROJECT/work/manifest.json")
echo "$IDS" > "$PROJECT/work/recovery_review/recovery_ids.csv"
```

Expected:

```text
52 comma-separated segment IDs
```

- [ ] **Step 2: targeted Korean script 실행**

Run:

```bash
uv run asmr-dub korean-script \
  --project runs/20260510T174455Z_RJ372458 \
  --confirm-rights \
  --only-segments "$IDS"
```

Expected:

```text
korean-script complete
```

- [ ] **Step 3: 기존 2031개 status 보존 검증**

Run:

```bash
PROJECT=runs/20260510T174455Z_RJ372458
jq '{
  targeted_scripted: ([.segments[] | select(.analysis.mock_translation_contamination_recovery and .status=="scripted")] | length),
  preserved_rvc_converted: ([.segments[] | select((.analysis.mock_translation_contamination_recovery | not) and .status=="rvc_converted" and .rvc.accepted==true)] | length),
  remaining_mock_script: ([.segments[] | select(.analysis.mock_translation_contamination_recovery and ((.script.tts_text // "") == "자연 번역, 부드럽게 속삭여 드릴게요."))] | length)
}' "$PROJECT/work/manifest.json"
```

Expected:

```json
{
  "targeted_scripted": 52,
  "preserved_rvc_converted": 2031,
  "remaining_mock_script": 0
}
```

---

## Task 6: 카운트다운 타이밍 실험 설계와 샘플 선정

**Files:**
- Create: `experiments/countdown_embedded_hybrid/README.md`
- Create: `experiments/countdown_embedded_hybrid/select_samples.py`
- Create: `experiments/countdown_embedded_hybrid/run_eval_matrix.py`

### Hypotheses

- **H1 plain synth pronunciation risk:** 일반 `synth`는 숫자 누락/오발음이 남을 수 있습니다.
- **H2 plain synth timing risk:** 일반 `synth`가 duration gate를 통과해도 숫자별 발화 시점은 source anchor와 크게 어긋날 수 있습니다.
- **H3 prompt/script repair limitation:** 번역과 스크립트에서 숫자를 보존해도 GPT-SoVITS가 숫자별 간격을 원본처럼 유지한다는 보장은 없습니다.
- **H4 hybrid renderer benefit:** 숫자 부분만 countdown token bank/source anchor로 렌더링하고 주변 문장은 일반 synth로 처리하면 숫자별 타이밍 오차와 누락률이 줄어듭니다.

### Sample strata

- `pure-ish`: `4 3 2 1 絶頂します ゼロ`, `9 8 ... 0`
- `prefix_context`: `ほら 3 2 1...`, `あと10 9 8...`
- `suffix_context`: `5 4 3 2 1 ... 行く`
- `long_context`: 앞뒤 문장이 길고 countdown이 중간에 있는 문장
- `interleaved_context`: `4 電子... 3 僕... 2...`

### Metrics

- `number_pronunciation_pass`: ASR 결과가 기대 한국어 숫자 토큰을 순서대로 포함하는지
- `number_missing_count`: 누락 숫자 개수
- `number_extra_count`: 불필요한 숫자 개수
- `mean_abs_timing_error_sec`: source anchor 대비 숫자별 평균 절대 오차
- `max_abs_timing_error_sec`: source anchor 대비 최대 절대 오차
- `gap_cv_delta`: 원본 숫자 간격 변동계수와 합성 숫자 간격 변동계수 차이
- `segment_duration_ratio`: 합성 길이 / 원본 세그먼트 길이
- `manual_review_count`: stage가 manual review 또는 failed로 끝난 개수

### Pass criteria

- pronunciation pass rate >= 0.95
- mean abs timing error <= 0.18 sec
- max abs timing error <= 0.35 sec
- manual review count == 0
- 사람이 듣는 spot check에서 카운트가 원본 의도와 맞을 것

- [ ] **Step 1: 샘플 선정 스크립트 작성**

`select_samples.py`는 52개에서 strata별 샘플 12개와 전체 52개 평가 목록을 만듭니다.

- [ ] **Step 2: 평가 매트릭스 정의**

`run_eval_matrix.py`는 아래 정책을 비교합니다.

```text
policy_a_plain_synth_real_translation
policy_b_number_preserving_script_plain_synth
policy_c_countdown_only_renderer_legacy
policy_d_hybrid_splice_source_anchor
```

---

## Task 7: 실험 실행 순서

**Files:**
- Create under: `experiments/countdown_embedded_hybrid/<timestamp>/`

- [ ] **Step 1: baseline plain synth**

Run targeted synth only for selected sample IDs:

```bash
uv run asmr-dub synth \
  --project runs/20260510T174455Z_RJ372458 \
  --refs refs/refs.json \
  --confirm-rights \
  --auto-gsv-server \
  --only-segments "$SAMPLE_IDS"
```

Expected:

```text
synth complete
```

- [ ] **Step 2: ASR/timing 평가**

Run:

```bash
uv run python experiments/countdown_embedded_hybrid/run_eval_matrix.py \
  --project runs/20260510T174455Z_RJ372458 \
  --policy plain_synth \
  --segment-ids "$SAMPLE_IDS"
```

Expected output files:

```text
experiments/countdown_embedded_hybrid/<timestamp>/plain_synth_metrics.json
experiments/countdown_embedded_hybrid/<timestamp>/plain_synth_rows.csv
```

- [ ] **Step 3: hybrid prototype 평가**

Hybrid prototype requirements:

```text
1. source_anchor_timeline에서 숫자 local slot 추출
2. 일반 synth 결과에서 주변 문장 오디오 생성
3. countdown token bank/pack으로 숫자 토큰 오디오 생성
4. 숫자 slot에 token audio를 place/crossfade
5. 주변 문장과 숫자가 겹치면 ducking 또는 split-splice 적용
```

Run:

```bash
uv run python experiments/countdown_embedded_hybrid/run_eval_matrix.py \
  --project runs/20260510T174455Z_RJ372458 \
  --policy hybrid_splice \
  --segment-ids "$SAMPLE_IDS"
```

Expected:

```text
hybrid_splice_metrics.json exists
manual_review_count == 0
```

- [ ] **Step 4: decision report 작성**

Report file:

```text
experiments/countdown_embedded_hybrid/<timestamp>/decision_report.md
```

Report must answer:

```text
1. plain synth로 충분한가?
2. 숫자 발음은 정확한가?
3. 숫자별 타이밍은 원본 anchor와 얼마나 다른가?
4. hybrid가 통계적으로/청감상 나은가?
5. production countdown-synth를 어떻게 바꿔야 하는가?
```

---

## Task 8: 복구 후 믹스까지 가는 조건부 명령

이 단계는 실험에서 어떤 합성 정책을 쓸지 결정한 뒤에만 실행합니다.

### 조건 A: plain synth가 통과한 경우

```bash
PROJECT=runs/20260510T174455Z_RJ372458
IDS=$(cat "$PROJECT/work/recovery_review/recovery_ids.csv")

uv run asmr-dub synth \
  --project "$PROJECT" \
  --refs refs/refs.json \
  --confirm-rights \
  --auto-gsv-server \
  --only-segments "$IDS"

uv run python - <<'PY'
from pathlib import Path
from asmr_dub_pipeline.pipeline.steps import rvc_step
project = Path("runs/20260510T174455Z_RJ372458")
ids = set((project / "work/recovery_review/recovery_ids.csv").read_text().strip().split(","))
rvc_step(project, confirm_rights=True, only_segment_ids=ids)
PY

uv run asmr-dub qc \
  --project "$PROJECT" \
  --gemma-backend mock \
  --confirm-rights

uv run asmr-dub mix \
  --project "$PROJECT" \
  --confirm-rights
```

### 조건 B: hybrid가 필요한 경우

먼저 production patch를 만든 뒤, 52개에 hybrid renderer를 적용하고 그 결과만 RVC/QC/Mix로 넘깁니다. plain synth 결과를 final manifest에 남기지 않습니다.
