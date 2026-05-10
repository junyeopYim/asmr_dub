#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _report_paths(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve()
        if resolved.is_file() and resolved.name == "asr_high_risk_report.json":
            paths.append(resolved)
        elif resolved.exists():
            paths.extend(resolved.rglob("work/transcribe/asr_high_risk_report.json"))
    return sorted(set(paths))


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"ASR audit report must be an object: {path}")
    return payload


def build_audit_report(roots: list[Path]) -> dict[str, Any]:
    reports = []
    items: list[dict[str, Any]] = []
    for path in _report_paths(roots):
        payload = _load_report(path)
        report_items = payload.get("items") or []
        if not isinstance(report_items, list):
            report_items = []
        reports.append({"path": str(path), "item_count": len(report_items)})
        for item in report_items:
            if isinstance(item, dict):
                items.append({**item, "report_path": str(path)})

    reason_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    replacement_source_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    for item in items:
        decision = str(item.get("decision") or "unknown")
        severity = str(item.get("severity") or "unknown")
        decision_counts[decision] += 1
        severity_counts[severity] += 1
        for reason in item.get("reasons") or []:
            reason_counts[str(reason)] += 1
        for hit in item.get("replacement_hits") or []:
            if not isinstance(hit, dict):
                continue
            source = str(hit.get("source") or "")
            if not source:
                continue
            replacement_source_counts[source] += int(hit.get("count") or 1)

    return {
        "summary": {
            "report_count": len(reports),
            "item_count": len(items),
            "severe": severity_counts["severe"],
            "warning": severity_counts["warning"],
        },
        "reports": reports,
        "reason_counts": dict(sorted(reason_counts.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "replacement_source_counts": dict(sorted(replacement_source_counts.items())),
        "top_review_candidates": [
            {"reason": reason, "count": count}
            for reason, count in reason_counts.most_common()
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ASR high-risk report JSON files.")
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_audit_report(args.roots)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, "utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
