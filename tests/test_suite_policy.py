from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

DEFAULT_MAX_TESTS_PER_FILE = 40
DEFAULT_MAX_LINES_PER_FILE = 2_000

# Existing legacy scenario suites are capped at their current size. New cases
# should move into fixtures, shared tables, or focused contract tests instead.
LEGACY_TEST_LIMITS = {
    Path("tests/test_text_translation_lane.py"): 223,
    Path("tests/test_gpt_sovits_few_shot.py"): 87,
    Path("tests/test_synth_gpt_sovits_omission_retry.py"): 57,
    Path("tests/test_voice_bank.py"): 45,
}
LEGACY_LINE_LIMITS = {
    Path("tests/test_text_translation_lane.py"): 10_500,
    Path("tests/test_gpt_sovits_few_shot.py"): 6_000,
    Path("tests/test_synth_gpt_sovits_omission_retry.py"): 5_500,
    Path("tests/test_voice_bank.py"): 2_500,
}


def _test_count(path: Path) -> int:
    return len(re.findall(r"(?m)^(?:def test_|class Test)", path.read_text(encoding="utf-8")))


def test_test_files_do_not_accumulate_unbounded_cases() -> None:
    offenders: list[str] = []

    for path in sorted(Path("tests").glob("test_*.py")):
        test_limit = LEGACY_TEST_LIMITS.get(path, DEFAULT_MAX_TESTS_PER_FILE)
        line_limit = LEGACY_LINE_LIMITS.get(path, DEFAULT_MAX_LINES_PER_FILE)
        test_count = _test_count(path)
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if test_count > test_limit:
            offenders.append(f"{path}: {test_count} tests > limit {test_limit}")
        if line_count > line_limit:
            offenders.append(f"{path}: {line_count} lines > limit {line_limit}")

    assert not offenders, (
        "Test files are growing into scenario archives. Move new cases into a fixture/table, "
        "or split out a small contract test and mark the bulky scenario suite as regression.\n"
        + "\n".join(offenders)
    )
