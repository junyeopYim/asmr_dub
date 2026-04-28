from __future__ import annotations

from pathlib import Path

from asmr_dub_pipeline.cli import app


def test_cli_help_mentions_rights(cli_runner) -> None:
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "permission" in result.output
    assert "DRM" in result.output


def test_no_shell_true_or_scraping_commands() -> None:
    roots = [Path("asmr_dub_pipeline"), Path("docs"), Path("README.md"), Path("pyproject.toml")]
    text_suffixes = {".py", ".md", ".toml"}
    text = "\n".join(
        path.read_text("utf-8")
        for root in roots
        for path in ([root] if root.is_file() else root.rglob("*"))
        if path.is_file() and path.suffix in text_suffixes
    )
    assert "shell=True" not in text
    assert "yt-dlp" not in text
    assert "youtube" not in text.lower()
