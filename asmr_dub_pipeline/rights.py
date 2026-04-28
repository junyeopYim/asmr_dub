from __future__ import annotations

import hashlib
from pathlib import Path

from .schemas import RightsAudit, utc_now

RIGHTS_MESSAGE = (
    "This command processes real media. Re-run with --confirm-rights to confirm "
    "you own or have permission/consent for the source content, voice references, "
    "and distribution."
)


class RightsError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_confirmed_rights(
    confirm_rights: bool,
    command: str,
    input_path: Path | None = None,
    metadata: dict[str, object] | None = None,
) -> RightsAudit:
    if not confirm_rights:
        raise RightsError(RIGHTS_MESSAGE)
    source_path = str(input_path) if input_path else None
    source_sha256 = sha256_file(input_path) if input_path and input_path.exists() else None
    confirmed_at = utc_now()
    entry = {
        "confirmed_at": confirmed_at.isoformat(),
        "command": command,
        "source_path": source_path,
        "source_sha256": source_sha256,
        **(metadata or {}),
    }
    return RightsAudit(
        confirmed=True,
        confirmed_at=confirmed_at,
        command=command,
        source_path=source_path,
        source_sha256=source_sha256,
        history=[entry],
    )


def merge_rights_audit(existing: RightsAudit, new: RightsAudit) -> RightsAudit:
    history = [*existing.history, *new.history]
    merged = new.model_copy(update={"history": history})
    return merged


def record_rights_reliance(
    existing: RightsAudit,
    command: str,
    input_path: Path | None = None,
    metadata: dict[str, object] | None = None,
) -> RightsAudit:
    if not existing.confirmed:
        raise RightsError(RIGHTS_MESSAGE)
    source_path = str(input_path) if input_path else existing.source_path
    source_sha256 = sha256_file(input_path) if input_path and input_path.exists() else existing.source_sha256
    entry = {
        "relied_at": utc_now().isoformat(),
        "command": command,
        "source_path": source_path,
        "source_sha256": source_sha256,
        **(metadata or {}),
    }
    return existing.model_copy(update={"history": [*existing.history, entry]})


def require_existing_or_confirmed_rights(
    existing: RightsAudit,
    confirm_rights: bool,
    command: str,
    input_path: Path | None = None,
    metadata: dict[str, object] | None = None,
) -> RightsAudit:
    if confirm_rights:
        return merge_rights_audit(
            existing,
            require_confirmed_rights(True, command, input_path, metadata=metadata),
        )
    return record_rights_reliance(existing, command, input_path, metadata=metadata)


def ensure_not_same_path(input_path: Path, output_path: Path) -> None:
    try:
        if input_path.exists() and output_path.exists() and input_path.samefile(output_path):
            raise RightsError("Refusing to overwrite the input file.")
        if input_path.resolve() == output_path.resolve():
            raise RightsError("Refusing to overwrite the input file.")
    except (FileNotFoundError, OSError) as exc:
        if input_path.absolute() == output_path.absolute():
            raise RightsError("Refusing to overwrite the input file.") from exc


def ensure_inside_project(project_dir: Path, path: Path) -> Path:
    root = project_dir.expanduser().resolve()
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RightsError(f"Refusing to write outside the project directory: {resolved}") from exc
    return resolved
