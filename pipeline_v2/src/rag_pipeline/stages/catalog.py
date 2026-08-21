from __future__ import annotations

import hashlib
import mimetypes
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5

from rag_pipeline.contracts import CatalogEntry, FileFormat


HASH_BUFFER_SIZE = 1024 * 1024

SUPPORTED_EXTENSIONS: dict[str, FileFormat] = {
    ".pdf": FileFormat.PDF,
    ".docx": FileFormat.DOCX,
    ".doc": FileFormat.DOC,
    ".png": FileFormat.IMAGE,
    ".jpg": FileFormat.IMAGE,
    ".jpeg": FileFormat.IMAGE,
    ".tif": FileFormat.IMAGE,
    ".tiff": FileFormat.IMAGE,
    ".bmp": FileFormat.IMAGE,
}


class CatalogError(ValueError):
    """Raised when an input file cannot safely enter Catalog."""


def compute_sha256(
    file_path: str | Path,
    buffer_size: int = HASH_BUFFER_SIZE,
) -> str:
    path = Path(file_path)

    if buffer_size <= 0:
        raise ValueError("buffer_size must be greater than zero")

    if not path.is_file():
        raise CatalogError(f"Not a readable file: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as source:
        while chunk := source.read(buffer_size):
            digest.update(chunk)

    return digest.hexdigest()


def detect_file_format(file_path: str | Path) -> FileFormat:
    suffix = Path(file_path).suffix.lower()
    return SUPPORTED_EXTENSIONS.get(suffix, FileFormat.UNKNOWN)


def create_catalog_id(
    relative_path: str,
    source_sha256: str,
) -> str:
    identity = (
        f"rag-pipeline-v2:"
        f"{relative_path.casefold()}:{source_sha256}"
    )

    return f"cat-{uuid5(NAMESPACE_URL, identity).hex}"


def catalog_file(
    source_path: str | Path,
    data_root: str | Path,
) -> CatalogEntry:
    try:
        resolved_root = Path(data_root).resolve(strict=True)
    except FileNotFoundError as error:
        raise CatalogError(
            f"Data root does not exist: {data_root}"
        ) from error

    if not resolved_root.is_dir():
        raise CatalogError(
            f"Data root is not a directory: {resolved_root}"
        )

    candidate = Path(source_path)

    if not candidate.is_absolute():
        candidate = resolved_root / candidate

    if candidate.is_symlink():
        raise CatalogError(
            "Symbolic-link inputs are not accepted"
        )

    try:
        resolved_source = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise CatalogError(
            f"Source file does not exist: {candidate}"
        ) from error

    if not resolved_source.is_file():
        raise CatalogError(
            f"Source path is not a file: {resolved_source}"
        )

    try:
        relative_path = resolved_source.relative_to(
            resolved_root
        )
    except ValueError as error:
        raise CatalogError(
            "Source file is outside data_root"
        ) from error

    stat_result = resolved_source.stat()

    if stat_result.st_size <= 0:
        raise CatalogError(
            f"Empty source file: {relative_path}"
        )

    source_sha256 = compute_sha256(resolved_source)
    portable_path = relative_path.as_posix()
    file_format = detect_file_format(resolved_source)

    guessed_mime_type, _ = mimetypes.guess_type(
        resolved_source.name
    )

    return CatalogEntry(
        catalog_id=create_catalog_id(
            relative_path=portable_path,
            source_sha256=source_sha256,
        ),
        source_name=resolved_source.name,
        source_path=portable_path,
        file_format=file_format,
        source_sha256=source_sha256,
        size_bytes=stat_result.st_size,
        mime_type=guessed_mime_type,
        modified_at=datetime.fromtimestamp(
            stat_result.st_mtime,
            tz=timezone.utc,
        ),
    )


def scan_data_root(
    data_root: str | Path,
    scan_root: str | Path | None = None,
    recursive: bool = True,
) -> list[CatalogEntry]:
    try:
        resolved_data_root = Path(data_root).resolve(
            strict=True
        )
    except FileNotFoundError as error:
        raise CatalogError(
            f"Data root does not exist: {data_root}"
        ) from error

    if not resolved_data_root.is_dir():
        raise CatalogError(
            "Data root is not a directory: "
            f"{resolved_data_root}"
        )

    if scan_root is None:
        scan_candidate = resolved_data_root
    else:
        scan_candidate = Path(scan_root)

        if not scan_candidate.is_absolute():
            scan_candidate = (
                resolved_data_root / scan_candidate
            )

    if scan_candidate.is_symlink():
        raise CatalogError(
            "Symbolic-link scan roots are not accepted"
        )

    try:
        resolved_scan_root = scan_candidate.resolve(
            strict=True
        )
    except FileNotFoundError as error:
        raise CatalogError(
            f"Scan root does not exist: {scan_candidate}"
        ) from error

    if not resolved_scan_root.is_dir():
        raise CatalogError(
            "Scan root is not a directory: "
            f"{resolved_scan_root}"
        )

    try:
        resolved_scan_root.relative_to(
            resolved_data_root
        )
    except ValueError as error:
        raise CatalogError(
            "Scan root is outside data_root"
        ) from error

    iterator = (
        resolved_scan_root.rglob("*")
        if recursive
        else resolved_scan_root.glob("*")
    )

    candidates = [
        path
        for path in iterator
        if (
            path.is_file()
            and not path.is_symlink()
            and not path.name.startswith("~$")
            and detect_file_format(path)
            != FileFormat.UNKNOWN
        )
    ]

    candidates.sort(
        key=lambda path: (
            path.relative_to(resolved_data_root)
            .as_posix()
            .casefold()
        )
    )

    return [
        catalog_file(
            source_path=path,
            data_root=resolved_data_root,
        )
        for path in candidates
    ]


def write_catalog_manifest(
    entries: Iterable[CatalogEntry],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for entry in entries:
                output_file.write(entry.model_dump_json())
                output_file.write("\n")

        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def load_catalog_manifest(
    manifest_path: str | Path,
) -> list[CatalogEntry]:
    path = Path(manifest_path)

    if not path.is_file():
        raise CatalogError(
            f"Catalog manifest does not exist: {path}"
        )

    entries: list[CatalogEntry] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(
            input_file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                entries.append(
                    CatalogEntry.model_validate_json(line)
                )
            except Exception as error:
                raise CatalogError(
                    "Invalid catalog record at "
                    f"line {line_number}"
                ) from error

    return entries