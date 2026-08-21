from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKER_SCHEMA_VERSION = "1.0"
WORKER_VERSION = "2026-08-17-5.1B.1"
ENGINE_NAME = "OPENDATALOADER_LOCAL"

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

Converter = Callable[..., Any]


def configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)

        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_key(value: object) -> str:
    return re.sub(r"[\s_-]+", "", str(value)).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def ensure_empty_artifact_directory(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(
            f"Artifact path is not a directory: {path}"
        )

    path.mkdir(parents=True, exist_ok=True)

    if any(path.iterdir()):
        raise ValueError(
            "Artifact directory must be empty to prevent mixed runs"
        )


def ensure_result_outside_artifacts(
    result_path: Path,
    artifact_directory: Path,
) -> None:
    if (
        result_path == artifact_directory
        or artifact_directory in result_path.parents
    ):
        raise ValueError(
            "Result JSON must be outside the raw artifact directory"
        )


def package_version() -> str:
    for distribution_name in (
        "opendataloader-pdf",
        "opendataloader_pdf",
    ):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue

    return "unknown"


def load_converter() -> Converter:
    import opendataloader_pdf

    return opendataloader_pdf.convert


def root_value(
    payload: dict[str, Any],
    expected_key: str,
) -> Any:
    normalized_expected = normalize_key(expected_key)

    for key, value in payload.items():
        if normalize_key(key) == normalized_expected:
            return value

    return None


def artifact_record(path: Path, artifact_directory: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "relative_path": path.relative_to(
            artifact_directory
        ).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inspect_engine_output(artifact_directory: Path) -> dict[str, Any]:
    all_files = sorted(
        path
        for path in artifact_directory.rglob("*")
        if path.is_file()
    )
    markdown_files = [
        path for path in all_files if path.suffix.casefold() == ".md"
    ]
    json_files = [
        path for path in all_files if path.suffix.casefold() == ".json"
    ]
    image_files = [
        path
        for path in all_files
        if path.suffix.casefold() in IMAGE_SUFFIXES
    ]

    if len(markdown_files) != 1:
        raise ValueError(
            "OpenDataLoader must create exactly one Markdown file; "
            f"found {len(markdown_files)}"
        )

    if len(json_files) != 1:
        raise ValueError(
            "OpenDataLoader must create exactly one structured JSON file; "
            f"found {len(json_files)}"
        )

    markdown_path = markdown_files[0]
    structured_json_path = json_files[0]

    markdown = markdown_path.read_text(encoding="utf-8")

    if not markdown.strip():
        raise ValueError("OpenDataLoader Markdown output is empty")

    structured_payload = json.loads(
        structured_json_path.read_text(encoding="utf-8-sig")
    )

    if not isinstance(structured_payload, dict):
        raise TypeError(
            "OpenDataLoader structured JSON root must be an object"
        )

    raw_page_count = root_value(
        structured_payload,
        "number of pages",
    )

    try:
        page_count = int(raw_page_count)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Structured JSON does not contain a valid page count"
        ) from error

    if page_count < 1:
        raise ValueError("Structured JSON page count must be positive")

    image_records = [
        artifact_record(path, artifact_directory)
        for path in image_files
    ]
    paths_by_hash: dict[str, list[str]] = {}

    for record in image_records:
        paths_by_hash.setdefault(record["sha256"], []).append(
            record["relative_path"]
        )

    image_hash_groups = [
        {
            "content_sha256": content_sha256,
            "physical_paths": sorted(physical_paths),
        }
        for content_sha256, physical_paths in sorted(
            paths_by_hash.items()
        )
    ]

    return {
        "artifact_directory": str(artifact_directory),
        "page_count": page_count,
        "total_file_count": len(all_files),
        "markdown_character_count": len(markdown),
        "markdown_broken_character_count": markdown.count("\ufffd"),
        "markdown": artifact_record(
            markdown_path,
            artifact_directory,
        ),
        "structured_json": artifact_record(
            structured_json_path,
            artifact_directory,
        ),
        "images": image_records,
        "physical_image_file_count": len(image_records),
        "unique_image_hash_count": len(paths_by_hash),
        "duplicate_image_hash_group_count": sum(
            1
            for physical_paths in paths_by_hash.values()
            if len(physical_paths) > 1
        ),
        "image_hash_groups": image_hash_groups,
    }


def validate_input_pdf(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Input PDF does not exist: {path}")

    if path.suffix.casefold() != ".pdf":
        raise ValueError("OpenDataLoader Local worker accepts PDF only")

    if path.stat().st_size < 1:
        raise ValueError("Input PDF is empty")


def run_conversion(
    input_pdf: Path,
    artifact_directory: Path,
    converter: Converter | None = None,
) -> dict[str, Any]:
    input_pdf = input_pdf.expanduser().resolve()
    artifact_directory = artifact_directory.expanduser().resolve()

    validate_input_pdf(input_pdf)
    ensure_empty_artifact_directory(artifact_directory)

    started_at = utc_now_iso()
    started = time.perf_counter()
    source_sha256 = sha256_file(input_pdf)

    active_converter = converter or load_converter()
    active_converter(
        input_path=[str(input_pdf)],
        output_dir=str(artifact_directory),
        format="markdown,json",
    )

    output = inspect_engine_output(artifact_directory)
    elapsed_seconds = time.perf_counter() - started

    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "worker_version": WORKER_VERSION,
        "status": "PASS",
        "engine": ENGINE_NAME,
        "engine_version": package_version(),
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "input": {
            "path": str(input_pdf),
            "name": input_pdf.name,
            "size_bytes": input_pdf.stat().st_size,
            "sha256": source_sha256,
        },
        "output": output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run OpenDataLoader PDF Local in its isolated Python "
            "environment and write one machine-readable result manifest."
        )
    )
    parser.add_argument(
        "--input-pdf",
        required=True,
        help="Absolute or relative path to one PDF.",
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="New or empty directory for raw engine artifacts.",
    )
    parser.add_argument(
        "--result-json",
        required=True,
        help="Manifest path outside the raw artifact directory.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_console()
    arguments = build_parser().parse_args(argv)

    input_pdf = Path(arguments.input_pdf).expanduser().resolve()
    artifact_directory = Path(
        arguments.artifact_dir
    ).expanduser().resolve()
    result_path = Path(arguments.result_json).expanduser().resolve()

    try:
        ensure_result_outside_artifacts(
            result_path,
            artifact_directory,
        )
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2

    try:
        result = run_conversion(
            input_pdf=input_pdf,
            artifact_directory=artifact_directory,
        )
        exit_code = 0
    except Exception as error:
        result = {
            "schema_version": WORKER_SCHEMA_VERSION,
            "worker_version": WORKER_VERSION,
            "status": "FAIL",
            "engine": ENGINE_NAME,
            "engine_version": package_version(),
            "finished_at": utc_now_iso(),
            "input": {
                "path": str(input_pdf),
                "name": input_pdf.name,
            },
            "output": {
                "artifact_directory": str(artifact_directory),
            },
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        traceback.print_exc(file=sys.stderr)
        exit_code = 1

    atomic_write_json(result_path, result)
    print(f"WORKER_RESULT_JSON={result_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
