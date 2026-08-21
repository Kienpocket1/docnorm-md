from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

WORKER_SCHEMA_VERSION = "1.0"
WORKER_VERSION = "2026-08-21-1.4.0"
ENGINE_NAME = "OPENDATALOADER_HYBRID"
HYBRID_BACKEND = "docling-fast"
HYBRID_MODE = "full"
OCR_LANGUAGES = ("vi", "en")

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
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def package_version() -> str:
    for distribution_name in ("opendataloader-pdf", "opendataloader_pdf"):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def load_converter() -> Converter:
    import opendataloader_pdf

    return opendataloader_pdf.convert


def validate_local_hybrid_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "http":
        raise ValueError("Hybrid backend URL must use local HTTP")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Hybrid backend must be bound to localhost")
    if parsed.username or parsed.password:
        raise ValueError("Hybrid backend URL cannot contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Hybrid backend URL cannot contain path, query, or fragment")
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Hybrid backend URL contains an invalid port") from error
    if port is None or not 1 <= port <= 65535:
        raise ValueError("Hybrid backend URL must contain a valid port")
    return value.rstrip("/")


def load_local_worker_helpers(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"OpenDataLoader local worker not found: {resolved}")
    spec = importlib.util.spec_from_file_location(
        "docnorm_opendataloader_local_worker",
        resolved,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OpenDataLoader worker helpers: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "validate_input_pdf",
        "ensure_empty_artifact_directory",
        "ensure_result_outside_artifacts",
        "inspect_engine_output",
        "sha256_file",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise ImportError(
            "OpenDataLoader local worker is missing required helpers: "
            + ", ".join(missing)
        )
    return module


def run_conversion(
    input_pdf: Path,
    artifact_directory: Path,
    *,
    local_worker_helpers: ModuleType,
    hybrid_url: str = "http://127.0.0.1:5002",
    hybrid_timeout_ms: int = 3_600_000,
    converter: Converter | None = None,
) -> dict[str, Any]:
    input_pdf = input_pdf.expanduser().resolve()
    artifact_directory = artifact_directory.expanduser().resolve()
    normalized_hybrid_url = validate_local_hybrid_url(hybrid_url)
    if hybrid_timeout_ms < 1:
        raise ValueError("hybrid_timeout_ms must be positive")

    local_worker_helpers.validate_input_pdf(input_pdf)
    local_worker_helpers.ensure_empty_artifact_directory(artifact_directory)
    started_at = utc_now_iso()
    started = time.perf_counter()
    source_sha256 = local_worker_helpers.sha256_file(input_pdf)
    active_converter = converter or load_converter()

    # The backend is launched by run_windows.ps1 with --force-ocr and
    # --ocr-lang vi,en. Java fallback is disabled so a backend failure cannot
    # silently publish a non-OCR result.
    active_converter(
        input_path=[str(input_pdf)],
        output_dir=str(artifact_directory),
        format="markdown,json",
        hybrid=HYBRID_BACKEND,
        hybrid_mode=HYBRID_MODE,
        hybrid_url=normalized_hybrid_url,
        hybrid_timeout=str(hybrid_timeout_ms),
        hybrid_fallback=False,
    )

    output = local_worker_helpers.inspect_engine_output(artifact_directory)
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
        "attempt_number": 1,
        "max_attempts": 1,
        "input": {
            "path": str(input_pdf),
            "name": input_pdf.name,
            "size_bytes": input_pdf.stat().st_size,
            "sha256": source_sha256,
        },
        "request": {
            "scope": "FULL_DOCUMENT",
            "hybrid_backend": HYBRID_BACKEND,
            "hybrid_mode": HYBRID_MODE,
            "hybrid_url": normalized_hybrid_url,
            "hybrid_timeout_ms": hybrid_timeout_ms,
            "hybrid_fallback": False,
            "force_ocr": True,
            "ocr_languages": list(OCR_LANGUAGES),
            "external_calls": 0,
        },
        "output": output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete scan PDF through the localhost OpenDataLoader "
            "Hybrid backend and write an auditable result manifest."
        )
    )
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--local-worker", required=True)
    parser.add_argument("--hybrid-url", default="http://127.0.0.1:5002")
    parser.add_argument("--hybrid-timeout-ms", type=int, default=3_600_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_utf8_console()
    arguments = build_parser().parse_args(argv)
    input_pdf = Path(arguments.input_pdf).expanduser().resolve()
    artifact_directory = Path(arguments.artifact_dir).expanduser().resolve()
    result_path = Path(arguments.result_json).expanduser().resolve()
    local_worker = Path(arguments.local_worker).expanduser().resolve()

    try:
        helpers = load_local_worker_helpers(local_worker)
        helpers.ensure_result_outside_artifacts(result_path, artifact_directory)
        result = run_conversion(
            input_pdf=input_pdf,
            artifact_directory=artifact_directory,
            local_worker_helpers=helpers,
            hybrid_url=arguments.hybrid_url,
            hybrid_timeout_ms=arguments.hybrid_timeout_ms,
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
            "input": {"path": str(input_pdf), "name": input_pdf.name},
            "request": {
                "scope": "FULL_DOCUMENT",
                "hybrid_backend": HYBRID_BACKEND,
                "hybrid_mode": HYBRID_MODE,
                "hybrid_url": arguments.hybrid_url,
                "hybrid_timeout_ms": arguments.hybrid_timeout_ms,
                "hybrid_fallback": False,
                "force_ocr": True,
                "ocr_languages": list(OCR_LANGUAGES),
                "external_calls": 0,
            },
            "output": {"artifact_directory": str(artifact_directory)},
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        traceback.print_exc(file=sys.stderr)
        exit_code = 1

    atomic_write_json(result_path, result)
    print(f"WORKER_RESULT_JSON={result_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
