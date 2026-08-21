from __future__ import annotations

import argparse
import re
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

try:
    from rag_pipeline.workers.opendataloader_local_worker import (
        atomic_write_json,
        configure_utf8_console,
        ensure_empty_artifact_directory,
        ensure_result_outside_artifacts,
        inspect_engine_output,
        package_version,
        sha256_file,
        utc_now_iso,
        validate_input_pdf,
    )
except ModuleNotFoundError:
    # The worker is intentionally executable from the isolated
    # OpenDataLoader environment where rag_pipeline is not installed.
    from opendataloader_local_worker import (  # type: ignore[no-redef]
        atomic_write_json,
        configure_utf8_console,
        ensure_empty_artifact_directory,
        ensure_result_outside_artifacts,
        inspect_engine_output,
        package_version,
        sha256_file,
        utc_now_iso,
        validate_input_pdf,
    )


WORKER_SCHEMA_VERSION = "1.0"
WORKER_VERSION = "2026-08-18-6.1B.1"
ENGINE_NAME = "OPENDATALOADER_HYBRID"
HYBRID_BACKEND = "docling-fast"
HYBRID_MODE = "full"
MAX_ATTEMPTS = 1

Converter = Callable[..., Any]


def normalize_key(value: object) -> str:
    return re.sub(r"[\s_-]+", "", str(value)).casefold()


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

    if parsed.query or parsed.fragment:
        raise ValueError("Hybrid backend URL cannot contain query or fragment")

    if parsed.path not in {"", "/"}:
        raise ValueError("Hybrid backend URL cannot contain a path")

    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("Hybrid backend URL contains an invalid port") from error

    if port is None or not 1 <= port <= 65535:
        raise ValueError("Hybrid backend URL must contain a valid port")

    return value.rstrip("/")


def validate_reason_codes(reason_codes: Sequence[str]) -> list[str]:
    normalized = sorted(set(reason_codes))

    if not normalized:
        raise ValueError("At least one fallback reason code is required")

    for reason_code in normalized:
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            reason_code,
        ):
            raise ValueError(f"Invalid fallback reason code: {reason_code!r}")

    return normalized


def collect_page_numbers(payload: Any) -> list[int]:
    page_numbers: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if normalize_key(key) == "pagenumber":
                    try:
                        page_number = int(child)
                    except (TypeError, ValueError):
                        continue

                    if page_number >= 1:
                        page_numbers.add(page_number)

                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return sorted(page_numbers)


def inspect_page_scoped_output(
    artifact_directory: Path,
    source_page_number: int,
) -> dict[str, Any]:
    output = inspect_engine_output(artifact_directory)
    structured_json_path = Path(output["structured_json"]["path"])

    import json

    structured_payload = json.loads(
        structured_json_path.read_text(encoding="utf-8-sig")
    )
    observed_page_numbers = collect_page_numbers(structured_payload)
    unexpected_page_numbers = sorted(
        set(observed_page_numbers) - {source_page_number}
    )

    if unexpected_page_numbers:
        raise ValueError(
            "Hybrid output escaped requested page scope: "
            f"{unexpected_page_numbers}"
        )

    output.update(
        {
            "requested_page_number": source_page_number,
            "selected_page_count": 1,
            "observed_page_numbers": observed_page_numbers,
            "unexpected_page_numbers": unexpected_page_numbers,
            "page_scope_verified": True,
            "page_scope_evidence": (
                "STRUCTURED_PAGE_FIELDS"
                if observed_page_numbers
                else "CONVERTER_PAGE_FILTER"
            ),
        }
    )
    return output


def run_conversion(
    input_pdf: Path,
    source_page_number: int,
    artifact_directory: Path,
    *,
    reason_codes: Sequence[str],
    hybrid_url: str = "http://127.0.0.1:5002",
    hybrid_timeout_ms: int = 120_000,
    converter: Converter | None = None,
) -> dict[str, Any]:
    input_pdf = input_pdf.expanduser().resolve()
    artifact_directory = artifact_directory.expanduser().resolve()

    validate_input_pdf(input_pdf)

    if source_page_number < 1:
        raise ValueError("source_page_number must be positive")

    normalized_reasons = validate_reason_codes(reason_codes)
    normalized_hybrid_url = validate_local_hybrid_url(hybrid_url)

    if hybrid_timeout_ms < 1:
        raise ValueError("hybrid_timeout_ms must be positive")

    ensure_empty_artifact_directory(artifact_directory)

    started_at = utc_now_iso()
    started = time.perf_counter()
    source_sha256 = sha256_file(input_pdf)
    active_converter = converter or load_converter()

    # full + pages=<N> forces exactly the failed source page through the
    # local Docling backend. Java fallback is deliberately disabled.
    active_converter(
        input_path=[str(input_pdf)],
        output_dir=str(artifact_directory),
        format="markdown,json",
        pages=str(source_page_number),
        hybrid=HYBRID_BACKEND,
        hybrid_mode=HYBRID_MODE,
        hybrid_url=normalized_hybrid_url,
        hybrid_timeout=str(hybrid_timeout_ms),
        hybrid_fallback=False,
    )

    output = inspect_page_scoped_output(
        artifact_directory,
        source_page_number,
    )
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
        "max_attempts": MAX_ATTEMPTS,
        "input": {
            "path": str(input_pdf),
            "name": input_pdf.name,
            "size_bytes": input_pdf.stat().st_size,
            "sha256": source_sha256,
        },
        "request": {
            "source_page_number": source_page_number,
            "reason_codes": normalized_reasons,
            "hybrid_backend": HYBRID_BACKEND,
            "hybrid_mode": HYBRID_MODE,
            "hybrid_url": normalized_hybrid_url,
            "hybrid_timeout_ms": hybrid_timeout_ms,
            "hybrid_fallback": False,
        },
        "output": output,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one failed PDF page through OpenDataLoader Hybrid in "
            "the isolated OpenDataLoader environment."
        )
    )
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--page-number", required=True, type=int)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument(
        "--reason-code",
        action="append",
        required=True,
        dest="reason_codes",
    )
    parser.add_argument(
        "--hybrid-url",
        default="http://127.0.0.1:5002",
    )
    parser.add_argument(
        "--hybrid-timeout-ms",
        default=120_000,
        type=int,
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
        ensure_result_outside_artifacts(result_path, artifact_directory)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2

    try:
        result = run_conversion(
            input_pdf=input_pdf,
            source_page_number=arguments.page_number,
            artifact_directory=artifact_directory,
            reason_codes=arguments.reason_codes,
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
            "attempt_number": 1,
            "max_attempts": MAX_ATTEMPTS,
            "input": {
                "path": str(input_pdf),
                "name": input_pdf.name,
            },
            "request": {
                "source_page_number": arguments.page_number,
                "reason_codes": sorted(set(arguments.reason_codes)),
                "hybrid_backend": HYBRID_BACKEND,
                "hybrid_mode": HYBRID_MODE,
                "hybrid_url": arguments.hybrid_url,
                "hybrid_timeout_ms": arguments.hybrid_timeout_ms,
                "hybrid_fallback": False,
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
