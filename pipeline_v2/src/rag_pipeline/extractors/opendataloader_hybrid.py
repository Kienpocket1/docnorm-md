from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

import fitz
from pydantic import Field, ValidationError, model_validator

from rag_pipeline.extractors.opendataloader_local import (
    ManifestModel,
    OpenDataLoaderAdapterError,
    OpenDataLoaderConfigurationError,
    OpenDataLoaderIntegrityError,
    OpenDataLoaderManifestError,
    OpenDataLoaderRunExistsError,
    OpenDataLoaderTimeoutError,
    OpenDataLoaderWorkerError,
    WorkerInputRecord,
    WorkerOutputRecord,
    read_log_tail,
)
from rag_pipeline.failure import ODL_HYBRID_MAX_ATTEMPTS
from rag_pipeline.workers.opendataloader_hybrid_page_worker import (
    HYBRID_BACKEND,
    HYBRID_MODE,
    validate_local_hybrid_url,
    validate_reason_codes,
)


HYBRID_ADAPTER_VERSION = "2026-08-18-6.1B.1"


class HybridWorkerRequestRecord(ManifestModel):
    source_page_number: int = Field(ge=1)
    reason_codes: list[str] = Field(min_length=1)
    hybrid_backend: Literal["docling-fast"]
    hybrid_mode: Literal["full"]
    hybrid_url: str = Field(min_length=1)
    hybrid_timeout_ms: int = Field(ge=1)
    hybrid_fallback: Literal[False]

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.reason_codes != sorted(set(self.reason_codes)):
            raise ValueError("reason_codes must be sorted and unique")

        validate_reason_codes(self.reason_codes)
        validate_local_hybrid_url(self.hybrid_url)
        return self


class HybridWorkerOutputRecord(WorkerOutputRecord):
    requested_page_number: int = Field(ge=1)
    selected_page_count: Literal[1]
    observed_page_numbers: list[int]
    unexpected_page_numbers: list[int]
    page_scope_verified: Literal[True]
    page_scope_evidence: Literal[
        "STRUCTURED_PAGE_FIELDS",
        "CONVERTER_PAGE_FILTER",
    ]

    @model_validator(mode="after")
    def validate_page_scope(self) -> Self:
        if self.observed_page_numbers != sorted(
            set(self.observed_page_numbers)
        ):
            raise ValueError(
                "observed_page_numbers must be sorted and unique"
            )

        if self.unexpected_page_numbers:
            raise ValueError("Hybrid output contains unexpected pages")

        if any(
            page_number != self.requested_page_number
            for page_number in self.observed_page_numbers
        ):
            raise ValueError("Observed page escaped requested scope")

        if (
            self.observed_page_numbers
            and self.page_scope_evidence != "STRUCTURED_PAGE_FIELDS"
        ):
            raise ValueError("Page scope evidence is inconsistent")

        if (
            not self.observed_page_numbers
            and self.page_scope_evidence != "CONVERTER_PAGE_FILTER"
        ):
            raise ValueError("Page scope evidence is inconsistent")

        return self


class OpenDataLoaderHybridWorkerManifest(ManifestModel):
    schema_version: Literal["1.0"]
    worker_version: str = Field(min_length=1)
    status: Literal["PASS"]
    engine: Literal["OPENDATALOADER_HYBRID"]
    engine_version: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float = Field(ge=0)
    attempt_number: Literal[1]
    max_attempts: Literal[1]
    input: WorkerInputRecord
    request: HybridWorkerRequestRecord
    output: HybridWorkerOutputRecord

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be before started_at")

        if (
            self.request.source_page_number
            != self.output.requested_page_number
        ):
            raise ValueError("Requested page differs between records")

        return self


class OpenDataLoaderHybridPageError(OpenDataLoaderAdapterError):
    """Base exception for page-scoped Hybrid fallback."""


class OpenDataLoaderHybridConfigurationError(
    OpenDataLoaderConfigurationError,
    OpenDataLoaderHybridPageError,
):
    pass


class OpenDataLoaderHybridRunExistsError(
    OpenDataLoaderRunExistsError,
    OpenDataLoaderHybridPageError,
):
    pass


class OpenDataLoaderHybridTimeoutError(
    OpenDataLoaderTimeoutError,
    OpenDataLoaderHybridPageError,
):
    pass


class OpenDataLoaderHybridWorkerError(
    OpenDataLoaderWorkerError,
    OpenDataLoaderHybridPageError,
):
    pass


class OpenDataLoaderHybridManifestError(
    OpenDataLoaderManifestError,
    OpenDataLoaderHybridPageError,
):
    pass


class OpenDataLoaderHybridIntegrityError(
    OpenDataLoaderIntegrityError,
    OpenDataLoaderHybridPageError,
):
    pass


@dataclass(frozen=True, slots=True)
class OpenDataLoaderHybridPageConfig:
    python_executable: Path
    worker_script: Path
    output_root: Path
    timeout_seconds: float = 300.0
    hybrid_url: str = "http://127.0.0.1:5002"
    hybrid_timeout_ms: int = 120_000
    expected_engine_version: str | None = "2.5.0"
    verify_artifact_hashes: bool = True


@dataclass(frozen=True, slots=True)
class OpenDataLoaderHybridPageRun:
    run_id: str
    source_page_number: int
    source_document_page_count: int
    source_sha256: str
    reason_codes: tuple[str, ...]
    run_directory: Path
    raw_artifact_directory: Path
    result_manifest_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    manifest: OpenDataLoaderHybridWorkerManifest


def utc_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class OpenDataLoaderHybridPageExtractor:
    def __init__(self, config: OpenDataLoaderHybridPageConfig) -> None:
        self.config = config

    def extract(
        self,
        input_pdf: Path | str,
        *,
        page_number: int,
        reason_codes: list[str] | tuple[str, ...],
        run_id: str | None = None,
    ) -> OpenDataLoaderHybridPageRun:
        python_executable = (
            self.config.python_executable.expanduser().resolve()
        )
        worker_script = self.config.worker_script.expanduser().resolve()
        output_root = self.config.output_root.expanduser().resolve()
        input_path = Path(input_pdf).expanduser().resolve()
        normalized_hybrid_url = validate_local_hybrid_url(
            self.config.hybrid_url
        )
        normalized_reasons = validate_reason_codes(reason_codes)

        self._validate_configuration(
            python_executable=python_executable,
            worker_script=worker_script,
            output_root=output_root,
        )
        page_count = self._validate_input(input_path, page_number)

        active_run_id = run_id or utc_run_id()
        self._validate_run_id(active_run_id)
        output_root.mkdir(parents=True, exist_ok=True)
        run_directory = output_root / active_run_id

        try:
            run_directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise OpenDataLoaderHybridRunExistsError(
                f"Hybrid fallback run already exists: {active_run_id}"
            ) from error

        raw_artifact_directory = run_directory / "raw"
        result_manifest_path = run_directory / "worker_result.json"
        stdout_log_path = run_directory / "worker_stdout.log"
        stderr_log_path = run_directory / "worker_stderr.log"
        source_sha256_before = sha256_file(input_path)

        command = [
            str(python_executable),
            str(worker_script),
            "--input-pdf",
            str(input_path),
            "--page-number",
            str(page_number),
            "--artifact-dir",
            str(raw_artifact_directory),
            "--result-json",
            str(result_manifest_path),
            "--hybrid-url",
            normalized_hybrid_url,
            "--hybrid-timeout-ms",
            str(self.config.hybrid_timeout_ms),
        ]

        for reason_code in normalized_reasons:
            command.extend(["--reason-code", reason_code])

        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"
        popen_arguments: dict = {
            "args": command,
            "cwd": str(worker_script.parent),
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "shell": False,
        }

        if os.name == "nt":
            popen_arguments["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            popen_arguments["start_new_session"] = True

        with (
            stdout_log_path.open("w", encoding="utf-8") as stdout_log,
            stderr_log_path.open("w", encoding="utf-8") as stderr_log,
        ):
            popen_arguments["stdout"] = stdout_log
            popen_arguments["stderr"] = stderr_log

            try:
                process = subprocess.Popen(**popen_arguments)
            except OSError as error:
                raise OpenDataLoaderHybridConfigurationError(
                    f"Unable to start Hybrid worker: {error}"
                ) from error

            try:
                return_code = process.wait(
                    timeout=self.config.timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                terminate_process_tree(process)
                raise OpenDataLoaderHybridTimeoutError(
                    "Hybrid worker exceeded timeout of "
                    f"{self.config.timeout_seconds} seconds"
                ) from error

        try:
            source_sha256_after = sha256_file(input_path)
        except OSError as error:
            raise OpenDataLoaderHybridIntegrityError(
                "Source PDF became unreadable during Hybrid fallback"
            ) from error

        if source_sha256_after != source_sha256_before:
            raise OpenDataLoaderHybridIntegrityError(
                "Source PDF changed during Hybrid fallback"
            )

        raw_manifest = self._read_raw_manifest(result_manifest_path)

        if return_code != 0 or raw_manifest.get("status") != "PASS":
            error_payload = raw_manifest.get("error")
            error_type = "WorkerFailure"
            error_message = "OpenDataLoader Hybrid worker failed"

            if isinstance(error_payload, dict):
                error_type = str(error_payload.get("type") or error_type)
                error_message = str(
                    error_payload.get("message") or error_message
                )

            stderr_tail = read_log_tail(stderr_log_path)
            detail = f"{error_type}: {error_message}"

            if stderr_tail:
                detail += f"\nWorker stderr tail:\n{stderr_tail}"

            raise OpenDataLoaderHybridWorkerError(detail)

        try:
            manifest = OpenDataLoaderHybridWorkerManifest.model_validate(
                raw_manifest
            )
        except ValidationError as error:
            raise OpenDataLoaderHybridManifestError(
                "Hybrid worker result manifest does not match schema"
            ) from error

        self._validate_integrity(
            manifest=manifest,
            input_pdf=input_path,
            page_number=page_number,
            reason_codes=normalized_reasons,
            source_sha256=source_sha256_before,
            raw_artifact_directory=raw_artifact_directory,
        )

        return OpenDataLoaderHybridPageRun(
            run_id=active_run_id,
            source_page_number=page_number,
            source_document_page_count=page_count,
            source_sha256=source_sha256_before,
            reason_codes=tuple(normalized_reasons),
            run_directory=run_directory,
            raw_artifact_directory=raw_artifact_directory,
            result_manifest_path=result_manifest_path,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            manifest=manifest,
        )

    def _validate_configuration(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        output_root: Path,
    ) -> None:
        if not python_executable.is_file():
            raise OpenDataLoaderHybridConfigurationError(
                f"Python executable not found: {python_executable}"
            )

        if not worker_script.is_file():
            raise OpenDataLoaderHybridConfigurationError(
                f"Hybrid worker script not found: {worker_script}"
            )

        if self.config.timeout_seconds <= 0:
            raise OpenDataLoaderHybridConfigurationError(
                "timeout_seconds must be positive"
            )

        if self.config.hybrid_timeout_ms < 1:
            raise OpenDataLoaderHybridConfigurationError(
                "hybrid_timeout_ms must be positive"
            )

        if output_root.exists() and not output_root.is_dir():
            raise OpenDataLoaderHybridConfigurationError(
                f"Output root is not a directory: {output_root}"
            )

    @staticmethod
    def _validate_input(input_pdf: Path, page_number: int) -> int:
        if not input_pdf.is_file():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        if input_pdf.suffix.casefold() != ".pdf":
            raise ValueError("OpenDataLoader Hybrid accepts PDF only")

        if input_pdf.stat().st_size < 1:
            raise ValueError("Input PDF is empty")

        try:
            with fitz.open(input_pdf) as document:
                page_count = document.page_count
        except Exception as error:
            raise ValueError("Input PDF cannot be opened") from error

        if page_number < 1 or page_number > page_count:
            raise ValueError(
                f"page_number must be between 1 and {page_count}"
            )

        return page_count

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
            return

        raise ValueError(
            "run_id must contain only letters, numbers, dot, "
            "underscore, or hyphen"
        )

    @staticmethod
    def _read_raw_manifest(path: Path) -> dict:
        if not path.is_file():
            raise OpenDataLoaderHybridWorkerError(
                "Hybrid worker did not create result manifest"
            )

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OpenDataLoaderHybridManifestError(
                "Hybrid result manifest is not valid UTF-8 JSON"
            ) from error

        if not isinstance(payload, dict):
            raise OpenDataLoaderHybridManifestError(
                "Hybrid result manifest root must be an object"
            )

        return payload

    def _validate_integrity(
        self,
        *,
        manifest: OpenDataLoaderHybridWorkerManifest,
        input_pdf: Path,
        page_number: int,
        reason_codes: list[str],
        source_sha256: str,
        raw_artifact_directory: Path,
    ) -> None:
        if manifest.input.path.resolve() != input_pdf:
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest input path does not match requested PDF"
            )

        if manifest.input.name != input_pdf.name:
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest input name does not match requested PDF"
            )

        if manifest.input.size_bytes != input_pdf.stat().st_size:
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest input size does not match requested PDF"
            )

        if manifest.input.sha256 != source_sha256:
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest input SHA256 does not match requested PDF"
            )

        if manifest.request.source_page_number != page_number:
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest page does not match requested page"
            )

        if manifest.request.reason_codes != reason_codes:
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest reason codes do not match routing request"
            )

        if manifest.request.hybrid_backend != HYBRID_BACKEND:
            raise OpenDataLoaderHybridIntegrityError(
                "Unexpected Hybrid backend"
            )

        if manifest.request.hybrid_mode != HYBRID_MODE:
            raise OpenDataLoaderHybridIntegrityError(
                "Hybrid fallback must use full mode"
            )

        if manifest.request.hybrid_fallback is not False:
            raise OpenDataLoaderHybridIntegrityError(
                "Java fallback must remain disabled"
            )

        if manifest.attempt_number != ODL_HYBRID_MAX_ATTEMPTS:
            raise OpenDataLoaderHybridIntegrityError(
                "Hybrid attempt count violates roadmap cap"
            )

        if manifest.max_attempts != ODL_HYBRID_MAX_ATTEMPTS:
            raise OpenDataLoaderHybridIntegrityError(
                "Hybrid max attempts violates roadmap cap"
            )

        if (
            self.config.expected_engine_version is not None
            and manifest.engine_version
            != self.config.expected_engine_version
        ):
            raise OpenDataLoaderHybridIntegrityError(
                "OpenDataLoader engine version mismatch: expected "
                f"{self.config.expected_engine_version}, got "
                f"{manifest.engine_version}"
            )

        expected_raw_directory = raw_artifact_directory.resolve()

        if (
            manifest.output.artifact_directory.resolve()
            != expected_raw_directory
        ):
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest artifact directory does not match run directory"
            )

        artifact_records = [
            manifest.output.markdown,
            manifest.output.structured_json,
            *manifest.output.images,
        ]
        recorded_relative_paths: set[str] = set()

        for artifact in artifact_records:
            relative_path = Path(artifact.relative_path)

            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise OpenDataLoaderHybridIntegrityError(
                    "Manifest contains an unsafe artifact path"
                )

            expected_path = (
                expected_raw_directory / relative_path
            ).resolve()

            if (
                expected_path == expected_raw_directory
                or expected_raw_directory not in expected_path.parents
            ):
                raise OpenDataLoaderHybridIntegrityError(
                    "Artifact path escapes raw run directory"
                )

            if artifact.path.resolve() != expected_path:
                raise OpenDataLoaderHybridIntegrityError(
                    "Absolute and relative artifact paths disagree"
                )

            if not expected_path.is_file():
                raise OpenDataLoaderHybridIntegrityError(
                    f"Recorded artifact is missing: {relative_path}"
                )

            normalized_path = relative_path.as_posix()

            if normalized_path in recorded_relative_paths:
                raise OpenDataLoaderHybridIntegrityError(
                    "Artifact relative paths must be unique"
                )

            recorded_relative_paths.add(normalized_path)

            if artifact.size_bytes != expected_path.stat().st_size:
                raise OpenDataLoaderHybridIntegrityError(
                    f"Artifact size mismatch: {relative_path}"
                )

            if (
                self.config.verify_artifact_hashes
                and artifact.sha256 != sha256_file(expected_path)
            ):
                raise OpenDataLoaderHybridIntegrityError(
                    f"Artifact SHA256 mismatch: {relative_path}"
                )

        actual_relative_paths = {
            path.relative_to(expected_raw_directory).as_posix()
            for path in expected_raw_directory.rglob("*")
            if path.is_file()
        }

        if actual_relative_paths != recorded_relative_paths:
            raise OpenDataLoaderHybridIntegrityError(
                "Raw artifact inventory differs from manifest"
            )

        if len(actual_relative_paths) != manifest.output.total_file_count:
            raise OpenDataLoaderHybridIntegrityError(
                "Manifest total_file_count is inconsistent"
            )
