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
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


Sha256Hex = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{64}$"),
]


class ManifestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class WorkerArtifactRecord(ManifestModel):
    path: Path
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    sha256: Sha256Hex


class WorkerImageHashGroup(ManifestModel):
    content_sha256: Sha256Hex
    physical_paths: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if len(self.physical_paths) != len(
            set(self.physical_paths)
        ):
            raise ValueError("physical_paths must be unique")

        return self


class WorkerInputRecord(ManifestModel):
    path: Path
    name: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    sha256: Sha256Hex


class WorkerOutputRecord(ManifestModel):
    artifact_directory: Path
    page_count: int = Field(ge=1)
    total_file_count: int = Field(ge=2)
    markdown_character_count: int = Field(ge=1)
    markdown_broken_character_count: int = Field(ge=0)
    markdown: WorkerArtifactRecord
    structured_json: WorkerArtifactRecord
    images: list[WorkerArtifactRecord] = Field(default_factory=list)
    physical_image_file_count: int = Field(ge=0)
    unique_image_hash_count: int = Field(ge=0)
    duplicate_image_hash_group_count: int = Field(ge=0)
    image_hash_groups: list[WorkerImageHashGroup] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_image_inventory(self) -> Self:
        if self.physical_image_file_count != len(self.images):
            raise ValueError(
                "physical_image_file_count must equal images length"
            )

        if self.unique_image_hash_count != len(
            self.image_hash_groups
        ):
            raise ValueError(
                "unique_image_hash_count must equal hash group count"
            )

        expected_duplicate_groups = sum(
            1
            for group in self.image_hash_groups
            if len(group.physical_paths) > 1
        )

        if (
            self.duplicate_image_hash_group_count
            != expected_duplicate_groups
        ):
            raise ValueError(
                "duplicate_image_hash_group_count is inconsistent"
            )

        image_paths = [image.relative_path for image in self.images]
        grouped_paths = [
            path
            for group in self.image_hash_groups
            for path in group.physical_paths
        ]

        if len(image_paths) != len(set(image_paths)):
            raise ValueError("Image relative paths must be unique")

        if sorted(image_paths) != sorted(grouped_paths):
            raise ValueError(
                "Image hash groups must cover every image exactly once"
            )

        image_hashes_by_path = {
            image.relative_path: image.sha256
            for image in self.images
        }

        for group in self.image_hash_groups:
            for physical_path in group.physical_paths:
                if (
                    image_hashes_by_path[physical_path]
                    != group.content_sha256
                ):
                    raise ValueError(
                        "Image hash group does not match artifact SHA256"
                    )

        if Path(self.markdown.relative_path).suffix.casefold() != ".md":
            raise ValueError("Markdown artifact must use .md extension")

        if (
            Path(self.structured_json.relative_path).suffix.casefold()
            != ".json"
        ):
            raise ValueError(
                "Structured artifact must use .json extension"
            )

        minimum_file_count = 2 + len(self.images)

        if self.total_file_count < minimum_file_count:
            raise ValueError(
                "total_file_count is smaller than recorded artifacts"
            )

        return self


class OpenDataLoaderWorkerManifest(ManifestModel):
    schema_version: Literal["1.0"]
    worker_version: str = Field(min_length=1)
    status: Literal["PASS"]
    engine: Literal["OPENDATALOADER_LOCAL"]
    engine_version: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime
    elapsed_seconds: float = Field(ge=0)
    input: WorkerInputRecord
    output: WorkerOutputRecord

    @model_validator(mode="after")
    def validate_timing(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError(
                "finished_at cannot be before started_at"
            )

        return self


class OpenDataLoaderAdapterError(RuntimeError):
    """Base exception for the isolated OpenDataLoader adapter."""


class OpenDataLoaderConfigurationError(OpenDataLoaderAdapterError):
    pass


class OpenDataLoaderRunExistsError(OpenDataLoaderAdapterError):
    pass


class OpenDataLoaderTimeoutError(OpenDataLoaderAdapterError):
    pass


class OpenDataLoaderWorkerError(OpenDataLoaderAdapterError):
    pass


class OpenDataLoaderManifestError(OpenDataLoaderAdapterError):
    pass


class OpenDataLoaderIntegrityError(OpenDataLoaderAdapterError):
    pass


@dataclass(frozen=True, slots=True)
class OpenDataLoaderLocalConfig:
    python_executable: Path
    worker_script: Path
    output_root: Path
    timeout_seconds: float = 180.0
    verify_artifact_hashes: bool = True


@dataclass(frozen=True, slots=True)
class OpenDataLoaderLocalRun:
    run_id: str
    run_directory: Path
    raw_artifact_directory: Path
    result_manifest_path: Path
    stdout_log_path: Path
    stderr_log_path: Path
    manifest: OpenDataLoaderWorkerManifest


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


def read_log_tail(path: Path, maximum_characters: int = 4000) -> str:
    if not path.is_file():
        return ""

    content = path.read_text(encoding="utf-8", errors="replace")
    return content[-maximum_characters:]


def terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            [
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
            ],
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


class OpenDataLoaderLocalExtractor:
    def __init__(self, config: OpenDataLoaderLocalConfig) -> None:
        self.config = config

    def extract(
        self,
        input_pdf: Path | str,
        *,
        run_id: str | None = None,
    ) -> OpenDataLoaderLocalRun:
        python_executable = (
            self.config.python_executable.expanduser().resolve()
        )
        worker_script = self.config.worker_script.expanduser().resolve()
        output_root = self.config.output_root.expanduser().resolve()
        input_path = Path(input_pdf).expanduser().resolve()

        self._validate_configuration(
            python_executable=python_executable,
            worker_script=worker_script,
            output_root=output_root,
        )
        self._validate_input(input_path)

        active_run_id = run_id or utc_run_id()
        self._validate_run_id(active_run_id)

        output_root.mkdir(parents=True, exist_ok=True)
        run_directory = output_root / active_run_id

        try:
            run_directory.mkdir(parents=False, exist_ok=False)
        except FileExistsError as error:
            raise OpenDataLoaderRunExistsError(
                f"Extraction run already exists: {active_run_id}"
            ) from error

        raw_artifact_directory = run_directory / "raw"
        result_manifest_path = run_directory / "worker_result.json"
        stdout_log_path = run_directory / "worker_stdout.log"
        stderr_log_path = run_directory / "worker_stderr.log"

        command = [
            str(python_executable),
            str(worker_script),
            "--input-pdf",
            str(input_path),
            "--artifact-dir",
            str(raw_artifact_directory),
            "--result-json",
            str(result_manifest_path),
        ]

        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONIOENCODING"] = "utf-8"

        popen_arguments: dict[str, Any] = {
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
                raise OpenDataLoaderConfigurationError(
                    f"Unable to start OpenDataLoader worker: {error}"
                ) from error

            try:
                return_code = process.wait(
                    timeout=self.config.timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                terminate_process_tree(process)
                raise OpenDataLoaderTimeoutError(
                    "OpenDataLoader worker exceeded timeout of "
                    f"{self.config.timeout_seconds} seconds"
                ) from error

        raw_manifest = self._read_raw_manifest(result_manifest_path)

        if return_code != 0 or raw_manifest.get("status") != "PASS":
            error_payload = raw_manifest.get("error")
            error_type = "WorkerFailure"
            error_message = "OpenDataLoader worker failed"

            if isinstance(error_payload, dict):
                error_type = str(error_payload.get("type") or error_type)
                error_message = str(
                    error_payload.get("message") or error_message
                )

            stderr_tail = read_log_tail(stderr_log_path)
            detail = f"{error_type}: {error_message}"

            if stderr_tail:
                detail += f"\nWorker stderr tail:\n{stderr_tail}"

            raise OpenDataLoaderWorkerError(detail)

        try:
            manifest = OpenDataLoaderWorkerManifest.model_validate(
                raw_manifest
            )
        except ValidationError as error:
            raise OpenDataLoaderManifestError(
                "Worker result manifest does not match schema"
            ) from error

        self._validate_integrity(
            manifest=manifest,
            input_pdf=input_path,
            raw_artifact_directory=raw_artifact_directory,
        )

        return OpenDataLoaderLocalRun(
            run_id=active_run_id,
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
            raise OpenDataLoaderConfigurationError(
                f"Python executable not found: {python_executable}"
            )

        if not worker_script.is_file():
            raise OpenDataLoaderConfigurationError(
                f"Worker script not found: {worker_script}"
            )

        if self.config.timeout_seconds <= 0:
            raise OpenDataLoaderConfigurationError(
                "timeout_seconds must be positive"
            )

        if output_root.exists() and not output_root.is_dir():
            raise OpenDataLoaderConfigurationError(
                f"Output root is not a directory: {output_root}"
            )

    @staticmethod
    def _validate_input(input_pdf: Path) -> None:
        if not input_pdf.is_file():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        if input_pdf.suffix.casefold() != ".pdf":
            raise ValueError("OpenDataLoader Local accepts PDF only")

        if input_pdf.stat().st_size < 1:
            raise ValueError("Input PDF is empty")

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
            return

        raise ValueError(
            "run_id must contain only letters, numbers, dot, "
            "underscore, or hyphen"
        )

    @staticmethod
    def _read_raw_manifest(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise OpenDataLoaderWorkerError(
                "Worker did not create result manifest"
            )

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OpenDataLoaderManifestError(
                "Worker result manifest is not valid UTF-8 JSON"
            ) from error

        if not isinstance(payload, dict):
            raise OpenDataLoaderManifestError(
                "Worker result manifest root must be an object"
            )

        return payload

    def _validate_integrity(
        self,
        *,
        manifest: OpenDataLoaderWorkerManifest,
        input_pdf: Path,
        raw_artifact_directory: Path,
    ) -> None:
        if manifest.input.path.resolve() != input_pdf:
            raise OpenDataLoaderIntegrityError(
                "Manifest input path does not match requested PDF"
            )

        if manifest.input.name != input_pdf.name:
            raise OpenDataLoaderIntegrityError(
                "Manifest input name does not match requested PDF"
            )

        if manifest.input.size_bytes != input_pdf.stat().st_size:
            raise OpenDataLoaderIntegrityError(
                "Manifest input size does not match requested PDF"
            )

        if manifest.input.sha256 != sha256_file(input_pdf):
            raise OpenDataLoaderIntegrityError(
                "Manifest input SHA256 does not match requested PDF"
            )

        expected_raw_directory = raw_artifact_directory.resolve()

        if (
            manifest.output.artifact_directory.resolve()
            != expected_raw_directory
        ):
            raise OpenDataLoaderIntegrityError(
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
                raise OpenDataLoaderIntegrityError(
                    "Manifest contains an unsafe relative artifact path"
                )

            expected_path = (
                expected_raw_directory / relative_path
            ).resolve()

            if (
                expected_path == expected_raw_directory
                or expected_raw_directory not in expected_path.parents
            ):
                raise OpenDataLoaderIntegrityError(
                    "Artifact path escapes the raw run directory"
                )

            if artifact.path.resolve() != expected_path:
                raise OpenDataLoaderIntegrityError(
                    "Absolute and relative artifact paths disagree"
                )

            if not expected_path.is_file():
                raise OpenDataLoaderIntegrityError(
                    f"Recorded artifact is missing: {relative_path}"
                )

            normalized_relative_path = relative_path.as_posix()

            if normalized_relative_path in recorded_relative_paths:
                raise OpenDataLoaderIntegrityError(
                    "Artifact relative paths must be unique"
                )

            recorded_relative_paths.add(normalized_relative_path)

            if artifact.size_bytes != expected_path.stat().st_size:
                raise OpenDataLoaderIntegrityError(
                    f"Artifact size mismatch: {relative_path}"
                )

            if (
                self.config.verify_artifact_hashes
                and artifact.sha256 != sha256_file(expected_path)
            ):
                raise OpenDataLoaderIntegrityError(
                    f"Artifact SHA256 mismatch: {relative_path}"
                )

        actual_relative_paths = {
            path.relative_to(expected_raw_directory).as_posix()
            for path in expected_raw_directory.rglob("*")
            if path.is_file()
        }

        if actual_relative_paths != recorded_relative_paths:
            raise OpenDataLoaderIntegrityError(
                "Raw artifact inventory differs from manifest"
            )

        if len(actual_relative_paths) != manifest.output.total_file_count:
            raise OpenDataLoaderIntegrityError(
                "Manifest total_file_count is inconsistent"
            )
