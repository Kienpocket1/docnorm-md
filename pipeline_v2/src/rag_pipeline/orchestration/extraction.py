from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.contracts import (
    ExtractedDocument,
    ExtractionQualityReport,
    FailureDomain,
    QualityDisposition,
    QualitySeverity,
    QuarantineRecord,
    QuarantineType,
    ResumeFrom,
    StageResult,
    StageStatus,
)
from rag_pipeline.extractors import (
    DocumentIdentity,
    OpenDataLoaderJsonMapper,
    OpenDataLoaderLocalRun,
    OpenDataLoaderMappingResult,
    OpenDataLoaderWorkerManifest,
)
from rag_pipeline.quality import (
    ExtractionQualityGate,
    ExtractionQualityGateResult,
)


EXTRACTION_ORCHESTRATOR_VERSION = "2026-08-17-5.1E.1"
EXTRACTION_COMMIT_SCHEMA_VERSION = "1.0"

DOCUMENT_FILENAME = "extracted_document.json"
QUALITY_REPORT_FILENAME = "extraction_quality_report.json"
SELECTION_FILENAME = "indexing_selection.json"
QUARANTINE_FILENAME = "quarantine_record.json"
STAGE_RESULT_FILENAME = "stage_result.json"
COMMIT_MANIFEST_FILENAME = "commit_manifest.json"

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NextAction = Literal[
    "NORMALIZE",
    "RUN_OPENDATALOADER_HYBRID",
    "OPEN_EXTRACTION_QUARANTINE",
]


class OrchestrationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class CommitArtifactRecord(OrchestrationModel):
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=1)
    sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_relative_path(self) -> Self:
        path = PurePosixPath(self.relative_path)

        if (
            path.is_absolute()
            or ".." in path.parts
            or self.relative_path != path.as_posix()
        ):
            raise ValueError("Artifact path must be safe and relative")

        return self


class IndexingSelection(OrchestrationModel):
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: Sha256Hex
    indexable_element_ids: list[str]
    filtered_element_ids: list[str]

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        indexable = self.indexable_element_ids
        filtered = self.filtered_element_ids

        if len(indexable) != len(set(indexable)):
            raise ValueError("indexable_element_ids must be unique")

        if len(filtered) != len(set(filtered)):
            raise ValueError("filtered_element_ids must be unique")

        if set(indexable) & set(filtered):
            raise ValueError("Indexable and filtered IDs must be disjoint")

        return self


class ExtractionCommitManifest(OrchestrationModel):
    schema_version: Literal["1.0"]
    orchestrator_version: str = Field(min_length=1)
    run_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: Sha256Hex
    local_run_id: str = Field(min_length=1)
    local_run_directory: str = Field(min_length=1)
    local_worker_manifest_path: str = Field(min_length=1)
    quality_disposition: QualityDisposition
    local_result_accepted: bool
    fallback_required: bool
    next_action: NextAction
    local_attempt_count: Literal[1] = 1
    automatic_retry_count: Literal[0] = 0
    quarantine_id: str | None = None
    artifact_count: int = Field(ge=4)
    artifacts: list[CommitArtifactRecord] = Field(min_length=4)
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_decision_and_inventory(self) -> Self:
        artifact_paths = [
            artifact.relative_path
            for artifact in self.artifacts
        ]

        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("Commit artifact paths must be unique")

        if self.artifact_count != len(self.artifacts):
            raise ValueError("artifact_count must equal artifacts length")

        accepted_dispositions = {
            QualityDisposition.PASS,
            QualityDisposition.PASS_WITH_WARNINGS,
        }

        if self.quality_disposition in accepted_dispositions:
            if (
                not self.local_result_accepted
                or self.fallback_required
                or self.next_action != "NORMALIZE"
                or self.quarantine_id is not None
            ):
                raise ValueError("Accepted commit decision is inconsistent")

        elif (
            self.quality_disposition
            == QualityDisposition.FALLBACK_REQUIRED
        ):
            if (
                self.local_result_accepted
                or not self.fallback_required
                or self.next_action
                != "RUN_OPENDATALOADER_HYBRID"
                or self.quarantine_id is not None
            ):
                raise ValueError("Fallback commit decision is inconsistent")

        elif (
            self.quality_disposition
            == QualityDisposition.QUARANTINE_REQUIRED
        ):
            if (
                self.local_result_accepted
                or self.fallback_required
                or self.next_action
                != "OPEN_EXTRACTION_QUARANTINE"
                or not self.quarantine_id
            ):
                raise ValueError(
                    "Quarantine commit decision is inconsistent"
                )

        return self


@dataclass(frozen=True, slots=True)
class ExtractionOrchestratorConfig:
    output_root: Path


@dataclass(frozen=True, slots=True)
class ExtractionCommitBundle:
    commit_directory: Path
    manifest: ExtractionCommitManifest
    document: ExtractedDocument
    quality_report: ExtractionQualityReport
    selection: IndexingSelection
    stage_result: StageResult
    quarantine_record: QuarantineRecord | None


@dataclass(frozen=True, slots=True)
class ExtractionOrchestrationResult:
    local_run: OpenDataLoaderLocalRun
    commit: ExtractionCommitBundle


class LocalExtractorLike(Protocol):
    def extract(
        self,
        input_pdf: Path | str,
        *,
        run_id: str | None = None,
    ) -> OpenDataLoaderLocalRun: ...


class MapperLike(Protocol):
    def map_run(
        self,
        run: OpenDataLoaderLocalRun,
        identity: DocumentIdentity,
    ) -> OpenDataLoaderMappingResult: ...


class QualityGateLike(Protocol):
    def evaluate(
        self,
        mapping: OpenDataLoaderMappingResult,
        manifest: OpenDataLoaderWorkerManifest,
    ) -> ExtractionQualityGateResult: ...


class ExtractionOrchestrationError(RuntimeError):
    """Base exception for extraction orchestration and commit loading."""


class ExtractionOrchestratorConfigurationError(
    ExtractionOrchestrationError
):
    pass


class ExtractionCommitExistsError(ExtractionOrchestrationError):
    pass


class ExtractionCommitIntegrityError(ExtractionOrchestrationError):
    pass


def utc_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    return f"extraction-{timestamp}-{uuid.uuid4().hex[:8]}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def model_json_bytes(model: BaseModel) -> bytes:
    return (model.model_dump_json(indent=2) + "\n").encode("utf-8")


def stable_identifier(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def validate_run_id(run_id: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        return

    raise ValueError(
        "run_id must contain only letters, numbers, dot, "
        "underscore, or hyphen"
    )


def write_payload(path: Path, payload: bytes) -> None:
    with path.open("xb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def artifact_record(path: Path, root: Path) -> CommitArtifactRecord:
    return CommitArtifactRecord(
        relative_path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def expected_state(
    disposition: QualityDisposition,
) -> tuple[bool, bool, NextAction, StageStatus]:
    if disposition in {
        QualityDisposition.PASS,
        QualityDisposition.PASS_WITH_WARNINGS,
    }:
        return True, False, "NORMALIZE", StageStatus.PASS

    if disposition == QualityDisposition.FALLBACK_REQUIRED:
        return (
            False,
            True,
            "RUN_OPENDATALOADER_HYBRID",
            StageStatus.FAIL,
        )

    return (
        False,
        False,
        "OPEN_EXTRACTION_QUARANTINE",
        StageStatus.QUARANTINED,
    )


class ExtractionOrchestrator:
    def __init__(
        self,
        config: ExtractionOrchestratorConfig,
        *,
        extractor: LocalExtractorLike | None = None,
        mapper: MapperLike | None = None,
        quality_gate: QualityGateLike | None = None,
    ) -> None:
        self.config = config
        self.extractor = extractor
        self.mapper = mapper or OpenDataLoaderJsonMapper()
        self.quality_gate = quality_gate or ExtractionQualityGate()

    def execute(
        self,
        input_pdf: Path | str,
        identity: DocumentIdentity,
        *,
        run_id: str | None = None,
        local_run_id: str | None = None,
    ) -> ExtractionOrchestrationResult:
        if self.extractor is None:
            raise ExtractionOrchestratorConfigurationError(
                "execute requires one configured Local extractor"
            )

        active_run_id = run_id or utc_run_id()
        validate_run_id(active_run_id)
        self._ensure_commit_target_is_available(active_run_id)
        active_local_run_id = (
            local_run_id or f"{active_run_id}-local"
        )

        local_run = self.extractor.extract(
            input_pdf,
            run_id=active_local_run_id,
        )

        return self.process_run(
            local_run,
            identity,
            run_id=active_run_id,
        )

    def process_run(
        self,
        local_run: OpenDataLoaderLocalRun,
        identity: DocumentIdentity,
        *,
        run_id: str | None = None,
    ) -> ExtractionOrchestrationResult:
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        active_run_id = run_id or utc_run_id()
        validate_run_id(active_run_id)
        self._ensure_commit_target_is_available(active_run_id)

        mapping = self.mapper.map_run(local_run, identity)
        decision = self.quality_gate.evaluate(
            mapping,
            local_run.manifest,
        )
        self._validate_mapping_and_decision(
            local_run=local_run,
            mapping=mapping,
            decision=decision,
        )
        selection = IndexingSelection(
            catalog_id=mapping.document.catalog_id,
            family_id=mapping.document.family_id,
            version_id=mapping.document.version_id,
            source_sha256=mapping.document.source_sha256,
            indexable_element_ids=list(
                decision.indexable_element_ids
            ),
            filtered_element_ids=list(
                decision.filtered_element_ids
            ),
        )
        quarantine_record = self._build_quarantine_record(
            decision.report
        )
        report = decision.report
        stage_result = self._build_stage_result(
            report=report,
            quarantine_record=quarantine_record,
            started_at=started_at,
            duration_seconds=time.monotonic() - started_monotonic,
        )
        bundle = self._commit(
            run_id=active_run_id,
            local_run=local_run,
            document=mapping.document,
            report=report,
            selection=selection,
            stage_result=stage_result,
            quarantine_record=quarantine_record,
        )

        return ExtractionOrchestrationResult(
            local_run=local_run,
            commit=bundle,
        )

    def _ensure_commit_target_is_available(self, run_id: str) -> None:
        output_root = self.config.output_root.expanduser().resolve()
        final_directory = output_root / run_id

        if final_directory.exists():
            raise ExtractionCommitExistsError(
                f"Extraction commit already exists: {run_id}"
            )

    @staticmethod
    def _validate_mapping_and_decision(
        *,
        local_run: OpenDataLoaderLocalRun,
        mapping: OpenDataLoaderMappingResult,
        decision: ExtractionQualityGateResult,
    ) -> None:
        document = mapping.document
        report = decision.report

        if document.source_sha256 != local_run.manifest.input.sha256:
            raise ExtractionOrchestrationError(
                "Mapped document SHA256 differs from Local manifest"
            )

        identity_values = (
            document.catalog_id,
            document.family_id,
            document.version_id,
            document.source_sha256,
        )
        report_identity_values = (
            report.catalog_id,
            report.family_id,
            report.version_id,
            report.source_sha256,
        )

        if identity_values != report_identity_values:
            raise ExtractionOrchestrationError(
                "Quality report identity differs from mapped document"
            )

        if (
            decision.indexable_element_ids
            != mapping.indexable_element_ids
            or decision.filtered_element_ids
            != mapping.filtered_element_ids
        ):
            raise ExtractionOrchestrationError(
                "Quality decision changed the mapper element selection"
            )

        expected_ids = {
            element.element_id
            for element in document.elements
        }
        selected_ids = set(decision.indexable_element_ids) | set(
            decision.filtered_element_ids
        )

        if expected_ids != selected_ids:
            raise ExtractionOrchestrationError(
                "Indexing selection does not cover every source element"
            )

        accepted, fallback, _, _ = expected_state(report.disposition)

        if (
            decision.local_result_accepted != accepted
            or decision.fallback_required != fallback
        ):
            raise ExtractionOrchestrationError(
                "Quality decision flags contradict its disposition"
            )

    @staticmethod
    def _build_quarantine_record(
        report: ExtractionQualityReport,
    ) -> QuarantineRecord | None:
        if (
            report.disposition
            != QualityDisposition.QUARANTINE_REQUIRED
        ):
            return None

        errors = [
            issue
            for issue in report.issues
            if issue.severity == QualitySeverity.ERROR
        ]
        primary_issue = errors[0]

        return QuarantineRecord(
            quarantine_id=stable_identifier(
                "extraction-quarantine",
                report.source_sha256,
                report.version_id,
                report.report_id,
            ),
            queue=QuarantineType.EXTRACTION,
            failure_domain=primary_issue.failure_domain,
            root_cause_stage=ResumeFrom.EXTRACTION_QUALITY,
            reason_code=primary_issue.reason_code,
            resume_from=ResumeFrom.EXTRACT,
            attempt_count=1,
            max_attempts=1,
            catalog_id=report.catalog_id,
            family_id=report.family_id,
            version_id=report.version_id,
            details={
                "quality_report_id": report.report_id,
                "issue_ids": [issue.issue_id for issue in errors],
                "reason_codes": [
                    issue.reason_code for issue in errors
                ],
                "automatic_retry_allowed": False,
            },
        )

    @staticmethod
    def _build_stage_result(
        *,
        report: ExtractionQualityReport,
        quarantine_record: QuarantineRecord | None,
        started_at: datetime,
        duration_seconds: float,
    ) -> StageResult:
        _, _, next_action, stage_status = expected_state(
            report.disposition
        )
        artifact_paths = [
            DOCUMENT_FILENAME,
            QUALITY_REPORT_FILENAME,
            SELECTION_FILENAME,
        ]

        if quarantine_record is not None:
            artifact_paths.append(QUARANTINE_FILENAME)

        message = f"Extraction quality next action: {next_action}"

        return StageResult(
            stage=ResumeFrom.EXTRACTION_QUALITY,
            status=stage_status,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_seconds=duration_seconds,
            artifact_paths=artifact_paths,
            metrics={
                "quality_disposition": report.disposition.value,
                "quality_issue_count": len(report.issues),
                "next_action": next_action,
                "local_attempt_count": 1,
                "automatic_retry_count": 0,
            },
            quarantine_id=(
                quarantine_record.quarantine_id
                if quarantine_record is not None
                else None
            ),
            message=message,
        )

    def _commit(
        self,
        *,
        run_id: str,
        local_run: OpenDataLoaderLocalRun,
        document: ExtractedDocument,
        report: ExtractionQualityReport,
        selection: IndexingSelection,
        stage_result: StageResult,
        quarantine_record: QuarantineRecord | None,
    ) -> ExtractionCommitBundle:
        output_root = self.config.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        final_directory = output_root / run_id

        if final_directory.exists():
            raise ExtractionCommitExistsError(
                f"Extraction commit already exists: {run_id}"
            )

        staging_directory = output_root / (
            f".{run_id}.staging-{uuid.uuid4().hex}"
        )
        staging_directory.mkdir(parents=False, exist_ok=False)

        try:
            models_by_filename: dict[str, BaseModel] = {
                DOCUMENT_FILENAME: document,
                QUALITY_REPORT_FILENAME: report,
                SELECTION_FILENAME: selection,
            }

            if quarantine_record is not None:
                models_by_filename[QUARANTINE_FILENAME] = (
                    quarantine_record
                )

            models_by_filename[STAGE_RESULT_FILENAME] = stage_result
            artifacts: list[CommitArtifactRecord] = []

            for filename, model in models_by_filename.items():
                artifact_path = staging_directory / filename
                write_payload(artifact_path, model_json_bytes(model))
                artifacts.append(
                    artifact_record(artifact_path, staging_directory)
                )

            accepted, fallback, next_action, _ = expected_state(
                report.disposition
            )
            manifest = ExtractionCommitManifest(
                schema_version=EXTRACTION_COMMIT_SCHEMA_VERSION,
                orchestrator_version=EXTRACTION_ORCHESTRATOR_VERSION,
                run_id=run_id,
                catalog_id=document.catalog_id,
                family_id=document.family_id,
                version_id=document.version_id,
                source_sha256=document.source_sha256,
                local_run_id=local_run.run_id,
                local_run_directory=str(local_run.run_directory),
                local_worker_manifest_path=str(
                    local_run.result_manifest_path
                ),
                quality_disposition=report.disposition,
                local_result_accepted=accepted,
                fallback_required=fallback,
                next_action=next_action,
                quarantine_id=(
                    quarantine_record.quarantine_id
                    if quarantine_record is not None
                    else None
                ),
                artifact_count=len(artifacts),
                artifacts=artifacts,
            )
            write_payload(
                staging_directory / COMMIT_MANIFEST_FILENAME,
                model_json_bytes(manifest),
            )
            load_extraction_commit(staging_directory)

            if final_directory.exists():
                raise ExtractionCommitExistsError(
                    f"Extraction commit already exists: {run_id}"
                )

            staging_directory.rename(final_directory)

        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
            raise

        return load_extraction_commit(final_directory)


def load_extraction_commit(
    commit_directory: Path | str,
) -> ExtractionCommitBundle:
    root = Path(commit_directory).expanduser().resolve()

    if not root.is_dir():
        raise ExtractionCommitIntegrityError(
            f"Extraction commit directory not found: {root}"
        )

    manifest_path = root / COMMIT_MANIFEST_FILENAME

    try:
        manifest = ExtractionCommitManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except Exception as error:
        raise ExtractionCommitIntegrityError(
            "Commit manifest is missing or invalid"
        ) from error

    recorded_paths = {
        artifact.relative_path
        for artifact in manifest.artifacts
    }
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_paths = recorded_paths | {COMMIT_MANIFEST_FILENAME}

    if actual_paths != expected_paths:
        raise ExtractionCommitIntegrityError(
            "Commit file inventory differs from manifest"
        )

    records_by_path = {
        artifact.relative_path: artifact
        for artifact in manifest.artifacts
    }

    for relative_path, record in records_by_path.items():
        artifact_path = (root / relative_path).resolve()

        if root not in artifact_path.parents:
            raise ExtractionCommitIntegrityError(
                "Commit artifact escapes its directory"
            )

        if artifact_path.stat().st_size != record.size_bytes:
            raise ExtractionCommitIntegrityError(
                f"Commit artifact size mismatch: {relative_path}"
            )

        if sha256_file(artifact_path) != record.sha256:
            raise ExtractionCommitIntegrityError(
                f"Commit artifact SHA256 mismatch: {relative_path}"
            )

    required_paths = {
        DOCUMENT_FILENAME,
        QUALITY_REPORT_FILENAME,
        SELECTION_FILENAME,
        STAGE_RESULT_FILENAME,
    }

    if manifest.quality_disposition == (
        QualityDisposition.QUARANTINE_REQUIRED
    ):
        required_paths.add(QUARANTINE_FILENAME)

    if recorded_paths != required_paths:
        raise ExtractionCommitIntegrityError(
            "Commit does not contain the required typed artifacts"
        )

    try:
        document = ExtractedDocument.model_validate_json(
            (root / DOCUMENT_FILENAME).read_text(encoding="utf-8")
        )
        report = ExtractionQualityReport.model_validate_json(
            (root / QUALITY_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        selection = IndexingSelection.model_validate_json(
            (root / SELECTION_FILENAME).read_text(encoding="utf-8")
        )
        stage_result = StageResult.model_validate_json(
            (root / STAGE_RESULT_FILENAME).read_text(encoding="utf-8")
        )
        quarantine_record = (
            QuarantineRecord.model_validate_json(
                (root / QUARANTINE_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            if QUARANTINE_FILENAME in recorded_paths
            else None
        )
    except Exception as error:
        raise ExtractionCommitIntegrityError(
            "Committed typed artifact is invalid"
        ) from error

    _validate_loaded_commit(
        manifest=manifest,
        document=document,
        report=report,
        selection=selection,
        stage_result=stage_result,
        quarantine_record=quarantine_record,
    )

    return ExtractionCommitBundle(
        commit_directory=root,
        manifest=manifest,
        document=document,
        quality_report=report,
        selection=selection,
        stage_result=stage_result,
        quarantine_record=quarantine_record,
    )


def _validate_loaded_commit(
    *,
    manifest: ExtractionCommitManifest,
    document: ExtractedDocument,
    report: ExtractionQualityReport,
    selection: IndexingSelection,
    stage_result: StageResult,
    quarantine_record: QuarantineRecord | None,
) -> None:
    identity = (
        manifest.catalog_id,
        manifest.family_id,
        manifest.version_id,
        manifest.source_sha256,
    )

    if identity != (
        document.catalog_id,
        document.family_id,
        document.version_id,
        document.source_sha256,
    ):
        raise ExtractionCommitIntegrityError(
            "Document identity differs from commit manifest"
        )

    if identity != (
        report.catalog_id,
        report.family_id,
        report.version_id,
        report.source_sha256,
    ):
        raise ExtractionCommitIntegrityError(
            "Quality identity differs from commit manifest"
        )

    if identity != (
        selection.catalog_id,
        selection.family_id,
        selection.version_id,
        selection.source_sha256,
    ):
        raise ExtractionCommitIntegrityError(
            "Selection identity differs from commit manifest"
        )

    if report.disposition != manifest.quality_disposition:
        raise ExtractionCommitIntegrityError(
            "Quality disposition differs from commit manifest"
        )

    document_ids = {
        element.element_id
        for element in document.elements
    }
    selection_ids = set(selection.indexable_element_ids) | set(
        selection.filtered_element_ids
    )

    if document_ids != selection_ids:
        raise ExtractionCommitIntegrityError(
            "Committed selection does not cover document elements"
        )

    _, _, next_action, expected_stage_status = expected_state(
        manifest.quality_disposition
    )

    if stage_result.stage != ResumeFrom.EXTRACTION_QUALITY:
        raise ExtractionCommitIntegrityError(
            "Stage result does not describe extraction quality"
        )

    if stage_result.status != expected_stage_status:
        raise ExtractionCommitIntegrityError(
            "Stage status differs from quality disposition"
        )

    if stage_result.metrics.get("next_action") != next_action:
        raise ExtractionCommitIntegrityError(
            "Stage next action differs from commit manifest"
        )

    expected_artifact_paths = [
        DOCUMENT_FILENAME,
        QUALITY_REPORT_FILENAME,
        SELECTION_FILENAME,
    ]

    if quarantine_record is not None:
        expected_artifact_paths.append(QUARANTINE_FILENAME)

    if stage_result.artifact_paths != expected_artifact_paths:
        raise ExtractionCommitIntegrityError(
            "Stage artifact paths differ from committed artifacts"
        )

    if manifest.quarantine_id is None:
        if quarantine_record is not None:
            raise ExtractionCommitIntegrityError(
                "Unexpected quarantine record in accepted/fallback commit"
            )
    elif (
        quarantine_record is None
        or quarantine_record.quarantine_id != manifest.quarantine_id
        or stage_result.quarantine_id != manifest.quarantine_id
    ):
        raise ExtractionCommitIntegrityError(
            "Quarantine identity differs across committed artifacts"
        )
