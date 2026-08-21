from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from rag_pipeline.contracts import (
    AssetReference,
    EngineName,
    ExtractedAsset,
    ExtractedDocument,
    ExtractionQualityIssue,
    QualitySeverity,
    ResumeFrom,
    SourceElement,
    StageResult,
    StageStatus,
)
from rag_pipeline.extractors import (
    DocumentIdentity,
    OpenDataLoaderHybridPageRun,
    OpenDataLoaderHybridWorkerManifest,
    OpenDataLoaderJsonMapper,
)
from rag_pipeline.failure import (
    MINERU_MAX_ATTEMPTS,
    ODL_HYBRID_MAX_ATTEMPTS,
    ExtractionRoutingPlan,
    FallbackAction,
    RouteKind,
)
from rag_pipeline.orchestration.extraction import (
    CommitArtifactRecord,
    artifact_record,
    model_json_bytes,
    sha256_file,
    stable_identifier,
    validate_run_id,
    write_payload,
)
from rag_pipeline.validators.extraction import ExtractionPageValidator


HYBRID_REVERIFY_STAGE_VERSION = "2026-08-21-6.1C.1"
HYBRID_REVERIFY_COMMIT_SCHEMA_VERSION = "1.0"

DOCUMENT_FILENAME = "reverified_document.json"
REPORT_FILENAME = "hybrid_reverification_report.json"
CONTINUATION_FILENAME = "fallback_continuation_plan.json"
STAGE_RESULT_FILENAME = "stage_result.json"
COMMIT_MANIFEST_FILENAME = "commit_manifest.json"

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NextAction = Literal[
    "NORMALIZE",
    "MINERU_PAGE",
    "RUN_REMAINING_FALLBACKS",
]
PageDecision = Literal["ACCEPT_HYBRID", "RUN_MINERU_PAGE"]


class HybridReverifyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class HybridPageReverificationResult(HybridReverifyModel):
    page_number: int = Field(ge=1)
    hybrid_run_id: str = Field(min_length=1)
    route_planned: bool
    route_ids: list[str]
    requested_reason_codes: list[str]
    worker_reason_codes: list[str] = Field(min_length=1)
    route_reason_match: bool
    original_element_ids: list[str]
    candidate_element_ids: list[str]
    original_element_count: int = Field(ge=0)
    candidate_element_count: int = Field(ge=0)
    mapping_issue_count: int = Field(ge=0)
    mapping_issue_reason_codes: list[str]
    quality_issue_reason_codes: list[str]
    provenance_coverage: float = Field(ge=0, le=1)
    passed: bool
    decision: PageDecision

    @model_validator(mode="after")
    def validate_counts_and_decision(self) -> Self:
        if self.original_element_count != len(self.original_element_ids):
            raise ValueError("original_element_count is inconsistent")

        if self.candidate_element_count != len(self.candidate_element_ids):
            raise ValueError("candidate_element_count is inconsistent")

        expected_decision: PageDecision = (
            "ACCEPT_HYBRID" if self.passed else "RUN_MINERU_PAGE"
        )

        if self.decision != expected_decision:
            raise ValueError("Page decision is inconsistent")

        return self


class MinerUPageRequest(HybridReverifyModel):
    page_number: int = Field(ge=1)
    route_ids: list[str] = Field(min_length=1)
    reason_codes: list[str] = Field(min_length=1)
    engine: Literal[EngineName.MINERU] = EngineName.MINERU
    max_attempts: Literal[1] = MINERU_MAX_ATTEMPTS
    condition: Literal["odl_hybrid_reverify_failed"] = (
        "odl_hybrid_reverify_failed"
    )


class FallbackContinuationPlan(HybridReverifyModel):
    next_action: NextAction
    mineru_pages: list[MinerUPageRequest]
    resolved_route_ids: list[str]
    unresolved_route_ids: list[str]
    external_model_required: bool
    provenance_repair_required: bool

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        mineru_page_numbers = [item.page_number for item in self.mineru_pages]

        if len(mineru_page_numbers) != len(set(mineru_page_numbers)):
            raise ValueError("MinerU page requests must be unique")

        if self.next_action == "MINERU_PAGE" and not self.mineru_pages:
            raise ValueError("MINERU_PAGE requires at least one page")

        if self.next_action != "MINERU_PAGE" and self.mineru_pages:
            raise ValueError("MinerU pages require MINERU_PAGE next action")

        if self.next_action == "NORMALIZE" and self.unresolved_route_ids:
            raise ValueError("NORMALIZE cannot contain unresolved routes")

        return self


class HybridReverificationReport(HybridReverifyModel):
    revalidator_version: str = Field(min_length=1)
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: Sha256Hex
    source_page_count: int = Field(ge=1)
    page_results: list[HybridPageReverificationResult] = Field(min_length=1)
    requested_page_numbers: list[int] = Field(min_length=1)
    accepted_page_numbers: list[int]
    mineru_page_numbers: list[int]
    original_element_count: int = Field(ge=1)
    merged_element_count: int = Field(ge=1)
    replaced_element_count: int = Field(ge=0)
    replacement_element_count: int = Field(ge=0)
    provenance_coverage: float = Field(ge=0, le=1)
    external_calls: Literal[0] = 0
    ocr_invocations: Literal[0] = 0
    next_action: NextAction

    @model_validator(mode="after")
    def validate_page_sets(self) -> Self:
        result_pages = [result.page_number for result in self.page_results]

        if result_pages != self.requested_page_numbers:
            raise ValueError("Requested pages differ from page results")

        if len(result_pages) != len(set(result_pages)):
            raise ValueError("Page results must be unique")

        accepted = [result.page_number for result in self.page_results if result.passed]
        failed = [result.page_number for result in self.page_results if not result.passed]

        if accepted != self.accepted_page_numbers:
            raise ValueError("accepted_page_numbers is inconsistent")

        if failed != self.mineru_page_numbers:
            raise ValueError("mineru_page_numbers is inconsistent")

        return self


class HybridInputRecord(HybridReverifyModel):
    run_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    worker_manifest_path: str = Field(min_length=1)
    worker_manifest_sha256: Sha256Hex
    raw_artifact_count: int = Field(ge=2)


class HybridReverificationCommitManifest(HybridReverifyModel):
    schema_version: Literal["1.0"]
    revalidator_version: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: Sha256Hex
    input_extraction_run_id: str = Field(min_length=1)
    input_extraction_commit_directory: str = Field(min_length=1)
    input_extraction_manifest_sha256: Sha256Hex
    input_hybrid_runs: list[HybridInputRecord] = Field(min_length=1)
    requested_page_numbers: list[int] = Field(min_length=1)
    accepted_page_numbers: list[int]
    mineru_page_numbers: list[int]
    next_action: NextAction
    artifact_count: Literal[4]
    artifacts: list[CommitArtifactRecord] = Field(min_length=4, max_length=4)
    committed_at: datetime

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        paths = [artifact.relative_path for artifact in self.artifacts]

        if len(paths) != len(set(paths)):
            raise ValueError("Commit artifact paths must be unique")

        if self.artifact_count != len(paths):
            raise ValueError("artifact_count is inconsistent")

        if self.requested_page_numbers != sorted(set(self.requested_page_numbers)):
            raise ValueError("requested_page_numbers must be sorted and unique")

        return self


@dataclass(frozen=True, slots=True)
class HybridReverificationConfig:
    output_root: Path


@dataclass(frozen=True, slots=True)
class HybridReverificationOutcome:
    document: ExtractedDocument
    report: HybridReverificationReport
    continuation_plan: FallbackContinuationPlan


@dataclass(frozen=True, slots=True)
class HybridReverificationCommitBundle:
    commit_directory: Path
    manifest: HybridReverificationCommitManifest
    document: ExtractedDocument
    report: HybridReverificationReport
    continuation_plan: FallbackContinuationPlan
    stage_result: StageResult


class HybridReverificationError(RuntimeError):
    """Base exception for Stage 6.1C re-verification."""


class HybridReverificationIntegrityError(HybridReverificationError):
    pass


class HybridReverificationCommitExistsError(HybridReverificationError):
    pass


def utc_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"stage6_1c_{stamp}_{uuid.uuid4().hex[:8]}"


def load_hybrid_page_run(run_directory: Path | str) -> OpenDataLoaderHybridPageRun:
    root = Path(run_directory).expanduser().resolve()

    if not root.is_dir():
        raise HybridReverificationIntegrityError(
            f"Hybrid run directory not found: {root}"
        )

    manifest_path = root / "worker_result.json"

    try:
        manifest = OpenDataLoaderHybridWorkerManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise HybridReverificationIntegrityError(
            "Hybrid worker manifest is missing or invalid"
        ) from error

    raw_root = (root / "raw").resolve()

    if manifest.output.artifact_directory.resolve() != raw_root:
        raise HybridReverificationIntegrityError(
            "Hybrid artifact directory differs from run/raw"
        )

    records = [
        manifest.output.markdown,
        manifest.output.structured_json,
        *manifest.output.images,
    ]
    recorded_paths: set[str] = set()

    for record in records:
        relative = PurePosixPath(record.relative_path)

        if relative.is_absolute() or ".." in relative.parts:
            raise HybridReverificationIntegrityError(
                "Hybrid manifest contains an unsafe artifact path"
            )

        expected_path = (raw_root / Path(*relative.parts)).resolve()

        if raw_root not in expected_path.parents:
            raise HybridReverificationIntegrityError(
                "Hybrid artifact escapes raw directory"
            )

        if record.path.resolve() != expected_path:
            raise HybridReverificationIntegrityError(
                "Hybrid artifact absolute and relative paths disagree"
            )

        if not expected_path.is_file():
            raise HybridReverificationIntegrityError(
                f"Hybrid artifact is missing: {relative.as_posix()}"
            )

        if expected_path.stat().st_size != record.size_bytes:
            raise HybridReverificationIntegrityError(
                f"Hybrid artifact size mismatch: {relative.as_posix()}"
            )

        if sha256_file(expected_path) != record.sha256:
            raise HybridReverificationIntegrityError(
                f"Hybrid artifact SHA256 mismatch: {relative.as_posix()}"
            )

        if relative.as_posix() in recorded_paths:
            raise HybridReverificationIntegrityError(
                "Hybrid artifact paths must be unique"
            )

        recorded_paths.add(relative.as_posix())

    actual_paths = {
        path.relative_to(raw_root).as_posix()
        for path in raw_root.rglob("*")
        if path.is_file()
    }

    if actual_paths != recorded_paths:
        raise HybridReverificationIntegrityError(
            "Hybrid raw inventory differs from worker manifest"
        )

    if len(records) != manifest.output.total_file_count:
        raise HybridReverificationIntegrityError(
            "Hybrid total file count is inconsistent"
        )

    return OpenDataLoaderHybridPageRun(
        run_id=root.name,
        source_page_number=manifest.request.source_page_number,
        source_document_page_count=manifest.output.page_count,
        source_sha256=manifest.input.sha256,
        reason_codes=tuple(manifest.request.reason_codes),
        run_directory=root,
        raw_artifact_directory=raw_root,
        result_manifest_path=manifest_path,
        stdout_log_path=root / "worker_stdout.log",
        stderr_log_path=root / "worker_stderr.log",
        manifest=manifest,
    )


class HybridReverifyMergeStage:
    """Re-verify Hybrid page output and merge only accepted replacements."""

    def __init__(
        self,
        config: HybridReverificationConfig,
        *,
        mapper: OpenDataLoaderJsonMapper | None = None,
        validator: ExtractionPageValidator | None = None,
    ) -> None:
        self.config = config
        self.mapper = mapper or OpenDataLoaderJsonMapper()
        self.validator = validator or ExtractionPageValidator()

    def reverify(
        self,
        base_document: ExtractedDocument,
        routing_plan: ExtractionRoutingPlan,
        hybrid_runs: list[OpenDataLoaderHybridPageRun],
        *,
        allow_unplanned_pilot_runs: bool = False,
    ) -> HybridReverificationOutcome:
        if base_document.page_count is None:
            raise ValueError("Hybrid page re-verification requires a paginated PDF")

        if not hybrid_runs:
            raise ValueError("At least one Hybrid page run is required")

        route_by_page = self._planned_hybrid_routes(routing_plan)
        run_by_page: dict[int, OpenDataLoaderHybridPageRun] = {}

        for run in hybrid_runs:
            page_number = run.source_page_number

            if page_number in run_by_page:
                raise ValueError(f"Duplicate Hybrid run for page {page_number}")

            if page_number not in route_by_page and not allow_unplanned_pilot_runs:
                raise ValueError(
                    f"Hybrid run page {page_number} is absent from routing plan"
                )

            self._validate_run_identity(base_document, run)
            run_by_page[page_number] = run

        missing_pages = sorted(set(route_by_page) - set(run_by_page))

        if missing_pages:
            raise ValueError(
                "Routing plan has Hybrid pages without runs: "
                + ", ".join(str(page) for page in missing_pages)
            )

        current_document = base_document
        page_results: list[HybridPageReverificationResult] = []
        resolved_route_ids: set[str] = set()
        mineru_requests: list[MinerUPageRequest] = []
        replaced_count = 0
        replacement_count = 0

        identity = DocumentIdentity(
            catalog_id=base_document.catalog_id,
            family_id=base_document.family_id,
            version_id=base_document.version_id,
        )

        for page_number, run in sorted(run_by_page.items()):
            planned = route_by_page.get(page_number)
            route_ids = (
                list(planned["route_ids"])
                if planned is not None
                else [f"pilot-page-{page_number}"]
            )
            requested_reasons = (
                list(planned["reason_codes"])
                if planned is not None
                else list(run.reason_codes)
            )
            worker_reasons = list(run.reason_codes)
            original_elements = self._page_elements(current_document, page_number)
            mapping_issues: list[ExtractionQualityIssue] = []
            quality_reason_codes: list[str] = []
            candidate_elements: list[SourceElement] = []
            candidate_assets: list[ExtractedAsset] = []
            page_provenance_coverage = 0.0
            accepted = False
            provisional: ExtractedDocument | None = None

            try:
                mapping = self.mapper.map_run(run, identity)
                mapping_issues = self._relevant_mapping_issues(
                    mapping.mapping_issues,
                    page_number=page_number,
                    candidate_element_ids={
                        element.element_id for element in mapping.document.elements
                    },
                    candidate_asset_ids={asset.asset_id for asset in mapping.document.assets},
                )
                candidate_elements = self._hybrid_elements(
                    mapping.document.elements,
                    run=run,
                    page_number=page_number,
                    route_ids=route_ids,
                    requested_reason_codes=requested_reasons,
                )
                candidate_assets = self._hybrid_assets(mapping.document.assets, run)

                if candidate_elements:
                    provisional = self._merge_page(
                        current_document,
                        page_number=page_number,
                        replacement_elements=candidate_elements,
                        replacement_assets=candidate_assets,
                    )
                    assessment = self.validator.evaluate(
                        provisional,
                        inherited_issues=mapping_issues,
                    )
                    page_quality = next(
                        (
                            result
                            for result in assessment.page_results
                            if result.page_number == page_number
                        ),
                        None,
                    )

                    if page_quality is None:
                        raise HybridReverificationIntegrityError(
                            "Validator omitted the requested page"
                        )

                    candidate_ids = {element.element_id for element in candidate_elements}
                    provenance_issues = [
                        issue
                        for issue in assessment.provenance.issues
                        if candidate_ids & set(issue.element_ids)
                    ]
                    uncovered_ids = {
                        element_id
                        for issue in provenance_issues
                        for element_id in issue.element_ids
                        if element_id in candidate_ids
                    }
                    page_provenance_coverage = (
                        (len(candidate_ids) - len(uncovered_ids)) / len(candidate_ids)
                    )
                    quality_issues = self._deduplicate_issues(
                        [*page_quality.issues, *mapping_issues]
                    )
                    quality_reason_codes = sorted(
                        {issue.reason_code for issue in quality_issues}
                    )
                    accepted = (
                        page_quality.passed
                        and page_provenance_coverage == 1.0
                        and not any(self._issue_fails(issue) for issue in mapping_issues)
                    )
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
                quality_reason_codes = [
                    f"hybrid_mapping_exception:{type(error).__name__}"
                ]

            if accepted and provisional is not None:
                current_document = provisional
                resolved_route_ids.update(route_ids)
                replaced_count += len(original_elements)
                replacement_count += len(candidate_elements)
            else:
                mineru_requests.append(
                    MinerUPageRequest(
                        page_number=page_number,
                        route_ids=sorted(route_ids),
                        reason_codes=sorted(
                            set(requested_reasons)
                            | set(quality_reason_codes)
                            | {"odl_hybrid_reverify_failed"}
                        ),
                    )
                )

            page_results.append(
                HybridPageReverificationResult(
                    page_number=page_number,
                    hybrid_run_id=run.run_id,
                    route_planned=planned is not None,
                    route_ids=sorted(route_ids),
                    requested_reason_codes=sorted(requested_reasons),
                    worker_reason_codes=sorted(worker_reasons),
                    route_reason_match=(
                        planned is not None
                        and set(requested_reasons) == set(worker_reasons)
                    ),
                    original_element_ids=[
                        element.element_id for element in original_elements
                    ],
                    candidate_element_ids=[
                        element.element_id for element in candidate_elements
                    ],
                    original_element_count=len(original_elements),
                    candidate_element_count=len(candidate_elements),
                    mapping_issue_count=len(mapping_issues),
                    mapping_issue_reason_codes=sorted(
                        {issue.reason_code for issue in mapping_issues}
                    ),
                    quality_issue_reason_codes=quality_reason_codes,
                    provenance_coverage=round(page_provenance_coverage, 8),
                    passed=accepted,
                    decision=(
                        "ACCEPT_HYBRID" if accepted else "RUN_MINERU_PAGE"
                    ),
                )
            )

        fallback_route_ids = {
            route.route_id
            for route in routing_plan.routes
            if route.fallback_required
        }
        unresolved_route_ids = fallback_route_ids - resolved_route_ids

        if mineru_requests:
            next_action: NextAction = "MINERU_PAGE"
            unresolved_route_ids.update(
                route_id
                for request in mineru_requests
                for route_id in request.route_ids
            )
        elif unresolved_route_ids:
            next_action = "RUN_REMAINING_FALLBACKS"
        else:
            next_action = "NORMALIZE"

        unresolved_routes = [
            route
            for route in routing_plan.routes
            if route.route_id in unresolved_route_ids
        ]
        continuation = FallbackContinuationPlan(
            next_action=next_action,
            mineru_pages=mineru_requests,
            resolved_route_ids=sorted(resolved_route_ids),
            unresolved_route_ids=sorted(unresolved_route_ids),
            external_model_required=any(
                route.external_model_required for route in unresolved_routes
            ),
            provenance_repair_required=any(
                route.route_kind == RouteKind.PROVENANCE
                for route in unresolved_routes
            ),
        )
        final_assessment = self.validator.evaluate(current_document)
        report = HybridReverificationReport(
            revalidator_version=HYBRID_REVERIFY_STAGE_VERSION,
            catalog_id=base_document.catalog_id,
            family_id=base_document.family_id,
            version_id=base_document.version_id,
            source_sha256=base_document.source_sha256,
            source_page_count=base_document.page_count,
            page_results=page_results,
            requested_page_numbers=[result.page_number for result in page_results],
            accepted_page_numbers=[
                result.page_number for result in page_results if result.passed
            ],
            mineru_page_numbers=[
                result.page_number for result in page_results if not result.passed
            ],
            original_element_count=len(base_document.elements),
            merged_element_count=len(current_document.elements),
            replaced_element_count=replaced_count,
            replacement_element_count=replacement_count,
            provenance_coverage=round(final_assessment.provenance.coverage, 8),
            next_action=next_action,
        )
        return HybridReverificationOutcome(
            document=current_document,
            report=report,
            continuation_plan=continuation,
        )

    def execute(
        self,
        base_document: ExtractedDocument,
        routing_plan: ExtractionRoutingPlan,
        hybrid_runs: list[OpenDataLoaderHybridPageRun],
        *,
        input_extraction_run_id: str,
        input_extraction_commit_directory: Path | str,
        input_extraction_manifest_sha256: str,
        run_id: str | None = None,
        allow_unplanned_pilot_runs: bool = False,
    ) -> HybridReverificationCommitBundle:
        active_run_id = run_id or utc_run_id()
        validate_run_id(active_run_id)
        started_at = datetime.now(timezone.utc)
        outcome = self.reverify(
            base_document,
            routing_plan,
            hybrid_runs,
            allow_unplanned_pilot_runs=allow_unplanned_pilot_runs,
        )
        finished_at = datetime.now(timezone.utc)
        stage_status = (
            StageStatus.PASS
            if outcome.report.next_action == "NORMALIZE"
            else StageStatus.FAIL
        )
        stage_result = StageResult(
            stage=ResumeFrom.EXTRACTION_QUALITY,
            status=stage_status,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=max(
                0.0, (finished_at - started_at).total_seconds()
            ),
            artifact_paths=[
                DOCUMENT_FILENAME,
                REPORT_FILENAME,
                CONTINUATION_FILENAME,
            ],
            metrics={
                "revalidator_version": HYBRID_REVERIFY_STAGE_VERSION,
                "requested_page_count": len(outcome.report.requested_page_numbers),
                "accepted_page_count": len(outcome.report.accepted_page_numbers),
                "mineru_page_count": len(outcome.report.mineru_page_numbers),
                "merged_element_count": len(outcome.document.elements),
                "provenance_coverage": outcome.report.provenance_coverage,
                "external_calls": 0,
                "ocr_invocations": 0,
                "next_action": outcome.report.next_action,
            },
            message=(
                "Hybrid pages accepted; merged document is ready for normalization"
                if stage_status == StageStatus.PASS
                else "One or more Hybrid pages require a remaining fallback"
            ),
        )
        return self._commit(
            outcome,
            stage_result,
            hybrid_runs=hybrid_runs,
            input_extraction_run_id=input_extraction_run_id,
            input_extraction_commit_directory=input_extraction_commit_directory,
            input_extraction_manifest_sha256=input_extraction_manifest_sha256,
            run_id=active_run_id,
            committed_at=finished_at,
        )

    def _commit(
        self,
        outcome: HybridReverificationOutcome,
        stage_result: StageResult,
        *,
        hybrid_runs: list[OpenDataLoaderHybridPageRun],
        input_extraction_run_id: str,
        input_extraction_commit_directory: Path | str,
        input_extraction_manifest_sha256: str,
        run_id: str,
        committed_at: datetime,
    ) -> HybridReverificationCommitBundle:
        if len(input_extraction_manifest_sha256) != 64:
            raise ValueError("input extraction manifest SHA256 is invalid")

        output_root = self.config.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        final_directory = output_root / run_id

        if final_directory.exists():
            raise HybridReverificationCommitExistsError(
                f"Stage 6.1C commit already exists: {run_id}"
            )

        staging_directory = output_root / f".{run_id}.staging-{uuid.uuid4().hex}"
        staging_directory.mkdir(parents=False, exist_ok=False)

        try:
            payloads = {
                DOCUMENT_FILENAME: model_json_bytes(outcome.document),
                REPORT_FILENAME: model_json_bytes(outcome.report),
                CONTINUATION_FILENAME: model_json_bytes(outcome.continuation_plan),
                STAGE_RESULT_FILENAME: model_json_bytes(stage_result),
            }

            for filename, payload in payloads.items():
                write_payload(staging_directory / filename, payload)

            artifacts = [
                artifact_record(staging_directory / filename, staging_directory)
                for filename in sorted(payloads)
            ]
            manifest = HybridReverificationCommitManifest(
                schema_version=HYBRID_REVERIFY_COMMIT_SCHEMA_VERSION,
                revalidator_version=HYBRID_REVERIFY_STAGE_VERSION,
                run_id=run_id,
                catalog_id=outcome.document.catalog_id,
                family_id=outcome.document.family_id,
                version_id=outcome.document.version_id,
                source_sha256=outcome.document.source_sha256,
                input_extraction_run_id=input_extraction_run_id,
                input_extraction_commit_directory=str(
                    Path(input_extraction_commit_directory).expanduser().resolve()
                ),
                input_extraction_manifest_sha256=input_extraction_manifest_sha256,
                input_hybrid_runs=[
                    HybridInputRecord(
                        run_id=run.run_id,
                        page_number=run.source_page_number,
                        worker_manifest_path=str(run.result_manifest_path.resolve()),
                        worker_manifest_sha256=sha256_file(run.result_manifest_path),
                        raw_artifact_count=run.manifest.output.total_file_count,
                    )
                    for run in sorted(hybrid_runs, key=lambda item: item.source_page_number)
                ],
                requested_page_numbers=outcome.report.requested_page_numbers,
                accepted_page_numbers=outcome.report.accepted_page_numbers,
                mineru_page_numbers=outcome.report.mineru_page_numbers,
                next_action=outcome.report.next_action,
                artifact_count=4,
                artifacts=artifacts,
                committed_at=committed_at,
            )
            write_payload(
                staging_directory / COMMIT_MANIFEST_FILENAME,
                model_json_bytes(manifest),
            )
            load_hybrid_reverification_commit(staging_directory)
            staging_directory.rename(final_directory)
        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
            raise

        return load_hybrid_reverification_commit(final_directory)

    @staticmethod
    def _planned_hybrid_routes(
        routing_plan: ExtractionRoutingPlan,
    ) -> dict[int, dict[str, tuple[str, ...]]]:
        pages: dict[int, dict[str, set[str]]] = {}

        for route in routing_plan.routes:
            steps = [
                step for step in route.steps if step.action == FallbackAction.ODL_HYBRID
            ]

            if not steps:
                continue

            if (
                route.route_kind != RouteKind.TEXT_TABLE
                or route.page_number is None
                or not route.fallback_required
                or len(steps) != 1
                or steps[0].engine != EngineName.OPENDATALOADER_HYBRID
                or steps[0].max_attempts != ODL_HYBRID_MAX_ATTEMPTS
            ):
                raise ValueError("Routing plan contains an invalid ODL Hybrid route")

            page = pages.setdefault(
                route.page_number,
                {"route_ids": set(), "reason_codes": set()},
            )
            page["route_ids"].add(route.route_id)
            page["reason_codes"].update(route.reason_codes)

        return {
            page_number: {
                "route_ids": tuple(sorted(values["route_ids"])),
                "reason_codes": tuple(sorted(values["reason_codes"])),
            }
            for page_number, values in sorted(pages.items())
        }

    @staticmethod
    def _validate_run_identity(
        document: ExtractedDocument,
        run: OpenDataLoaderHybridPageRun,
    ) -> None:
        manifest = run.manifest

        if run.source_sha256 != document.source_sha256:
            raise HybridReverificationIntegrityError(
                "Hybrid source SHA256 differs from base document"
            )

        if run.source_document_page_count != document.page_count:
            raise HybridReverificationIntegrityError(
                "Hybrid source page count differs from base document"
            )

        if manifest.output.page_count != document.page_count:
            raise HybridReverificationIntegrityError(
                "Hybrid manifest page count differs from base document"
            )

        if manifest.request.source_page_number != run.source_page_number:
            raise HybridReverificationIntegrityError(
                "Hybrid run page differs from worker manifest"
            )

        if (
            manifest.output.selected_page_count != 1
            or manifest.output.page_scope_verified is not True
            or manifest.request.hybrid_fallback is not False
            or manifest.attempt_number != ODL_HYBRID_MAX_ATTEMPTS
            or manifest.max_attempts != ODL_HYBRID_MAX_ATTEMPTS
        ):
            raise HybridReverificationIntegrityError(
                "Hybrid fail-closed page contract is invalid"
            )

    @staticmethod
    def _page_elements(
        document: ExtractedDocument,
        page_number: int,
    ) -> list[SourceElement]:
        return [
            element
            for element in document.elements
            if element.source_anchor.page_number == page_number
        ]

    @staticmethod
    def _hybrid_elements(
        elements: list[SourceElement],
        *,
        run: OpenDataLoaderHybridPageRun,
        page_number: int,
        route_ids: list[str],
        requested_reason_codes: list[str],
    ) -> list[SourceElement]:
        output: list[SourceElement] = []

        for element in elements:
            anchor = element.source_anchor

            if anchor.page_number != page_number:
                raise HybridReverificationIntegrityError(
                    "Mapped Hybrid element escaped the requested page"
                )

            content_hash = hashlib.sha256(
                element.content.encode("utf-8")
            ).hexdigest()
            element_id = stable_identifier(
                "element",
                run.source_sha256,
                "OPENDATALOADER_HYBRID",
                page_number,
                anchor.block_index,
                element.element_type.value,
                content_hash,
            )
            hybrid_anchor = anchor.model_copy(
                update={
                    "anchor_id": stable_identifier(
                        "anchor",
                        run.source_sha256,
                        "OPENDATALOADER_HYBRID",
                        page_number,
                        anchor.block_index,
                        element.element_type.value,
                        content_hash,
                    )
                }
            )
            metadata = dict(element.metadata)
            metadata.update(
                {
                    "fallback_source": "ODL_HYBRID",
                    "hybrid_run_id": run.run_id,
                    "fallback_route_ids": sorted(route_ids),
                    "fallback_reason_codes": sorted(requested_reason_codes),
                    "reverified_page_number": page_number,
                }
            )
            output.append(
                element.model_copy(
                    update={
                        "element_id": element_id,
                        "source_anchor": hybrid_anchor,
                        "engine": EngineName.OPENDATALOADER_HYBRID,
                        "metadata": metadata,
                    }
                )
            )

        return output

    @staticmethod
    def _hybrid_assets(
        assets: list[ExtractedAsset],
        run: OpenDataLoaderHybridPageRun,
    ) -> list[ExtractedAsset]:
        output: list[ExtractedAsset] = []
        prefix = f"hybrid_runs/{run.run_id}"

        for asset in assets:
            path_map = {
                path: f"{prefix}/{PurePosixPath(path).as_posix()}"
                for path in asset.physical_paths
            }
            references: list[AssetReference] = []

            for reference in asset.references:
                references.append(
                    reference.model_copy(
                        update={
                            "reference_id": stable_identifier(
                                "asset-reference",
                                run.source_sha256,
                                "OPENDATALOADER_HYBRID",
                                run.source_page_number,
                                reference.reference_id,
                            ),
                            "source_path": path_map[reference.source_path],
                            "metadata": {
                                **reference.metadata,
                                "hybrid_run_id": run.run_id,
                            },
                        }
                    )
                )

            metadata = dict(asset.metadata)
            metadata.update(
                {
                    "hybrid_run_id": run.run_id,
                    "physical_file_count": len(path_map),
                    "reference_count": len(references),
                }
            )
            output.append(
                asset.model_copy(
                    update={
                        "physical_paths": sorted(path_map.values()),
                        "references": references,
                        "metadata": metadata,
                    }
                )
            )

        return output

    @staticmethod
    def _merge_page(
        document: ExtractedDocument,
        *,
        page_number: int,
        replacement_elements: list[SourceElement],
        replacement_assets: list[ExtractedAsset],
    ) -> ExtractedDocument:
        original_indices = [
            index
            for index, element in enumerate(document.elements)
            if element.source_anchor.page_number == page_number
        ]
        insertion_index = original_indices[0] if original_indices else len(document.elements)
        retained = [
            element
            for element in document.elements
            if element.source_anchor.page_number != page_number
        ]
        retained_before = sum(
            1
            for index, element in enumerate(document.elements)
            if index < insertion_index and element.source_anchor.page_number != page_number
        )
        elements = [
            *retained[:retained_before],
            *replacement_elements,
            *retained[retained_before:],
        ]
        assets = HybridReverifyMergeStage._merge_assets(
            document.assets,
            replacement_assets,
            page_number=page_number,
        )
        engine_chain = list(document.engine_chain)

        if EngineName.OPENDATALOADER_HYBRID not in engine_chain:
            engine_chain.append(EngineName.OPENDATALOADER_HYBRID)

        return document.model_copy(
            update={
                "engine_chain": engine_chain,
                "elements": elements,
                "assets": assets,
            }
        )

    @staticmethod
    def _merge_assets(
        base_assets: list[ExtractedAsset],
        candidate_assets: list[ExtractedAsset],
        *,
        page_number: int,
    ) -> list[ExtractedAsset]:
        merged: dict[str, ExtractedAsset] = {}

        for asset in base_assets:
            references = [
                reference
                for reference in asset.references
                if (
                    reference.source_anchor is None
                    or reference.source_anchor.page_number != page_number
                )
            ]
            metadata = dict(asset.metadata)
            metadata["reference_count"] = len(references)
            merged[asset.asset_id] = asset.model_copy(
                update={"references": references, "metadata": metadata}
            )

        for candidate in candidate_assets:
            existing = merged.get(candidate.asset_id)

            if existing is None:
                merged[candidate.asset_id] = candidate
                continue

            if existing.content_sha256 != candidate.content_sha256:
                raise HybridReverificationIntegrityError(
                    "Canonical asset ID collision has different content"
                )

            paths = sorted(set(existing.physical_paths) | set(candidate.physical_paths))
            references_by_id = {
                reference.reference_id: reference for reference in existing.references
            }

            for reference in candidate.references:
                references_by_id[reference.reference_id] = reference

            references = list(references_by_id.values())
            metadata = {**existing.metadata, **candidate.metadata}
            metadata.update(
                {
                    "physical_file_count": len(paths),
                    "reference_count": len(references),
                }
            )
            merged[candidate.asset_id] = existing.model_copy(
                update={
                    "physical_paths": paths,
                    "references": references,
                    "metadata": metadata,
                }
            )

        return list(merged.values())

    @staticmethod
    def _relevant_mapping_issues(
        issues: tuple[ExtractionQualityIssue, ...],
        *,
        page_number: int,
        candidate_element_ids: set[str],
        candidate_asset_ids: set[str],
    ) -> list[ExtractionQualityIssue]:
        return [
            issue
            for issue in issues
            if (
                page_number in issue.page_numbers
                or bool(candidate_element_ids & set(issue.element_ids))
                or bool(candidate_asset_ids & set(issue.asset_ids))
                or (
                    not issue.page_numbers
                    and not issue.element_ids
                    and not issue.asset_ids
                )
            )
        ]

    @staticmethod
    def _issue_fails(issue: ExtractionQualityIssue) -> bool:
        return issue.triggers_fallback or issue.severity == QualitySeverity.ERROR

    @staticmethod
    def _deduplicate_issues(
        issues: list[ExtractionQualityIssue],
    ) -> list[ExtractionQualityIssue]:
        unique: dict[str, ExtractionQualityIssue] = {}

        for issue in issues:
            unique.setdefault(issue.issue_id, issue)

        return list(unique.values())


def load_hybrid_reverification_commit(
    commit_directory: Path | str,
) -> HybridReverificationCommitBundle:
    root = Path(commit_directory).expanduser().resolve()

    if not root.is_dir():
        raise HybridReverificationIntegrityError(
            f"Stage 6.1C commit directory not found: {root}"
        )

    try:
        manifest = HybridReverificationCommitManifest.model_validate_json(
            (root / COMMIT_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise HybridReverificationIntegrityError(
            "Stage 6.1C commit manifest is missing or invalid"
        ) from error

    recorded_paths = {artifact.relative_path for artifact in manifest.artifacts}
    expected_paths = {
        DOCUMENT_FILENAME,
        REPORT_FILENAME,
        CONTINUATION_FILENAME,
        STAGE_RESULT_FILENAME,
    }

    if recorded_paths != expected_paths:
        raise HybridReverificationIntegrityError(
            "Stage 6.1C commit does not contain required artifacts"
        )

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }

    if actual_paths != recorded_paths | {COMMIT_MANIFEST_FILENAME}:
        raise HybridReverificationIntegrityError(
            "Stage 6.1C commit inventory differs from manifest"
        )

    for record in manifest.artifacts:
        path = (root / record.relative_path).resolve()

        if root not in path.parents:
            raise HybridReverificationIntegrityError(
                "Stage 6.1C artifact escapes commit directory"
            )

        if path.stat().st_size != record.size_bytes or sha256_file(path) != record.sha256:
            raise HybridReverificationIntegrityError(
                f"Stage 6.1C artifact integrity mismatch: {record.relative_path}"
            )

    try:
        document = ExtractedDocument.model_validate_json(
            (root / DOCUMENT_FILENAME).read_text(encoding="utf-8")
        )
        report = HybridReverificationReport.model_validate_json(
            (root / REPORT_FILENAME).read_text(encoding="utf-8")
        )
        continuation = FallbackContinuationPlan.model_validate_json(
            (root / CONTINUATION_FILENAME).read_text(encoding="utf-8")
        )
        stage_result = StageResult.model_validate_json(
            (root / STAGE_RESULT_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise HybridReverificationIntegrityError(
            "Stage 6.1C typed artifact is invalid"
        ) from error

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
    ) or identity != (
        report.catalog_id,
        report.family_id,
        report.version_id,
        report.source_sha256,
    ):
        raise HybridReverificationIntegrityError(
            "Stage 6.1C artifact identity differs from manifest"
        )

    if (
        manifest.requested_page_numbers != report.requested_page_numbers
        or manifest.accepted_page_numbers != report.accepted_page_numbers
        or manifest.mineru_page_numbers != report.mineru_page_numbers
        or manifest.next_action != report.next_action
        or continuation.next_action != report.next_action
    ):
        raise HybridReverificationIntegrityError(
            "Stage 6.1C decision artifacts disagree"
        )

    expected_status = (
        StageStatus.PASS if report.next_action == "NORMALIZE" else StageStatus.FAIL
    )

    if (
        stage_result.stage != ResumeFrom.EXTRACTION_QUALITY
        or stage_result.status != expected_status
        or stage_result.metrics.get("next_action") != report.next_action
    ):
        raise HybridReverificationIntegrityError(
            "Stage 6.1C stage result is inconsistent"
        )

    return HybridReverificationCommitBundle(
        commit_directory=root,
        manifest=manifest,
        document=document,
        report=report,
        continuation_plan=continuation,
        stage_result=stage_result,
    )
