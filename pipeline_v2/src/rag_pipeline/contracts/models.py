from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.contracts.enums import (
    AssetClassification,
    AssetRole,
    ElementType,
    EngineName,
    FailureDomain,
    FileFormat,
    PublicationState,
    QualityDisposition,
    QualitySeverity,
    QuarantineStatus,
    QuarantineType,
    ResolutionOutcome,
    ResumeFrom,
    RouteType,
    StageStatus,
    VersionAction,
)


Sha256Hex = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{64}$"),
]

NonEmptyString = Annotated[
    str,
    Field(min_length=1),
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class BoundingBox(ContractModel):
    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_coordinate_order(self) -> Self:
        if self.x1 <= self.x0:
            raise ValueError("x1 must be greater than x0")

        if self.y1 <= self.y0:
            raise ValueError("y1 must be greater than y0")

        return self


class CatalogEntry(ContractModel):
    catalog_id: NonEmptyString
    source_name: NonEmptyString
    source_path: NonEmptyString
    file_format: FileFormat
    source_sha256: Sha256Hex
    size_bytes: int = Field(ge=1)
    mime_type: str | None = None
    modified_at: datetime | None = None
    discovered_at: datetime = Field(default_factory=utc_now)


class ResolutionDecision(ContractModel):
    decision_id: NonEmptyString
    catalog_id: NonEmptyString
    outcome: ResolutionOutcome

    # Family đã khớp hoặc family mới được đề xuất.
    family_id: str | None = None

    # Chỉ sử dụng khi outcome là SAME_VERSION.
    matched_version_id: str | None = None

    # Các family cần con người phân xử khi CONFLICT.
    candidate_family_ids: list[str] = Field(
        default_factory=list
    )

    confidence: float = Field(ge=0, le=1)
    reason_code: NonEmptyString
    requires_human_review: bool = False

    # Chỉ chứa metadata giải thích quyết định,
    # không chứa nội dung tài liệu.
    evidence: dict[str, Any] = Field(
        default_factory=dict
    )

    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_resolution_references(self) -> Self:
        if len(self.candidate_family_ids) != len(
            set(self.candidate_family_ids)
        ):
            raise ValueError(
                "candidate_family_ids must be unique"
            )

        if self.outcome == ResolutionOutcome.SAME_VERSION:
            if not self.family_id or not self.matched_version_id:
                raise ValueError(
                    "SAME_VERSION requires family_id "
                    "and matched_version_id"
                )

        if self.outcome == ResolutionOutcome.NEW_VERSION:
            if not self.family_id:
                raise ValueError(
                    "NEW_VERSION requires family_id"
                )

        if self.outcome == ResolutionOutcome.UNRELATED:
            if not self.family_id:
                raise ValueError(
                    "UNRELATED requires a proposed family_id"
                )

        if self.outcome == ResolutionOutcome.CONFLICT:
            if not self.requires_human_review:
                raise ValueError(
                    "CONFLICT must require human review"
                )

            if not self.candidate_family_ids:
                raise ValueError(
                    "CONFLICT requires at least one "
                    "candidate family"
                )

        return self


class FamilyRecord(ContractModel):
    family_id: str = Field(
        pattern=r"^[a-z0-9][a-z0-9-]{0,63}$"
    )
    canonical_name: NonEmptyString
    document_code: str | None = None
    aliases: list[str] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_aliases(self) -> Self:
        if len(self.aliases) != len(set(self.aliases)):
            raise ValueError(
                "Family aliases must be unique"
            )

        return self


class VersionRecord(ContractModel):
    family_id: NonEmptyString
    version_id: NonEmptyString
    catalog_id: NonEmptyString
    source_sha256: Sha256Hex
    version_number: int = Field(ge=1)
    version_label: NonEmptyString
    publication_state: PublicationState = (
        PublicationState.DRAFT
    )
    supersedes_version_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_version_identity(self) -> Self:
        expected_label = f"v{self.version_number:03d}"

        if self.version_label != expected_label:
            raise ValueError(
                "version_label must equal "
                f"'{expected_label}'"
            )

        expected_version_id = (
            f"{self.family_id}-"
            f"{expected_label}-"
            f"{self.source_sha256[:12]}"
        )

        if self.version_id != expected_version_id:
            raise ValueError(
                "version_id must equal "
                "'<family_id>-vNNN-<sha12>'"
            )

        if self.supersedes_version_id == self.version_id:
            raise ValueError(
                "A version cannot supersede itself"
            )

        return self


class VersionAssignment(ContractModel):
    assignment_id: NonEmptyString
    decision_id: NonEmptyString
    catalog_id: NonEmptyString
    family_id: NonEmptyString
    version_id: NonEmptyString
    version_number: int = Field(ge=1)
    version_label: NonEmptyString
    source_sha256: Sha256Hex
    action: VersionAction
    assigned_at: datetime = Field(default_factory=utc_now)

class ResolverIndexEntry(ContractModel):
    """Metadata of a known version used by Resolver."""

    family_id: NonEmptyString
    version_id: NonEmptyString
    source_sha256: Sha256Hex
    source_name: NonEmptyString
    normalized_name: NonEmptyString
    document_code: str | None = None

class ResolverReviewRecord(ContractModel):
    review_id: NonEmptyString
    original_decision_id: NonEmptyString
    catalog_id: NonEmptyString
    outcome: ResolutionOutcome
    family_id: str | None = None
    matched_version_id: str | None = None
    reviewer: NonEmptyString
    reason_code: NonEmptyString
    reviewed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_review_outcome(self) -> Self:
        if self.outcome == ResolutionOutcome.CONFLICT:
            raise ValueError(
                "Manual review cannot return CONFLICT"
            )

        if self.outcome == ResolutionOutcome.SAME_VERSION:
            if not self.family_id or not self.matched_version_id:
                raise ValueError(
                    "SAME_VERSION review requires family_id "
                    "and matched_version_id"
                )

        if self.outcome in {
            ResolutionOutcome.NEW_VERSION,
            ResolutionOutcome.UNRELATED,
        }:
            if not self.family_id:
                raise ValueError(
                    f"{self.outcome.value} review "
                    "requires family_id"
                )

            if self.matched_version_id is not None:
                raise ValueError(
                    "Only SAME_VERSION may contain "
                    "matched_version_id"
                )

        if self.outcome == ResolutionOutcome.REJECTED:
            if (
                self.family_id is not None
                or self.matched_version_id is not None
            ):
                raise ValueError(
                    "REJECTED review cannot assign "
                    "family or version"
                )

        return self

class RouteDecision(ContractModel):
    route_id: NonEmptyString
    catalog_id: NonEmptyString
    version_id: NonEmptyString
    file_format: FileFormat
    route_type: RouteType
    primary_engine: EngineName
    fallback_engines: list[EngineName] = Field(
        default_factory=list
    )
    reason_code: NonEmptyString
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_route(self) -> Self:
        if self.primary_engine in self.fallback_engines:
            raise ValueError(
                "primary_engine must not appear "
                "in fallback_engines"
            )

        if len(self.fallback_engines) != len(
            set(self.fallback_engines)
        ):
            raise ValueError(
                "fallback_engines must be unique"
            )

        expected_mapping = {
            RouteType.PDF_OPENDATALOADER: (
                FileFormat.PDF,
                EngineName.OPENDATALOADER_LOCAL,
            ),
            RouteType.DOCX_NATIVE: (
                FileFormat.DOCX,
                EngineName.DOCX_NATIVE,
            ),
            RouteType.DOC_CONVERT: (
                FileFormat.DOC,
                EngineName.LIBREOFFICE,
            ),
            RouteType.IMAGE_VISUAL: (
                FileFormat.IMAGE,
                EngineName.QWEN3_VL,
            ),
            RouteType.UNSUPPORTED: (
                FileFormat.UNKNOWN,
                EngineName.MANUAL,
            ),
        }

        expected = expected_mapping[self.route_type]

        if self.file_format != expected[0]:
            raise ValueError(
                f"{self.route_type.value} requires "
                f"{expected[0].value} format"
            )

        if self.primary_engine != expected[1]:
            raise ValueError(
                f"{self.route_type.value} requires "
                f"{expected[1].value} as primary engine"
            )

        return self


class SourceAnchor(ContractModel):
    anchor_id: NonEmptyString
    source_sha256: Sha256Hex
    page_number: int | None = Field(default=None, ge=1)
    block_index: int = Field(ge=0)
    bbox: BoundingBox | None = None
    section_path: list[str] = Field(default_factory=list)


class AssetReference(ContractModel):
    """One logical JSON or Markdown reference to a physical asset file."""

    reference_id: NonEmptyString
    source_path: NonEmptyString
    source_anchor: SourceAnchor | None = None
    referenced_in_markdown: bool = False
    referenced_in_structure: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reference_origin(self) -> Self:
        if not (
            self.referenced_in_markdown
            or self.referenced_in_structure
        ):
            raise ValueError(
                "Asset reference must originate from Markdown "
                "or structured output"
            )

        if (
            self.referenced_in_structure
            and self.source_anchor is None
        ):
            raise ValueError(
                "Structured asset references require a source anchor"
            )

        return self


class ExtractedAsset(ContractModel):
    """Canonical asset grouped by identical content SHA256."""

    asset_id: NonEmptyString
    source_sha256: Sha256Hex
    content_sha256: Sha256Hex
    media_type: NonEmptyString
    size_bytes: int = Field(ge=1)
    classification: AssetClassification = (
        AssetClassification.UNKNOWN
    )
    role: AssetRole = AssetRole.UNKNOWN
    physical_paths: list[NonEmptyString] = Field(min_length=1)
    references: list[AssetReference] = Field(
        default_factory=list
    )
    retain_raw: bool = True
    indexable: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_asset_policy(self) -> Self:
        if len(self.physical_paths) != len(
            set(self.physical_paths)
        ):
            raise ValueError("physical_paths must be unique")

        reference_ids = [
            reference.reference_id
            for reference in self.references
        ]

        if len(reference_ids) != len(set(reference_ids)):
            raise ValueError(
                "Asset reference_id values must be unique"
            )

        physical_paths = set(self.physical_paths)

        for reference in self.references:
            if reference.source_path not in physical_paths:
                raise ValueError(
                    "Asset reference path must exist in "
                    "physical_paths"
                )

            anchor = reference.source_anchor

            if (
                anchor is not None
                and anchor.source_sha256 != self.source_sha256
            ):
                raise ValueError(
                    "Asset anchor SHA256 does not match document"
                )

        decorative_roles = {
            AssetRole.BRAND_LOGO,
            AssetRole.PAGE_DECORATION,
        }

        if (
            self.role in decorative_roles
            and self.classification
            != AssetClassification.DECORATIVE
        ):
            raise ValueError(
                "Decorative asset roles require DECORATIVE "
                "classification"
            )

        if (
            self.indexable
            and self.classification
            != AssetClassification.INFORMATIVE
        ):
            raise ValueError(
                "Only INFORMATIVE assets may be indexable"
            )

        if not self.retain_raw:
            raise ValueError(
                "Raw extracted assets must be retained for audit"
            )

        return self


class SourceElement(ContractModel):
    element_id: NonEmptyString
    element_type: ElementType
    content: str = ""
    source_anchor: SourceAnchor
    engine: EngineName
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        empty_content_allowed = {
            ElementType.IMAGE,
            ElementType.FLOWCHART,
        }

        if (
            not self.content
            and self.element_type not in empty_content_allowed
        ):
            raise ValueError(
                "Non-visual source elements must contain content"
            )

        return self


class ExtractedDocument(ContractModel):
    catalog_id: NonEmptyString
    family_id: NonEmptyString
    version_id: NonEmptyString
    source_sha256: Sha256Hex
    page_count: int | None = Field(default=None, ge=1)
    engine_chain: list[EngineName] = Field(min_length=1)
    elements: list[SourceElement] = Field(min_length=1)
    assets: list[ExtractedAsset] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_elements(self) -> Self:
        element_ids = [element.element_id for element in self.elements]

        if len(element_ids) != len(set(element_ids)):
            raise ValueError("element_id values must be unique")

        asset_ids = [asset.asset_id for asset in self.assets]

        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("asset_id values must be unique")

        for element in self.elements:
            anchor = element.source_anchor

            if anchor.source_sha256 != self.source_sha256:
                raise ValueError(
                    "Element source SHA256 does not match document"
                )

            if (
                self.page_count is not None
                and anchor.page_number is not None
                and anchor.page_number > self.page_count
            ):
                raise ValueError(
                    "Source anchor page exceeds page_count"
                )

        for asset in self.assets:
            if asset.source_sha256 != self.source_sha256:
                raise ValueError(
                    "Asset source SHA256 does not match document"
                )

            for reference in asset.references:
                anchor = reference.source_anchor

                if (
                    self.page_count is not None
                    and anchor is not None
                    and anchor.page_number is not None
                    and anchor.page_number > self.page_count
                ):
                    raise ValueError(
                        "Asset anchor page exceeds page_count"
                    )

        return self


class ExtractionQualityIssue(ContractModel):
    issue_id: NonEmptyString
    reason_code: NonEmptyString
    severity: QualitySeverity
    failure_domain: FailureDomain
    message: NonEmptyString
    page_numbers: list[int] = Field(default_factory=list)
    element_ids: list[str] = Field(default_factory=list)
    asset_ids: list[str] = Field(default_factory=list)
    triggers_fallback: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_issue_references(self) -> Self:
        if any(page_number < 1 for page_number in self.page_numbers):
            raise ValueError("page_numbers must be one-based")

        for field_name, values in (
            ("page_numbers", self.page_numbers),
            ("element_ids", self.element_ids),
            ("asset_ids", self.asset_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must be unique")

        if (
            self.triggers_fallback
            and self.severity == QualitySeverity.INFO
        ):
            raise ValueError(
                "INFO quality issues cannot trigger fallback"
            )

        return self


class ExtractionQualityReport(ContractModel):
    report_id: NonEmptyString
    catalog_id: NonEmptyString
    family_id: NonEmptyString
    version_id: NonEmptyString
    source_sha256: Sha256Hex
    engine: EngineName
    disposition: QualityDisposition
    issues: list[ExtractionQualityIssue] = Field(
        default_factory=list
    )
    metrics: dict[str, Any] = Field(default_factory=dict)
    fallback_engine: EngineName | None = None
    evaluated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_disposition(self) -> Self:
        issue_ids = [issue.issue_id for issue in self.issues]

        if len(issue_ids) != len(set(issue_ids)):
            raise ValueError("issue_id values must be unique")

        warnings = [
            issue
            for issue in self.issues
            if issue.severity == QualitySeverity.WARNING
        ]
        errors = [
            issue
            for issue in self.issues
            if issue.severity == QualitySeverity.ERROR
        ]
        fallback_issues = [
            issue
            for issue in self.issues
            if issue.triggers_fallback
        ]

        if self.disposition == QualityDisposition.PASS:
            if warnings or errors or fallback_issues:
                raise ValueError(
                    "PASS cannot contain warnings, errors, "
                    "or fallback issues"
                )

        elif (
            self.disposition
            == QualityDisposition.PASS_WITH_WARNINGS
        ):
            if not warnings:
                raise ValueError(
                    "PASS_WITH_WARNINGS requires a warning"
                )

            if errors or fallback_issues:
                raise ValueError(
                    "PASS_WITH_WARNINGS cannot contain errors "
                    "or fallback issues"
                )

        elif (
            self.disposition
            == QualityDisposition.FALLBACK_REQUIRED
        ):
            if not fallback_issues:
                raise ValueError(
                    "FALLBACK_REQUIRED requires a triggering issue"
                )

            if self.fallback_engine is None:
                raise ValueError(
                    "FALLBACK_REQUIRED requires fallback_engine"
                )

            if self.fallback_engine == self.engine:
                raise ValueError(
                    "fallback_engine must differ from engine"
                )

        elif (
            self.disposition
            == QualityDisposition.QUARANTINE_REQUIRED
        ):
            if not errors:
                raise ValueError(
                    "QUARANTINE_REQUIRED requires an error"
                )

        if (
            self.disposition
            != QualityDisposition.FALLBACK_REQUIRED
            and self.fallback_engine is not None
        ):
            raise ValueError(
                "fallback_engine is only valid when fallback "
                "is required"
            )

        return self


class NormalizedSpan(ContractModel):
    span_id: NonEmptyString
    source_element_id: NonEmptyString
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    source_anchor: SourceAnchor

    @model_validator(mode="after")
    def validate_span_order(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError(
                "char_end must be greater than char_start"
            )

        return self


class NormalizedBlock(ContractModel):
    block_id: NonEmptyString
    heading_path: list[str] = Field(default_factory=list)
    markdown: NonEmptyString
    source_spans: list[NormalizedSpan] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_spans(self) -> Self:
        span_ids = [span.span_id for span in self.source_spans]

        if len(span_ids) != len(set(span_ids)):
            raise ValueError("span_id values must be unique")

        for span in self.source_spans:
            if span.char_end > len(self.markdown):
                raise ValueError(
                    "Source span exceeds normalized Markdown length"
                )

        return self


class NormalizedDocument(ContractModel):
    catalog_id: NonEmptyString
    family_id: NonEmptyString
    version_id: NonEmptyString
    source_sha256: Sha256Hex
    blocks: list[NormalizedBlock] = Field(min_length=1)
    normalized_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        block_ids = [block.block_id for block in self.blocks]

        if len(block_ids) != len(set(block_ids)):
            raise ValueError("block_id values must be unique")

        return self


class ChunkRecord(ContractModel):
    family_id: NonEmptyString
    version_id: NonEmptyString
    chunk_id: NonEmptyString
    chunk_instance_id: NonEmptyString
    source_sha256: Sha256Hex
    ordinal: int = Field(ge=1)
    heading_path: list[str] = Field(default_factory=list)
    text: NonEmptyString
    token_count: int = Field(ge=1)
    source_anchors: list[SourceAnchor] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_chunk_identifiers(self) -> Self:
        expected_prefix = f"{self.family_id}#"

        if not self.chunk_id.startswith(expected_prefix):
            raise ValueError(
                "chunk_id must begin with '<family_id>#'"
            )

        expected_instance_id = (
            f"{self.version_id}::{self.chunk_id}"
        )

        if self.chunk_instance_id != expected_instance_id:
            raise ValueError(
                "chunk_instance_id must equal "
                "'<version_id>::<chunk_id>'"
            )

        return self


class QuarantineRecord(ContractModel):
    quarantine_id: NonEmptyString
    queue: QuarantineType
    failure_domain: FailureDomain
    root_cause_stage: ResumeFrom
    reason_code: NonEmptyString
    resume_from: ResumeFrom
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=0, ge=0)
    status: QuarantineStatus = QuarantineStatus.OPEN
    catalog_id: str | None = None
    family_id: str | None = None
    version_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_attempt_count(self) -> Self:
        if self.attempt_count > self.max_attempts:
            raise ValueError(
                "attempt_count cannot exceed max_attempts"
            )

        return self


class PublishReceipt(ContractModel):
    transaction_id: NonEmptyString
    family_id: NonEmptyString
    version_id: NonEmptyString
    previous_active_version_id: str | None = None
    new_state: PublicationState = PublicationState.ACTIVE
    vector_namespace: NonEmptyString
    active_pointer: NonEmptyString
    old_vector_cleanup_scheduled: bool = False
    committed_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_publish_state(self) -> Self:
        if self.new_state != PublicationState.ACTIVE:
            raise ValueError(
                "A committed publish receipt must be ACTIVE"
            )

        if self.previous_active_version_id == self.version_id:
            raise ValueError(
                "Previous and new ACTIVE versions must differ"
            )

        return self


class RunContext(ContractModel):
    run_id: NonEmptyString
    pipeline_version: NonEmptyString
    catalog_id: str | None = None
    current_stage: ResumeFrom
    status: StageStatus = StageStatus.RUNNING
    started_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class StageResult(ContractModel):
    stage: ResumeFrom
    status: StageStatus
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)
    artifact_paths: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    quarantine_id: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def validate_stage_result(self) -> Self:
        if self.finished_at < self.started_at:
            raise ValueError(
                "finished_at cannot be before started_at"
            )

        if (
            self.status == StageStatus.QUARANTINED
            and not self.quarantine_id
        ):
            raise ValueError(
                "QUARANTINED result requires quarantine_id"
            )

        return self
