from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class ConversionMode(StrEnum):
    AUTO = "auto"
    STRUCTURE = "structure"
    FAST = "fast"


class RepairProfile(StrEnum):
    AUTO = "auto"
    GENERIC = "generic"
    PETROLIMEX_MATERIALS = "petrolimex_materials"
    MATH_EXAM = "math_exam"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobStage(StrEnum):
    UPLOAD = "UPLOAD"
    EXTRACT = "EXTRACT"
    VERIFY = "VERIFY"
    NORMALIZE = "NORMALIZE"
    EXPORT = "EXPORT"
    DONE = "DONE"


class QualityIssue(BaseModel):
    severity: str = "INFO"
    code: str
    message: str
    page_numbers: list[int] = Field(default_factory=list)


class DocumentMetrics(BaseModel):
    pages: int | None = None
    elements: int = 0
    headings: int = 0
    paragraphs: int = 0
    tables: int = 0
    lists: int = 0
    images: int = 0
    markdown_characters: int = 0
    broken_characters: int = 0
    provenance_coverage: float = 0.0
    ocr_invocations: int = 0
    vlm_invocations: int = 0
    vlm_pages: int = 0


class QualitySummary(BaseModel):
    disposition: str = "PASS"
    issues: list[QualityIssue] = Field(default_factory=list)
    engine_chain: list[str] = Field(default_factory=list)
    repair_profile_requested: RepairProfile = RepairProfile.AUTO
    repair_profile_applied: RepairProfile = RepairProfile.GENERIC
    repair_version: str | None = None
    repair_fingerprint_score: float = Field(default=0.0, ge=0.0, le=1.0)
    resolved_issue_codes: list[str] = Field(default_factory=list)


class JobRecord(BaseModel):
    job_id: str
    filename: str
    size_bytes: int
    content_type: str
    mode: ConversionMode
    repair_profile: RepairProfile = RepairProfile.AUTO
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.UPLOAD
    progress_percent: int = Field(default=0, ge=0, le=100)
    message: str = "Đã tiếp nhận tài liệu"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed_seconds: float = 0.0
    source_path: str
    run_directory: str
    markdown_path: str | None = None
    report_path: str | None = None
    bundle_path: str | None = None
    metrics: DocumentMetrics = Field(default_factory=DocumentMetrics)
    quality: QualitySummary = Field(default_factory=QualitySummary)
    error_code: str | None = None
    error_detail: str | None = None

    def public_dict(self, *, include_paths: bool = False) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if not include_paths:
            for field in (
                "source_path",
                "run_directory",
                "markdown_path",
                "report_path",
                "bundle_path",
                "error_detail",
            ):
                payload.pop(field, None)
        return payload


class MarkdownUpdate(BaseModel):
    content: str = Field(min_length=1, max_length=20_000_000)
