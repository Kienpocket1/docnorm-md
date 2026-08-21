from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.contracts import (
    ExtractedDocument,
    ExtractionQualityIssue,
    FailureDomain,
    QualitySeverity,
)


PROVENANCE_VALIDATOR_VERSION = "2026-08-18-6.1A.1"
MAX_PROVENANCE_REPAIR_ATTEMPTS = 2
PROVENANCE_REPAIR_STRATEGIES = (
    "source_anchor+page+block",
    "page+bbox_overlap+text_similarity",
)


class ProvenanceValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validator_version: str
    element_count: int = Field(ge=0)
    covered_element_count: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    issues: list[ExtractionQualityIssue]
    repair_strategies: list[str]
    max_repair_attempts: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class ProvenanceValidatorConfig:
    require_pdf_page_number: bool = True
    require_pdf_bbox: bool = True
    maximum_coordinate: float = 100_000.0


def _issue_id(
    source_sha256: str,
    reason_code: str,
    *parts: object,
) -> str:
    payload = "|".join(
        [source_sha256, reason_code, *(str(part) for part in parts)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"quality-issue-{digest}"


def _issue(
    document: ExtractedDocument,
    *,
    reason_code: str,
    message: str,
    element_ids: list[str],
    page_numbers: list[int] | None = None,
    details: dict[str, Any] | None = None,
) -> ExtractionQualityIssue:
    pages = sorted(set(page_numbers or []))
    elements = sorted(set(element_ids))
    return ExtractionQualityIssue(
        issue_id=_issue_id(
            document.source_sha256,
            reason_code,
            ",".join(elements),
            ",".join(str(page) for page in pages),
        ),
        reason_code=reason_code,
        severity=QualitySeverity.ERROR,
        failure_domain=FailureDomain.PROVENANCE,
        message=message,
        page_numbers=pages,
        element_ids=elements,
        triggers_fallback=True,
        details=details or {},
    )


class ProvenanceValidator:
    def __init__(
        self,
        config: ProvenanceValidatorConfig | None = None,
    ) -> None:
        self.config = config or ProvenanceValidatorConfig()

        if self.config.maximum_coordinate <= 0:
            raise ValueError("maximum_coordinate must be positive")

    def validate(
        self,
        document: ExtractedDocument,
    ) -> ProvenanceValidationResult:
        issues: list[ExtractionQualityIssue] = []
        covered = 0
        seen_anchor_ids: dict[str, str] = {}
        is_paginated = document.page_count is not None

        for element in document.elements:
            anchor = element.source_anchor
            element_issues_before = len(issues)

            if anchor.source_sha256 != document.source_sha256:
                issues.append(
                    _issue(
                        document,
                        reason_code="source_anchor_sha256_mismatch",
                        message=(
                            "Source anchor SHA256 differs from document"
                        ),
                        element_ids=[element.element_id],
                    )
                )

            prior_element = seen_anchor_ids.get(anchor.anchor_id)

            if prior_element is not None:
                issues.append(
                    _issue(
                        document,
                        reason_code="duplicate_source_anchor",
                        message=(
                            "Multiple elements reuse one source anchor"
                        ),
                        element_ids=[prior_element, element.element_id],
                    )
                )
            else:
                seen_anchor_ids[anchor.anchor_id] = element.element_id

            if (
                is_paginated
                and self.config.require_pdf_page_number
                and anchor.page_number is None
            ):
                issues.append(
                    _issue(
                        document,
                        reason_code="source_anchor_page_missing",
                        message=(
                            "Paginated source element has no page number"
                        ),
                        element_ids=[element.element_id],
                    )
                )

            if (
                anchor.page_number is not None
                and document.page_count is not None
                and anchor.page_number > document.page_count
            ):
                issues.append(
                    _issue(
                        document,
                        reason_code="source_anchor_page_out_of_range",
                        message=(
                            "Source anchor page exceeds document page count"
                        ),
                        element_ids=[element.element_id],
                        page_numbers=[anchor.page_number],
                    )
                )

            if (
                is_paginated
                and self.config.require_pdf_bbox
                and anchor.bbox is None
            ):
                issues.append(
                    _issue(
                        document,
                        reason_code="source_anchor_bbox_missing",
                        message=(
                            "Paginated source element has no bounding box"
                        ),
                        element_ids=[element.element_id],
                        page_numbers=(
                            [anchor.page_number]
                            if anchor.page_number is not None
                            else []
                        ),
                    )
                )

            if anchor.bbox is not None:
                coordinates = (
                    anchor.bbox.x0,
                    anchor.bbox.y0,
                    anchor.bbox.x1,
                    anchor.bbox.y1,
                )

                if any(
                    coordinate > self.config.maximum_coordinate
                    for coordinate in coordinates
                ):
                    issues.append(
                        _issue(
                            document,
                            reason_code="source_anchor_bbox_out_of_range",
                            message=(
                                "Source anchor bounding box is outside "
                                "the accepted coordinate range"
                            ),
                            element_ids=[element.element_id],
                            page_numbers=(
                                [anchor.page_number]
                                if anchor.page_number is not None
                                else []
                            ),
                            details={
                                "maximum_coordinate": (
                                    self.config.maximum_coordinate
                                ),
                                "coordinates": list(coordinates),
                            },
                        )
                    )

            if len(issues) == element_issues_before:
                covered += 1

        element_count = len(document.elements)
        coverage = covered / element_count if element_count else 0.0
        return ProvenanceValidationResult(
            validator_version=PROVENANCE_VALIDATOR_VERSION,
            element_count=element_count,
            covered_element_count=covered,
            coverage=coverage,
            issues=self._deduplicate(issues),
            repair_strategies=list(PROVENANCE_REPAIR_STRATEGIES),
            max_repair_attempts=MAX_PROVENANCE_REPAIR_ATTEMPTS,
        )

    @staticmethod
    def _deduplicate(
        issues: list[ExtractionQualityIssue],
    ) -> list[ExtractionQualityIssue]:
        unique: dict[str, ExtractionQualityIssue] = {}

        for issue in issues:
            unique.setdefault(issue.issue_id, issue)

        return list(unique.values())
