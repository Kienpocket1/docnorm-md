from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.contracts import (
    AssetClassification,
    AssetRole,
    ElementType,
    ExtractedDocument,
    ExtractionQualityIssue,
    FailureDomain,
    QualitySeverity,
    SourceElement,
)
from rag_pipeline.validators.provenance import (
    ProvenanceValidationResult,
    ProvenanceValidator,
)


PAGE_QUALITY_VALIDATOR_VERSION = "2026-08-18-6.1A.1"
_MOJIBAKE_MARKERS = ("\ufffd", "Ã", "Â", "â€", "ðŸ")


class PageQualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope_id: str
    page_number: int | None = None
    section_index: int | None = None
    element_ids: list[str]
    text_character_count: int = Field(ge=0)
    mojibake_ratio: float = Field(ge=0, le=1)
    repetition_ratio: float = Field(ge=0, le=1)
    table_count: int = Field(ge=0)
    visual_count: int = Field(ge=0)
    passed: bool
    issues: list[ExtractionQualityIssue]


class DocumentPageQualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validator_version: str
    page_results: list[PageQualityResult]
    issues: list[ExtractionQualityIssue]
    provenance: ProvenanceValidationResult
    indexable_element_ids: list[str]
    filtered_element_ids: list[str]
    pending_visual_element_ids: list[str]
    metrics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExtractionPageValidatorConfig:
    minimum_text_characters: int = 12
    maximum_mojibake_ratio: float = 0.02
    repetition_minimum_characters: int = 300
    repetition_minimum_lines: int = 8
    maximum_repetition_ratio: float = 0.70


def _stable_issue_id(
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
    severity: QualitySeverity,
    failure_domain: FailureDomain,
    message: str,
    triggers_fallback: bool,
    page_number: int | None = None,
    element_ids: Iterable[str] = (),
    details: dict[str, Any] | None = None,
) -> ExtractionQualityIssue:
    elements = sorted(set(element_ids))
    pages = [page_number] if page_number is not None else []
    return ExtractionQualityIssue(
        issue_id=_stable_issue_id(
            document.source_sha256,
            reason_code,
            page_number,
            ",".join(elements),
        ),
        reason_code=reason_code,
        severity=severity,
        failure_domain=failure_domain,
        message=message,
        page_numbers=pages,
        element_ids=elements,
        triggers_fallback=triggers_fallback,
        details=details or {},
    )


def _heading_level(element: SourceElement) -> int | None:
    value = element.metadata.get("heading_level")

    if isinstance(value, int) and value >= 1:
        return value

    raw = element.metadata.get("heading_level_raw")

    if isinstance(raw, int) and raw >= 1:
        return raw

    if isinstance(raw, str):
        match = re.search(r"\d+", raw)

        if match:
            return max(1, int(match.group(0)))

    return None


def _gfm_table_is_valid(content: str) -> bool:
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    if len(lines) < 2:
        return False

    def cells(line: str) -> list[str]:
        stripped = line.strip().strip("|")
        return [
            cell.strip()
            for cell in re.split(r"(?<!\\)\|", stripped)
        ]

    widths = [len(cells(line)) for line in lines]

    if not widths or min(widths) < 1 or len(set(widths)) != 1:
        return False

    separator = cells(lines[1])
    return all(
        re.fullmatch(r":?-{3,}:?", cell) is not None
        for cell in separator
    )


class ExtractionPageValidator:
    def __init__(
        self,
        config: ExtractionPageValidatorConfig | None = None,
        *,
        provenance_validator: ProvenanceValidator | None = None,
    ) -> None:
        self.config = config or ExtractionPageValidatorConfig()
        self.provenance_validator = (
            provenance_validator or ProvenanceValidator()
        )

        if self.config.minimum_text_characters < 0:
            raise ValueError(
                "minimum_text_characters must be non-negative"
            )

        for field_name in (
            "maximum_mojibake_ratio",
            "maximum_repetition_ratio",
        ):
            value = getattr(self.config, field_name)

            if not 0 <= value <= 1:
                raise ValueError(f"{field_name} must be between 0 and 1")

    def evaluate(
        self,
        document: ExtractedDocument,
        *,
        inherited_issues: Iterable[ExtractionQualityIssue] = (),
    ) -> DocumentPageQualityResult:
        provenance = self.provenance_validator.validate(document)
        issues = [*inherited_issues, *provenance.issues]
        indexable: list[str] = []
        filtered: list[str] = []
        pending_visuals: list[str] = []
        asset_by_id = {asset.asset_id: asset for asset in document.assets}

        for element in document.elements:
            explicitly_indexable = element.metadata.get("indexable")

            if element.element_type in {
                ElementType.IMAGE,
                ElementType.FLOWCHART,
            }:
                asset = asset_by_id.get(
                    str(element.metadata.get("asset_id") or "")
                )
                classification = str(
                    element.metadata.get("asset_classification")
                    or (
                        asset.classification.value
                        if asset is not None
                        else ""
                    )
                ).upper()
                role = str(
                    element.metadata.get("asset_role")
                    or (
                        asset.role.value
                        if asset is not None
                        else ""
                    )
                ).upper()
                decorative = (
                    classification == AssetClassification.DECORATIVE.value
                    or role
                    in {
                        AssetRole.BRAND_LOGO.value,
                        AssetRole.PAGE_DECORATION.value,
                    }
                )

                if decorative:
                    filtered.append(element.element_id)
                else:
                    pending_visuals.append(element.element_id)

                continue

            if explicitly_indexable is False:
                filtered.append(element.element_id)
            else:
                indexable.append(element.element_id)

        scopes = self._build_scopes(document)
        page_results: list[PageQualityResult] = []

        for scope_id, page_number, section_index, elements in scopes:
            scope_issues = self._evaluate_scope(
                document,
                elements,
                page_number=page_number,
            )
            issues.extend(scope_issues)
            text_parts = [
                element.content
                for element in elements
                if element.content
                and element.element_type
                not in {ElementType.IMAGE, ElementType.FLOWCHART}
            ]
            text = "\n".join(text_parts)
            mojibake_count = sum(
                text.count(marker)
                for marker in _MOJIBAKE_MARKERS
            )
            mojibake_ratio = mojibake_count / max(len(text), 1)
            repetition_ratio = self._repetition_ratio(elements)
            relevant_provenance = [
                issue
                for issue in provenance.issues
                if (
                    page_number is not None
                    and page_number in issue.page_numbers
                )
                or bool(
                    set(issue.element_ids)
                    & {element.element_id for element in elements}
                )
            ]
            combined_scope_issues = self._deduplicate(
                [*scope_issues, *relevant_provenance]
            )
            page_results.append(
                PageQualityResult(
                    scope_id=scope_id,
                    page_number=page_number,
                    section_index=section_index,
                    element_ids=[
                        element.element_id
                        for element in elements
                    ],
                    text_character_count=len(text),
                    mojibake_ratio=round(mojibake_ratio, 8),
                    repetition_ratio=round(repetition_ratio, 8),
                    table_count=sum(
                        element.element_type == ElementType.TABLE
                        for element in elements
                    ),
                    visual_count=sum(
                        element.element_type
                        in {ElementType.IMAGE, ElementType.FLOWCHART}
                        for element in elements
                    ),
                    passed=not any(
                        issue.triggers_fallback
                        or issue.severity == QualitySeverity.ERROR
                        for issue in combined_scope_issues
                    ),
                    issues=combined_scope_issues,
                )
            )

        heading_issues = self._heading_hierarchy_issues(document)
        issues.extend(heading_issues)
        issues = self._deduplicate(issues)
        updated_page_results: list[PageQualityResult] = []

        for result in page_results:
            result_element_ids = set(result.element_ids)
            relevant_heading_issues = [
                issue
                for issue in heading_issues
                if (
                    result.page_number is not None
                    and result.page_number in issue.page_numbers
                )
                or bool(result_element_ids & set(issue.element_ids))
            ]
            combined = self._deduplicate(
                [*result.issues, *relevant_heading_issues]
            )
            updated_page_results.append(
                result.model_copy(
                    update={
                        "issues": combined,
                        "passed": not any(
                            issue.triggers_fallback
                            or issue.severity
                            == QualitySeverity.ERROR
                            for issue in combined
                        ),
                    }
                )
            )

        page_results = updated_page_results
        failed_scopes = sum(not result.passed for result in page_results)
        return DocumentPageQualityResult(
            validator_version=PAGE_QUALITY_VALIDATOR_VERSION,
            page_results=page_results,
            issues=issues,
            provenance=provenance,
            indexable_element_ids=indexable,
            filtered_element_ids=filtered,
            pending_visual_element_ids=pending_visuals,
            metrics={
                "scope_count": len(page_results),
                "passed_scope_count": len(page_results) - failed_scopes,
                "failed_scope_count": failed_scopes,
                "empty_scope_count": sum(
                    not result.element_ids
                    for result in page_results
                ),
                "quality_issue_count": len(issues),
                "provenance_coverage": provenance.coverage,
                "indexable_element_count": len(indexable),
                "filtered_element_count": len(filtered),
                "pending_visual_element_count": len(pending_visuals),
            },
        )

    def _build_scopes(
        self,
        document: ExtractedDocument,
    ) -> list[tuple[str, int | None, int | None, list[SourceElement]]]:
        if document.page_count is not None:
            by_page: dict[int, list[SourceElement]] = {
                page: []
                for page in range(1, document.page_count + 1)
            }
            unanchored: list[SourceElement] = []

            for element in document.elements:
                page = element.source_anchor.page_number

                if page in by_page:
                    by_page[page].append(element)
                else:
                    unanchored.append(element)

            scopes = [
                (f"page-{page}", page, None, elements)
                for page, elements in by_page.items()
            ]

            if unanchored:
                scopes.append(("page-unanchored", None, None, unanchored))

            return scopes

        by_section: dict[int, list[SourceElement]] = {}

        for element in document.elements:
            raw_section = element.metadata.get("section_index", 0)
            section_index = (
                raw_section
                if isinstance(raw_section, int) and raw_section >= 0
                else 0
            )
            by_section.setdefault(section_index, []).append(element)

        return [
            (f"section-{index}", None, index, by_section[index])
            for index in sorted(by_section)
        ]

    def _evaluate_scope(
        self,
        document: ExtractedDocument,
        elements: list[SourceElement],
        *,
        page_number: int | None,
    ) -> list[ExtractionQualityIssue]:
        issues: list[ExtractionQualityIssue] = []

        if not elements:
            return [
                _issue(
                    document,
                    reason_code="empty_page",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.TEXT,
                    message="Page contains no extracted elements",
                    triggers_fallback=True,
                    page_number=page_number,
                )
            ]

        text_elements = [
            element
            for element in elements
            if element.element_type
            not in {
                ElementType.IMAGE,
                ElementType.FLOWCHART,
                ElementType.HEADER,
                ElementType.FOOTER,
            }
            and element.content
        ]
        text = "\n".join(element.content for element in text_elements)
        has_structure_or_visual = any(
            element.element_type
            in {
                ElementType.TABLE,
                ElementType.LIST,
                ElementType.IMAGE,
                ElementType.FLOWCHART,
            }
            for element in elements
        )

        if (
            len(text.strip()) < self.config.minimum_text_characters
            and not has_structure_or_visual
        ):
            issues.append(
                _issue(
                    document,
                    reason_code="page_text_too_short",
                    severity=QualitySeverity.WARNING,
                    failure_domain=FailureDomain.TEXT,
                    message="Page contains too little extracted text",
                    triggers_fallback=True,
                    page_number=page_number,
                    element_ids=(
                        element.element_id
                        for element in elements
                    ),
                    details={
                        "actual_characters": len(text.strip()),
                        "minimum_characters": (
                            self.config.minimum_text_characters
                        ),
                    },
                )
            )

        mojibake_count = sum(
            text.count(marker)
            for marker in _MOJIBAKE_MARKERS
        )
        mojibake_ratio = mojibake_count / max(len(text), 1)

        if mojibake_ratio > self.config.maximum_mojibake_ratio:
            issues.append(
                _issue(
                    document,
                    reason_code="page_mojibake_ratio_exceeded",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.TEXT,
                    message="Page mojibake ratio exceeds threshold",
                    triggers_fallback=True,
                    page_number=page_number,
                    element_ids=(
                        element.element_id
                        for element in text_elements
                    ),
                    details={
                        "actual_ratio": mojibake_ratio,
                        "maximum_ratio": (
                            self.config.maximum_mojibake_ratio
                        ),
                    },
                )
            )

        repetition_ratio = self._repetition_ratio(elements)

        if (
            len(text) >= self.config.repetition_minimum_characters
            and repetition_ratio
            > self.config.maximum_repetition_ratio
        ):
            issues.append(
                _issue(
                    document,
                    reason_code="page_text_repetition_abnormal",
                    severity=QualitySeverity.WARNING,
                    failure_domain=FailureDomain.TEXT,
                    message="Page contains abnormally repeated text",
                    triggers_fallback=True,
                    page_number=page_number,
                    element_ids=(
                        element.element_id
                        for element in text_elements
                    ),
                    details={
                        "actual_ratio": repetition_ratio,
                        "maximum_ratio": (
                            self.config.maximum_repetition_ratio
                        ),
                    },
                )
            )

        block_indices = [
            element.source_anchor.block_index
            for element in elements
        ]

        if block_indices != sorted(block_indices) or len(
            block_indices
        ) != len(set(block_indices)):
            issues.append(
                _issue(
                    document,
                    reason_code="reading_order_invalid",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.TEXT,
                    message="Element block order is not strictly increasing",
                    triggers_fallback=True,
                    page_number=page_number,
                    element_ids=(
                        element.element_id
                        for element in elements
                    ),
                )
            )

        for table in (
            element
            for element in elements
            if element.element_type == ElementType.TABLE
        ):
            if not self._table_is_valid(table):
                issues.append(
                    _issue(
                        document,
                        reason_code="table_grid_invalid",
                        severity=QualitySeverity.ERROR,
                        failure_domain=FailureDomain.TABLE,
                        message=(
                            "Table row/column grid or GFM rendering "
                            "is invalid"
                        ),
                        triggers_fallback=True,
                        page_number=page_number,
                        element_ids=[table.element_id],
                    )
                )

        return issues

    def _repetition_ratio(
        self,
        elements: list[SourceElement],
    ) -> float:
        lines = [
            " ".join(element.content.casefold().split())
            for element in elements
            if element.element_type
            in {
                ElementType.TITLE,
                ElementType.HEADING,
                ElementType.PARAGRAPH,
                ElementType.CAPTION,
            }
            and element.content.strip()
        ]

        if len(lines) < self.config.repetition_minimum_lines:
            return 0.0

        counts = Counter(lines)
        repeated = sum(count - 1 for count in counts.values())
        return repeated / len(lines)

    @staticmethod
    def _table_is_valid(table: SourceElement) -> bool:
        metadata = table.metadata
        declared_rows = metadata.get("declared_rows")
        declared_columns = metadata.get("declared_columns")
        actual_rows = metadata.get("actual_rows")
        row_count = metadata.get("row_count")
        column_count = metadata.get("column_count")

        for value in (
            declared_rows,
            declared_columns,
            actual_rows,
            row_count,
            column_count,
        ):
            if value is not None and (
                not isinstance(value, int) or value < 1
            ):
                return False

        if (
            isinstance(declared_rows, int)
            and isinstance(actual_rows, int)
            and declared_rows != actual_rows
        ):
            return False

        for key in (
            "invalid_positions",
            "collisions",
            "missing_slots",
        ):
            value = metadata.get(key, 0)

            if isinstance(value, int) and value > 0:
                return False

        return _gfm_table_is_valid(table.content)

    @staticmethod
    def _heading_hierarchy_issues(
        document: ExtractedDocument,
    ) -> list[ExtractionQualityIssue]:
        issues: list[ExtractionQualityIssue] = []
        previous_level: int | None = None

        for element in document.elements:
            if element.element_type != ElementType.HEADING:
                continue

            level = _heading_level(element)

            if level is None:
                continue

            if (
                previous_level is not None
                and level > previous_level + 1
            ):
                issues.append(
                    _issue(
                        document,
                        reason_code="heading_hierarchy_jump",
                        severity=QualitySeverity.WARNING,
                        failure_domain=FailureDomain.TEXT,
                        message="Heading hierarchy skips one or more levels",
                        triggers_fallback=True,
                        page_number=element.source_anchor.page_number,
                        element_ids=[element.element_id],
                        details={
                            "previous_level": previous_level,
                            "current_level": level,
                        },
                    )
                )

            previous_level = level

        return issues

    @staticmethod
    def _deduplicate(
        issues: Iterable[ExtractionQualityIssue],
    ) -> list[ExtractionQualityIssue]:
        unique: dict[str, ExtractionQualityIssue] = {}

        for issue in issues:
            unique.setdefault(issue.issue_id, issue)

        return list(unique.values())
