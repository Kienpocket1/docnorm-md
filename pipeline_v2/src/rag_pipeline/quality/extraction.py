from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from rag_pipeline.contracts import (
    AssetClassification,
    AssetRole,
    EngineName,
    ExtractionQualityIssue,
    ExtractionQualityReport,
    FailureDomain,
    QualityDisposition,
    QualitySeverity,
)
from rag_pipeline.extractors.opendataloader_local import (
    OpenDataLoaderWorkerManifest,
)
from rag_pipeline.extractors.opendataloader_mapping import (
    OpenDataLoaderMappingResult,
)


EXTRACTION_QUALITY_GATE_VERSION = "2026-08-17-5.1D.1"


@dataclass(frozen=True, slots=True)
class ExtractionQualityGateResult:
    report: ExtractionQualityReport
    local_result_accepted: bool
    fallback_required: bool
    indexable_element_ids: tuple[str, ...]
    filtered_element_ids: tuple[str, ...]


def stable_issue_id(
    source_sha256: str,
    reason_code: str,
    *parts: object,
) -> str:
    payload = "|".join(
        [source_sha256, reason_code, *(str(part) for part in parts)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"quality-issue-{digest}"


class ExtractionQualityGate:
    def __init__(self, minimum_bbox_coverage: float = 0.98) -> None:
        if not 0 <= minimum_bbox_coverage <= 1:
            raise ValueError(
                "minimum_bbox_coverage must be between zero and one"
            )

        self.minimum_bbox_coverage = minimum_bbox_coverage

    def evaluate(
        self,
        mapping: OpenDataLoaderMappingResult,
        manifest: OpenDataLoaderWorkerManifest,
    ) -> ExtractionQualityGateResult:
        document = mapping.document
        issues = list(mapping.mapping_issues)

        if manifest.output.markdown_broken_character_count > 0:
            issues.append(
                self._issue(
                    source_sha256=document.source_sha256,
                    reason_code="markdown_encoding_error",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.TEXT,
                    message="Markdown contains Unicode replacement characters",
                    triggers_fallback=True,
                    details={
                        "broken_character_count": (
                            manifest.output.markdown_broken_character_count
                        )
                    },
                )
            )

        if document.page_count != manifest.output.page_count:
            issues.append(
                self._issue(
                    source_sha256=document.source_sha256,
                    reason_code="page_count_mismatch",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.PROVENANCE,
                    message=(
                        "Mapped document page count differs from worker "
                        "manifest"
                    ),
                    triggers_fallback=True,
                )
            )

        expected_pages = set(range(1, manifest.output.page_count + 1))
        mapped_pages = set(mapping.metrics["pages_with_elements"])
        missing_pages = sorted(expected_pages - mapped_pages)

        if missing_pages:
            issues.append(
                self._issue(
                    source_sha256=document.source_sha256,
                    reason_code="pages_without_mapped_elements",
                    severity=QualitySeverity.WARNING,
                    failure_domain=FailureDomain.PROVENANCE,
                    message=(
                        "One or more PDF pages contain no mapped elements"
                    ),
                    page_numbers=missing_pages,
                    triggers_fallback=True,
                )
            )

        bbox_coverage = float(mapping.metrics["bbox_coverage"])

        if bbox_coverage < self.minimum_bbox_coverage:
            issues.append(
                self._issue(
                    source_sha256=document.source_sha256,
                    reason_code="insufficient_bbox_coverage",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.PROVENANCE,
                    message="Mapped element bounding-box coverage is too low",
                    triggers_fallback=True,
                    details={
                        "actual": bbox_coverage,
                        "minimum": self.minimum_bbox_coverage,
                    },
                )
            )

        if not document.elements:
            issues.append(
                self._issue(
                    source_sha256=document.source_sha256,
                    reason_code="no_mapped_elements",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.TEXT,
                    message="Extraction produced no mapped source elements",
                    triggers_fallback=True,
                )
            )

        for asset in document.assets:
            referenced_paths = {
                reference.source_path
                for reference in asset.references
            }
            unreferenced_count = (
                len(asset.physical_paths) - len(referenced_paths)
            )

            if (
                asset.classification == AssetClassification.DECORATIVE
                and len(asset.physical_paths) > 1
            ):
                reason_code = "repeated_decorative_asset"

                if asset.role == AssetRole.BRAND_LOGO:
                    reason_code = "repeated_brand_logo"

                issues.append(
                    self._issue(
                        source_sha256=document.source_sha256,
                        reason_code=reason_code,
                        severity=QualitySeverity.WARNING,
                        failure_domain=FailureDomain.VISUAL,
                        message=(
                            "Repeated decorative asset will be excluded "
                            "from indexing"
                        ),
                        asset_ids=[asset.asset_id],
                        triggers_fallback=False,
                        details={
                            "content_sha256": asset.content_sha256,
                            "physical_file_count": len(
                                asset.physical_paths
                            ),
                            "reference_count": len(asset.references),
                        },
                    )
                )

            if unreferenced_count > 0:
                issues.append(
                    self._issue(
                        source_sha256=document.source_sha256,
                        reason_code="unreferenced_extracted_assets",
                        severity=QualitySeverity.WARNING,
                        failure_domain=FailureDomain.VISUAL,
                        message=(
                            "Some physical image artifacts are not "
                            "referenced by JSON or Markdown"
                        ),
                        asset_ids=[asset.asset_id],
                        triggers_fallback=False,
                        details={
                            "unreferenced_file_count": unreferenced_count,
                            "physical_file_count": len(
                                asset.physical_paths
                            ),
                        },
                    )
                )

        issues = self._deduplicate_issues(issues)
        fallback_issues = [
            issue for issue in issues if issue.triggers_fallback
        ]
        errors = [
            issue
            for issue in issues
            if issue.severity == QualitySeverity.ERROR
        ]
        warnings = [
            issue
            for issue in issues
            if issue.severity == QualitySeverity.WARNING
        ]

        fallback_engine: EngineName | None = None

        if fallback_issues:
            disposition = QualityDisposition.FALLBACK_REQUIRED
            fallback_engine = EngineName.OPENDATALOADER_HYBRID
        elif errors:
            disposition = QualityDisposition.QUARANTINE_REQUIRED
        elif warnings:
            disposition = QualityDisposition.PASS_WITH_WARNINGS
        else:
            disposition = QualityDisposition.PASS

        report = ExtractionQualityReport(
            report_id=self._report_id(
                document.source_sha256,
                document.version_id,
            ),
            catalog_id=document.catalog_id,
            family_id=document.family_id,
            version_id=document.version_id,
            source_sha256=document.source_sha256,
            engine=EngineName.OPENDATALOADER_LOCAL,
            disposition=disposition,
            issues=issues,
            metrics={
                **mapping.metrics,
                "quality_gate_version": (
                    EXTRACTION_QUALITY_GATE_VERSION
                ),
                "worker_page_count": manifest.output.page_count,
                "worker_total_file_count": (
                    manifest.output.total_file_count
                ),
                "quality_issue_count": len(issues),
                "quality_warning_count": len(warnings),
                "quality_error_count": len(errors),
                "fallback_issue_count": len(fallback_issues),
            },
            fallback_engine=fallback_engine,
        )

        return ExtractionQualityGateResult(
            report=report,
            local_result_accepted=not fallback_issues and not errors,
            fallback_required=bool(fallback_issues),
            indexable_element_ids=mapping.indexable_element_ids,
            filtered_element_ids=mapping.filtered_element_ids,
        )

    @staticmethod
    def _report_id(source_sha256: str, version_id: str) -> str:
        digest = hashlib.sha256(
            f"{source_sha256}|{version_id}|quality".encode("utf-8")
        ).hexdigest()[:24]
        return f"extraction-quality-{digest}"

    @staticmethod
    def _deduplicate_issues(
        issues: list[ExtractionQualityIssue],
    ) -> list[ExtractionQualityIssue]:
        output: list[ExtractionQualityIssue] = []
        seen: set[str] = set()

        for issue in issues:
            if issue.issue_id in seen:
                continue

            seen.add(issue.issue_id)
            output.append(issue)

        return output

    @staticmethod
    def _issue(
        *,
        source_sha256: str,
        reason_code: str,
        severity: QualitySeverity,
        failure_domain: FailureDomain,
        message: str,
        triggers_fallback: bool,
        page_numbers: list[int] | None = None,
        asset_ids: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> ExtractionQualityIssue:
        pages = sorted(set(page_numbers or []))
        assets = sorted(set(asset_ids or []))

        return ExtractionQualityIssue(
            issue_id=stable_issue_id(
                source_sha256,
                reason_code,
                ",".join(map(str, pages)),
                ",".join(assets),
            ),
            reason_code=reason_code,
            severity=severity,
            failure_domain=failure_domain,
            message=message,
            page_numbers=pages,
            asset_ids=assets,
            triggers_fallback=triggers_fallback,
            details=details or {},
        )
