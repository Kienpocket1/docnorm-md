from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from rag_pipeline.contracts import (
    EngineName,
    ExtractedDocument,
    ExtractionQualityIssue,
    ExtractionQualityReport,
    FailureDomain,
    QualityDisposition,
    QualitySeverity,
    ResumeFrom,
    StageResult,
    StageStatus,
)
from rag_pipeline.failure import ExtractionRoutingPlan, FailureRouter
from rag_pipeline.validators import (
    DocumentPageQualityResult,
    ExtractionPageValidator,
    NarrativeCoverageResult,
    NarrativeCoverageValidator,
)


EXTRACTION_QUALITY_STAGE_VERSION = "2026-08-18-6.1A.1"


@dataclass(frozen=True, slots=True)
class ExtractionQualityStageOutcome:
    assessment: DocumentPageQualityResult
    narrative_results: tuple[NarrativeCoverageResult, ...]
    routing_plan: ExtractionRoutingPlan
    report: ExtractionQualityReport
    stage_result: StageResult


def _stable_identifier(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


class ExtractionQualityStage:
    """Run Stage 6 quality preflight and produce per-page routing plans."""

    def __init__(
        self,
        *,
        page_validator: ExtractionPageValidator | None = None,
        narrative_validator: NarrativeCoverageValidator | None = None,
        failure_router: FailureRouter | None = None,
    ) -> None:
        self.page_validator = page_validator or ExtractionPageValidator()
        self.narrative_validator = (
            narrative_validator or NarrativeCoverageValidator()
        )
        self.failure_router = failure_router or FailureRouter()

    def evaluate(
        self,
        document: ExtractedDocument,
        *,
        inherited_issues: Iterable[ExtractionQualityIssue] = (),
    ) -> ExtractionQualityStageOutcome:
        started_at = datetime.now(timezone.utc)
        started_timer = time.perf_counter()
        assessment = self.page_validator.evaluate(
            document,
            inherited_issues=inherited_issues,
        )
        narrative_results = tuple(
            self.narrative_validator.evaluate(document, element_id)
            for element_id in assessment.pending_visual_element_ids
        )
        narrative_issues = self._narrative_issues(
            document,
            narrative_results,
        )

        if narrative_issues:
            assessment = assessment.model_copy(
                update={
                    "issues": self._deduplicate_issues(
                        [*assessment.issues, *narrative_issues]
                    ),
                    "metrics": {
                        **assessment.metrics,
                        "quality_issue_count": len(
                            self._deduplicate_issues(
                                [
                                    *assessment.issues,
                                    *narrative_issues,
                                ]
                            )
                        ),
                    },
                }
            )

        routing_plan = self.failure_router.route(
            assessment,
            source_sha256=document.source_sha256,
            narrative_results=list(narrative_results),
        )
        issues = self._deduplicate_issues(assessment.issues)
        fallback_issues = [
            issue
            for issue in issues
            if issue.triggers_fallback
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

        if routing_plan.fallback_required:
            disposition = QualityDisposition.FALLBACK_REQUIRED
            fallback_engine = self._fallback_engine(
                document,
                routing_plan,
            )
        elif errors:
            disposition = QualityDisposition.QUARANTINE_REQUIRED
        elif warnings:
            disposition = QualityDisposition.PASS_WITH_WARNINGS
        else:
            disposition = QualityDisposition.PASS

        current_engine = document.engine_chain[-1]
        report = ExtractionQualityReport(
            report_id=_stable_identifier(
                "extraction-quality",
                document.source_sha256,
                document.version_id,
                EXTRACTION_QUALITY_STAGE_VERSION,
            ),
            catalog_id=document.catalog_id,
            family_id=document.family_id,
            version_id=document.version_id,
            source_sha256=document.source_sha256,
            engine=current_engine,
            disposition=disposition,
            issues=issues,
            metrics={
                **assessment.metrics,
                **routing_plan.metrics,
                "quality_stage_version": (
                    EXTRACTION_QUALITY_STAGE_VERSION
                ),
                "quality_warning_count": len(warnings),
                "quality_error_count": len(errors),
                "fallback_issue_count": len(fallback_issues),
                "narrative_check_count": len(narrative_results),
                "external_calls": 0,
                "ocr_invocations": 0,
            },
            fallback_engine=fallback_engine,
        )
        finished_at = datetime.now(timezone.utc)
        status = StageStatus.PASS

        if disposition == QualityDisposition.FALLBACK_REQUIRED:
            status = StageStatus.FAIL
        elif disposition == QualityDisposition.QUARANTINE_REQUIRED:
            status = StageStatus.QUARANTINED

        quarantine_id = (
            _stable_identifier(
                "extraction-quarantine",
                document.source_sha256,
                document.version_id,
                report.report_id,
            )
            if status == StageStatus.QUARANTINED
            else None
        )

        stage_result = StageResult(
            stage=ResumeFrom.EXTRACTION_QUALITY,
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=time.perf_counter() - started_timer,
            artifact_paths=[
                "page_quality.json",
                "routing_plan.json",
                "extraction_quality_report.json",
            ],
            metrics={
                "quality_disposition": disposition.value,
                "next_action": routing_plan.next_action,
                "scope_count": len(assessment.page_results),
                "quality_issue_count": len(issues),
                "fallback_route_count": routing_plan.metrics[
                    "fallback_route_count"
                ],
                "external_calls": 0,
                "ocr_invocations": 0,
            },
            quarantine_id=quarantine_id,
            message=(
                "Extraction quality routing complete; next action: "
                f"{routing_plan.next_action}"
            ),
        )
        return ExtractionQualityStageOutcome(
            assessment=assessment,
            narrative_results=narrative_results,
            routing_plan=routing_plan,
            report=report,
            stage_result=stage_result,
        )

    @staticmethod
    def _narrative_issues(
        document: ExtractedDocument,
        results: tuple[NarrativeCoverageResult, ...],
    ) -> list[ExtractionQualityIssue]:
        by_id = {
            element.element_id: element
            for element in document.elements
        }
        issues: list[ExtractionQualityIssue] = []

        for result in results:
            element = by_id[result.visual_element_id]
            reason_code = (
                "visual_narrative_coverage_sufficient"
                if result.sufficient
                else "visual_narrative_coverage_insufficient"
            )
            issues.append(
                ExtractionQualityIssue(
                    issue_id=_stable_identifier(
                        "quality-issue",
                        document.source_sha256,
                        reason_code,
                        element.element_id,
                    ),
                    reason_code=reason_code,
                    severity=(
                        QualitySeverity.INFO
                        if result.sufficient
                        else QualitySeverity.WARNING
                    ),
                    failure_domain=FailureDomain.VISUAL,
                    message=(
                        "Visual has sufficient nearby narrative"
                        if result.sufficient
                        else "Visual requires Qwen flowchart extraction"
                    ),
                    page_numbers=(
                        [element.source_anchor.page_number]
                        if element.source_anchor.page_number is not None
                        else []
                    ),
                    element_ids=[element.element_id],
                    triggers_fallback=not result.sufficient,
                    details=result.model_dump(mode="json"),
                )
            )

        return issues

    @staticmethod
    def _fallback_engine(
        document: ExtractedDocument,
        routing_plan: ExtractionRoutingPlan,
    ) -> EngineName:
        ordered_engines = [
            step.engine
            for route in routing_plan.routes
            if route.fallback_required
            for step in route.steps
            if step.engine is not None
        ]

        for engine in ordered_engines:
            if engine != document.engine_chain[-1]:
                return engine

        return EngineName.MANUAL

    @staticmethod
    def _deduplicate_issues(
        issues: Iterable[ExtractionQualityIssue],
    ) -> list[ExtractionQualityIssue]:
        unique: dict[str, ExtractionQualityIssue] = {}

        for issue in issues:
            unique.setdefault(issue.issue_id, issue)

        return list(unique.values())
