from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline.contracts import (
    ExtractionQualityReport,
    QualityDisposition,
    QualitySeverity,
    ResumeFrom,
    StageResult,
    StageStatus,
)
from rag_pipeline.normalization.contract_native import (
    ContractNativeNormalizer,
    NormalizationCommitBundle,
    NormalizationCommitExistsError,
    NormalizerConfig,
)
from rag_pipeline.orchestration import (
    EXTRACTION_COMMIT_SCHEMA_VERSION,
    ExtractionCommitBundle,
    ExtractionCommitExistsError,
    ExtractionCommitManifest,
    IndexingSelection,
    load_extraction_commit,
)
from rag_pipeline.orchestration.extraction import (
    COMMIT_MANIFEST_FILENAME,
    DOCUMENT_FILENAME,
    QUALITY_REPORT_FILENAME,
    SELECTION_FILENAME,
    STAGE_RESULT_FILENAME,
    artifact_record,
    model_json_bytes,
    sha256_file,
    stable_identifier,
    validate_run_id,
    write_payload,
)
from rag_pipeline.stages.hybrid_reverify import (
    HybridReverificationCommitBundle,
    load_hybrid_reverification_commit,
)
from rag_pipeline.validators.extraction import ExtractionPageValidator


REVERIFIED_NORMALIZATION_VERSION = "2026-08-21-7.1A.1"


@dataclass(frozen=True, slots=True)
class ReverifiedNormalizationConfig:
    handoff_output_root: Path
    normalization_output_root: Path


@dataclass(frozen=True, slots=True)
class ReverifiedNormalizationResult:
    reverified_commit: HybridReverificationCommitBundle
    extraction_handoff: ExtractionCommitBundle
    normalization_commit: NormalizationCommitBundle


class ReverifiedNormalizationError(RuntimeError):
    """Base exception for the Stage 6.1C to normalization handoff."""


class ReverifiedNormalizationInputError(ReverifiedNormalizationError):
    pass


def utc_normalization_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"stage7_1a_{stamp}_{uuid.uuid4().hex[:8]}"


class ReverifiedNormalizationOrchestrator:
    """Convert an accepted Stage 6.1C commit into a chunk-ready normalization."""

    def __init__(
        self,
        config: ReverifiedNormalizationConfig,
        *,
        validator: ExtractionPageValidator | None = None,
    ) -> None:
        self.config = config
        self.validator = validator or ExtractionPageValidator()

    def normalize_reverified_commit(
        self,
        reverified_commit_directory: Path | str,
        *,
        run_id: str | None = None,
        handoff_run_id: str | None = None,
    ) -> ReverifiedNormalizationResult:
        reverified = load_hybrid_reverification_commit(
            reverified_commit_directory
        )
        self._validate_reverified_input(reverified)
        active_run_id = run_id or utc_normalization_run_id()
        active_handoff_run_id = (
            handoff_run_id or f"{active_run_id}-handoff"
        )
        validate_run_id(active_run_id)
        validate_run_id(active_handoff_run_id)
        normalization_target = (
            self.config.normalization_output_root.expanduser().resolve()
            / active_run_id
        )

        if normalization_target.exists():
            raise NormalizationCommitExistsError(
                f"Normalization commit already exists: {active_run_id}"
            )

        handoff = self._create_extraction_handoff(
            reverified,
            run_id=active_handoff_run_id,
        )
        normalizer = ContractNativeNormalizer(
            NormalizerConfig(
                output_root=self.config.normalization_output_root
            )
        )
        normalization = normalizer.normalize_commit(
            handoff.commit_directory,
            run_id=active_run_id,
        )
        return ReverifiedNormalizationResult(
            reverified_commit=reverified,
            extraction_handoff=handoff,
            normalization_commit=normalization,
        )

    @staticmethod
    def _validate_reverified_input(
        reverified: HybridReverificationCommitBundle,
    ) -> None:
        report = reverified.report

        if report.next_action != "NORMALIZE":
            raise ReverifiedNormalizationInputError(
                "Stage 6.1C commit is not routed to NORMALIZE"
            )

        if reverified.stage_result.status != StageStatus.PASS:
            raise ReverifiedNormalizationInputError(
                "Stage 6.1C stage result must be PASS"
            )

        if report.mineru_page_numbers:
            raise ReverifiedNormalizationInputError(
                "Stage 6.1C still contains pending MinerU pages"
            )

        if report.provenance_coverage != 1.0:
            raise ReverifiedNormalizationInputError(
                "Stage 6.1C provenance coverage must be 100 percent"
            )

        if report.external_calls != 0 or report.ocr_invocations != 0:
            raise ReverifiedNormalizationInputError(
                "Stage 6.1C re-verification must remain local and OCR-free"
            )

    def _create_extraction_handoff(
        self,
        reverified: HybridReverificationCommitBundle,
        *,
        run_id: str,
    ) -> ExtractionCommitBundle:
        output_root = self.config.handoff_output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        final_directory = output_root / run_id

        if final_directory.exists():
            raise ExtractionCommitExistsError(
                f"Extraction handoff commit already exists: {run_id}"
            )

        document = reverified.document
        assessment = self.validator.evaluate(document)
        failing_issues = [
            issue
            for issue in assessment.issues
            if issue.triggers_fallback
            or issue.severity == QualitySeverity.ERROR
        ]

        if failing_issues or any(
            not result.passed for result in assessment.page_results
        ):
            reasons = sorted(
                {issue.reason_code for issue in failing_issues}
            )
            raise ReverifiedNormalizationInputError(
                "Reverified document failed the normalization handoff: "
                + ",".join(reasons)
            )

        if assessment.provenance.coverage != 1.0:
            raise ReverifiedNormalizationInputError(
                "Reverified document provenance is incomplete"
            )

        if assessment.pending_visual_element_ids:
            raise ReverifiedNormalizationInputError(
                "Reverified document contains unresolved informative visuals"
            )

        element_ids = [element.element_id for element in document.elements]
        selected_ids = [*assessment.indexable_element_ids]
        filtered_ids = [*assessment.filtered_element_ids]

        if set(element_ids) != set(selected_ids) | set(filtered_ids):
            raise ReverifiedNormalizationInputError(
                "Normalization selection does not cover every source element"
            )

        expected_selected_order = [
            element_id
            for element_id in element_ids
            if element_id in set(selected_ids)
        ]

        if selected_ids != expected_selected_order:
            raise ReverifiedNormalizationInputError(
                "Normalization selection changed document order"
            )

        source_manifest_path = (
            reverified.commit_directory / COMMIT_MANIFEST_FILENAME
        )
        source_manifest_sha256 = sha256_file(source_manifest_path)
        warnings = [
            issue
            for issue in assessment.issues
            if issue.severity == QualitySeverity.WARNING
        ]
        disposition = (
            QualityDisposition.PASS_WITH_WARNINGS
            if warnings
            else QualityDisposition.PASS
        )
        report = ExtractionQualityReport(
            report_id=stable_identifier(
                "extraction-quality-report",
                document.source_sha256,
                document.version_id,
                reverified.manifest.run_id,
                REVERIFIED_NORMALIZATION_VERSION,
            ),
            catalog_id=document.catalog_id,
            family_id=document.family_id,
            version_id=document.version_id,
            source_sha256=document.source_sha256,
            engine=document.engine_chain[-1],
            disposition=disposition,
            issues=assessment.issues,
            metrics={
                **assessment.metrics,
                "handoff_version": REVERIFIED_NORMALIZATION_VERSION,
                "input_reverification_run_id": reverified.manifest.run_id,
                "input_reverification_manifest_sha256": (
                    source_manifest_sha256
                ),
                "next_action": "NORMALIZE",
            },
            evaluated_at=reverified.manifest.committed_at,
        )
        selection = IndexingSelection(
            catalog_id=document.catalog_id,
            family_id=document.family_id,
            version_id=document.version_id,
            source_sha256=document.source_sha256,
            indexable_element_ids=selected_ids,
            filtered_element_ids=filtered_ids,
        )
        stage_result = StageResult(
            stage=ResumeFrom.EXTRACTION_QUALITY,
            status=StageStatus.PASS,
            started_at=reverified.manifest.committed_at,
            finished_at=reverified.manifest.committed_at,
            duration_seconds=0.0,
            artifact_paths=[
                DOCUMENT_FILENAME,
                QUALITY_REPORT_FILENAME,
                SELECTION_FILENAME,
            ],
            metrics={
                "quality_disposition": disposition.value,
                "quality_issue_count": len(report.issues),
                "next_action": "NORMALIZE",
                "local_attempt_count": 1,
                "automatic_retry_count": 0,
                "handoff_version": REVERIFIED_NORMALIZATION_VERSION,
            },
            message="Accepted reverified extraction handoff to normalization",
        )
        staging_directory = output_root / (
            f".{run_id}.staging-{uuid.uuid4().hex}"
        )
        staging_directory.mkdir(parents=False, exist_ok=False)

        try:
            models = {
                DOCUMENT_FILENAME: document,
                QUALITY_REPORT_FILENAME: report,
                SELECTION_FILENAME: selection,
                STAGE_RESULT_FILENAME: stage_result,
            }

            for filename, model in models.items():
                write_payload(
                    staging_directory / filename,
                    model_json_bytes(model),
                )

            artifacts = [
                artifact_record(staging_directory / filename, staging_directory)
                for filename in sorted(models)
            ]
            manifest = ExtractionCommitManifest(
                schema_version=EXTRACTION_COMMIT_SCHEMA_VERSION,
                orchestrator_version=REVERIFIED_NORMALIZATION_VERSION,
                run_id=run_id,
                catalog_id=document.catalog_id,
                family_id=document.family_id,
                version_id=document.version_id,
                source_sha256=document.source_sha256,
                local_run_id=(
                    f"{reverified.manifest.run_id}-"
                    f"{source_manifest_sha256[:12]}"
                ),
                local_run_directory=str(reverified.commit_directory),
                local_worker_manifest_path=str(source_manifest_path),
                quality_disposition=disposition,
                local_result_accepted=True,
                fallback_required=False,
                next_action="NORMALIZE",
                local_attempt_count=1,
                automatic_retry_count=0,
                quarantine_id=None,
                artifact_count=len(artifacts),
                artifacts=artifacts,
                committed_at=reverified.manifest.committed_at,
            )
            write_payload(
                staging_directory / COMMIT_MANIFEST_FILENAME,
                model_json_bytes(manifest),
            )
            load_extraction_commit(staging_directory)

            if final_directory.exists():
                raise ExtractionCommitExistsError(
                    f"Extraction handoff commit already exists: {run_id}"
                )

            staging_directory.rename(final_directory)
        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
            raise

        return load_extraction_commit(final_directory)
