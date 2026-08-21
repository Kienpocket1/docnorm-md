from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from rag_pipeline.contracts import (
    EngineName,
    ResumeFrom,
    RouteType,
    StageResult,
    StageStatus,
)
from rag_pipeline.extractors.base import (
    ExtractionContractError,
    ExtractionOutput,
    ExtractionRequest,
    provenance_coverage,
    validate_unified_document,
)
from rag_pipeline.extractors.docx_native import DocxNativeExtractor
from rag_pipeline.extractors.opendataloader_mapping import DocumentIdentity


EXTRACT_STAGE_VERSION = "2026-08-18-5.0.1"


class ExtractStageError(RuntimeError):
    """Base failure raised by the official Stage 5 handler."""


class ExtractStageConfigurationError(ExtractStageError):
    pass


class UnsupportedExtractionRouteError(ExtractStageError):
    pass


class PdfLocalExtractorLike(Protocol):
    def extract(
        self,
        input_pdf: Path | str,
        *,
        run_id: str | None = None,
    ) -> Any: ...


class PdfMapperLike(Protocol):
    def map_run(
        self,
        run: Any,
        identity: DocumentIdentity,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class ExtractStageOutcome:
    extraction: ExtractionOutput
    stage_result: StageResult


class ExtractStage:
    """Dispatch PDF and DOCX while keeping quality/fallback in Stage 6."""

    def __init__(
        self,
        *,
        pdf_extractor: PdfLocalExtractorLike | None = None,
        pdf_mapper: PdfMapperLike | None = None,
        docx_extractor: DocxNativeExtractor | None = None,
    ) -> None:
        self.pdf_extractor = pdf_extractor
        self.pdf_mapper = pdf_mapper
        self.docx_extractor = docx_extractor or DocxNativeExtractor()

    def run(self, request: ExtractionRequest) -> ExtractStageOutcome:
        started_at = datetime.now(timezone.utc)
        started_timer = time.perf_counter()

        if request.route.route_type == RouteType.PDF_OPENDATALOADER:
            extraction = self._extract_pdf(request)
        elif request.route.route_type == RouteType.DOCX_NATIVE:
            extraction = self.docx_extractor.extract(request)
        else:
            raise UnsupportedExtractionRouteError(
                "Stage 5 supports PDF_OPENDATALOADER and DOCX_NATIVE; "
                f"received {request.route.route_type.value}"
            )

        expected_engine = request.route.primary_engine
        validate_unified_document(
            request,
            extraction.document,
            expected_engine=expected_engine,
        )
        finished_at = datetime.now(timezone.utc)
        duration_seconds = time.perf_counter() - started_timer
        metrics = {
            **dict(extraction.metrics),
            "extract_stage_version": EXTRACT_STAGE_VERSION,
            "route_type": request.route.route_type.value,
            "primary_engine": expected_engine.value,
            "source_element_count": len(extraction.document.elements),
            "canonical_asset_count": len(extraction.document.assets),
            "provenance_coverage": provenance_coverage(
                extraction.document
            ),
            "quality_gate_invocations": 0,
            "fallback_invocations": 0,
        }
        stage_result = StageResult(
            stage=ResumeFrom.EXTRACT,
            status=StageStatus.PASS,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            artifact_paths=[
                str(path)
                for path in extraction.artifact_paths
            ],
            metrics=metrics,
            message=(
                "Unified extraction complete; continue to "
                "EXTRACTION_QUALITY"
            ),
        )
        return ExtractStageOutcome(
            extraction=ExtractionOutput(
                document=extraction.document,
                engine_version=extraction.engine_version,
                elapsed_seconds=duration_seconds,
                artifact_paths=extraction.artifact_paths,
                metrics=metrics,
                raw_handle=extraction.raw_handle,
            ),
            stage_result=stage_result,
        )

    def _extract_pdf(
        self,
        request: ExtractionRequest,
    ) -> ExtractionOutput:
        if self.pdf_extractor is None or self.pdf_mapper is None:
            raise ExtractStageConfigurationError(
                "PDF route requires both pdf_extractor and pdf_mapper"
            )

        started = time.perf_counter()
        local_run = self.pdf_extractor.extract(
            request.source_path,
            run_id=request.run_id,
        )
        mapping = self.pdf_mapper.map_run(
            local_run,
            DocumentIdentity(
                catalog_id=request.catalog_id,
                family_id=request.family_id,
                version_id=request.version_id,
            ),
        )
        manifest = local_run.manifest
        records = [
            manifest.output.markdown,
            manifest.output.structured_json,
            *manifest.output.images,
        ]
        artifact_paths = [
            Path(record.path).resolve()
            for record in records
        ]

        for path in (
            local_run.result_manifest_path,
            local_run.stdout_log_path,
            local_run.stderr_log_path,
        ):
            resolved = Path(path).resolve()

            if resolved.is_file():
                artifact_paths.append(resolved)

        if len(artifact_paths) != len(set(artifact_paths)):
            raise ExtractionContractError(
                "OpenDataLoader artifact paths must be unique"
            )

        metrics = {
            **dict(mapping.metrics),
            "extractor_version": EXTRACT_STAGE_VERSION,
            "document_format": "PDF",
            "worker_version": manifest.worker_version,
            "worker_engine_version": manifest.engine_version,
            "worker_elapsed_seconds": manifest.elapsed_seconds,
            "mapping_issue_count": len(mapping.mapping_issues),
            "external_calls": 0,
            "ocr_invocations": 0,
        }
        output = ExtractionOutput(
            document=mapping.document,
            engine_version=manifest.engine_version,
            elapsed_seconds=time.perf_counter() - started,
            artifact_paths=tuple(artifact_paths),
            metrics=metrics,
            raw_handle=local_run,
        )
        validate_unified_document(
            request,
            output.document,
            expected_engine=EngineName.OPENDATALOADER_LOCAL,
        )
        return output


def run_extract(
    request: ExtractionRequest,
    *,
    pdf_extractor: PdfLocalExtractorLike | None = None,
    pdf_mapper: PdfMapperLike | None = None,
    docx_extractor: DocxNativeExtractor | None = None,
) -> ExtractStageOutcome:
    return ExtractStage(
        pdf_extractor=pdf_extractor,
        pdf_mapper=pdf_mapper,
        docx_extractor=docx_extractor,
    ).run(request)
