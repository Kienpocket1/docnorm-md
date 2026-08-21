from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rag_pipeline.contracts import (
    BoundingBox,
    ElementType,
    EngineName,
    ExtractedDocument,
    ExtractionQualityIssue,
    ExtractionQualityReport,
    FailureDomain,
    QualityDisposition,
    QualitySeverity,
    QuarantineType,
    SourceAnchor,
    SourceElement,
    StageStatus,
)
from rag_pipeline.extractors import (
    DocumentIdentity,
    OpenDataLoaderLocalRun,
    OpenDataLoaderMappingResult,
)
from rag_pipeline.extractors.opendataloader_local import (
    OpenDataLoaderWorkerManifest,
)
from rag_pipeline.orchestration import (
    ExtractionCommitExistsError,
    ExtractionCommitIntegrityError,
    ExtractionOrchestrationError,
    ExtractionOrchestrator,
    ExtractionOrchestratorConfig,
    load_extraction_commit,
)
from rag_pipeline.quality import ExtractionQualityGateResult


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_record(path: Path, raw_directory: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": path.relative_to(raw_directory).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_local_run(tmp_path: Path) -> OpenDataLoaderLocalRun:
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\n% orchestration test\n")
    source_sha256 = sha256_file(input_pdf)
    run_directory = tmp_path / "local-run"
    raw_directory = run_directory / "raw"
    raw_directory.mkdir(parents=True)
    markdown_path = raw_directory / "sample.md"
    structured_path = raw_directory / "sample.json"
    markdown_path.write_text("# Test\n", encoding="utf-8")
    structured_path.write_text(
        '{"number of pages": 1, "kids": []}',
        encoding="utf-8",
    )
    now = datetime.now(timezone.utc)
    manifest = OpenDataLoaderWorkerManifest.model_validate(
        {
            "schema_version": "1.0",
            "worker_version": "test-worker",
            "status": "PASS",
            "engine": "OPENDATALOADER_LOCAL",
            "engine_version": "test",
            "started_at": now,
            "finished_at": now,
            "elapsed_seconds": 0.1,
            "input": {
                "path": str(input_pdf),
                "name": input_pdf.name,
                "size_bytes": input_pdf.stat().st_size,
                "sha256": source_sha256,
            },
            "output": {
                "artifact_directory": str(raw_directory),
                "page_count": 1,
                "total_file_count": 2,
                "markdown_character_count": 7,
                "markdown_broken_character_count": 0,
                "markdown": artifact_record(
                    markdown_path,
                    raw_directory,
                ),
                "structured_json": artifact_record(
                    structured_path,
                    raw_directory,
                ),
                "images": [],
                "physical_image_file_count": 0,
                "unique_image_hash_count": 0,
                "duplicate_image_hash_group_count": 0,
                "image_hash_groups": [],
            },
        }
    )
    manifest_path = run_directory / "worker_result.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return OpenDataLoaderLocalRun(
        run_id="local-test",
        run_directory=run_directory,
        raw_artifact_directory=raw_directory,
        result_manifest_path=manifest_path,
        stdout_log_path=run_directory / "worker_stdout.log",
        stderr_log_path=run_directory / "worker_stderr.log",
        manifest=manifest,
    )


def make_anchor(
    source_sha256: str,
    block_index: int,
) -> SourceAnchor:
    return SourceAnchor(
        anchor_id=f"anchor-{block_index}",
        source_sha256=source_sha256,
        page_number=1,
        block_index=block_index,
        bbox=BoundingBox(
            x0=10,
            y0=10 + block_index * 50,
            x1=200,
            y1=40 + block_index * 50,
        ),
    )


def build_mapping(
    run: OpenDataLoaderLocalRun,
) -> OpenDataLoaderMappingResult:
    source_sha256 = run.manifest.input.sha256
    paragraph = SourceElement(
        element_id="element-text",
        element_type=ElementType.PARAGRAPH,
        content="Indexable procedure text",
        source_anchor=make_anchor(source_sha256, 0),
        engine=EngineName.OPENDATALOADER_LOCAL,
        metadata={"indexable": True},
    )
    image = SourceElement(
        element_id="element-logo",
        element_type=ElementType.IMAGE,
        content="",
        source_anchor=make_anchor(source_sha256, 1),
        engine=EngineName.OPENDATALOADER_LOCAL,
        metadata={"indexable": False},
    )
    document = ExtractedDocument(
        catalog_id="catalog-001",
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v001-aaaaaaaaaaaa",
        source_sha256=source_sha256,
        page_count=1,
        engine_chain=[EngineName.OPENDATALOADER_LOCAL],
        elements=[paragraph, image],
    )
    return OpenDataLoaderMappingResult(
        document=document,
        mapping_issues=(),
        indexable_element_ids=("element-text",),
        filtered_element_ids=("element-logo",),
        metrics={
            "mapped_element_count": 2,
            "indexable_element_count": 1,
            "filtered_element_count": 1,
            "bbox_coverage": 1.0,
            "pages_with_elements": [1],
        },
    )


def make_quality_issue(
    source_sha256: str,
    *,
    severity: QualitySeverity,
    triggers_fallback: bool,
) -> ExtractionQualityIssue:
    return ExtractionQualityIssue(
        issue_id=(
            f"issue-{severity.value.lower()}-"
            f"{str(triggers_fallback).lower()}"
        ),
        reason_code=(
            "recoverable_visual_gap"
            if triggers_fallback
            else "nonrecoverable_contract_error"
        ),
        severity=severity,
        failure_domain=(
            FailureDomain.VISUAL
            if triggers_fallback
            else FailureDomain.PROVENANCE
        ),
        message="Synthetic extraction-quality issue",
        triggers_fallback=triggers_fallback,
    )


def build_decision(
    mapping: OpenDataLoaderMappingResult,
    disposition: QualityDisposition,
) -> ExtractionQualityGateResult:
    issues: list[ExtractionQualityIssue] = []
    fallback_engine = None

    if disposition == QualityDisposition.PASS_WITH_WARNINGS:
        issues = [
            ExtractionQualityIssue(
                issue_id="issue-known-logo",
                reason_code="repeated_brand_logo",
                severity=QualitySeverity.WARNING,
                failure_domain=FailureDomain.VISUAL,
                message="Known brand logo is excluded",
                triggers_fallback=False,
            )
        ]
    elif disposition == QualityDisposition.FALLBACK_REQUIRED:
        issues = [
            make_quality_issue(
                mapping.document.source_sha256,
                severity=QualitySeverity.WARNING,
                triggers_fallback=True,
            )
        ]
        fallback_engine = EngineName.OPENDATALOADER_HYBRID
    elif disposition == QualityDisposition.QUARANTINE_REQUIRED:
        issues = [
            make_quality_issue(
                mapping.document.source_sha256,
                severity=QualitySeverity.ERROR,
                triggers_fallback=False,
            )
        ]

    report = ExtractionQualityReport(
        report_id=f"report-{disposition.value.lower()}",
        catalog_id=mapping.document.catalog_id,
        family_id=mapping.document.family_id,
        version_id=mapping.document.version_id,
        source_sha256=mapping.document.source_sha256,
        engine=EngineName.OPENDATALOADER_LOCAL,
        disposition=disposition,
        issues=issues,
        metrics=mapping.metrics,
        fallback_engine=fallback_engine,
    )
    local_result_accepted = disposition in {
        QualityDisposition.PASS,
        QualityDisposition.PASS_WITH_WARNINGS,
    }
    return ExtractionQualityGateResult(
        report=report,
        local_result_accepted=local_result_accepted,
        fallback_required=(
            disposition == QualityDisposition.FALLBACK_REQUIRED
        ),
        indexable_element_ids=mapping.indexable_element_ids,
        filtered_element_ids=mapping.filtered_element_ids,
    )


class StaticMapper:
    def __init__(self, mapping: OpenDataLoaderMappingResult) -> None:
        self.mapping = mapping
        self.call_count = 0

    def map_run(self, run, identity):
        self.call_count += 1
        return self.mapping


class StaticQualityGate:
    def __init__(self, decision: ExtractionQualityGateResult) -> None:
        self.decision = decision
        self.call_count = 0

    def evaluate(self, mapping, manifest):
        self.call_count += 1
        return self.decision


class CountingExtractor:
    def __init__(self, run: OpenDataLoaderLocalRun) -> None:
        self.run = run
        self.call_count = 0

    def extract(self, input_pdf, *, run_id=None):
        self.call_count += 1
        return self.run


def identity() -> DocumentIdentity:
    return DocumentIdentity(
        catalog_id="catalog-001",
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v001-aaaaaaaaaaaa",
    )


def make_orchestrator(
    tmp_path: Path,
    run: OpenDataLoaderLocalRun,
    disposition: QualityDisposition,
    *,
    extractor=None,
) -> ExtractionOrchestrator:
    mapping = build_mapping(run)
    decision = build_decision(mapping, disposition)
    return ExtractionOrchestrator(
        ExtractionOrchestratorConfig(
            output_root=tmp_path / "commits"
        ),
        extractor=extractor,
        mapper=StaticMapper(mapping),
        quality_gate=StaticQualityGate(decision),
    )


def test_accepted_commit_is_atomic_and_resumable(tmp_path: Path) -> None:
    run = build_local_run(tmp_path)
    orchestrator = make_orchestrator(
        tmp_path,
        run,
        QualityDisposition.PASS_WITH_WARNINGS,
    )

    result = orchestrator.process_run(
        run,
        identity(),
        run_id="accepted-run",
    )
    commit = result.commit

    assert commit.manifest.next_action == "NORMALIZE"
    assert commit.manifest.local_result_accepted is True
    assert commit.manifest.fallback_required is False
    assert commit.manifest.local_attempt_count == 1
    assert commit.manifest.automatic_retry_count == 0
    assert commit.stage_result.status == StageStatus.PASS
    assert commit.quarantine_record is None
    assert commit.selection.indexable_element_ids == ["element-text"]
    assert commit.selection.filtered_element_ids == ["element-logo"]
    assert len(list(commit.commit_directory.iterdir())) == 5
    assert not list((tmp_path / "commits").glob("*.staging-*"))

    loaded = load_extraction_commit(commit.commit_directory)
    assert loaded.manifest == commit.manifest
    assert loaded.document == commit.document


def test_fallback_commit_does_not_retry_local(tmp_path: Path) -> None:
    run = build_local_run(tmp_path)
    orchestrator = make_orchestrator(
        tmp_path,
        run,
        QualityDisposition.FALLBACK_REQUIRED,
    )

    result = orchestrator.process_run(
        run,
        identity(),
        run_id="fallback-run",
    )
    commit = result.commit

    assert commit.manifest.next_action == (
        "RUN_OPENDATALOADER_HYBRID"
    )
    assert commit.manifest.fallback_required is True
    assert commit.manifest.automatic_retry_count == 0
    assert commit.stage_result.status == StageStatus.FAIL
    assert commit.quarantine_record is None


def test_quarantine_commit_creates_typed_record(tmp_path: Path) -> None:
    run = build_local_run(tmp_path)
    orchestrator = make_orchestrator(
        tmp_path,
        run,
        QualityDisposition.QUARANTINE_REQUIRED,
    )

    result = orchestrator.process_run(
        run,
        identity(),
        run_id="quarantine-run",
    )
    commit = result.commit
    quarantine = commit.quarantine_record

    assert commit.manifest.next_action == (
        "OPEN_EXTRACTION_QUARANTINE"
    )
    assert commit.stage_result.status == StageStatus.QUARANTINED
    assert quarantine is not None
    assert quarantine.queue == QuarantineType.EXTRACTION
    assert quarantine.attempt_count == 1
    assert quarantine.max_attempts == 1
    assert quarantine.details["automatic_retry_allowed"] is False
    assert len(list(commit.commit_directory.iterdir())) == 6


def test_existing_commit_is_never_overwritten(tmp_path: Path) -> None:
    run = build_local_run(tmp_path)
    orchestrator = make_orchestrator(
        tmp_path,
        run,
        QualityDisposition.PASS,
    )
    first = orchestrator.process_run(
        run,
        identity(),
        run_id="immutable-run",
    )
    manifest_path = (
        first.commit.commit_directory / "commit_manifest.json"
    )
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(ExtractionCommitExistsError):
        orchestrator.process_run(
            run,
            identity(),
            run_id="immutable-run",
        )

    assert manifest_path.read_bytes() == original_manifest


def test_commit_reader_rejects_tampering(tmp_path: Path) -> None:
    run = build_local_run(tmp_path)
    orchestrator = make_orchestrator(
        tmp_path,
        run,
        QualityDisposition.PASS,
    )
    result = orchestrator.process_run(
        run,
        identity(),
        run_id="tamper-run",
    )
    selection_path = (
        result.commit.commit_directory / "indexing_selection.json"
    )
    selection_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ExtractionCommitIntegrityError):
        load_extraction_commit(result.commit.commit_directory)


def test_execute_calls_local_extractor_exactly_once(
    tmp_path: Path,
) -> None:
    run = build_local_run(tmp_path)
    extractor = CountingExtractor(run)
    orchestrator = make_orchestrator(
        tmp_path,
        run,
        QualityDisposition.FALLBACK_REQUIRED,
        extractor=extractor,
    )

    result = orchestrator.execute(
        run.manifest.input.path,
        identity(),
        run_id="single-attempt-run",
    )

    assert extractor.call_count == 1
    assert result.commit.manifest.next_action == (
        "RUN_OPENDATALOADER_HYBRID"
    )
    assert result.commit.manifest.automatic_retry_count == 0


def test_incomplete_selection_is_rejected_before_commit(
    tmp_path: Path,
) -> None:
    run = build_local_run(tmp_path)
    mapping = build_mapping(run)
    decision = build_decision(mapping, QualityDisposition.PASS)
    invalid_decision = ExtractionQualityGateResult(
        report=decision.report,
        local_result_accepted=True,
        fallback_required=False,
        indexable_element_ids=("element-text",),
        filtered_element_ids=(),
    )
    orchestrator = ExtractionOrchestrator(
        ExtractionOrchestratorConfig(
            output_root=tmp_path / "commits"
        ),
        mapper=StaticMapper(mapping),
        quality_gate=StaticQualityGate(invalid_decision),
    )

    with pytest.raises(ExtractionOrchestrationError):
        orchestrator.process_run(
            run,
            identity(),
            run_id="invalid-selection-run",
        )

    assert not (tmp_path / "commits" / "invalid-selection-run").exists()
