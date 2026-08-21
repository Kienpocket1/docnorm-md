import pytest
from pydantic import ValidationError

from rag_pipeline.contracts import (
    AssetClassification,
    AssetReference,
    AssetRole,
    BoundingBox,
    CatalogEntry,
    ChunkRecord,
    ElementType,
    EngineName,
    ExtractedAsset,
    ExtractedDocument,
    ExtractionQualityIssue,
    ExtractionQualityReport,
    FailureDomain,
    FileFormat,
    NormalizedBlock,
    NormalizedSpan,
    PublicationState,
    PublishReceipt,
    QualityDisposition,
    QualitySeverity,
    QuarantineRecord,
    QuarantineType,
    ResolutionDecision,
    ResolutionOutcome,
    ResumeFrom,
    SourceAnchor,
    SourceElement,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
ASSET_SHA = "c" * 64


def make_anchor(
    source_sha256: str = SHA_A,
) -> SourceAnchor:
    return SourceAnchor(
        anchor_id="page-1-block-0",
        source_sha256=source_sha256,
        page_number=1,
        block_index=0,
        bbox=BoundingBox(
            x0=10,
            y0=20,
            x1=100,
            y1=80,
        ),
    )


def make_brand_logo_asset(
    source_sha256: str = SHA_A,
) -> ExtractedAsset:
    physical_paths = [
        f"document_images/imageFile{index}.png"
        for index in range(1, 17)
    ]
    referenced_indices = (1, 2, 15, 16)
    references = [
        AssetReference(
            reference_id=f"logo-reference-{index}",
            source_path=(
                f"document_images/imageFile{index}.png"
            ),
            source_anchor=SourceAnchor(
                anchor_id=f"page-{index}-logo",
                source_sha256=source_sha256,
                page_number=index,
                block_index=0,
                bbox=BoundingBox(
                    x0=10,
                    y0=10,
                    x1=200,
                    y1=80,
                ),
            ),
            referenced_in_markdown=True,
            referenced_in_structure=True,
        )
        for index in referenced_indices
    ]

    return ExtractedAsset(
        asset_id="asset-brand-logo",
        source_sha256=source_sha256,
        content_sha256=ASSET_SHA,
        media_type="image/png",
        size_bytes=44_492,
        classification=AssetClassification.DECORATIVE,
        role=AssetRole.BRAND_LOGO,
        physical_paths=physical_paths,
        references=references,
        retain_raw=True,
        indexable=False,
    )


def test_catalog_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError):
        CatalogEntry(
            catalog_id="catalog-001",
            source_name="test.pdf",
            source_path="data_local/test.pdf",
            file_format=FileFormat.PDF,
            source_sha256="invalid",
            size_bytes=100,
        )


def test_bbox_rejects_invalid_order() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(
            x0=100,
            y0=20,
            x1=10,
            y1=80,
        )


def test_same_version_requires_references() -> None:
    with pytest.raises(ValidationError):
        ResolutionDecision(
            decision_id="decision-001",
            catalog_id="catalog-001",
            outcome=ResolutionOutcome.SAME_VERSION,
            confidence=1.0,
            reason_code="same_sha256",
        )


def test_conflict_requires_human_review() -> None:
    with pytest.raises(ValidationError):
        ResolutionDecision(
            decision_id="decision-002",
            catalog_id="catalog-001",
            outcome=ResolutionOutcome.CONFLICT,
            confidence=0.5,
            reason_code="family_conflict",
            requires_human_review=False,
        )


def test_extracted_document_rejects_mismatched_sha() -> None:
    element = SourceElement(
        element_id="element-001",
        element_type=ElementType.PARAGRAPH,
        content="Nội dung kiểm thử",
        source_anchor=make_anchor(SHA_B),
        engine=EngineName.OPENDATALOADER_LOCAL,
    )

    with pytest.raises(ValidationError):
        ExtractedDocument(
            catalog_id="catalog-001",
            family_id="qt-tckt-01",
            version_id="qt-tckt-01-v01",
            source_sha256=SHA_A,
            page_count=1,
            engine_chain=[
                EngineName.OPENDATALOADER_LOCAL
            ],
            elements=[element],
        )


def test_normalized_span_cannot_exceed_text() -> None:
    span = NormalizedSpan(
        span_id="span-001",
        source_element_id="element-001",
        char_start=0,
        char_end=50,
        source_anchor=make_anchor(),
    )

    with pytest.raises(ValidationError):
        NormalizedBlock(
            block_id="block-001",
            markdown="Nội dung ngắn",
            source_spans=[span],
        )


def test_chunk_identifier_is_stable() -> None:
    version_id = "qt-tckt-01-v01"
    chunk_id = "qt-tckt-01#vi#001"

    chunk = ChunkRecord(
        family_id="qt-tckt-01",
        version_id=version_id,
        chunk_id=chunk_id,
        chunk_instance_id=(
            f"{version_id}::{chunk_id}"
        ),
        source_sha256=SHA_A,
        ordinal=1,
        text="Nội dung chunk",
        token_count=4,
        source_anchors=[make_anchor()],
    )

    assert chunk.chunk_id == chunk_id


def test_quarantine_rejects_excess_attempts() -> None:
    with pytest.raises(ValidationError):
        QuarantineRecord(
            quarantine_id="quarantine-001",
            queue=QuarantineType.EXTRACTION,
            failure_domain=FailureDomain.PROVENANCE,
            root_cause_stage=ResumeFrom.EXTRACT,
            reason_code="source_anchor_mapping_failed",
            resume_from=ResumeFrom.EXTRACT,
            attempt_count=3,
            max_attempts=2,
        )


def test_source_provenance_uses_extraction_queue() -> None:
    record = QuarantineRecord(
        quarantine_id="quarantine-002",
        queue=QuarantineType.EXTRACTION,
        failure_domain=FailureDomain.PROVENANCE,
        root_cause_stage=ResumeFrom.EXTRACT,
        reason_code="source_anchor_mapping_failed",
        resume_from=ResumeFrom.EXTRACT,
        attempt_count=2,
        max_attempts=2,
    )

    assert record.queue == QuarantineType.EXTRACTION
    assert record.resume_from == ResumeFrom.EXTRACT


def test_publish_rejects_same_previous_version() -> None:
    with pytest.raises(ValidationError):
        PublishReceipt(
            transaction_id="transaction-001",
            family_id="qt-tckt-01",
            version_id="qt-tckt-01-v02",
            previous_active_version_id="qt-tckt-01-v02",
            new_state=PublicationState.ACTIVE,
            vector_namespace="qt-tckt-01-v02",
            active_pointer="qt-tckt-01-v02",
        )


def test_brand_logo_keeps_physical_files_and_logical_references() -> None:
    asset = make_brand_logo_asset()

    assert len(asset.physical_paths) == 16
    assert len(asset.references) == 4
    assert asset.classification == AssetClassification.DECORATIVE
    assert asset.role == AssetRole.BRAND_LOGO
    assert asset.retain_raw is True
    assert asset.indexable is False


def test_decorative_asset_cannot_be_indexable() -> None:
    with pytest.raises(ValidationError):
        ExtractedAsset(
            asset_id="asset-invalid-logo",
            source_sha256=SHA_A,
            content_sha256=ASSET_SHA,
            media_type="image/png",
            size_bytes=100,
            classification=AssetClassification.DECORATIVE,
            role=AssetRole.BRAND_LOGO,
            physical_paths=["images/logo.png"],
            indexable=True,
        )


def test_asset_reference_must_point_to_physical_file() -> None:
    reference = AssetReference(
        reference_id="reference-missing-file",
        source_path="images/missing.png",
        source_anchor=make_anchor(),
        referenced_in_structure=True,
    )

    with pytest.raises(ValidationError):
        ExtractedAsset(
            asset_id="asset-missing-file",
            source_sha256=SHA_A,
            content_sha256=ASSET_SHA,
            media_type="image/png",
            size_bytes=100,
            classification=AssetClassification.UNKNOWN,
            role=AssetRole.UNKNOWN,
            physical_paths=["images/present.png"],
            references=[reference],
        )


def test_extracted_document_rejects_mismatched_asset_sha() -> None:
    element = SourceElement(
        element_id="element-asset-test",
        element_type=ElementType.PARAGRAPH,
        content="Nội dung kiểm thử",
        source_anchor=make_anchor(SHA_A),
        engine=EngineName.OPENDATALOADER_LOCAL,
    )

    with pytest.raises(ValidationError):
        ExtractedDocument(
            catalog_id="catalog-001",
            family_id="qt-tckt-01",
            version_id="qt-tckt-01-v01",
            source_sha256=SHA_A,
            page_count=16,
            engine_chain=[EngineName.OPENDATALOADER_LOCAL],
            elements=[element],
            assets=[make_brand_logo_asset(SHA_B)],
        )


def test_quality_report_accepts_non_fallback_logo_warning() -> None:
    issue = ExtractionQualityIssue(
        issue_id="issue-repeated-brand-logo",
        reason_code="repeated_brand_logo",
        severity=QualitySeverity.WARNING,
        failure_domain=FailureDomain.VISUAL,
        message="Repeated brand logo requires normalization filtering",
        page_numbers=[1, 2, 15, 16],
        asset_ids=["asset-brand-logo"],
        triggers_fallback=False,
        details={
            "physical_file_count": 16,
            "referenced_file_count": 4,
            "unique_content_hash_count": 1,
        },
    )
    report = ExtractionQualityReport(
        report_id="quality-report-001",
        catalog_id="catalog-001",
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v01",
        source_sha256=SHA_A,
        engine=EngineName.OPENDATALOADER_LOCAL,
        disposition=QualityDisposition.PASS_WITH_WARNINGS,
        issues=[issue],
        metrics={
            "page_count": 16,
            "bbox_coverage": 1.0,
        },
    )

    assert report.fallback_engine is None
    assert report.issues[0].triggers_fallback is False


def test_fallback_report_requires_fallback_engine() -> None:
    issue = ExtractionQualityIssue(
        issue_id="issue-table-loss",
        reason_code="table_structure_loss",
        severity=QualitySeverity.ERROR,
        failure_domain=FailureDomain.TABLE,
        message="Structured table extraction failed",
        page_numbers=[10],
        triggers_fallback=True,
    )

    with pytest.raises(ValidationError):
        ExtractionQualityReport(
            report_id="quality-report-002",
            catalog_id="catalog-001",
            family_id="qt-tckt-01",
            version_id="qt-tckt-01-v01",
            source_sha256=SHA_A,
            engine=EngineName.OPENDATALOADER_LOCAL,
            disposition=QualityDisposition.FALLBACK_REQUIRED,
            issues=[issue],
        )


def test_fallback_report_accepts_different_engine() -> None:
    issue = ExtractionQualityIssue(
        issue_id="issue-table-loss",
        reason_code="table_structure_loss",
        severity=QualitySeverity.ERROR,
        failure_domain=FailureDomain.TABLE,
        message="Structured table extraction failed",
        page_numbers=[10],
        triggers_fallback=True,
    )
    report = ExtractionQualityReport(
        report_id="quality-report-003",
        catalog_id="catalog-001",
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v01",
        source_sha256=SHA_A,
        engine=EngineName.OPENDATALOADER_LOCAL,
        disposition=QualityDisposition.FALLBACK_REQUIRED,
        issues=[issue],
        fallback_engine=EngineName.OPENDATALOADER_HYBRID,
    )

    assert (
        report.fallback_engine
        == EngineName.OPENDATALOADER_HYBRID
    )
