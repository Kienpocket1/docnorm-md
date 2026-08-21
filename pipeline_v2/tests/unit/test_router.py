import pytest

from rag_pipeline.contracts import (
    CatalogEntry,
    EngineName,
    FailureDomain,
    FileFormat,
    QuarantineType,
    ResumeFrom,
    RouteType,
    VersionAction,
    VersionAssignment,
)
from rag_pipeline.stages.router import (
    RouterError,
    route_assignments,
    route_catalog_entry,
)


SHA_A = "a" * 64


def make_catalog(
    file_format: FileFormat,
    source_name: str,
) -> CatalogEntry:
    return CatalogEntry(
        catalog_id="cat-001",
        source_name=source_name,
        source_path=f"04/{source_name}",
        file_format=file_format,
        source_sha256=SHA_A,
        size_bytes=100,
    )


def make_assignment(
    action: VersionAction = (
        VersionAction.CREATE_FAMILY_AND_VERSION
    ),
    source_sha256: str = SHA_A,
) -> VersionAssignment:
    return VersionAssignment(
        assignment_id="va-001",
        decision_id="res-001",
        catalog_id="cat-001",
        family_id="family-001",
        version_id=(
            "family-001-v001-aaaaaaaaaaaa"
        ),
        version_number=1,
        version_label="v001",
        source_sha256=source_sha256,
        action=action,
    )


def test_pdf_routes_to_opendataloader_local() -> None:
    route = route_catalog_entry(
        make_catalog(FileFormat.PDF, "document.pdf"),
        make_assignment(),
    )

    assert (
        route.route_type
        == RouteType.PDF_OPENDATALOADER
    )
    assert (
        route.primary_engine
        == EngineName.OPENDATALOADER_LOCAL
    )
    assert route.fallback_engines == [
        EngineName.OPENDATALOADER_HYBRID,
        EngineName.MINERU,
    ]


def test_docx_routes_to_native_extractor() -> None:
    route = route_catalog_entry(
        make_catalog(FileFormat.DOCX, "document.docx"),
        make_assignment(),
    )

    assert route.route_type == RouteType.DOCX_NATIVE
    assert (
        route.primary_engine
        == EngineName.DOCX_NATIVE
    )


def test_doc_routes_to_libreoffice() -> None:
    route = route_catalog_entry(
        make_catalog(FileFormat.DOC, "document.doc"),
        make_assignment(),
    )

    assert route.route_type == RouteType.DOC_CONVERT
    assert (
        route.primary_engine
        == EngineName.LIBREOFFICE
    )


def test_image_routes_to_qwen() -> None:
    route = route_catalog_entry(
        make_catalog(FileFormat.IMAGE, "image.png"),
        make_assignment(),
    )

    assert route.route_type == RouteType.IMAGE_VISUAL
    assert (
        route.primary_engine
        == EngineName.QWEN3_VL
    )


def test_reused_version_skips_extraction() -> None:
    result = route_assignments(
        catalog_entries=[
            make_catalog(
                FileFormat.PDF,
                "document.pdf",
            )
        ],
        assignments=[
            make_assignment(
                action=VersionAction.REUSE_VERSION
            )
        ],
    )

    assert not result.routes_to_extract
    assert len(result.reused_assignments) == 1


def test_unknown_format_enters_extraction_quarantine() -> None:
    result = route_assignments(
        catalog_entries=[
            make_catalog(
                FileFormat.UNKNOWN,
                "document.xyz",
            )
        ],
        assignments=[make_assignment()],
    )

    assert not result.routes_to_extract
    assert len(result.unsupported_routes) == 1
    assert len(result.routing_quarantine) == 1

    record = result.routing_quarantine[0]

    assert record.queue == QuarantineType.EXTRACTION
    assert (
        record.failure_domain
        == FailureDomain.ROUTING
    )
    assert record.resume_from == ResumeFrom.ROUTE


def test_router_rejects_sha_mismatch() -> None:
    with pytest.raises(
        RouterError,
        match="SHA256 mismatch",
    ):
        route_catalog_entry(
            make_catalog(
                FileFormat.PDF,
                "document.pdf",
            ),
            make_assignment(
                source_sha256="b" * 64
            ),
        )