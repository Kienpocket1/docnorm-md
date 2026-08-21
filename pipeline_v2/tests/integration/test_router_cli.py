from typer.testing import CliRunner

from rag_pipeline.cli import app
from rag_pipeline.contracts import (
    CatalogEntry,
    EngineName,
    FileFormat,
    RouteType,
    VersionAction,
    VersionAssignment,
)
from rag_pipeline.stages.catalog import (
    write_catalog_manifest,
)
from rag_pipeline.stages.router import (
    load_route_decisions_manifest,
    load_routing_quarantine_manifest,
)
from rag_pipeline.stages.version_manager import (
    load_version_assignments_manifest,
    write_version_assignments_manifest,
)


runner = CliRunner()


def make_catalog(
    *,
    catalog_id: str,
    source_name: str,
    source_sha256: str,
    file_format: FileFormat,
) -> CatalogEntry:
    return CatalogEntry(
        catalog_id=catalog_id,
        source_name=source_name,
        source_path=f"04/{source_name}",
        file_format=file_format,
        source_sha256=source_sha256,
        size_bytes=100,
    )


def make_assignment(
    *,
    catalog_id: str,
    family_id: str,
    source_sha256: str,
    action: VersionAction,
) -> VersionAssignment:
    return VersionAssignment(
        assignment_id=f"va-{catalog_id}",
        decision_id=f"res-{catalog_id}",
        catalog_id=catalog_id,
        family_id=family_id,
        version_id=(
            f"{family_id}-v001-"
            f"{source_sha256[:12]}"
        ),
        version_number=1,
        version_label="v001",
        source_sha256=source_sha256,
        action=action,
    )


def test_router_cli_routes_three_pdfs(
    tmp_path,
) -> None:
    catalog_path = tmp_path / "catalog.jsonl"
    assignment_path = tmp_path / "assignments.jsonl"
    output_dir = tmp_path / "router"

    hashes = [
        "a" * 64,
        "b" * 64,
        "c" * 64,
    ]

    catalogs = [
        make_catalog(
            catalog_id=f"cat-{index}",
            source_name=f"document-{index}.pdf",
            source_sha256=source_hash,
            file_format=FileFormat.PDF,
        )
        for index, source_hash in enumerate(
            hashes,
            start=1,
        )
    ]

    assignments = [
        make_assignment(
            catalog_id=catalog.catalog_id,
            family_id=f"family-{index}",
            source_sha256=catalog.source_sha256,
            action=(
                VersionAction
                .CREATE_FAMILY_AND_VERSION
            ),
        )
        for index, catalog in enumerate(
            catalogs,
            start=1,
        )
    ]

    write_catalog_manifest(
        catalogs,
        catalog_path,
    )
    write_version_assignments_manifest(
        assignments,
        assignment_path,
    )

    result = runner.invoke(
        app,
        [
            "route",
            "--catalog-manifest",
            str(catalog_path),
            "--assignments-manifest",
            str(assignment_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    routes = load_route_decisions_manifest(
        output_dir / "routes_to_extract.jsonl"
    )
    quarantine = load_routing_quarantine_manifest(
        output_dir / "routing_quarantine.jsonl"
    )

    assert len(routes) == 3
    assert not quarantine

    assert all(
        route.route_type
        == RouteType.PDF_OPENDATALOADER
        for route in routes
    )

    assert all(
        route.primary_engine
        == EngineName.OPENDATALOADER_LOCAL
        for route in routes
    )


def test_router_cli_separates_reuse_and_unsupported(
    tmp_path,
) -> None:
    catalog_path = tmp_path / "catalog.jsonl"
    assignment_path = tmp_path / "assignments.jsonl"
    output_dir = tmp_path / "router"

    pdf_hash = "a" * 64
    unknown_hash = "b" * 64

    catalogs = [
        make_catalog(
            catalog_id="cat-pdf",
            source_name="document.pdf",
            source_sha256=pdf_hash,
            file_format=FileFormat.PDF,
        ),
        make_catalog(
            catalog_id="cat-unknown",
            source_name="document.xyz",
            source_sha256=unknown_hash,
            file_format=FileFormat.UNKNOWN,
        ),
    ]

    assignments = [
        make_assignment(
            catalog_id="cat-pdf",
            family_id="family-pdf",
            source_sha256=pdf_hash,
            action=VersionAction.REUSE_VERSION,
        ),
        make_assignment(
            catalog_id="cat-unknown",
            family_id="family-unknown",
            source_sha256=unknown_hash,
            action=(
                VersionAction
                .CREATE_FAMILY_AND_VERSION
            ),
        ),
    ]

    write_catalog_manifest(
        catalogs,
        catalog_path,
    )
    write_version_assignments_manifest(
        assignments,
        assignment_path,
    )

    result = runner.invoke(
        app,
        [
            "route",
            "--catalog-manifest",
            str(catalog_path),
            "--assignments-manifest",
            str(assignment_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    extract_routes = load_route_decisions_manifest(
        output_dir / "routes_to_extract.jsonl"
    )
    unsupported = load_route_decisions_manifest(
        output_dir / "unsupported_routes.jsonl"
    )
    reused = load_version_assignments_manifest(
        output_dir / "reused_assignments.jsonl"
    )
    quarantine = load_routing_quarantine_manifest(
        output_dir / "routing_quarantine.jsonl"
    )

    assert not extract_routes
    assert len(unsupported) == 1
    assert len(reused) == 1
    assert len(quarantine) == 1