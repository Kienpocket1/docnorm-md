from typer.testing import CliRunner

from rag_pipeline.cli import app
from rag_pipeline.contracts import (
    CatalogEntry,
    FileFormat,
    ResolutionOutcome,
    ResolverIndexEntry,
)
from rag_pipeline.stages.catalog import (
    write_catalog_manifest,
)
from rag_pipeline.stages.resolver import (
    extract_document_code,
    load_resolution_manifest,
    load_resolver_quarantine_manifest,
    normalize_document_name,
    write_resolver_index_manifest,
)


runner = CliRunner()

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def make_catalog(
    catalog_id: str,
    source_name: str,
    source_sha256: str,
) -> CatalogEntry:
    return CatalogEntry(
        catalog_id=catalog_id,
        source_name=source_name,
        source_path=(
            "04. Quy trinh quan ly vat tu/"
            f"{source_name}"
        ),
        file_format=FileFormat.PDF,
        source_sha256=source_sha256,
        size_bytes=100,
        mime_type="application/pdf",
    )


def make_reference(
    *,
    family_id: str,
    version_id: str,
    source_name: str,
    source_sha256: str,
) -> ResolverIndexEntry:
    return ResolverIndexEntry(
        family_id=family_id,
        version_id=version_id,
        source_sha256=source_sha256,
        source_name=source_name,
        normalized_name=normalize_document_name(
            source_name
        ),
        document_code=extract_document_code(
            source_name
        ),
    )


def test_resolve_first_ingestion_cli(
    tmp_path,
) -> None:
    catalog_path = tmp_path / "catalog.jsonl"
    output_dir = tmp_path / "resolver"

    entries = [
        make_catalog(
            "cat-001",
            "34 HDQT Quy che lua chon nha thau.pdf",
            SHA_A,
        ),
        make_catalog(
            "cat-002",
            "QD 171100920181630437521ID627.pdf",
            SHA_B,
        ),
        make_catalog(
            "cat-003",
            "QT.TCKT.01 - QT QL VAT TU.pdf",
            SHA_C,
        ),
    ]

    write_catalog_manifest(
        entries,
        catalog_path,
    )

    result = runner.invoke(
        app,
        [
            "resolve",
            "--catalog-manifest",
            str(catalog_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    decisions = load_resolution_manifest(
        output_dir / "resolver_decisions.jsonl"
    )
    continuing = load_resolution_manifest(
        output_dir / "continue_to_version.jsonl"
    )
    quarantined = (
        load_resolver_quarantine_manifest(
            output_dir / "resolver_quarantine.jsonl"
        )
    )
    rejected = load_resolution_manifest(
        output_dir / "terminal_rejected.jsonl"
    )

    assert len(decisions) == 3
    assert len(continuing) == 3
    assert not quarantined
    assert not rejected

    assert {
        decision.outcome
        for decision in decisions
    } == {
        ResolutionOutcome.UNRELATED
    }

    assert {
        decision.family_id
        for decision in decisions
    } == {
        "34-hdqt",
        "id627",
        "qt-tckt-01",
    }


def test_resolve_conflict_cli_enters_quarantine(
    tmp_path,
) -> None:
    catalog_path = tmp_path / "catalog.jsonl"
    index_path = tmp_path / "resolver_index.jsonl"
    output_dir = tmp_path / "resolver"

    entry = make_catalog(
        "cat-004",
        "QT.TCKT.01 - Ban moi.pdf",
        SHA_C,
    )

    references = [
        make_reference(
            family_id="qt-tckt-01",
            version_id="qt-tckt-01-v01",
            source_name="QT.TCKT.01 - Ban A.pdf",
            source_sha256=SHA_A,
        ),
        make_reference(
            family_id="legacy-qt-tckt-01",
            version_id="legacy-v01",
            source_name="QT.TCKT.01 - Ban B.pdf",
            source_sha256=SHA_B,
        ),
    ]

    write_catalog_manifest(
        [entry],
        catalog_path,
    )
    write_resolver_index_manifest(
        references,
        index_path,
    )

    result = runner.invoke(
        app,
        [
            "resolve",
            "--catalog-manifest",
            str(catalog_path),
            "--resolver-index",
            str(index_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    decisions = load_resolution_manifest(
        output_dir / "resolver_decisions.jsonl"
    )
    continuing = load_resolution_manifest(
        output_dir / "continue_to_version.jsonl"
    )
    quarantined = (
        load_resolver_quarantine_manifest(
            output_dir / "resolver_quarantine.jsonl"
        )
    )

    assert len(decisions) == 1
    assert decisions[0].outcome == (
        ResolutionOutcome.CONFLICT
    )
    assert not continuing
    assert len(quarantined) == 1