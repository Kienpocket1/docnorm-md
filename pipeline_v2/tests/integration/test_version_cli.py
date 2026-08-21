from typer.testing import CliRunner

from rag_pipeline.cli import app
from rag_pipeline.contracts import (
    CatalogEntry,
    FileFormat,
    PublicationState,
    ResolutionDecision,
    ResolutionOutcome,
    VersionAction,
)
from rag_pipeline.stages.catalog import (
    write_catalog_manifest,
)
from rag_pipeline.stages.resolver import (
    write_resolution_manifest,
)
from rag_pipeline.stages.version_manager import (
    load_family_registry_manifest,
    load_resolver_index_candidate_manifest,
    load_version_assignments_manifest,
    load_version_registry_manifest,
)


runner = CliRunner()
SHA_A = "a" * 64


def prepare_input(tmp_path):
    catalog_path = tmp_path / "catalog.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"

    entry = CatalogEntry(
        catalog_id="cat-001",
        source_name="QT.TCKT.01 - QT QL VAT TU.pdf",
        source_path=(
            "04. Quy trinh quan ly vat tu/"
            "QT.TCKT.01 - QT QL VAT TU.pdf"
        ),
        file_format=FileFormat.PDF,
        source_sha256=SHA_A,
        size_bytes=100,
        mime_type="application/pdf",
    )

    decision = ResolutionDecision(
        decision_id="res-001",
        catalog_id="cat-001",
        outcome=ResolutionOutcome.UNRELATED,
        family_id="qt-tckt-01",
        confidence=1.0,
        reason_code="no_known_family_match",
    )

    write_catalog_manifest(
        [entry],
        catalog_path,
    )
    write_resolution_manifest(
        [decision],
        decisions_path,
    )

    return catalog_path, decisions_path


def test_version_cli_creates_candidate_state(
    tmp_path,
) -> None:
    catalog_path, decisions_path = prepare_input(
        tmp_path
    )
    output_dir = tmp_path / "version-state"

    result = runner.invoke(
        app,
        [
            "version",
            "--catalog-manifest",
            str(catalog_path),
            "--decisions-manifest",
            str(decisions_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    families = load_family_registry_manifest(
        output_dir
        / "family_registry_candidate.jsonl"
    )
    versions = load_version_registry_manifest(
        output_dir
        / "version_registry_candidate.jsonl"
    )
    assignments = (
        load_version_assignments_manifest(
            output_dir / "version_assignments.jsonl"
        )
    )
    resolver_index = (
        load_resolver_index_candidate_manifest(
            output_dir
            / "resolver_index_candidate.jsonl"
        )
    )

    assert len(families) == 1
    assert len(versions) == 1
    assert len(assignments) == 1
    assert len(resolver_index) == 1

    assert (
        versions[0].publication_state
        == PublicationState.DRAFT
    )
    assert (
        assignments[0].action
        == VersionAction.CREATE_FAMILY_AND_VERSION
    )

    # Candidate chưa được ghi thành canonical state.
    assert not (
        output_dir / "resolver_index.jsonl"
    ).exists()


def test_version_cli_rerun_is_idempotent(
    tmp_path,
) -> None:
    catalog_path, decisions_path = prepare_input(
        tmp_path
    )

    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first_result = runner.invoke(
        app,
        [
            "version",
            "--catalog-manifest",
            str(catalog_path),
            "--decisions-manifest",
            str(decisions_path),
            "--output-dir",
            str(first_output),
        ],
    )

    assert first_result.exit_code == 0

    second_result = runner.invoke(
        app,
        [
            "version",
            "--catalog-manifest",
            str(catalog_path),
            "--decisions-manifest",
            str(decisions_path),
            "--family-registry",
            str(
                first_output
                / "family_registry_candidate.jsonl"
            ),
            "--version-registry",
            str(
                first_output
                / "version_registry_candidate.jsonl"
            ),
            "--resolver-index",
            str(
                first_output
                / "resolver_index_candidate.jsonl"
            ),
            "--output-dir",
            str(second_output),
        ],
    )

    assert second_result.exit_code == 0

    assignments = (
        load_version_assignments_manifest(
            second_output
            / "version_assignments.jsonl"
        )
    )
    created_versions = (
        load_version_registry_manifest(
            second_output
            / "created_versions.jsonl"
        )
    )

    assert len(assignments) == 1
    assert (
        assignments[0].action
        == VersionAction.REUSE_VERSION
    )
    assert not created_versions