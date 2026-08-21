import pytest

from rag_pipeline.contracts import (
    CatalogEntry,
    FileFormat,
    ResolutionDecision,
    ResolutionOutcome,
    ResolverIndexEntry,
    VersionAction,
    VersionAssignment,
)
from rag_pipeline.stages.version_manager import (
    VersionManagerError,
    assign_versions,
    build_resolver_index_candidate,
    load_family_registry_manifest,
    load_resolver_index_candidate_manifest,
    load_version_assignments_manifest,
    load_version_registry_manifest,
    write_family_registry_manifest,
    write_resolver_index_candidate_manifest,
    write_version_assignments_manifest,
    write_version_registry_manifest,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def make_first_batch():
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

    result = assign_versions(
        catalog_entries=[entry],
        decisions=[decision],
    )

    return entry, result


def test_build_resolver_index_candidate() -> None:
    entry, result = make_first_batch()

    candidate = build_resolver_index_candidate(
        catalog_entries=[entry],
        assignments=result.assignments,
    )

    assert len(candidate) == 1

    reference = candidate[0]

    assert reference.family_id == "qt-tckt-01"
    assert (
        reference.version_id
        == result.versions[0].version_id
    )
    assert reference.document_code == "qt-tckt-01"
    assert reference.source_sha256 == SHA_A


def test_resolver_index_merge_is_idempotent() -> None:
    entry, result = make_first_batch()

    first_candidate = (
        build_resolver_index_candidate(
            catalog_entries=[entry],
            assignments=result.assignments,
        )
    )

    second_candidate = (
        build_resolver_index_candidate(
            catalog_entries=[entry],
            assignments=result.assignments,
            existing_index=first_candidate,
        )
    )

    assert second_candidate == first_candidate


def test_resolver_index_rejects_sha_collision() -> None:
    entry, _ = make_first_batch()

    existing_reference = ResolverIndexEntry(
        family_id="family-a",
        version_id="family-a-v001-aaaaaaaaaaaa",
        source_sha256=SHA_A,
        source_name="Document A.pdf",
        normalized_name="document a",
    )

    conflicting_assignment = VersionAssignment(
        assignment_id="va-conflict",
        decision_id="res-conflict",
        catalog_id=entry.catalog_id,
        family_id="family-b",
        version_id="family-b-v001-aaaaaaaaaaaa",
        version_number=1,
        version_label="v001",
        source_sha256=SHA_A,
        action=VersionAction.CREATE_FAMILY_AND_VERSION,
    )

    with pytest.raises(
        VersionManagerError,
        match="multiple version_id",
    ):
        build_resolver_index_candidate(
            catalog_entries=[entry],
            assignments=[conflicting_assignment],
            existing_index=[existing_reference],
        )


def test_version_state_artifacts_round_trip(
    tmp_path,
) -> None:
    entry, result = make_first_batch()

    resolver_index = (
        build_resolver_index_candidate(
            catalog_entries=[entry],
            assignments=result.assignments,
        )
    )

    family_path = tmp_path / "families.jsonl"
    version_path = tmp_path / "versions.jsonl"
    assignment_path = tmp_path / "assignments.jsonl"
    index_path = tmp_path / "resolver-index.jsonl"

    write_family_registry_manifest(
        result.families,
        family_path,
    )
    write_version_registry_manifest(
        result.versions,
        version_path,
    )
    write_version_assignments_manifest(
        result.assignments,
        assignment_path,
    )
    write_resolver_index_candidate_manifest(
        resolver_index,
        index_path,
    )

    assert load_family_registry_manifest(
        family_path
    ) == result.families

    assert load_version_registry_manifest(
        version_path
    ) == result.versions

    assert load_version_assignments_manifest(
        assignment_path
    ) == result.assignments

    assert load_resolver_index_candidate_manifest(
        index_path
    ) == resolver_index