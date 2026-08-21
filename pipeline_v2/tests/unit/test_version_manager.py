import pytest

from rag_pipeline.contracts import (
    CatalogEntry,
    FamilyRecord,
    FileFormat,
    PublicationState,
    ResolutionDecision,
    ResolutionOutcome,
    VersionAction,
    VersionRecord,
)
from rag_pipeline.stages.version_manager import (
    VersionManagerError,
    assign_versions,
    create_version_id,
)


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
        source_path=f"04/{source_name}",
        file_format=FileFormat.PDF,
        source_sha256=source_sha256,
        size_bytes=100,
        mime_type="application/pdf",
    )


def make_unrelated(
    decision_id: str,
    catalog_id: str,
    family_id: str,
) -> ResolutionDecision:
    return ResolutionDecision(
        decision_id=decision_id,
        catalog_id=catalog_id,
        outcome=ResolutionOutcome.UNRELATED,
        family_id=family_id,
        confidence=1.0,
        reason_code="no_known_family_match",
    )


def make_family(
    family_id: str = "qt-tckt-01",
) -> FamilyRecord:
    return FamilyRecord(
        family_id=family_id,
        canonical_name="QT.TCKT.01",
        document_code="qt-tckt-01",
        aliases=["QT.TCKT.01.pdf"],
    )


def make_version(
    *,
    family_id: str = "qt-tckt-01",
    catalog_id: str = "cat-old",
    source_sha256: str = SHA_A,
    version_number: int = 1,
    state: PublicationState = PublicationState.ACTIVE,
) -> VersionRecord:
    return VersionRecord(
        family_id=family_id,
        version_id=create_version_id(
            family_id,
            version_number,
            source_sha256,
        ),
        catalog_id=catalog_id,
        source_sha256=source_sha256,
        version_number=version_number,
        version_label=f"v{version_number:03d}",
        publication_state=state,
    )


def test_first_ingestion_creates_three_families() -> None:
    catalogs = [
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

    decisions = [
        make_unrelated(
            "res-001",
            "cat-001",
            "34-hdqt",
        ),
        make_unrelated(
            "res-002",
            "cat-002",
            "id627",
        ),
        make_unrelated(
            "res-003",
            "cat-003",
            "qt-tckt-01",
        ),
    ]

    result = assign_versions(
        catalogs,
        decisions,
    )

    assert len(result.created_families) == 3
    assert len(result.created_versions) == 3
    assert len(result.assignments) == 3

    assert {
        assignment.action
        for assignment in result.assignments
    } == {
        VersionAction.CREATE_FAMILY_AND_VERSION
    }

    assert {
        version.version_label
        for version in result.created_versions
    } == {"v001"}

    assert all(
        version.publication_state
        == PublicationState.DRAFT
        for version in result.created_versions
    )


def test_new_version_is_draft_and_supersedes_active() -> None:
    family = make_family()
    active_version = make_version()

    catalog = make_catalog(
        "cat-new",
        "QT.TCKT.01 - Ban moi.pdf",
        SHA_B,
    )

    decision = ResolutionDecision(
        decision_id="res-new",
        catalog_id="cat-new",
        outcome=ResolutionOutcome.NEW_VERSION,
        family_id="qt-tckt-01",
        confidence=0.98,
        reason_code="same_document_code",
    )

    result = assign_versions(
        [catalog],
        [decision],
        existing_families=[family],
        existing_versions=[active_version],
    )

    new_version = result.created_versions[0]

    assert new_version.version_number == 2
    assert new_version.version_label == "v002"
    assert (
        new_version.publication_state
        == PublicationState.DRAFT
    )
    assert (
        new_version.supersedes_version_id
        == active_version.version_id
    )

    old_version = next(
        version
        for version in result.versions
        if version.version_id
        == active_version.version_id
    )

    assert (
        old_version.publication_state
        == PublicationState.ACTIVE
    )


def test_same_version_reuses_existing_version() -> None:
    family = make_family()
    existing_version = make_version()

    catalog = make_catalog(
        "cat-copy",
        "QT.TCKT.01 - Copy.pdf",
        SHA_A,
    )

    decision = ResolutionDecision(
        decision_id="res-same",
        catalog_id="cat-copy",
        outcome=ResolutionOutcome.SAME_VERSION,
        family_id="qt-tckt-01",
        matched_version_id=(
            existing_version.version_id
        ),
        confidence=1.0,
        reason_code="exact_sha256_match",
    )

    result = assign_versions(
        [catalog],
        [decision],
        existing_families=[family],
        existing_versions=[existing_version],
    )

    assert not result.created_versions
    assert (
        result.assignments[0].action
        == VersionAction.REUSE_VERSION
    )


def test_version_assignment_is_idempotent() -> None:
    catalog = make_catalog(
        "cat-001",
        "QT.TCKT.01.pdf",
        SHA_A,
    )
    decision = make_unrelated(
        "res-001",
        "cat-001",
        "qt-tckt-01",
    )

    first = assign_versions(
        [catalog],
        [decision],
    )

    second = assign_versions(
        [catalog],
        [decision],
        existing_families=first.families,
        existing_versions=first.versions,
    )

    assert not second.created_versions
    assert (
        second.assignments[0].action
        == VersionAction.REUSE_VERSION
    )


def test_conflict_cannot_enter_version_manager() -> None:
    catalog = make_catalog(
        "cat-001",
        "Document.pdf",
        SHA_A,
    )

    conflict = ResolutionDecision(
        decision_id="res-conflict",
        catalog_id="cat-001",
        outcome=ResolutionOutcome.CONFLICT,
        candidate_family_ids=["family-a"],
        confidence=0.9,
        reason_code="requires_review",
        requires_human_review=True,
    )

    with pytest.raises(
        VersionManagerError,
        match="cannot enter",
    ):
        assign_versions(
            [catalog],
            [conflict],
        )


def test_multiple_active_versions_are_rejected() -> None:
    family = make_family()

    first = make_version(
        source_sha256=SHA_A,
        version_number=1,
    )
    second = make_version(
        source_sha256=SHA_B,
        version_number=2,
    )

    with pytest.raises(
        VersionManagerError,
        match="Multiple ACTIVE",
    ):
        assign_versions(
            [],
            [],
            existing_families=[family],
            existing_versions=[first, second],
        )


def test_same_sha_cannot_cross_families() -> None:
    existing_family = make_family("family-a")
    existing_version = make_version(
        family_id="family-a",
        source_sha256=SHA_A,
    )

    new_catalog = make_catalog(
        "cat-new",
        "Different document.pdf",
        SHA_A,
    )
    new_decision = make_unrelated(
        "res-new",
        "cat-new",
        "family-b",
    )

    with pytest.raises(
        VersionManagerError,
        match="different families",
    ):
        assign_versions(
            [new_catalog],
            [new_decision],
            existing_families=[existing_family],
            existing_versions=[existing_version],
        )