import hashlib

from rag_pipeline.contracts import (
    CatalogEntry,
    FileFormat,
    ResolutionOutcome,
    ResolverIndexEntry,
)
from rag_pipeline.stages.resolver import (
    extract_document_code,
    normalize_document_name,
    resolve_catalog_entry,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def make_catalog(
    source_name: str,
    source_sha256: str,
) -> CatalogEntry:
    catalog_suffix = hashlib.sha256(
        source_name.encode("utf-8")
    ).hexdigest()[:32]

    return CatalogEntry(
        catalog_id=f"cat-{catalog_suffix}",
        source_name=source_name,
        source_path=f"04/{source_name}",
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


def test_normalize_vietnamese_document_name() -> None:
    assert normalize_document_name(
        "Quy chế lựa chọn nhà thầu.pdf"
    ) == "quy che lua chon nha thau"


def test_extract_pilot_document_codes() -> None:
    assert extract_document_code(
        "QT.TCKT.01 - QT QL VAT TU.pdf"
    ) == "qt-tckt-01"

    assert extract_document_code(
        "QD 171100920181630437521ID627.pdf"
    ) == "id627"

    assert extract_document_code(
        "34 HDQT Quy che lua chon nha thau.pdf"
    ) == "34-hdqt"


def test_first_ingestion_proposes_new_family() -> None:
    entry = make_catalog(
        "QT.TCKT.01 - QT QL VAT TU.pdf",
        SHA_A,
    )

    decision = resolve_catalog_entry(
        entry=entry,
        references=[],
    )

    assert decision.outcome == (
        ResolutionOutcome.UNRELATED
    )
    assert decision.family_id == "qt-tckt-01"


def test_exact_sha_is_same_version() -> None:
    entry = make_catalog(
        "QT.TCKT.01 - renamed.pdf",
        SHA_A,
    )
    reference = make_reference(
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v01",
        source_name=(
            "QT.TCKT.01 - QT QL VAT TU.pdf"
        ),
        source_sha256=SHA_A,
    )

    decision = resolve_catalog_entry(
        entry=entry,
        references=[reference],
    )

    assert decision.outcome == (
        ResolutionOutcome.SAME_VERSION
    )
    assert decision.matched_version_id == (
        "qt-tckt-01-v01"
    )


def test_same_code_with_new_sha_is_new_version() -> None:
    entry = make_catalog(
        "QT.TCKT.01 - Ban cap nhat.pdf",
        SHA_B,
    )
    reference = make_reference(
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v01",
        source_name=(
            "QT.TCKT.01 - QT QL VAT TU.pdf"
        ),
        source_sha256=SHA_A,
    )

    decision = resolve_catalog_entry(
        entry=entry,
        references=[reference],
    )

    assert decision.outcome == (
        ResolutionOutcome.NEW_VERSION
    )
    assert decision.family_id == "qt-tckt-01"


def test_code_collision_is_conflict() -> None:
    entry = make_catalog(
        "QT.TCKT.01 - Ban moi.pdf",
        SHA_C,
    )
    references = [
        make_reference(
            family_id="qt-tckt-01",
            version_id="family-a-v01",
            source_name="QT.TCKT.01 - A.pdf",
            source_sha256=SHA_A,
        ),
        make_reference(
            family_id="legacy-qt-tckt-01",
            version_id="family-b-v01",
            source_name="QT.TCKT.01 - B.pdf",
            source_sha256=SHA_B,
        ),
    ]

    decision = resolve_catalog_entry(
        entry=entry,
        references=references,
    )

    assert decision.outcome == (
        ResolutionOutcome.CONFLICT
    )
    assert decision.requires_human_review is True
    assert set(decision.candidate_family_ids) == {
        "qt-tckt-01",
        "legacy-qt-tckt-01",
    }


def test_fuzzy_match_requires_review() -> None:
    entry = make_catalog(
        "Quy che mua sam noi bo ban sua doi.pdf",
        SHA_B,
    )
    reference = make_reference(
        family_id="quy-che-mua-sam",
        version_id="quy-che-mua-sam-v01",
        source_name="Quy che mua sam noi bo.pdf",
        source_sha256=SHA_A,
    )

    decision = resolve_catalog_entry(
        entry=entry,
        references=[reference],
    )

    assert decision.outcome == (
        ResolutionOutcome.CONFLICT
    )
    assert decision.requires_human_review is True


def test_unmatched_name_stays_unrelated() -> None:
    entry = make_catalog(
        "Huong dan bao duong may bay.pdf",
        SHA_B,
    )
    reference = make_reference(
        family_id="quy-che-mua-sam",
        version_id="quy-che-mua-sam-v01",
        source_name="Quy che mua sam noi bo.pdf",
        source_sha256=SHA_A,
    )

    decision = resolve_catalog_entry(
        entry=entry,
        references=[reference],
    )

    assert decision.outcome == (
        ResolutionOutcome.UNRELATED
    )
    assert decision.family_id is not None