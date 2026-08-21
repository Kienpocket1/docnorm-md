import hashlib

import pytest

from rag_pipeline.contracts import FileFormat
from rag_pipeline.stages.catalog import (
    CatalogError,
    catalog_file,
    compute_sha256,
    load_catalog_manifest,
    scan_data_root,
    write_catalog_manifest,
)


def test_compute_sha256(tmp_path) -> None:
    source = tmp_path / "document.pdf"
    content = b"internal-rag-test"
    source.write_bytes(content)

    assert compute_sha256(source) == (
        hashlib.sha256(content).hexdigest()
    )


def test_catalog_id_is_stable(tmp_path) -> None:
    data_root = tmp_path / "data_local"
    source = data_root / "04" / "QT.TCKT.01.pdf"

    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake-pdf-content")

    first = catalog_file(source, data_root)
    second = catalog_file(source, data_root)

    assert first.catalog_id == second.catalog_id
    assert first.file_format == FileFormat.PDF
    assert first.source_path == "04/QT.TCKT.01.pdf"


def test_identical_files_are_not_collapsed(tmp_path) -> None:
    data_root = tmp_path / "data_local"
    first_path = data_root / "folder-a" / "document.pdf"
    second_path = data_root / "folder-b" / "document.pdf"

    first_path.parent.mkdir(parents=True)
    second_path.parent.mkdir(parents=True)

    content = b"same-document-content"
    first_path.write_bytes(content)
    second_path.write_bytes(content)

    entries = scan_data_root(data_root)

    assert len(entries) == 2
    assert (
        entries[0].source_sha256
        == entries[1].source_sha256
    )
    assert entries[0].catalog_id != entries[1].catalog_id


def test_scan_ignores_temporary_and_unsupported_files(
    tmp_path,
) -> None:
    data_root = tmp_path / "data_local"
    data_root.mkdir()

    (data_root / "valid.pdf").write_bytes(b"valid")
    (data_root / "~$temporary.docx").write_bytes(b"temp")
    (data_root / "notes.txt").write_text(
        "unsupported",
        encoding="utf-8",
    )

    entries = scan_data_root(data_root)

    assert len(entries) == 1
    assert entries[0].source_name == "valid.pdf"


def test_catalog_rejects_file_outside_data_root(
    tmp_path,
) -> None:
    data_root = tmp_path / "data_local"
    data_root.mkdir()

    outside_file = tmp_path / "outside.pdf"
    outside_file.write_bytes(b"outside")

    with pytest.raises(
        CatalogError,
        match="outside data_root",
    ):
        catalog_file(outside_file, data_root)


def test_catalog_manifest_round_trip(tmp_path) -> None:
    data_root = tmp_path / "data_local"
    source = data_root / "document.docx"

    data_root.mkdir()
    source.write_bytes(b"fake-docx-content")

    original_entries = scan_data_root(data_root)
    manifest_path = tmp_path / "catalog.jsonl"

    write_catalog_manifest(
        original_entries,
        manifest_path,
    )

    loaded_entries = load_catalog_manifest(
        manifest_path
    )

    assert loaded_entries == original_entries
def test_scan_scope_preserves_data_root_relative_path(
    tmp_path,
) -> None:
    data_root = tmp_path / "data_local"
    pilot_root = (
        data_root / "04. Quy trinh quan ly vat tu"
    )
    unrelated_root = data_root / "05. Other"

    pilot_root.mkdir(parents=True)
    unrelated_root.mkdir(parents=True)

    (pilot_root / "pilot.pdf").write_bytes(
        b"pilot-document"
    )
    (unrelated_root / "other.pdf").write_bytes(
        b"other-document"
    )

    entries = scan_data_root(
        data_root=data_root,
        scan_root=pilot_root,
    )

    assert len(entries) == 1
    assert entries[0].source_path == (
        "04. Quy trinh quan ly vat tu/pilot.pdf"
    )