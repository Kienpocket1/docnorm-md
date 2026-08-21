from typer.testing import CliRunner

from rag_pipeline.cli import app
from rag_pipeline.stages.catalog import (
    load_catalog_manifest,
)


runner = CliRunner()


def test_catalog_cli_scans_only_requested_folder(
    tmp_path,
) -> None:
    data_root = tmp_path / "data_local"
    pilot_root = (
        data_root / "04. Quy trinh quan ly vat tu"
    )
    other_root = data_root / "05. Other"

    pilot_root.mkdir(parents=True)
    other_root.mkdir(parents=True)

    (pilot_root / "pilot.pdf").write_bytes(
        b"pilot-pdf"
    )
    (other_root / "other.pdf").write_bytes(
        b"other-pdf"
    )

    manifest = tmp_path / "catalog.jsonl"

    result = runner.invoke(
        app,
        [
            "catalog",
            "--data-root",
            str(data_root),
            "--scan-root",
            str(pilot_root),
            "--output",
            str(manifest),
        ],
    )

    assert result.exit_code == 0, result.output

    entries = load_catalog_manifest(manifest)

    assert len(entries) == 1
    assert entries[0].source_path == (
        "04. Quy trinh quan ly vat tu/pilot.pdf"
    )