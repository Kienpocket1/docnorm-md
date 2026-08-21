from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_pipeline.workers import opendataloader_local_worker as worker


def write_input_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.7\n% worker unit test\n")


def fake_convert(
    *,
    input_path: list[str],
    output_dir: str,
    format: str,
) -> None:
    assert len(input_path) == 1
    assert format == "markdown,json"

    output = Path(output_dir)
    image_directory = output / "sample_images"
    image_directory.mkdir(parents=True)

    (output / "sample.md").write_text(
        "# Quy trình\n\nNội dung kiểm thử tiếng Việt.\n",
        encoding="utf-8",
    )
    (output / "sample.json").write_text(
        json.dumps(
            {
                "file name": "sample.pdf",
                "number of pages": 2,
                "kids": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (image_directory / "imageFile1.png").write_bytes(
        b"identical-logo"
    )
    (image_directory / "imageFile2.png").write_bytes(
        b"identical-logo"
    )
    (image_directory / "imageFile3.png").write_bytes(
        b"different-image"
    )


def test_run_conversion_writes_inventory_without_document_text(
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    artifact_directory = tmp_path / "artifacts"
    write_input_pdf(input_pdf)

    result = worker.run_conversion(
        input_pdf=input_pdf,
        artifact_directory=artifact_directory,
        converter=fake_convert,
    )

    assert result["status"] == "PASS"
    assert result["output"]["page_count"] == 2
    assert result["output"]["total_file_count"] == 5
    assert result["output"]["physical_image_file_count"] == 3
    assert result["output"]["unique_image_hash_count"] == 2
    assert result["output"]["duplicate_image_hash_group_count"] == 1
    assert "Nội dung kiểm thử" not in json.dumps(
        result,
        ensure_ascii=False,
    )


def test_worker_refuses_nonempty_artifact_directory(
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    (artifact_directory / "old-run.txt").write_text(
        "old",
        encoding="utf-8",
    )
    write_input_pdf(input_pdf)

    with pytest.raises(ValueError, match="must be empty"):
        worker.run_conversion(
            input_pdf=input_pdf,
            artifact_directory=artifact_directory,
            converter=fake_convert,
        )


def test_result_manifest_must_be_outside_raw_artifacts(
    tmp_path: Path,
) -> None:
    artifact_directory = (tmp_path / "artifacts").resolve()
    result_path = artifact_directory / "worker-result.json"

    with pytest.raises(ValueError, match="must be outside"):
        worker.ensure_result_outside_artifacts(
            result_path,
            artifact_directory,
        )


def test_invalid_result_location_does_not_pollute_raw_artifacts(
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    artifact_directory = tmp_path / "artifacts"
    result_path = artifact_directory / "worker-result.json"
    write_input_pdf(input_pdf)

    exit_code = worker.main(
        [
            "--input-pdf",
            str(input_pdf),
            "--artifact-dir",
            str(artifact_directory),
            "--result-json",
            str(result_path),
        ]
    )

    assert exit_code == 2
    assert not result_path.exists()


def test_structured_json_requires_positive_page_count(
    tmp_path: Path,
) -> None:
    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    (artifact_directory / "sample.md").write_text(
        "# Heading",
        encoding="utf-8",
    )
    (artifact_directory / "sample.json").write_text(
        json.dumps({"number of pages": 0}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be positive"):
        worker.inspect_engine_output(artifact_directory)


def test_failure_writes_machine_readable_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    artifact_directory = tmp_path / "artifacts"
    result_path = tmp_path / "manifests" / "result.json"
    write_input_pdf(input_pdf)

    def fail_conversion(**_: Any) -> None:
        raise RuntimeError("synthetic conversion failure")

    monkeypatch.setattr(worker, "load_converter", lambda: fail_conversion)

    exit_code = worker.main(
        [
            "--input-pdf",
            str(input_pdf),
            "--artifact-dir",
            str(artifact_directory),
            "--result-json",
            str(result_path),
        ]
    )

    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "FAIL"
    assert payload["error"]["type"] == "RuntimeError"
    assert payload["error"]["message"] == (
        "synthetic conversion failure"
    )
