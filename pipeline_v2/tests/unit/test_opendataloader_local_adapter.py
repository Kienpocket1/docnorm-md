from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_pipeline.extractors import (
    OpenDataLoaderIntegrityError,
    OpenDataLoaderLocalConfig,
    OpenDataLoaderLocalExtractor,
    OpenDataLoaderRunExistsError,
    OpenDataLoaderTimeoutError,
    OpenDataLoaderWorkerError,
)
from rag_pipeline.extractors import opendataloader_local as adapter_module
from rag_pipeline.workers import opendataloader_local_worker


FAKE_OPENDATALOADER_MODULE = '''
import json
import os
import time
from pathlib import Path


def convert(*, input_path, output_dir, format):
    if os.environ.get("FAKE_ODL_SLEEP"):
        time.sleep(float(os.environ["FAKE_ODL_SLEEP"]))

    if os.environ.get("FAKE_ODL_FAIL"):
        raise RuntimeError("synthetic OpenDataLoader failure")

    assert len(input_path) == 1
    assert format == "markdown,json"

    output = Path(output_dir)
    image_directory = output / "sample_images"
    image_directory.mkdir(parents=True)

    (output / "sample.md").write_text(
        "# Heading\\n\\nNội dung kiểm thử.\\n",
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
    (image_directory / "imageFile1.png").write_bytes(b"same")
    (image_directory / "imageFile2.png").write_bytes(b"same")
'''


def write_input_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.7\n% adapter unit test\n")


def build_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    timeout_seconds: float = 5.0,
) -> OpenDataLoaderLocalExtractor:
    fake_module_directory = tmp_path / "fake_module"
    fake_module_directory.mkdir()
    (fake_module_directory / "opendataloader_pdf.py").write_text(
        FAKE_OPENDATALOADER_MODULE,
        encoding="utf-8",
    )

    existing_python_path = os.environ.get("PYTHONPATH")
    python_path_parts = [str(fake_module_directory)]

    if existing_python_path:
        python_path_parts.append(existing_python_path)

    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(python_path_parts),
    )

    return OpenDataLoaderLocalExtractor(
        OpenDataLoaderLocalConfig(
            python_executable=Path(sys.executable),
            worker_script=Path(
                opendataloader_local_worker.__file__
            ),
            output_root=tmp_path / "runs",
            timeout_seconds=timeout_seconds,
        )
    )


def test_adapter_runs_isolated_worker_and_validates_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)

    run = extractor.extract(input_pdf, run_id="successful-run")

    assert run.manifest.status == "PASS"
    assert run.manifest.output.page_count == 2
    assert run.manifest.output.total_file_count == 4
    assert run.manifest.output.physical_image_file_count == 2
    assert run.manifest.output.unique_image_hash_count == 1
    assert run.result_manifest_path.is_file()
    assert run.stdout_log_path.is_file()
    assert run.stderr_log_path.is_file()


def test_adapter_rejects_duplicate_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)

    extractor.extract(input_pdf, run_id="stable-run")

    with pytest.raises(OpenDataLoaderRunExistsError):
        extractor.extract(input_pdf, run_id="stable-run")


def test_adapter_surfaces_worker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_ODL_FAIL", "1")

    with pytest.raises(
        OpenDataLoaderWorkerError,
        match="synthetic OpenDataLoader failure",
    ):
        extractor.extract(input_pdf, run_id="failed-run")


def test_adapter_kills_timed_out_worker_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)
    extractor = build_extractor(
        tmp_path,
        monkeypatch,
        timeout_seconds=0.1,
    )
    monkeypatch.setenv("FAKE_ODL_SLEEP", "5")

    with pytest.raises(OpenDataLoaderTimeoutError):
        extractor.extract(input_pdf, run_id="timeout-run")


def test_adapter_rechecks_input_sha256_at_trust_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)

    original_sha256_file = adapter_module.sha256_file

    def mismatched_input_hash(path: Path) -> str:
        if path.resolve() == input_pdf.resolve():
            return "0" * 64

        return original_sha256_file(path)

    monkeypatch.setattr(
        adapter_module,
        "sha256_file",
        mismatched_input_hash,
    )

    with pytest.raises(
        OpenDataLoaderIntegrityError,
        match="input SHA256",
    ):
        extractor.extract(input_pdf, run_id="tampered-input-run")


def test_manifest_rejects_false_image_hash_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)
    run = extractor.extract(input_pdf, run_id="hash-group-run")
    payload = run.manifest.model_dump(mode="json")
    payload["output"]["image_hash_groups"][0][
        "content_sha256"
    ] = "0" * 64

    with pytest.raises(
        ValidationError,
        match="does not match artifact SHA256",
    ):
        adapter_module.OpenDataLoaderWorkerManifest.model_validate(
            payload
        )
