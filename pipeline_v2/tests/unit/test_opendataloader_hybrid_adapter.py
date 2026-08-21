from __future__ import annotations

import os
import sys
from pathlib import Path

import fitz
import pytest

from rag_pipeline.extractors import (
    OpenDataLoaderHybridIntegrityError,
    OpenDataLoaderHybridPageConfig,
    OpenDataLoaderHybridPageExtractor,
    OpenDataLoaderHybridRunExistsError,
    OpenDataLoaderHybridTimeoutError,
    OpenDataLoaderHybridWorkerError,
)
from rag_pipeline.extractors import opendataloader_hybrid as adapter_module
from rag_pipeline.workers import opendataloader_hybrid_page_worker


FAKE_OPENDATALOADER_MODULE = '''
import json
import os
import time
from pathlib import Path


def convert(
    *,
    input_path,
    output_dir,
    format,
    pages,
    hybrid,
    hybrid_mode,
    hybrid_url,
    hybrid_timeout,
    hybrid_fallback,
):
    if os.environ.get("FAKE_HYBRID_SLEEP"):
        time.sleep(float(os.environ["FAKE_HYBRID_SLEEP"]))

    if os.environ.get("FAKE_HYBRID_FAIL"):
        raise RuntimeError("synthetic Hybrid backend failure")

    assert len(input_path) == 1
    assert format == "markdown,json"
    assert hybrid == "docling-fast"
    assert hybrid_mode == "full"
    assert hybrid_url == "http://127.0.0.1:5002"
    assert hybrid_timeout == "120000"
    assert hybrid_fallback is False

    page_number = int(pages)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sample.md").write_text(
        "# Hybrid result\\n\\nNội dung trang được chọn.\\n",
        encoding="utf-8",
    )
    (output / "sample.json").write_text(
        json.dumps(
            {
                "file name": "sample.pdf",
                "number of pages": 3,
                "kids": [
                    {
                        "type": "paragraph",
                        "page number": page_number,
                        "content": "Nội dung trang được chọn.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
'''


def write_pdf(path: Path, page_count: int = 3) -> None:
    document = fitz.open()

    for index in range(page_count):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {index + 1}")

    document.save(path)
    document.close()


def build_extractor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    timeout_seconds: float = 5.0,
) -> OpenDataLoaderHybridPageExtractor:
    fake_module_directory = tmp_path / "fake_module"
    fake_module_directory.mkdir()
    (fake_module_directory / "opendataloader_pdf.py").write_text(
        FAKE_OPENDATALOADER_MODULE,
        encoding="utf-8",
    )
    distribution = (
        fake_module_directory / "opendataloader_pdf-2.5.0.dist-info"
    )
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: opendataloader-pdf\nVersion: 2.5.0\n",
        encoding="utf-8",
    )

    existing_python_path = os.environ.get("PYTHONPATH")
    python_path_parts = [str(fake_module_directory)]

    if existing_python_path:
        python_path_parts.append(existing_python_path)

    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(python_path_parts))
    return OpenDataLoaderHybridPageExtractor(
        OpenDataLoaderHybridPageConfig(
            python_executable=Path(sys.executable),
            worker_script=Path(
                opendataloader_hybrid_page_worker.__file__
            ),
            output_root=tmp_path / "runs",
            timeout_seconds=timeout_seconds,
        )
    )


def test_adapter_runs_one_page_and_validates_fail_closed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)

    run = extractor.extract(
        input_pdf,
        page_number=2,
        reason_codes=["table_shape_invalid"],
        run_id="hybrid-page-2",
    )

    assert run.source_page_number == 2
    assert run.source_document_page_count == 3
    assert run.manifest.engine == "OPENDATALOADER_HYBRID"
    assert run.manifest.engine_version == "2.5.0"
    assert run.manifest.attempt_number == 1
    assert run.manifest.max_attempts == 1
    assert run.manifest.request.hybrid_mode == "full"
    assert run.manifest.request.hybrid_fallback is False
    assert run.manifest.output.observed_page_numbers == [2]
    assert run.manifest.output.unexpected_page_numbers == []
    assert run.manifest.output.page_scope_verified is True
    assert run.result_manifest_path.is_file()


def test_adapter_rejects_duplicate_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)
    extractor.extract(
        input_pdf,
        page_number=1,
        reason_codes=["text_too_short"],
        run_id="stable-run",
    )

    with pytest.raises(OpenDataLoaderHybridRunExistsError):
        extractor.extract(
            input_pdf,
            page_number=1,
            reason_codes=["text_too_short"],
            run_id="stable-run",
        )


def test_adapter_rejects_page_outside_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_pdf(input_pdf, page_count=2)
    extractor = build_extractor(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="between 1 and 2"):
        extractor.extract(
            input_pdf,
            page_number=3,
            reason_codes=["empty_page"],
        )


def test_adapter_surfaces_backend_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_HYBRID_FAIL", "1")

    with pytest.raises(
        OpenDataLoaderHybridWorkerError,
        match="synthetic Hybrid backend failure",
    ):
        extractor.extract(
            input_pdf,
            page_number=1,
            reason_codes=["empty_page"],
            run_id="failed-run",
        )


def test_adapter_kills_timed_out_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_pdf(input_pdf)
    extractor = build_extractor(
        tmp_path,
        monkeypatch,
        timeout_seconds=0.1,
    )
    monkeypatch.setenv("FAKE_HYBRID_SLEEP", "5")

    with pytest.raises(OpenDataLoaderHybridTimeoutError):
        extractor.extract(
            input_pdf,
            page_number=1,
            reason_codes=["empty_page"],
            run_id="timeout-run",
        )


def test_adapter_rechecks_source_sha256_at_trust_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_pdf(input_pdf)
    extractor = build_extractor(tmp_path, monkeypatch)
    original_sha256_file = adapter_module.sha256_file

    def mismatched_hash(path: Path) -> str:
        if path.resolve() == input_pdf.resolve():
            return "0" * 64

        return original_sha256_file(path)

    monkeypatch.setattr(adapter_module, "sha256_file", mismatched_hash)

    with pytest.raises(
        OpenDataLoaderHybridIntegrityError,
        match="input SHA256",
    ):
        extractor.extract(
            input_pdf,
            page_number=2,
            reason_codes=["table_shape_invalid"],
            run_id="tampered-source",
        )
