from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rag_pipeline.workers import opendataloader_hybrid_page_worker as worker


def write_input_pdf(path: Path) -> None:
    path.write_bytes(b"%PDF-1.7\n% hybrid worker unit test\n")


def fake_convert_factory(page_number: int):
    def fake_convert(**kwargs: Any) -> None:
        assert kwargs["input_path"]
        assert kwargs["format"] == "markdown,json"
        assert kwargs["pages"] == str(page_number)
        assert kwargs["hybrid"] == "docling-fast"
        assert kwargs["hybrid_mode"] == "full"
        assert kwargs["hybrid_url"] == "http://127.0.0.1:5002"
        assert kwargs["hybrid_timeout"] == "120000"
        assert kwargs["hybrid_fallback"] is False

        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "sample.md").write_text(
            "# Bảng kiểm thử\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n",
            encoding="utf-8",
        )
        (output / "sample.json").write_text(
            json.dumps(
                {
                    "file name": "sample.pdf",
                    "number of pages": 16,
                    "kids": [
                        {
                            "type": "paragraph",
                            "page number": page_number,
                            "content": "Nội dung Hybrid",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    return fake_convert


def test_worker_forces_exact_page_full_hybrid_without_java_fallback(
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)

    result = worker.run_conversion(
        input_pdf=input_pdf,
        source_page_number=15,
        artifact_directory=tmp_path / "raw",
        reason_codes=["invalid_table_shape", "invalid_table_shape"],
        converter=fake_convert_factory(15),
    )

    assert result["status"] == "PASS"
    assert result["engine"] == "OPENDATALOADER_HYBRID"
    assert result["attempt_number"] == 1
    assert result["max_attempts"] == 1
    assert result["request"]["hybrid_mode"] == "full"
    assert result["request"]["hybrid_fallback"] is False
    assert result["request"]["reason_codes"] == [
        "invalid_table_shape"
    ]
    assert result["output"]["observed_page_numbers"] == [15]
    assert result["output"]["page_scope_verified"] is True
    assert "Nội dung Hybrid" not in json.dumps(
        result,
        ensure_ascii=False,
    )


def test_worker_rejects_output_from_another_page(tmp_path: Path) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)

    def wrong_page_convert(**kwargs: Any) -> None:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "sample.md").write_text(
            "# Wrong page",
            encoding="utf-8",
        )
        (output / "sample.json").write_text(
            json.dumps(
                {
                    "number of pages": 16,
                    "kids": [
                        {
                            "type": "paragraph",
                            "page number": 14,
                            "content": "Wrong page",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="escaped requested page"):
        worker.run_conversion(
            input_pdf=input_pdf,
            source_page_number=15,
            artifact_directory=tmp_path / "raw",
            reason_codes=["invalid_table_shape"],
            converter=wrong_page_convert,
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:5002",
        "http://192.168.1.10:5002",
        "http://example.com:5002",
        "http://127.0.0.1",
        "http://user:secret@127.0.0.1:5002",
    ],
)
def test_worker_rejects_nonlocal_or_unsafe_backend_url(url: str) -> None:
    with pytest.raises(ValueError):
        worker.validate_local_hybrid_url(url)


def test_empty_structured_page_still_preserves_converter_scope(
    tmp_path: Path,
) -> None:
    input_pdf = tmp_path / "sample.pdf"
    write_input_pdf(input_pdf)

    def empty_convert(**kwargs: Any) -> None:
        output = Path(kwargs["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "sample.md").write_text(
            "# Empty page fallback result",
            encoding="utf-8",
        )
        (output / "sample.json").write_text(
            json.dumps({"number of pages": 16, "kids": []}),
            encoding="utf-8",
        )

    result = worker.run_conversion(
        input_pdf=input_pdf,
        source_page_number=2,
        artifact_directory=tmp_path / "raw",
        reason_codes=["empty_page"],
        converter=empty_convert,
    )

    assert result["output"]["observed_page_numbers"] == []
    assert result["output"]["page_scope_evidence"] == (
        "CONVERTER_PAGE_FILTER"
    )
