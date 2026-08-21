from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docnorm_web.opendataloader_hybrid_document_worker import (
    run_conversion,
    validate_local_hybrid_url,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, root: Path) -> dict:
    return {
        "path": str(path),
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def test_full_document_hybrid_worker_is_local_and_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "scan.pdf"
    source.write_bytes(b"%PDF-1.7\nmock scan\n%%EOF\n")
    raw = tmp_path / "raw"
    observed: dict = {}

    def fake_converter(**kwargs):
        observed.update(kwargs)
        output = Path(kwargs["output_dir"])
        markdown = output / "document.md"
        structured = output / "document.json"
        markdown.write_text("# OCR result\n", encoding="utf-8")
        structured.write_text(
            json.dumps({"number of pages": 2, "kids": []}),
            encoding="utf-8",
        )

    def ensure_empty(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=False)

    def inspect(path: Path) -> dict:
        markdown = path / "document.md"
        structured = path / "document.json"
        return {
            "artifact_directory": str(path),
            "page_count": 2,
            "total_file_count": 2,
            "markdown_character_count": len(markdown.read_text(encoding="utf-8")),
            "markdown_broken_character_count": 0,
            "markdown": _artifact(markdown, path),
            "structured_json": _artifact(structured, path),
            "images": [],
            "physical_image_file_count": 0,
            "unique_image_hash_count": 0,
            "duplicate_image_hash_group_count": 0,
            "image_hash_groups": [],
        }

    helpers = SimpleNamespace(
        validate_input_pdf=lambda path: None,
        ensure_empty_artifact_directory=ensure_empty,
        inspect_engine_output=inspect,
        sha256_file=_sha256,
    )
    manifest = run_conversion(
        source,
        raw,
        local_worker_helpers=helpers,
        hybrid_url="http://127.0.0.1:5002",
        hybrid_timeout_ms=123_456,
        converter=fake_converter,
    )
    assert manifest["status"] == "PASS"
    assert manifest["engine"] == "OPENDATALOADER_HYBRID"
    assert manifest["request"]["scope"] == "FULL_DOCUMENT"
    assert manifest["request"]["ocr_languages"] == ["vi", "en"]
    assert manifest["request"]["external_calls"] == 0
    assert observed["hybrid"] == "docling-fast"
    assert observed["hybrid_mode"] == "full"
    assert observed["hybrid_fallback"] is False
    assert "pages" not in observed


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:5002",
        "http://example.com:5002",
        "http://127.0.0.1:5002/path",
        "http://user:pass@127.0.0.1:5002",
    ],
)
def test_hybrid_worker_rejects_non_local_or_credentialed_url(url: str) -> None:
    with pytest.raises(ValueError):
        validate_local_hybrid_url(url)
