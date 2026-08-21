from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from docnorm_web.config import Settings
from docnorm_web.normalizer import sha256_file
from docnorm_web.pipeline_bridge import PipelineBridge


def _pipeline_root() -> Path:
    configured = os.getenv("DOCNORM_PIPELINE_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2] / "pipeline_v2"


def _record(path: Path) -> dict:
    return {
        "path": str(path),
        "relative_path": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def test_hybrid_mapping_relabels_engine_and_counts_one_ocr_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_root = _pipeline_root()
    if not (pipeline_root / "src" / "rag_pipeline").is_dir():
        pytest.skip("pipeline_v2 source is not available")
    product_root = Path(__file__).resolve().parents[1]
    settings = Settings(
        product_root=product_root,
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
        pipeline_root=pipeline_root,
        odl_python=Path(os.sys.executable),
        odl_worker=(
            pipeline_root
            / "src"
            / "rag_pipeline"
            / "workers"
            / "opendataloader_local_worker.py"
        ),
        hybrid_worker=(
            product_root / "docnorm_web" / "opendataloader_hybrid_document_worker.py"
        ),
        hybrid_executable=Path(os.sys.executable),
        vlm_python=Path(os.sys.executable),
        math_worker=product_root / "docnorm_web" / "qwen_math_worker.py",
    )
    bridge = PipelineBridge(settings)
    modules = bridge._load_pipeline()
    assert modules is not None, bridge._import_error

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    markdown = raw / "document.md"
    markdown.write_text(
        "# OCR title\n\nOCR paragraph with enough text.\n",
        encoding="utf-8",
    )
    structured = raw / "document.json"
    structured.write_text(
        json.dumps(
            {
                "number of pages": 1,
                "kids": [
                    {
                        "type": "heading",
                        "page number": 1,
                        "bounding box": [72, 72, 400, 100],
                        "content": "OCR title",
                        "font": "Arial",
                        "font size": 16,
                        "text color": "#000000",
                        "heading level": "Doctitle",
                        "id": "h1",
                        "pdfua_tag": "H1",
                    },
                    {
                        "type": "paragraph",
                        "page number": 1,
                        "bounding box": [72, 110, 500, 170],
                        "content": "OCR paragraph with enough text.",
                        "font": "Arial",
                        "font size": 11,
                        "text color": "#000000",
                        "id": "p1",
                        "pdfua_tag": "P",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    input_record = modules["WorkerInputRecord"].model_validate(
        {
            "path": str(source),
            "name": source.name,
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    )
    output_record = modules["WorkerOutputRecord"].model_validate(
        {
            "artifact_directory": str(raw),
            "page_count": 1,
            "total_file_count": 2,
            "markdown_character_count": len(markdown.read_text(encoding="utf-8")),
            "markdown_broken_character_count": 0,
            "markdown": _record(markdown),
            "structured_json": _record(structured),
            "images": [],
            "physical_image_file_count": 0,
            "unique_image_hash_count": 0,
            "duplicate_image_hash_group_count": 0,
            "image_hash_groups": [],
        }
    )
    run = SimpleNamespace(
        manifest=SimpleNamespace(input=input_record, output=output_record),
        raw_artifact_directory=raw,
    )
    monkeypatch.setattr(
        bridge,
        "_run_hybrid_document_worker",
        lambda *args, **kwargs: run,
    )
    result = bridge._convert_pdf_hybrid(
        source,
        tmp_path / "work",
        modules,
        lambda *args: None,
        route_reason="TEST_SCAN",
    )
    assert result.engine_chain == ["OPENDATALOADER_HYBRID"]
    assert result.backend == "RAG_PIPELINE_HYBRID_OCR"
    assert result.metrics.ocr_invocations == 1
    assert result.metrics.provenance_coverage == 1.0
    assert result.issues[0].code == "LOCAL_HYBRID_OCR_USED"
