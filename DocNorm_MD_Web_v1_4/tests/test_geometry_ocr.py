from __future__ import annotations

from pathlib import Path

import pymupdf

from docnorm_web.config import Settings
from docnorm_web.geometry_ocr import (
    cluster_rows,
    render_grid_table,
    render_page_markdown,
)
from docnorm_web.models import ConversionMode, DocumentMetrics, RepairProfile
from docnorm_web.normalizer import ConversionDocument
from docnorm_web.pipeline_bridge import PipelineBridge


def detection(text: str, bbox: list[int], confidence: float = 0.9) -> dict:
    return {"text": text, "bbox": bbox, "confidence": confidence}


def test_cluster_rows_restores_top_to_bottom_left_to_right_order() -> None:
    items = [
        detection("bay Petrolimex", [310, 100, 470, 126]),
        detection("Công ty cổ phần", [40, 100, 210, 126]),
        detection("Quyết định", [180, 160, 320, 194]),
        detection("nhiên liệu", [215, 100, 305, 126]),
    ]
    rows = cluster_rows(items)
    assert rows[0]["text"] == "Công ty cổ phần nhiên liệu bay Petrolimex"
    assert rows[1]["text"] == "Quyết định"


def test_grid_table_assigns_text_to_geometric_cells() -> None:
    grid = {
        "bbox": [0, 0, 200, 100],
        "x_lines": [0, 100, 200],
        "y_lines": [0, 50, 100],
    }
    items = [
        detection("TT", [10, 10, 30, 30]),
        detection("Nội dung", [110, 10, 180, 30]),
        detection("1", [10, 60, 30, 80]),
        detection("Mua sắm", [110, 60, 180, 80]),
    ]
    table = render_grid_table(grid, items)
    assert "| TT | Nội dung |" in table
    assert "| --- | --- |" in table
    assert "| 1 | Mua sắm |" in table
    assert "Cột 1" not in table


def test_page_renderer_keeps_heading_and_table_in_vertical_order() -> None:
    grid = {
        "bbox": [20, 180, 380, 280],
        "x_lines": [20, 200, 380],
        "y_lines": [180, 230, 280],
    }
    items = [
        detection("QUYẾT ĐỊNH", [120, 50, 280, 80]),
        detection("Nội dung văn bản.", [40, 110, 360, 135]),
        detection("A", [40, 195, 80, 215]),
        detection("B", [220, 195, 260, 215]),
        detection("1", [40, 245, 80, 265]),
        detection("2", [220, 245, 260, 265]),
    ]
    markdown, metrics = render_page_markdown(
        items,
        page_width=400,
        page_height=500,
        table_grids=[grid],
    )
    assert markdown.index("## QUYẾT ĐỊNH") < markdown.index("Nội dung văn bản")
    assert markdown.index("Nội dung văn bản") < markdown.index("| A | B |")
    assert metrics["tables"] == 1


def test_signature_role_is_not_promoted_to_heading() -> None:
    markdown, _ = render_page_markdown(
        [detection("TỔNG GIÁM ĐỐC", [120, 350, 280, 380])],
        page_width=400,
        page_height=500,
    )
    assert markdown.strip() == "TỔNG GIÁM ĐỐC"


def test_scanned_pdf_routes_to_geometry_before_fast_mode_rejects(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "scan.pdf"
    document = pymupdf.open()
    page = document.new_page()
    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB,
        pymupdf.IRect(0, 0, 200, 280),
        False,
    )
    pixmap.clear_with(245)
    page.insert_image(page.rect, pixmap=pixmap)
    document.save(source)
    document.close()
    executable = tmp_path / "python.exe"
    worker = tmp_path / "worker.py"
    executable.write_bytes(b"stub")
    worker.write_text("# stub\n", encoding="utf-8")
    settings = Settings(
        product_root=tmp_path,
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
        pipeline_root=tmp_path / "pipeline",
        odl_python=executable,
        odl_worker=worker,
        hybrid_worker=worker,
        hybrid_executable=executable,
        vlm_python=executable,
        math_worker=worker,
        scan_python=executable,
        scan_worker=worker,
    )
    bridge = PipelineBridge(settings)
    called = []

    def fake_geometry(_source, _directory, _progress):
        called.append(True)
        markdown = "<!-- page: 1 -->\n\n## QUYẾT ĐỊNH\n\nNội dung đúng thứ tự.\n"
        return ConversionDocument(
            markdown=markdown,
            metrics=DocumentMetrics(
                pages=1,
                elements=2,
                headings=1,
                paragraphs=1,
                provenance_coverage=1.0,
                ocr_invocations=1,
            ),
            issues=[],
            engine_chain=["EASYOCR_GEOMETRY"],
            backend="EASYOCR_GEOMETRY",
            quality_disposition="PASS_WITH_WARNINGS",
            source_sha256="test",
            page_markdown={1: markdown},
        )

    monkeypatch.setattr(bridge, "_convert_pdf_scan_geometry", fake_geometry)
    result = bridge.convert(
        source,
        tmp_path / "job",
        ConversionMode.FAST,
        RepairProfile.GENERIC,
        lambda *_args: None,
    )
    assert called == [True]
    assert result.backend == "EASYOCR_GEOMETRY"
