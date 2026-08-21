from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline.contracts import (
    AssetClassification,
    AssetRole,
    ElementType,
    QualityDisposition,
)
from rag_pipeline.extractors import (
    AssetPolicy,
    DocumentIdentity,
    OpenDataLoaderJsonMapper,
    OpenDataLoaderLocalRun,
)
from rag_pipeline.extractors.opendataloader_local import (
    OpenDataLoaderWorkerManifest,
)
from rag_pipeline.extractors.opendataloader_mapping import render_list
from rag_pipeline.quality import ExtractionQualityGate


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def artifact_record(path: Path, raw_directory: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": path.relative_to(raw_directory).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def paragraph(
    content: str,
    *,
    page: int,
    identifier: str,
) -> dict[str, object]:
    return {
        "type": "paragraph",
        "id": identifier,
        "page number": page,
        "bounding box": [10, 10, 200, 40],
        "content": content,
    }


def table_cell(
    *,
    row: int,
    column: int,
    content: str,
    column_span: int = 1,
) -> dict[str, object]:
    return {
        "type": "table cell",
        "row number": row,
        "column number": column,
        "row span": 1,
        "column span": column_span,
        "bounding box": [10, 10, 100, 30],
        "page number": 2,
        "kids": [
            paragraph(
                content,
                page=2,
                identifier=f"cell-{row}-{column}",
            )
        ],
    }


def build_run(
    tmp_path: Path,
    *,
    malformed_table: bool = False,
) -> tuple[OpenDataLoaderLocalRun, str]:
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\n% mapping test\n")
    source_sha256 = sha256_file(input_pdf)

    run_directory = tmp_path / "run"
    raw_directory = run_directory / "raw"
    image_directory = raw_directory / "sample_images"
    image_directory.mkdir(parents=True)

    logo_bytes = b"synthetic-brand-logo"
    logo_sha256 = sha256_bytes(logo_bytes)
    image_paths = []

    for index in range(1, 4):
        image_path = image_directory / f"imageFile{index}.png"
        image_path.write_bytes(logo_bytes)
        image_paths.append(image_path)

    second_row_cells = [
        table_cell(row=2, column=1, content="D"),
        table_cell(row=2, column=2, content="E"),
        table_cell(row=2, column=3, content="F"),
    ]

    if malformed_table:
        second_row_cells.pop()

    structured_payload = {
        "file name": "sample.pdf",
        "number of pages": 2,
        "kids": [
            {
                "type": "heading",
                "id": "heading-1",
                "page number": 1,
                "bounding box": [10, 10, 200, 40],
                "heading level": "Doctitle",
                "content": "Synthetic procedure",
            },
            paragraph(
                "Opening paragraph",
                page=1,
                identifier="paragraph-1",
            ),
            {
                "type": "list",
                "id": "list-1",
                "page number": 2,
                "bounding box": [10, 50, 300, 150],
                "numbering style": "unordered",
                "number of list items": 1,
                "list items": [
                    {
                        "type": "list item",
                        "page number": 2,
                        "bounding box": [10, 50, 250, 80],
                        "content": "Parent item",
                        "kids": [
                            {
                                "type": "list",
                                "page number": 2,
                                "bounding box": [20, 80, 250, 120],
                                "numbering style": "unordered",
                                "number of list items": 1,
                                "list items": [
                                    {
                                        "type": "list item",
                                        "page number": 2,
                                        "bounding box": [20, 80, 250, 100],
                                        "content": "Nested item",
                                        "kids": [],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "type": "table",
                "id": "table-1",
                "page number": 2,
                "bounding box": [10, 160, 500, 300],
                "number of rows": 2,
                "number of columns": 3,
                "rows": [
                    {
                        "type": "table row",
                        "row number": 1,
                        "cells": [
                            table_cell(
                                row=1,
                                column=1,
                                content="A",
                                column_span=2,
                            ),
                            table_cell(
                                row=1,
                                column=3,
                                content="C",
                            ),
                        ],
                    },
                    {
                        "type": "table row",
                        "row number": 2,
                        "cells": second_row_cells,
                    },
                ],
            },
            {
                "type": "image",
                "id": "image-1",
                "page number": 1,
                "bounding box": [300, 10, 500, 100],
                "source": "sample_images/imageFile1.png",
                "alt_source": "missing",
            },
        ],
    }

    markdown = "# Synthetic procedure\n\n![](sample_images/imageFile1.png)\n"
    markdown_path = raw_directory / "sample.md"
    structured_json_path = raw_directory / "sample.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    structured_json_path.write_text(
        json.dumps(structured_payload, ensure_ascii=False),
        encoding="utf-8",
    )

    image_records = [
        artifact_record(path, raw_directory)
        for path in image_paths
    ]
    now = datetime.now(timezone.utc)
    manifest = OpenDataLoaderWorkerManifest.model_validate(
        {
            "schema_version": "1.0",
            "worker_version": "test-worker",
            "status": "PASS",
            "engine": "OPENDATALOADER_LOCAL",
            "engine_version": "test",
            "started_at": now,
            "finished_at": now,
            "elapsed_seconds": 0.1,
            "input": {
                "path": str(input_pdf),
                "name": input_pdf.name,
                "size_bytes": input_pdf.stat().st_size,
                "sha256": source_sha256,
            },
            "output": {
                "artifact_directory": str(raw_directory),
                "page_count": 2,
                "total_file_count": 5,
                "markdown_character_count": len(markdown),
                "markdown_broken_character_count": 0,
                "markdown": artifact_record(
                    markdown_path,
                    raw_directory,
                ),
                "structured_json": artifact_record(
                    structured_json_path,
                    raw_directory,
                ),
                "images": image_records,
                "physical_image_file_count": 3,
                "unique_image_hash_count": 1,
                "duplicate_image_hash_group_count": 1,
                "image_hash_groups": [
                    {
                        "content_sha256": logo_sha256,
                        "physical_paths": [
                            record["relative_path"]
                            for record in image_records
                        ],
                    }
                ],
            },
        }
    )
    run = OpenDataLoaderLocalRun(
        run_id="mapping-test",
        run_directory=run_directory,
        raw_artifact_directory=raw_directory,
        result_manifest_path=run_directory / "worker_result.json",
        stdout_log_path=run_directory / "worker_stdout.log",
        stderr_log_path=run_directory / "worker_stderr.log",
        manifest=manifest,
    )
    return run, logo_sha256


def known_logo_mapper(logo_sha256: str) -> OpenDataLoaderJsonMapper:
    return OpenDataLoaderJsonMapper(
        {
            logo_sha256: AssetPolicy(
                classification=AssetClassification.DECORATIVE,
                role=AssetRole.BRAND_LOGO,
                indexable=False,
                reason_code="synthetic_known_logo",
            )
        }
    )


def identity() -> DocumentIdentity:
    return DocumentIdentity(
        catalog_id="catalog-001",
        family_id="sample-family",
        version_id="sample-family-v001-aaaaaaaaaaaa",
    )


def test_list_renderer_preserves_numbering_styles() -> None:
    items = [
        {"type": "list item", "content": "First", "kids": []},
        {"type": "list item", "content": "Second", "kids": []},
    ]

    assert render_list(
        {"numbering style": "roman numbers", "list items": items}
    ).splitlines() == ["i. First", "ii. Second"]
    assert render_list(
        {"numbering style": "english letters", "list items": items}
    ).splitlines() == ["a. First", "b. Second"]
    assert render_list(
        {
            "numbering style": "english letters in upper case",
            "list items": items,
        }
    ).splitlines() == ["A. First", "B. Second"]


def test_mapper_absorbs_nested_nodes_and_preserves_merged_table(
    tmp_path: Path,
) -> None:
    run, logo_sha256 = build_run(tmp_path)
    mapping = known_logo_mapper(logo_sha256).map_run(run, identity())

    assert mapping.metrics["root_node_count"] == 5
    assert mapping.metrics["mapped_element_count"] == 5
    assert mapping.metrics["mapped_table_count"] == 1
    assert mapping.metrics["mapped_list_count"] == 1
    assert mapping.metrics["merged_cell_count"] == 1
    assert mapping.metrics["invalid_table_slot_count"] == 0
    assert mapping.metrics["bbox_coverage"] == 1.0

    table = next(
        element
        for element in mapping.document.elements
        if element.element_type == ElementType.TABLE
    )
    assert table.metadata["missing_slots"] == 0
    assert "| A |  | C |" in table.content

    rendered_list = next(
        element
        for element in mapping.document.elements
        if element.element_type == ElementType.LIST
    )
    assert rendered_list.content.count("Parent item") == 1
    assert rendered_list.content.count("Nested item") == 1


def test_known_repeated_logo_is_filtered_without_fallback(
    tmp_path: Path,
) -> None:
    run, logo_sha256 = build_run(tmp_path)
    mapping = known_logo_mapper(logo_sha256).map_run(run, identity())
    decision = ExtractionQualityGate().evaluate(
        mapping,
        run.manifest,
    )

    assert len(mapping.document.assets) == 1
    asset = mapping.document.assets[0]
    assert len(asset.physical_paths) == 3
    assert len(asset.references) == 1
    assert asset.role == AssetRole.BRAND_LOGO
    assert asset.indexable is False
    assert len(mapping.filtered_element_ids) == 1
    assert decision.report.disposition == (
        QualityDisposition.PASS_WITH_WARNINGS
    )
    assert decision.fallback_required is False
    assert decision.local_result_accepted is True
    assert {
        issue.reason_code
        for issue in decision.report.issues
    } == {
        "repeated_brand_logo",
        "unreferenced_extracted_assets",
    }


def test_unknown_visual_asset_requires_hybrid_fallback(
    tmp_path: Path,
) -> None:
    run, _ = build_run(tmp_path)
    mapping = OpenDataLoaderJsonMapper({}).map_run(run, identity())
    decision = ExtractionQualityGate().evaluate(
        mapping,
        run.manifest,
    )

    assert decision.report.disposition == (
        QualityDisposition.FALLBACK_REQUIRED
    )
    assert decision.fallback_required is True
    assert "unclassified_visual_asset" in {
        issue.reason_code
        for issue in decision.report.issues
    }


def test_incomplete_table_grid_requires_fallback(
    tmp_path: Path,
) -> None:
    run, logo_sha256 = build_run(
        tmp_path,
        malformed_table=True,
    )
    mapping = known_logo_mapper(logo_sha256).map_run(run, identity())
    decision = ExtractionQualityGate().evaluate(
        mapping,
        run.manifest,
    )

    assert mapping.metrics["invalid_table_slot_count"] == 1
    assert decision.fallback_required is True
    assert "table_grid_mapping_failed" in {
        issue.reason_code
        for issue in decision.report.issues
    }


def test_mapping_identifiers_are_stable_for_same_source(
    tmp_path: Path,
) -> None:
    run, logo_sha256 = build_run(tmp_path)
    mapper = known_logo_mapper(logo_sha256)
    first = mapper.map_run(run, identity())
    second = mapper.map_run(run, identity())

    assert [
        element.element_id
        for element in first.document.elements
    ] == [
        element.element_id
        for element in second.document.elements
    ]
    assert [
        asset.asset_id
        for asset in first.document.assets
    ] == [
        asset.asset_id
        for asset in second.document.assets
    ]


def test_bbox_coverage_counts_only_emitted_elements(
    tmp_path: Path,
) -> None:
    run, logo_sha256 = build_run(tmp_path)
    structured_path = run.manifest.output.structured_json.path
    payload = json.loads(structured_path.read_text(encoding="utf-8"))
    payload["kids"].append(
        {
            "type": "paragraph",
            "id": "empty-paragraph",
            "page number": 2,
            "bounding box": [10, 310, 300, 340],
            "content": "",
        }
    )
    structured_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    mapping = known_logo_mapper(logo_sha256).map_run(run, identity())

    assert mapping.metrics["root_node_count"] == 6
    assert mapping.metrics["mapped_element_count"] == 5
    assert mapping.metrics["bbox_element_count"] == 5
    assert mapping.metrics["bbox_coverage"] == 1.0
