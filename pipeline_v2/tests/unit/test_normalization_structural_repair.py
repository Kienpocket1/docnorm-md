from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from rag_pipeline.contracts import (
    BoundingBox,
    ElementType,
    EngineName,
    ExtractedDocument,
    ExtractionQualityReport,
    QualityDisposition,
    ResumeFrom,
    SourceAnchor,
    SourceElement,
    StageResult,
    StageStatus,
)
from rag_pipeline.normalization.contract_native import (
    ContractNativeNormalizer,
    NormalizerConfig,
    is_valid_gfm_table,
    load_normalization_commit,
    render_element,
)
from rag_pipeline.normalization.structural_repair import (
    FLOWCHART_NOTE_1,
    FLOWCHART_NOTE_2,
    STRUCTURAL_REPAIR_VERSION,
    StructuralRepairConfig,
    StructuralRepairOrchestrator,
    repair_document,
)
from rag_pipeline.orchestration import (
    EXTRACTION_COMMIT_SCHEMA_VERSION,
    ExtractionCommitManifest,
    IndexingSelection,
    load_extraction_commit,
)
from rag_pipeline.orchestration.extraction import (
    COMMIT_MANIFEST_FILENAME,
    DOCUMENT_FILENAME,
    QUALITY_REPORT_FILENAME,
    SELECTION_FILENAME,
    STAGE_RESULT_FILENAME,
    artifact_record,
    model_json_bytes,
    write_payload,
)


SOURCE_SHA256 = "a" * 64
GENERATED_AT = datetime(2026, 8, 21, tzinfo=timezone.utc)


def _element(
    number: int,
    content: str,
    *,
    element_type: ElementType = ElementType.PARAGRAPH,
    page: int = 3,
) -> SourceElement:
    return SourceElement(
        element_id=f"element-{number}",
        element_type=element_type,
        content=content,
        source_anchor=SourceAnchor(
            anchor_id=f"anchor-{number}",
            source_sha256=SOURCE_SHA256,
            page_number=page,
            block_index=number,
            bbox=BoundingBox(
                x0=10.0,
                y0=float(number + 1),
                x1=100.0,
                y1=float(number + 2),
            ),
            section_path=[],
        ),
        engine=EngineName.OPENDATALOADER_LOCAL,
        metadata={"indexable": True},
    )


def _source_document() -> ExtractedDocument:
    elements: list[SourceElement] = [
        _element(1, "a. x", page=1),
        _element(2, "vụ cho sản xuất kinh doanh."),
        _element(3, "- Kho vật tư: Kho vật tư của Công ty tại Chi nhánh."),
        _element(
            4,
            """i. I. MỤC ĐÍCH: Quản lý vật tư.
ii. II. PHẠM VI ÁP DỤNG:
- - Áp dụng tại Công ty.
iii. III. ĐỊNH NGHĨA, THUẬT NGỮ VÀ TỪ VIẾT TẮT:
- - Công ty: Công ty Cổ phần Nhiên liệu bay Petrolimex.
- - Vật tư: bao gồm nguyên liệu, vật liệu, công cụ dụng cụ, phụ tùng thay thế phục
iv. IV. Qui định về quản lý vật tư:
- - Vật tư phải được quản lý.""",
            element_type=ElementType.LIST,
        ),
        _element(5, "V. LƯU ĐỒ", element_type=ElementType.HEADING, page=5),
    ]
    flow_text = [
        "1. Nhập vật tư",
        "Tiếp nhận vật tư, hồ sơ",
        "P.HH P.QLKT P.KT-TH",
        "Kiểm tra, xử lý",
        "Không đạt",
        "Không tiếp nhận",
        "Đạt",
        "Lưu kho",
        "Quản lý, theo dõi",
        "Báo cáo",
        "2. Xuất cấp vật tư",
        "Thông tin đề xuất",
        "Xử lý đề xuất",
        "Không đạt",
        "Không bàn giao",
        "Đạt",
        "Bàn giao",
        "Quản lý, theo dõi",
        "Báo cáo",
    ]
    elements.extend(
        _element(6 + index, text, page=5)
        for index, text in enumerate(flow_text)
    )
    next_number = 6 + len(flow_text)
    elements.extend(
        [
            _element(
                next_number,
                "i. VI. DIỄN GIẢI LƯU ĐỒ",
                element_type=ElementType.HEADING,
                page=6,
            ),
            _element(
                next_number + 1,
                "1. 1. Nhập vật tư",
                element_type=ElementType.LIST,
                page=6,
            ),
            _element(
                next_number + 2,
                """| TT | Nội dung | Diễn giải | Hồ sơ |
| --- | --- | --- | --- |
| 1 | Tiếp nhận | Nội dung 1 | BM.01 |
| 2 | Kiểm tra | Nội dung 2 | BM.02 |""",
                element_type=ElementType.TABLE,
                page=6,
            ),
            _element(
                next_number + 3,
                """| 3 | Bàn giao | Nội dung 3 | BM.03 |
| --- | --- | --- | --- |
| 4 | Theo dõi | Nội dung 4 | BM.04 |
| 5 | Báo cáo | Nội dung 5 | BM.05 |""",
                element_type=ElementType.TABLE,
                page=7,
            ),
            _element(
                next_number + 4,
                "2. 2. Xuất cấp vật tư",
                element_type=ElementType.LIST,
                page=7,
            ),
            _element(
                next_number + 5,
                """| TT | Nội dung | Diễn giải | Hồ sơ |
| --- | --- | --- | --- |
| 1 | Đề xuất | Nội dung 1 | BM.01 |
| 2 | Xử lý | Nội dung 2 | BM.02 |
| 3 | Bàn giao | Nội dung 3 | BM.03 |""",
                element_type=ElementType.TABLE,
                page=7,
            ),
            _element(
                next_number + 6,
                """| 4 | Quản lý | Nội dung 4 | BM.04 |
| --- | --- | --- | --- |
| 5 | Báo cáo | Nội dung 5 | BM.05 |""",
                element_type=ElementType.TABLE,
                page=8,
            ),
            _element(
                next_number + 7,
                "3. Luân chuyển, kiểm kê vật tư",
                element_type=ElementType.HEADING,
                page=8,
            ),
            _element(
                next_number + 8,
                """| TT | Nội dung | Ghi chú | Hồ sơ |
| --- | --- | --- | --- |
| 1 | Kiểm kê<br>A. − Ý thứ nhất<br>B. − Ý thứ hai | Giữ nguyên | BM.06 |""",
                element_type=ElementType.TABLE,
                page=14,
            ),
            _element(
                next_number + 9,
                "PETROLIMEX AVIATION",
                page=14,
            ),
        ]
    )
    return ExtractedDocument(
        catalog_id="catalog-1",
        family_id="family-1",
        version_id="family-1-v001-aaaaaaaaaaaa",
        source_sha256=SOURCE_SHA256,
        page_count=16,
        engine_chain=[EngineName.OPENDATALOADER_LOCAL],
        elements=elements,
        assets=[],
        extracted_at=GENERATED_AT,
    )


def _fully_paginated_source_document() -> ExtractedDocument:
    source = _source_document()
    occupied = {
        element.source_anchor.page_number for element in source.elements
    }
    required_filler_pages = sorted(set(range(1, 17)) - occupied | {1})
    fillers = [
        _element(
            100 + page,
            f"Nội dung hợp lệ giữ provenance cho trang {page}.",
            page=page,
        )
        for page in required_filler_pages
    ]
    elements = sorted(
        [*source.elements, *fillers],
        key=lambda element: (
            element.source_anchor.page_number or 0,
            element.source_anchor.block_index,
        ),
    )
    return source.model_copy(update={"elements": elements})


def _commit_extraction(root: Path, document: ExtractedDocument) -> Path:
    root.mkdir(parents=True)
    selected_ids = [element.element_id for element in document.elements]
    report = ExtractionQualityReport(
        report_id="quality-report-input",
        catalog_id=document.catalog_id,
        family_id=document.family_id,
        version_id=document.version_id,
        source_sha256=document.source_sha256,
        engine=EngineName.OPENDATALOADER_LOCAL,
        disposition=QualityDisposition.PASS,
        issues=[],
        metrics={"fixture": True},
        evaluated_at=GENERATED_AT,
    )
    selection = IndexingSelection(
        catalog_id=document.catalog_id,
        family_id=document.family_id,
        version_id=document.version_id,
        source_sha256=document.source_sha256,
        indexable_element_ids=selected_ids,
        filtered_element_ids=[],
    )
    stage_result = StageResult(
        stage=ResumeFrom.EXTRACTION_QUALITY,
        status=StageStatus.PASS,
        started_at=GENERATED_AT,
        finished_at=GENERATED_AT,
        duration_seconds=0.0,
        artifact_paths=[
            DOCUMENT_FILENAME,
            QUALITY_REPORT_FILENAME,
            SELECTION_FILENAME,
        ],
        metrics={"next_action": "NORMALIZE"},
    )
    models = {
        DOCUMENT_FILENAME: document,
        QUALITY_REPORT_FILENAME: report,
        SELECTION_FILENAME: selection,
        STAGE_RESULT_FILENAME: stage_result,
    }

    for filename, model in models.items():
        write_payload(root / filename, model_json_bytes(model))

    artifacts = [
        artifact_record(root / filename, root) for filename in sorted(models)
    ]
    manifest = ExtractionCommitManifest(
        schema_version=EXTRACTION_COMMIT_SCHEMA_VERSION,
        orchestrator_version="fixture-1.0",
        run_id="input-extraction",
        catalog_id=document.catalog_id,
        family_id=document.family_id,
        version_id=document.version_id,
        source_sha256=document.source_sha256,
        local_run_id="fixture-local",
        local_run_directory=str(root),
        local_worker_manifest_path=str(root / "fixture-worker.json"),
        quality_disposition=QualityDisposition.PASS,
        local_result_accepted=True,
        fallback_required=False,
        next_action="NORMALIZE",
        artifact_count=len(artifacts),
        artifacts=artifacts,
        committed_at=GENERATED_AT,
    )
    write_payload(root / COMMIT_MANIFEST_FILENAME, model_json_bytes(manifest))
    load_extraction_commit(root)
    return root


def _inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_repairs_all_pre_chunk_audit_issues() -> None:
    source = _source_document()
    build = repair_document(
        source,
        input_normalization_run_id="stage7-1a-input",
        generated_at=GENERATED_AT,
    )
    markdown = "\n\n".join(
        render_element(element) for element in build.document.elements
    )

    assert build.report.repair_version == STRUCTURAL_REPAIR_VERSION
    assert build.report.next_action == "CHUNK"
    assert build.report.chunking_invocations == 0
    assert build.report.removed_noise_element_ids == [
        "element-1",
        f"element-{6 + 19 + 9}",
    ]
    assert "a. x" not in markdown
    assert "PETROLIMEX AVIATION" not in markdown
    assert "i. I." not in markdown
    assert "ii. II." not in markdown
    assert "- -" not in markdown
    assert "<br>A. −" not in markdown
    assert "<br>B. −" not in markdown
    assert "Vật tư:" in markdown
    assert "phục vụ cho sản xuất kinh doanh." in markdown
    assert markdown.index("phục vụ cho sản xuất kinh doanh.") < markdown.index(
        "Kho vật tư:"
    )
    assert markdown.count("| TT | Nội dung | Diễn giải | Hồ sơ |") == 2
    assert markdown.count("| Đơn vị xử lý | Nội dung | Biểu mẫu |") == 2
    assert FLOWCHART_NOTE_1 in markdown
    assert FLOWCHART_NOTE_2 in markdown


def test_repaired_tables_are_valid_gfm_and_lineage_is_retained() -> None:
    build = repair_document(
        _source_document(),
        input_normalization_run_id="stage7-1a-input",
        generated_at=GENERATED_AT,
    )
    tables = [
        element
        for element in build.document.elements
        if element.element_type == ElementType.TABLE
    ]

    assert tables
    assert all(is_valid_gfm_table(element.content) for element in tables)
    assert set(build.report.merged_table_groups) == {
        "nhap_vat_tu",
        "xuat_cap_vat_tu",
    }
    assert all(
        "merged_source_element_ids" in element.metadata
        for element in tables
        if element.metadata.get("structural_repair_action")
        in {"flowchart_reconstruction", "page_split_table_merge"}
    )
    anchor_ids = [
        element.source_anchor.anchor_id for element in build.document.elements
    ]
    assert len(anchor_ids) == len(set(anchor_ids))


def test_repair_is_deterministic() -> None:
    first = repair_document(
        _source_document(),
        input_normalization_run_id="stage7-1a-input",
        generated_at=GENERATED_AT,
    )
    second = repair_document(
        _source_document(),
        input_normalization_run_id="stage7-1a-input",
        generated_at=GENERATED_AT,
    )

    assert first == second


def test_definition_fragments_may_share_one_source_list_element() -> None:
    source = _source_document()
    source_elements = {element.element_id: element for element in source.elements}
    definition = source_elements["element-4"]
    combined_content = definition.content.replace(
        "iii. III. ĐỊNH NGHĨA, THUẬT NGỮ VÀ TỪ VIẾT TẮT:\n",
        "iii. III. ĐỊNH NGHĨA, THUẬT NGỮ VÀ TỪ VIẾT TẮT:\n"
        "  vụ cho sản xuất kinh doanh.\n"
        "  - Kho vật tư: Kho vật tư của Công ty tại Chi nhánh.\n",
    )
    combined = definition.model_copy(update={"content": combined_content})
    elements = [
        combined if element.element_id == combined.element_id else element
        for element in source.elements
        if element.element_id not in {"element-2", "element-3"}
    ]
    build = repair_document(
        source.model_copy(update={"elements": elements}),
        input_normalization_run_id="stage7-1a-input",
        generated_at=GENERATED_AT,
    )
    markdown = "\n\n".join(
        render_element(element) for element in build.document.elements
    )

    assert build.report.definition_fragment_count == 3
    assert build.report.merged_definition_element_ids == ["element-4"]
    assert markdown.index("phục vụ cho sản xuất kinh doanh.") < markdown.index(
        "Kho vật tư:"
    )
    assert markdown.index("Kho vật tư:") < markdown.index("IV. Qui định")


def test_orchestrator_commits_reproducible_normalization_and_stops(
    tmp_path: Path,
) -> None:
    extraction_directory = _commit_extraction(
        tmp_path / "input-extraction",
        _fully_paginated_source_document(),
    )
    input_normalization = ContractNativeNormalizer(
        NormalizerConfig(output_root=tmp_path / "input-normalization")
    ).normalize_commit(extraction_directory, run_id="stage7-1a-input")
    input_before = _inventory(input_normalization.commit_directory)
    extraction_before = _inventory(extraction_directory)
    result = StructuralRepairOrchestrator(
        StructuralRepairConfig(
            handoff_output_root=tmp_path / "repaired-handoff",
            normalization_output_root=tmp_path / "repaired-normalization",
        )
    ).repair_normalization_commit(
        input_normalization.commit_directory,
        run_id="stage7-1b-output",
        handoff_run_id="stage7-1b-output-handoff",
    )
    loaded = load_normalization_commit(
        result.normalization_commit.commit_directory
    )

    assert _inventory(input_normalization.commit_directory) == input_before
    assert _inventory(extraction_directory) == extraction_before
    assert loaded.manifest.next_action == "CHUNK"
    assert loaded.stage_result.status == StageStatus.PASS
    assert loaded.report.provenance_coverage == 1.0
    assert loaded.report.page_numbers == list(range(1, 17))
    assert loaded.report.invalid_gfm_table_count == 0
    assert result.repair_report.chunking_invocations == 0
    assert not list(loaded.commit_directory.glob("chunks*"))
