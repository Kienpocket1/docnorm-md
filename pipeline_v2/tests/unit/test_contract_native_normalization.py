from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rag_pipeline.contracts import (
    BoundingBox,
    ElementType,
    EngineName,
    ExtractedDocument,
    ExtractionQualityReport,
    QualityDisposition,
    SourceAnchor,
    SourceElement,
    StageStatus,
)
from rag_pipeline.extractors import (
    DocumentIdentity,
    OpenDataLoaderLocalRun,
    OpenDataLoaderMappingResult,
)
from rag_pipeline.extractors.opendataloader_local import (
    OpenDataLoaderWorkerManifest,
)
from rag_pipeline.normalization import (
    ContractNativeNormalizer,
    NormalizationCommitExistsError,
    NormalizationCommitIntegrityError,
    NormalizationInputError,
    NormalizationStructureError,
    NormalizerConfig,
    build_normalization,
    load_normalization_commit,
)
from rag_pipeline.orchestration import (
    ExtractionOrchestrator,
    ExtractionOrchestratorConfig,
    load_extraction_commit,
)
from rag_pipeline.quality import ExtractionQualityGateResult


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            sha256_file(path),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def artifact_record(path: Path, raw_directory: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "relative_path": path.relative_to(raw_directory).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_local_run(tmp_path: Path) -> OpenDataLoaderLocalRun:
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.7\n% normalization test\n")
    run_directory = tmp_path / "local-run"
    raw_directory = run_directory / "raw"
    raw_directory.mkdir(parents=True)
    markdown_path = raw_directory / "sample.md"
    structured_path = raw_directory / "sample.json"
    markdown_path.write_text("# Test\n", encoding="utf-8")
    structured_path.write_text(
        '{"number of pages": 2, "kids": []}',
        encoding="utf-8",
    )
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
                "sha256": sha256_file(input_pdf),
            },
            "output": {
                "artifact_directory": str(raw_directory),
                "page_count": 2,
                "total_file_count": 2,
                "markdown_character_count": 7,
                "markdown_broken_character_count": 0,
                "markdown": artifact_record(
                    markdown_path,
                    raw_directory,
                ),
                "structured_json": artifact_record(
                    structured_path,
                    raw_directory,
                ),
                "images": [],
                "physical_image_file_count": 0,
                "unique_image_hash_count": 0,
                "duplicate_image_hash_group_count": 0,
                "image_hash_groups": [],
            },
        }
    )
    manifest_path = run_directory / "worker_result.json"
    manifest_path.write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return OpenDataLoaderLocalRun(
        run_id="local-normalization-test",
        run_directory=run_directory,
        raw_artifact_directory=raw_directory,
        result_manifest_path=manifest_path,
        stdout_log_path=run_directory / "worker_stdout.log",
        stderr_log_path=run_directory / "worker_stderr.log",
        manifest=manifest,
    )


def make_anchor(
    source_sha256: str,
    block_index: int,
    page_number: int,
    section_path: list[str],
) -> SourceAnchor:
    return SourceAnchor(
        anchor_id=f"anchor-{block_index}",
        source_sha256=source_sha256,
        page_number=page_number,
        block_index=block_index,
        bbox=BoundingBox(
            x0=10,
            y0=10 + block_index * 20,
            x1=300,
            y1=25 + block_index * 20,
        ),
        section_path=section_path,
    )


def build_elements(
    source_sha256: str,
    *,
    table_content: str | None = None,
) -> list[SourceElement]:
    table = table_content or (
        "| Key | Value |\n"
        "| --- | --- |\n"
        "| A \\| B | C |"
    )
    return [
        SourceElement(
            element_id="element-title",
            element_type=ElementType.TITLE,
            content="Material Procedure",
            source_anchor=make_anchor(
                source_sha256,
                0,
                1,
                ["Material Procedure"],
            ),
            engine=EngineName.OPENDATALOADER_LOCAL,
            metadata={"indexable": True},
        ),
        SourceElement(
            element_id="element-paragraph",
            element_type=ElementType.PARAGRAPH,
            content="Cafe\u0301    material   text",
            source_anchor=make_anchor(
                source_sha256,
                1,
                1,
                ["Material Procedure"],
            ),
            engine=EngineName.OPENDATALOADER_LOCAL,
            metadata={"indexable": True},
        ),
        SourceElement(
            element_id="element-list",
            element_type=ElementType.LIST,
            content="i. First item\n  - Nested item",
            source_anchor=make_anchor(
                source_sha256,
                2,
                1,
                ["Material Procedure", "Scope"],
            ),
            engine=EngineName.OPENDATALOADER_LOCAL,
            metadata={"indexable": True},
        ),
        SourceElement(
            element_id="element-table",
            element_type=ElementType.TABLE,
            content=table,
            source_anchor=make_anchor(
                source_sha256,
                3,
                2,
                ["Material Procedure", "Records"],
            ),
            engine=EngineName.OPENDATALOADER_LOCAL,
            metadata={"indexable": True},
        ),
        SourceElement(
            element_id="element-logo",
            element_type=ElementType.IMAGE,
            content="",
            source_anchor=make_anchor(
                source_sha256,
                4,
                2,
                ["Material Procedure", "Records"],
            ),
            engine=EngineName.OPENDATALOADER_LOCAL,
            metadata={"indexable": False},
        ),
    ]


class StaticMapper:
    def __init__(self, mapping: OpenDataLoaderMappingResult) -> None:
        self.mapping = mapping

    def map_run(self, run, identity):
        return self.mapping


class StaticQualityGate:
    def __init__(self, decision: ExtractionQualityGateResult) -> None:
        self.decision = decision

    def evaluate(self, mapping, manifest):
        return self.decision


def identity() -> DocumentIdentity:
    return DocumentIdentity(
        catalog_id="catalog-001",
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v001-aaaaaaaaaaaa",
    )


def build_extraction_commit(
    tmp_path: Path,
    *,
    table_content: str | None = None,
    include_empty_image: bool = False,
    reorder_selection: bool = False,
):
    run = build_local_run(tmp_path)
    elements = build_elements(
        run.manifest.input.sha256,
        table_content=table_content,
    )
    document = ExtractedDocument(
        catalog_id="catalog-001",
        family_id="qt-tckt-01",
        version_id="qt-tckt-01-v001-aaaaaaaaaaaa",
        source_sha256=run.manifest.input.sha256,
        page_count=2,
        engine_chain=[EngineName.OPENDATALOADER_LOCAL],
        elements=elements,
    )
    indexable_ids = [
        element.element_id
        for element in elements
        if include_empty_image or element.element_id != "element-logo"
    ]

    if reorder_selection:
        indexable_ids[0], indexable_ids[1] = (
            indexable_ids[1],
            indexable_ids[0],
        )

    filtered_ids = (
        [] if include_empty_image else ["element-logo"]
    )
    mapping = OpenDataLoaderMappingResult(
        document=document,
        mapping_issues=(),
        indexable_element_ids=tuple(indexable_ids),
        filtered_element_ids=tuple(filtered_ids),
        metrics={
            "mapped_element_count": len(elements),
            "indexable_element_count": len(indexable_ids),
            "filtered_element_count": len(filtered_ids),
            "bbox_coverage": 1.0,
            "pages_with_elements": [1, 2],
        },
    )
    report = ExtractionQualityReport(
        report_id="quality-report-pass",
        catalog_id=document.catalog_id,
        family_id=document.family_id,
        version_id=document.version_id,
        source_sha256=document.source_sha256,
        engine=EngineName.OPENDATALOADER_LOCAL,
        disposition=QualityDisposition.PASS,
        issues=[],
        metrics=mapping.metrics,
    )
    decision = ExtractionQualityGateResult(
        report=report,
        local_result_accepted=True,
        fallback_required=False,
        indexable_element_ids=mapping.indexable_element_ids,
        filtered_element_ids=mapping.filtered_element_ids,
    )
    orchestrator = ExtractionOrchestrator(
        ExtractionOrchestratorConfig(
            output_root=tmp_path / "extraction-commits"
        ),
        mapper=StaticMapper(mapping),
        quality_gate=StaticQualityGate(decision),
    )
    result = orchestrator.process_run(
        run,
        identity(),
        run_id="input-extraction-run",
    )
    return result.commit


def make_normalizer(tmp_path: Path) -> ContractNativeNormalizer:
    return ContractNativeNormalizer(
        NormalizerConfig(
            output_root=tmp_path / "normalization-commits"
        )
    )


def test_build_preserves_structure_and_provenance(
    tmp_path: Path,
) -> None:
    extraction = build_extraction_commit(tmp_path)
    build = build_normalization(extraction)

    assert len(build.document.blocks) == 4
    assert [
        block.source_spans[0].source_element_id
        for block in build.document.blocks
    ] == [
        "element-title",
        "element-paragraph",
        "element-list",
        "element-table",
    ]
    assert build.document.blocks[0].markdown == "# Material Procedure"
    assert build.document.blocks[1].markdown == "Café material text"
    assert build.document.blocks[2].markdown == (
        "i. First item\n  - Nested item"
    )
    assert build.document.blocks[3].markdown == (
        "| Key | Value |\n"
        "| --- | --- |\n"
        "| A \\| B | C |"
    )
    assert build.document.blocks[3].source_spans[0].source_anchor == (
        extraction.document.elements[3].source_anchor
    )
    assert build.report.table_count == 1
    assert build.report.valid_gfm_table_count == 1
    assert build.report.list_count == 1
    assert build.report.filtered_source_element_count == 1
    assert build.report.page_numbers == [1, 2]
    assert build.report.provenance_coverage == 1.0
    assert build.report.ocr_invocation_count == 0


def test_normalization_is_deterministic_for_same_commit(
    tmp_path: Path,
) -> None:
    extraction = build_extraction_commit(tmp_path)
    first = build_normalization(extraction)
    second = build_normalization(extraction)

    assert first == second
    assert first.document.normalized_at == (
        extraction.manifest.committed_at
    )


def test_atomic_commit_is_resumable_and_does_not_mutate_input(
    tmp_path: Path,
) -> None:
    extraction = build_extraction_commit(tmp_path)
    input_before = file_inventory(extraction.commit_directory)
    normalizer = make_normalizer(tmp_path)
    commit = normalizer.normalize_commit(
        extraction.commit_directory,
        run_id="normalized-run",
    )
    input_after = file_inventory(extraction.commit_directory)

    assert input_before == input_after
    assert len(list(commit.commit_directory.iterdir())) == 5
    assert commit.manifest.next_action == "CHUNK"
    assert commit.manifest.ocr_invocation_count == 0
    assert commit.manifest.artifact_count == 4
    assert commit.stage_result.status == StageStatus.PASS
    assert commit.stage_result.metrics["next_action"] == "CHUNK"

    loaded = load_normalization_commit(commit.commit_directory)
    assert loaded.document == commit.document
    assert loaded.markdown == commit.markdown
    assert loaded.report == commit.report


def test_invalid_table_is_rejected_without_partial_commit(
    tmp_path: Path,
) -> None:
    extraction = build_extraction_commit(
        tmp_path,
        table_content="| A | B |\n| not-a-separator | B |",
    )
    normalizer = make_normalizer(tmp_path)

    with pytest.raises(NormalizationStructureError):
        normalizer.normalize_commit(
            extraction.commit_directory,
            run_id="invalid-table-run",
        )

    assert not (
        tmp_path / "normalization-commits" / "invalid-table-run"
    ).exists()


def test_selected_empty_visual_is_rejected(tmp_path: Path) -> None:
    extraction = build_extraction_commit(
        tmp_path,
        include_empty_image=True,
    )

    with pytest.raises(NormalizationStructureError):
        build_normalization(extraction)


def test_reordered_selection_is_rejected(tmp_path: Path) -> None:
    extraction = build_extraction_commit(
        tmp_path,
        reorder_selection=True,
    )

    with pytest.raises(NormalizationInputError):
        build_normalization(extraction)


def test_existing_normalization_commit_is_not_overwritten(
    tmp_path: Path,
) -> None:
    extraction = build_extraction_commit(tmp_path)
    normalizer = make_normalizer(tmp_path)
    first = normalizer.normalize_commit(
        extraction.commit_directory,
        run_id="immutable-normalization",
    )
    manifest_path = first.commit_directory / "commit_manifest.json"
    original_bytes = manifest_path.read_bytes()

    with pytest.raises(NormalizationCommitExistsError):
        normalizer.normalize_commit(
            extraction.commit_directory,
            run_id="immutable-normalization",
        )

    assert manifest_path.read_bytes() == original_bytes


def test_normalization_reader_rejects_tampering(
    tmp_path: Path,
) -> None:
    extraction = build_extraction_commit(tmp_path)
    commit = make_normalizer(tmp_path).normalize_commit(
        extraction.commit_directory,
        run_id="tampered-normalization",
    )
    markdown_path = commit.commit_directory / "document.normalized.md"
    markdown_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(NormalizationCommitIntegrityError):
        load_normalization_commit(commit.commit_directory)


def test_extraction_commit_remains_valid_after_normalization(
    tmp_path: Path,
) -> None:
    extraction = build_extraction_commit(tmp_path)
    make_normalizer(tmp_path).normalize_commit(
        extraction.commit_directory,
        run_id="input-integrity-run",
    )

    reloaded = load_extraction_commit(extraction.commit_directory)
    assert reloaded.manifest == extraction.manifest
