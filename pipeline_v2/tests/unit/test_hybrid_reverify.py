from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rag_pipeline.contracts import (
    BoundingBox,
    ElementType,
    EngineName,
    ExtractedDocument,
    SourceAnchor,
    SourceElement,
    StageStatus,
)
from rag_pipeline.extractors.opendataloader_hybrid import (
    HybridWorkerOutputRecord,
    HybridWorkerRequestRecord,
    OpenDataLoaderHybridPageRun,
    OpenDataLoaderHybridWorkerManifest,
)
from rag_pipeline.extractors.opendataloader_local import (
    WorkerArtifactRecord,
    WorkerInputRecord,
)
from rag_pipeline.failure import (
    ExtractionRoutingPlan,
    FallbackAction,
    FallbackStep,
    RouteKind,
    ScopeRoutePlan,
)
from rag_pipeline.stages.hybrid_reverify import (
    HybridReverificationConfig,
    HybridReverificationIntegrityError,
    HybridReverifyMergeStage,
    load_hybrid_page_run,
    load_hybrid_reverification_commit,
)


SOURCE_SHA256 = "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, root: Path) -> WorkerArtifactRecord:
    return WorkerArtifactRecord(
        path=path.resolve(),
        relative_path=path.relative_to(root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _element(page: int, text: str) -> SourceElement:
    return SourceElement(
        element_id=f"base-page-{page}",
        element_type=ElementType.PARAGRAPH,
        content=text,
        source_anchor=SourceAnchor(
            anchor_id=f"base-anchor-{page}",
            source_sha256=SOURCE_SHA256,
            page_number=page,
            block_index=0,
            bbox=BoundingBox(x0=10, y0=10, x1=300, y1=80),
        ),
        engine=EngineName.OPENDATALOADER_LOCAL,
        metadata={"indexable": True},
    )


def _base_document() -> ExtractedDocument:
    return ExtractedDocument(
        catalog_id="catalog-1",
        family_id="family-1",
        version_id="version-1",
        source_sha256=SOURCE_SHA256,
        page_count=2,
        engine_chain=[EngineName.OPENDATALOADER_LOCAL],
        elements=[
            _element(1, "Stable content on the first page."),
            _element(2, "Original content on the second page."),
        ],
    )


def _routing_plan(*, hybrid_page: int | None = 2) -> ExtractionRoutingPlan:
    routes: list[ScopeRoutePlan] = []

    if hybrid_page is not None:
        routes.append(
            ScopeRoutePlan(
                route_id=f"route-page-{hybrid_page}",
                scope_id=f"page-{hybrid_page}",
                page_number=hybrid_page,
                route_kind=RouteKind.TEXT_TABLE,
                reason_codes=["table_grid_invalid"],
                failure_domains=["TABLE"],
                steps=[
                    FallbackStep(
                        ordinal=1,
                        action=FallbackAction.ODL_HYBRID,
                        max_attempts=1,
                        engine=EngineName.OPENDATALOADER_HYBRID,
                        condition="local_page_failed",
                    ),
                    FallbackStep(
                        ordinal=2,
                        action=FallbackAction.REVERIFY,
                        max_attempts=1,
                        condition="after_odl_hybrid",
                    ),
                    FallbackStep(
                        ordinal=3,
                        action=FallbackAction.MINERU_PAGE,
                        max_attempts=1,
                        engine=EngineName.MINERU,
                        condition="odl_hybrid_reverify_failed",
                    ),
                ],
                fallback_required=True,
                external_model_required=False,
            )
        )

    return ExtractionRoutingPlan(
        router_version="test-router",
        routes=routes,
        fallback_required=bool(routes),
        next_action="RUN_PAGE_FALLBACKS" if routes else "NORMALIZE",
        metrics={},
    )


def _hybrid_run(
    tmp_path: Path,
    *,
    include_bbox: bool = True,
    reason_codes: list[str] | None = None,
) -> OpenDataLoaderHybridPageRun:
    root = tmp_path / "hybrid-page-2"
    raw = root / "raw"
    raw.mkdir(parents=True)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"synthetic-pdf")
    markdown = raw / "page.md"
    structured = raw / "page.json"
    markdown.write_text(
        "Replacement content from the Hybrid backend.\n",
        encoding="utf-8",
    )
    node = {
        "type": "paragraph",
        "page number": 2,
        "content": "Replacement content from the Hybrid backend.",
    }

    if include_bbox:
        node["bounding box"] = [10, 10, 320, 90]

    structured.write_text(
        json.dumps(
            {
                "file name": "source.pdf",
                "number of pages": 2,
                "kids": [node],
            }
        ),
        encoding="utf-8",
    )
    active_reasons = reason_codes or ["table_grid_invalid"]
    now = datetime.now(timezone.utc)
    manifest = OpenDataLoaderHybridWorkerManifest(
        schema_version="1.0",
        worker_version="test-worker",
        status="PASS",
        engine="OPENDATALOADER_HYBRID",
        engine_version="2.5.0",
        started_at=now,
        finished_at=now,
        elapsed_seconds=0,
        attempt_number=1,
        max_attempts=1,
        input=WorkerInputRecord(
            path=source.resolve(),
            name=source.name,
            size_bytes=source.stat().st_size,
            sha256=SOURCE_SHA256,
        ),
        request=HybridWorkerRequestRecord(
            source_page_number=2,
            reason_codes=active_reasons,
            hybrid_backend="docling-fast",
            hybrid_mode="full",
            hybrid_url="http://127.0.0.1:5002",
            hybrid_timeout_ms=120000,
            hybrid_fallback=False,
        ),
        output=HybridWorkerOutputRecord(
            artifact_directory=raw.resolve(),
            page_count=2,
            total_file_count=2,
            markdown_character_count=len(markdown.read_text(encoding="utf-8")),
            markdown_broken_character_count=0,
            markdown=_artifact(markdown, raw),
            structured_json=_artifact(structured, raw),
            images=[],
            physical_image_file_count=0,
            unique_image_hash_count=0,
            duplicate_image_hash_group_count=0,
            image_hash_groups=[],
            requested_page_number=2,
            selected_page_count=1,
            observed_page_numbers=[2],
            unexpected_page_numbers=[],
            page_scope_verified=True,
            page_scope_evidence="STRUCTURED_PAGE_FIELDS",
        ),
    )
    manifest_path = root / "worker_result.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return OpenDataLoaderHybridPageRun(
        run_id=root.name,
        source_page_number=2,
        source_document_page_count=2,
        source_sha256=SOURCE_SHA256,
        reason_codes=tuple(active_reasons),
        run_directory=root,
        raw_artifact_directory=raw,
        result_manifest_path=manifest_path,
        stdout_log_path=root / "worker_stdout.log",
        stderr_log_path=root / "worker_stderr.log",
        manifest=manifest,
    )


def test_accepts_valid_hybrid_page_and_preserves_other_pages(tmp_path: Path) -> None:
    stage = HybridReverifyMergeStage(
        HybridReverificationConfig(output_root=tmp_path / "output")
    )
    base = _base_document()
    run = _hybrid_run(tmp_path)
    first = stage.reverify(base, _routing_plan(), [run])
    second = stage.reverify(base, _routing_plan(), [run])

    assert first.document.model_dump(mode="json") == second.document.model_dump(mode="json")
    assert first.report.model_dump(mode="json") == second.report.model_dump(mode="json")
    assert first.continuation_plan.model_dump(mode="json") == second.continuation_plan.model_dump(mode="json")
    assert first.report.accepted_page_numbers == [2]
    assert first.report.mineru_page_numbers == []
    assert first.report.next_action == "NORMALIZE"
    assert first.document.elements[0] == base.elements[0]
    assert first.document.elements[1].engine == EngineName.OPENDATALOADER_HYBRID
    assert "Replacement content" in first.document.elements[1].content
    assert EngineName.OPENDATALOADER_HYBRID in first.document.engine_chain


def test_rejects_candidate_without_bbox_and_retains_original_page(tmp_path: Path) -> None:
    stage = HybridReverifyMergeStage(
        HybridReverificationConfig(output_root=tmp_path / "output")
    )
    base = _base_document()
    outcome = stage.reverify(
        base,
        _routing_plan(),
        [_hybrid_run(tmp_path, include_bbox=False)],
    )

    assert outcome.document == base
    assert outcome.report.accepted_page_numbers == []
    assert outcome.report.mineru_page_numbers == [2]
    assert outcome.report.next_action == "MINERU_PAGE"
    assert outcome.continuation_plan.mineru_pages[0].max_attempts == 1
    assert "source_anchor_bbox_missing" in (
        outcome.report.page_results[0].quality_issue_reason_codes
    )


def test_unplanned_pilot_run_is_explicitly_gated(tmp_path: Path) -> None:
    stage = HybridReverifyMergeStage(
        HybridReverificationConfig(output_root=tmp_path / "output")
    )
    run = _hybrid_run(tmp_path, reason_codes=["stage6_1b_adapter_smoke"])

    with pytest.raises(ValueError, match="absent from routing plan"):
        stage.reverify(_base_document(), _routing_plan(hybrid_page=None), [run])

    outcome = stage.reverify(
        _base_document(),
        _routing_plan(hybrid_page=None),
        [run],
        allow_unplanned_pilot_runs=True,
    )
    result = outcome.report.page_results[0]
    assert result.route_planned is False
    assert result.route_reason_match is False
    assert result.passed is True


def test_loads_hybrid_run_and_commits_integrity_checked_result(tmp_path: Path) -> None:
    original_run = _hybrid_run(tmp_path)
    loaded_run = load_hybrid_page_run(original_run.run_directory)
    stage = HybridReverifyMergeStage(
        HybridReverificationConfig(output_root=tmp_path / "commits")
    )
    bundle = stage.execute(
        _base_document(),
        _routing_plan(),
        [loaded_run],
        input_extraction_run_id="stage5-test",
        input_extraction_commit_directory=tmp_path / "stage5-commit",
        input_extraction_manifest_sha256="b" * 64,
        run_id="stage6-1c-test",
    )
    reloaded = load_hybrid_reverification_commit(bundle.commit_directory)

    assert reloaded.stage_result.status == StageStatus.PASS
    assert reloaded.manifest.next_action == "NORMALIZE"
    assert reloaded.manifest.artifact_count == 4
    assert len(list(reloaded.commit_directory.iterdir())) == 5


def test_hybrid_loader_rejects_raw_artifact_tampering(tmp_path: Path) -> None:
    run = _hybrid_run(tmp_path)
    run.manifest.output.markdown.path.write_text("tampered", encoding="utf-8")

    with pytest.raises(HybridReverificationIntegrityError, match="mismatch"):
        load_hybrid_page_run(run.run_directory)
