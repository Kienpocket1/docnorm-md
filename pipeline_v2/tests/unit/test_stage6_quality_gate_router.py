from __future__ import annotations

from pathlib import Path

from rag_pipeline.contracts import (
    BoundingBox,
    ElementType,
    EngineName,
    ExtractedDocument,
    ExtractionQualityIssue,
    FailureDomain,
    QualityDisposition,
    QualitySeverity,
    SourceAnchor,
    SourceElement,
    StageStatus,
)
from rag_pipeline.failure import (
    MAX_PROVENANCE_REPAIR_ATTEMPTS,
    MINERU_MAX_ATTEMPTS,
    ODL_HYBRID_MAX_ATTEMPTS,
    QWEN_MAX_ATTEMPTS,
    FailureRouter,
    FallbackAction,
    RouteKind,
)
from rag_pipeline.stages.extraction_quality import ExtractionQualityStage
from rag_pipeline.validators import (
    ExtractionPageValidator,
    GraphTopologyValidator,
    NarrativeCoverageValidator,
)


SOURCE_SHA256 = "a" * 64


def element(
    element_id: str,
    element_type: ElementType,
    content: str,
    *,
    page: int | None,
    block: int,
    metadata: dict | None = None,
    bbox: bool = True,
) -> SourceElement:
    return SourceElement(
        element_id=element_id,
        element_type=element_type,
        content=content,
        source_anchor=SourceAnchor(
            anchor_id=f"anchor-{element_id}",
            source_sha256=SOURCE_SHA256,
            page_number=page,
            block_index=block,
            bbox=(
                BoundingBox(
                    x0=10,
                    y0=10 + block * 10,
                    x1=500,
                    y1=40 + block * 10,
                )
                if bbox
                else None
            ),
            section_path=["Test"],
        ),
        engine=EngineName.OPENDATALOADER_LOCAL,
        metadata=metadata or {"indexable": True},
    )


def document(
    elements: list[SourceElement],
    *,
    page_count: int = 1,
) -> ExtractedDocument:
    return ExtractedDocument(
        catalog_id="catalog-stage6",
        family_id="family-stage6",
        version_id="family-stage6-v001",
        source_sha256=SOURCE_SHA256,
        page_count=page_count,
        engine_chain=[EngineName.OPENDATALOADER_LOCAL],
        elements=elements,
    )


def actions(outcome) -> set[FallbackAction]:
    return {
        step.action
        for route in outcome.routing_plan.routes
        for step in route.steps
    }


def test_clean_pages_pass_without_model_or_fallback() -> None:
    extracted = document(
        [
            element(
                "page-1-text",
                ElementType.PARAGRAPH,
                "Nội dung hợp lệ của trang thứ nhất.",
                page=1,
                block=0,
            ),
            element(
                "page-2-text",
                ElementType.PARAGRAPH,
                "Nội dung hợp lệ của trang thứ hai.",
                page=2,
                block=1,
            ),
        ],
        page_count=2,
    )
    outcome = ExtractionQualityStage().evaluate(extracted)

    assert outcome.report.disposition == QualityDisposition.PASS
    assert outcome.stage_result.status == StageStatus.PASS
    assert outcome.routing_plan.next_action == "NORMALIZE"
    assert outcome.routing_plan.fallback_required is False
    assert outcome.routing_plan.metrics["pass_route_count"] == 2
    assert outcome.routing_plan.metrics["external_model_required"] is False
    assert actions(outcome) == {FallbackAction.PASS}


def test_decorative_logo_and_existing_warnings_remain_accepted() -> None:
    extracted = document(
        [
            element(
                "body",
                ElementType.PARAGRAPH,
                "Nội dung nghiệp vụ đủ để lập chỉ mục.",
                page=1,
                block=0,
            ),
            element(
                "logo",
                ElementType.IMAGE,
                "",
                page=1,
                block=1,
                metadata={
                    "indexable": False,
                    "asset_classification": "DECORATIVE",
                    "asset_role": "BRAND_LOGO",
                },
            ),
        ]
    )
    warning = ExtractionQualityIssue(
        issue_id="quality-existing-logo-warning",
        reason_code="repeated_brand_logo",
        severity=QualitySeverity.WARNING,
        failure_domain=FailureDomain.VISUAL,
        message="Known logo is filtered",
        element_ids=["logo"],
        triggers_fallback=False,
    )
    outcome = ExtractionQualityStage().evaluate(
        extracted,
        inherited_issues=[warning],
    )

    assert (
        outcome.report.disposition
        == QualityDisposition.PASS_WITH_WARNINGS
    )
    assert outcome.routing_plan.fallback_required is False
    assert outcome.assessment.filtered_element_ids == ["logo"]
    assert outcome.assessment.pending_visual_element_ids == []
    assert FallbackAction.QWEN_FLOWCHART not in actions(outcome)


def test_empty_page_routes_hybrid_then_mineru_then_quarantine() -> None:
    extracted = document(
        [
            element(
                "page-1",
                ElementType.PARAGRAPH,
                "Trang một có nội dung hợp lệ.",
                page=1,
                block=0,
            )
        ],
        page_count=2,
    )
    outcome = ExtractionQualityStage().evaluate(extracted)
    empty_route = next(
        route
        for route in outcome.routing_plan.routes
        if route.page_number == 2
    )

    assert outcome.report.disposition == QualityDisposition.FALLBACK_REQUIRED
    assert empty_route.route_kind == RouteKind.TEXT_TABLE
    assert [step.action for step in empty_route.steps] == [
        FallbackAction.ODL_HYBRID,
        FallbackAction.REVERIFY,
        FallbackAction.MINERU_PAGE,
        FallbackAction.REVERIFY,
        FallbackAction.EXTRACTION_QUARANTINE,
    ]
    assert empty_route.steps[0].max_attempts == 1
    assert empty_route.steps[2].max_attempts == 1
    assert FallbackAction.QWEN_FLOWCHART not in {
        step.action for step in empty_route.steps
    }


def test_invalid_table_uses_text_table_fallback_only() -> None:
    extracted = document(
        [
            element(
                "bad-table",
                ElementType.TABLE,
                "| A | B |\n| --- | --- |\n| 1 | 2 |",
                page=1,
                block=0,
                metadata={
                    "indexable": True,
                    "declared_rows": 0,
                    "declared_columns": 2,
                    "actual_rows": 2,
                },
            )
        ]
    )
    outcome = ExtractionQualityStage().evaluate(extracted)

    assert FallbackAction.ODL_HYBRID in actions(outcome)
    assert FallbackAction.MINERU_PAGE in actions(outcome)
    assert FallbackAction.QWEN_FLOWCHART not in actions(outcome)


def test_visual_without_narrative_routes_qwen_never_mineru() -> None:
    extracted = document(
        [
            element(
                "flowchart",
                ElementType.FLOWCHART,
                "",
                page=1,
                block=0,
                metadata={
                    "indexable": False,
                    "narrative_coverage": "insufficient",
                },
            )
        ]
    )
    outcome = ExtractionQualityStage().evaluate(extracted)

    assert outcome.report.disposition == QualityDisposition.FALLBACK_REQUIRED
    assert FallbackAction.QWEN_FLOWCHART in actions(outcome)
    assert FallbackAction.GRAPH_VERIFY in actions(outcome)
    assert FallbackAction.MINERU_PAGE not in actions(outcome)
    assert outcome.routing_plan.metrics["external_model_required"] is True


def test_visual_with_narrative_uses_substitution_without_model() -> None:
    extracted = document(
        [
            element(
                "flowchart",
                ElementType.FLOWCHART,
                "",
                page=1,
                block=0,
                metadata={
                    "indexable": False,
                    "narrative_coverage": "sufficient",
                },
            )
        ]
    )
    outcome = ExtractionQualityStage().evaluate(extracted)

    assert outcome.report.disposition == QualityDisposition.PASS
    assert FallbackAction.NARRATIVE_SUBSTITUTION in actions(outcome)
    assert FallbackAction.QWEN_FLOWCHART not in actions(outcome)
    assert outcome.routing_plan.fallback_required is False
    assert outcome.routing_plan.next_action == "NORMALIZE"


def test_missing_pdf_bbox_routes_provenance_repair() -> None:
    extracted = document(
        [
            element(
                "missing-bbox",
                ElementType.PARAGRAPH,
                "Nội dung có anchor nhưng thiếu bounding box.",
                page=1,
                block=0,
                bbox=False,
            )
        ]
    )
    outcome = ExtractionQualityStage().evaluate(extracted)
    provenance_route = next(
        route
        for route in outcome.routing_plan.routes
        if route.route_kind == RouteKind.PROVENANCE
    )

    assert provenance_route.steps[0].action == (
        FallbackAction.PROVENANCE_REPAIR
    )
    assert provenance_route.steps[0].max_attempts == 2
    assert FallbackAction.MINERU_PAGE not in {
        step.action for step in provenance_route.steps
    }


def test_invalid_graph_routes_qwen_and_never_mineru() -> None:
    extracted = document(
        [
            element(
                "graph-element",
                ElementType.PARAGRAPH,
                "Graph extraction placeholder.",
                page=1,
                block=0,
            )
        ]
    )
    graph_result = GraphTopologyValidator().validate(
        {
            "nodes": [
                {"id": "N1", "type": "decision"},
                {"id": "N2", "type": "process"},
            ],
            "edges": [{"source": "N1", "target": "missing"}],
        },
        source_sha256=SOURCE_SHA256,
        page_number=1,
        element_id="graph-element",
    )
    assessment = ExtractionPageValidator().evaluate(
        extracted,
        inherited_issues=graph_result.issues,
    )
    plan = FailureRouter().route(
        assessment,
        source_sha256=SOURCE_SHA256,
    )
    route = next(
        route
        for route in plan.routes
        if route.route_kind == RouteKind.GRAPH_TOPOLOGY
    )

    assert graph_result.valid is False
    assert route.steps[0].action == FallbackAction.QWEN_FLOWCHART
    assert route.steps[0].max_attempts == 2
    assert FallbackAction.MINERU_PAGE not in {
        step.action for step in route.steps
    }


def test_narrative_coverage_can_be_inferred_from_nearby_text() -> None:
    extracted = document(
        [
            element(
                "flowchart",
                ElementType.FLOWCHART,
                "",
                page=1,
                block=0,
                metadata={"indexable": False},
            ),
            element(
                "narrative",
                ElementType.PARAGRAPH,
                (
                    "Lưu đồ mô tả quy trình gồm các bước kiểm tra. "
                    "Nếu hồ sơ đạt thì chuyển sang bước tiếp theo; "
                    "nếu không đạt thì trả lại để bổ sung."
                ),
                page=1,
                block=1,
            ),
        ]
    )
    result = NarrativeCoverageValidator().evaluate(
        extracted,
        "flowchart",
    )

    assert result.sufficient is True
    assert result.evidence_element_ids == ["narrative"]
    assert "lưu đồ" in result.matched_keywords
    assert "quy trình" in result.matched_keywords


def test_retry_caps_are_frozen_by_roadmap() -> None:
    assert ODL_HYBRID_MAX_ATTEMPTS == 1
    assert MINERU_MAX_ATTEMPTS == 1
    assert QWEN_MAX_ATTEMPTS == 2
    assert MAX_PROVENANCE_REPAIR_ATTEMPTS == 2
