from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_pipeline.contracts import EngineName, FailureDomain
from rag_pipeline.failure import (
    ExtractionRoutingPlan,
    FallbackAction,
    FallbackStep,
    RouteKind,
    ScopeRoutePlan,
)
from rag_pipeline.stages.opendataloader_hybrid_fallback import (
    OpenDataLoaderHybridFallbackStage,
)


class FakeHybridExtractor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def extract(self, input_pdf, **kwargs):
        self.calls.append(
            {"input_pdf": Path(input_pdf), **kwargs}
        )
        return SimpleNamespace(
            run_id=kwargs["run_id"],
            source_page_number=kwargs["page_number"],
        )


def hybrid_route(
    route_id: str,
    *,
    page_number: int,
    reason_codes: list[str],
    max_attempts: int = 1,
) -> ScopeRoutePlan:
    return ScopeRoutePlan(
        route_id=route_id,
        scope_id=f"page-{page_number}",
        page_number=page_number,
        route_kind=RouteKind.TEXT_TABLE,
        reason_codes=reason_codes,
        failure_domains=[FailureDomain.TABLE],
        steps=[
            FallbackStep(
                ordinal=1,
                action=FallbackAction.ODL_HYBRID,
                max_attempts=max_attempts,
                engine=EngineName.OPENDATALOADER_HYBRID,
                condition="local_page_failed",
            )
        ],
        fallback_required=True,
        external_model_required=False,
    )


def routing_plan(routes: list[ScopeRoutePlan]) -> ExtractionRoutingPlan:
    return ExtractionRoutingPlan(
        router_version="2026-08-18-6.1A.1",
        routes=routes,
        fallback_required=any(route.fallback_required for route in routes),
        next_action="RUN_PAGE_FALLBACKS",
        metrics={},
    )


def test_stage_executes_each_flagged_page_once_in_page_order() -> None:
    extractor = FakeHybridExtractor()
    stage = OpenDataLoaderHybridFallbackStage(extractor)
    plan = routing_plan(
        [
            hybrid_route(
                "route-page-5-b",
                page_number=5,
                reason_codes=["table_shape_invalid"],
            ),
            hybrid_route(
                "route-page-2",
                page_number=2,
                reason_codes=["empty_page"],
            ),
            hybrid_route(
                "route-page-5-a",
                page_number=5,
                reason_codes=["text_too_short"],
            ),
        ]
    )

    result = stage.execute(
        "source.pdf",
        plan,
        execution_id="stage6_1b_test",
    )

    assert result.page_numbers == (2, 5)
    assert result.next_action == "REVERIFY"
    assert [call["page_number"] for call in extractor.calls] == [2, 5]
    assert extractor.calls[1]["reason_codes"] == [
        "table_shape_invalid",
        "text_too_short",
    ]
    assert extractor.calls[1]["run_id"] == (
        "stage6_1b_test_page_0005"
    )


def test_stage_does_nothing_when_plan_has_no_hybrid_route() -> None:
    extractor = FakeHybridExtractor()
    stage = OpenDataLoaderHybridFallbackStage(extractor)
    pass_route = ScopeRoutePlan(
        route_id="route-pass",
        scope_id="page-1",
        page_number=1,
        route_kind=RouteKind.PASS,
        reason_codes=[],
        failure_domains=[],
        steps=[
            FallbackStep(
                ordinal=1,
                action=FallbackAction.PASS,
                max_attempts=0,
                engine=None,
                condition="quality_gate_passed",
            )
        ],
        fallback_required=False,
        external_model_required=False,
    )

    result = stage.execute(
        "source.pdf",
        routing_plan([pass_route]),
        execution_id="stage6_1b_noop",
    )

    assert result.attempts == ()
    assert result.next_action == "NO_HYBRID_REQUIRED"
    assert extractor.calls == []


def test_stage_rejects_retry_cap_above_one() -> None:
    stage = OpenDataLoaderHybridFallbackStage(FakeHybridExtractor())

    with pytest.raises(ValueError, match="retry cap"):
        stage.execute(
            "source.pdf",
            routing_plan(
                [
                    hybrid_route(
                        "route-invalid-cap",
                        page_number=1,
                        reason_codes=["empty_page"],
                        max_attempts=2,
                    )
                ]
            ),
            execution_id="stage6_1b_invalid",
        )
