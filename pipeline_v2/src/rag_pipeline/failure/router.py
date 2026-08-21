from __future__ import annotations

import hashlib
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.contracts import EngineName, FailureDomain
from rag_pipeline.validators.extraction import DocumentPageQualityResult
from rag_pipeline.validators.graph import NarrativeCoverageResult


FAILURE_ROUTER_VERSION = "2026-08-18-6.1A.1"
ODL_HYBRID_MAX_ATTEMPTS = 1
MINERU_MAX_ATTEMPTS = 1
QWEN_MAX_ATTEMPTS = 2
MAX_PROVENANCE_REPAIR_ATTEMPTS = 2


class FallbackAction(StrEnum):
    PASS = "PASS"
    NARRATIVE_SUBSTITUTION = "NARRATIVE_SUBSTITUTION"
    ODL_HYBRID = "ODL_HYBRID"
    MINERU_PAGE = "MINERU_PAGE"
    QWEN_FLOWCHART = "QWEN_FLOWCHART"
    REVERIFY = "REVERIFY"
    GRAPH_VERIFY = "GRAPH_VERIFY"
    PROVENANCE_VERIFY = "PROVENANCE_VERIFY"
    PROVENANCE_REPAIR = "PROVENANCE_REPAIR"
    EXTRACTION_QUARANTINE = "EXTRACTION_QUARANTINE"


class RouteKind(StrEnum):
    PASS = "PASS"
    TEXT_TABLE = "TEXT_TABLE"
    VISUAL_NARRATIVE = "VISUAL_NARRATIVE"
    VISUAL_QWEN = "VISUAL_QWEN"
    GRAPH_TOPOLOGY = "GRAPH_TOPOLOGY"
    PROVENANCE = "PROVENANCE"


class FallbackStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ordinal: int = Field(ge=1)
    action: FallbackAction
    max_attempts: int = Field(ge=0)
    engine: EngineName | None = None
    condition: str


class ScopeRoutePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: str
    scope_id: str
    page_number: int | None = None
    route_kind: RouteKind
    reason_codes: list[str]
    failure_domains: list[FailureDomain]
    steps: list[FallbackStep]
    fallback_required: bool
    external_model_required: bool


class ExtractionRoutingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    router_version: str
    routes: list[ScopeRoutePlan]
    fallback_required: bool
    next_action: str
    metrics: dict[str, int | bool]


def _route_id(
    source_sha256: str,
    scope_id: str,
    route_kind: RouteKind,
    reason_codes: list[str],
) -> str:
    payload = "|".join(
        [
            source_sha256,
            scope_id,
            route_kind.value,
            ",".join(reason_codes),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"fallback-route-{digest}"


def _steps(
    values: list[tuple[FallbackAction, int, EngineName | None, str]],
) -> list[FallbackStep]:
    return [
        FallbackStep(
            ordinal=ordinal,
            action=action,
            max_attempts=max_attempts,
            engine=engine,
            condition=condition,
        )
        for ordinal, (
            action,
            max_attempts,
            engine,
            condition,
        ) in enumerate(values, start=1)
    ]


class FailureRouter:
    """Build deterministic per-scope fallback plans without executing them."""

    def route(
        self,
        assessment: DocumentPageQualityResult,
        *,
        source_sha256: str,
        narrative_results: list[NarrativeCoverageResult] | None = None,
    ) -> ExtractionRoutingPlan:
        narratives = narrative_results or []
        routes: list[ScopeRoutePlan] = []
        page_by_element = {
            element_id: result
            for result in assessment.page_results
            for element_id in result.element_ids
        }

        for scope in assessment.page_results:
            scope_ids = set(scope.element_ids)
            triggering_issues = [
                issue
                for issue in assessment.issues
                if issue.triggers_fallback
                and (
                    (
                        scope.page_number is not None
                        and scope.page_number in issue.page_numbers
                    )
                    or bool(scope_ids & set(issue.element_ids))
                    or (
                        not issue.page_numbers
                        and not issue.element_ids
                        and scope.scope_id == assessment.page_results[0].scope_id
                    )
                )
            ]
            structural = [
                issue
                for issue in triggering_issues
                if issue.failure_domain
                in {FailureDomain.TEXT, FailureDomain.TABLE}
            ]
            provenance = [
                issue
                for issue in triggering_issues
                if issue.failure_domain == FailureDomain.PROVENANCE
            ]
            graph = [
                issue
                for issue in triggering_issues
                if issue.reason_code == "graph_topology_invalid"
            ]

            if structural:
                routes.append(
                    self._structural_route(
                        source_sha256,
                        scope.scope_id,
                        scope.page_number,
                        structural,
                    )
                )

            if provenance:
                routes.append(
                    self._provenance_route(
                        source_sha256,
                        scope.scope_id,
                        scope.page_number,
                        provenance,
                    )
                )

            if graph:
                routes.append(
                    self._graph_route(
                        source_sha256,
                        scope.scope_id,
                        scope.page_number,
                        graph,
                    )
                )

        for narrative in narratives:
            scope = page_by_element.get(narrative.visual_element_id)
            scope_id = scope.scope_id if scope is not None else "document"
            page_number = scope.page_number if scope is not None else None
            routes.append(
                self._visual_route(
                    source_sha256,
                    scope_id,
                    page_number,
                    narrative,
                )
            )

        covered_scope_ids = {
            route.scope_id
            for route in routes
        }

        for scope in assessment.page_results:
            if scope.scope_id in covered_scope_ids:
                continue

            routes.append(
                ScopeRoutePlan(
                    route_id=_route_id(
                        source_sha256,
                        scope.scope_id,
                        RouteKind.PASS,
                        [],
                    ),
                    scope_id=scope.scope_id,
                    page_number=scope.page_number,
                    route_kind=RouteKind.PASS,
                    reason_codes=[],
                    failure_domains=[],
                    steps=_steps(
                        [
                            (
                                FallbackAction.PASS,
                                0,
                                None,
                                "quality_gate_passed",
                            )
                        ]
                    ),
                    fallback_required=False,
                    external_model_required=False,
                )
            )

        routes.sort(
            key=lambda route: (
                route.page_number is None,
                route.page_number or 0,
                route.scope_id,
                route.route_kind.value,
            )
        )
        fallback_required = any(
            route.fallback_required
            for route in routes
        )
        return ExtractionRoutingPlan(
            router_version=FAILURE_ROUTER_VERSION,
            routes=routes,
            fallback_required=fallback_required,
            next_action=(
                "RUN_PAGE_FALLBACKS"
                if fallback_required
                else "NORMALIZE"
            ),
            metrics={
                "route_count": len(routes),
                "pass_route_count": sum(
                    route.route_kind == RouteKind.PASS
                    for route in routes
                ),
                "fallback_route_count": sum(
                    route.fallback_required
                    for route in routes
                ),
                "odl_hybrid_route_count": self._count_action(
                    routes,
                    FallbackAction.ODL_HYBRID,
                ),
                "mineru_route_count": self._count_action(
                    routes,
                    FallbackAction.MINERU_PAGE,
                ),
                "qwen_route_count": self._count_action(
                    routes,
                    FallbackAction.QWEN_FLOWCHART,
                ),
                "narrative_substitution_count": self._count_action(
                    routes,
                    FallbackAction.NARRATIVE_SUBSTITUTION,
                ),
                "provenance_repair_route_count": self._count_action(
                    routes,
                    FallbackAction.PROVENANCE_REPAIR,
                ),
                "external_model_required": any(
                    route.external_model_required
                    for route in routes
                ),
            },
        )

    @staticmethod
    def _structural_route(
        source_sha256: str,
        scope_id: str,
        page_number: int | None,
        issues: list,
    ) -> ScopeRoutePlan:
        reasons = sorted({issue.reason_code for issue in issues})
        domains = sorted(
            {issue.failure_domain for issue in issues},
            key=lambda value: value.value,
        )
        return ScopeRoutePlan(
            route_id=_route_id(
                source_sha256,
                scope_id,
                RouteKind.TEXT_TABLE,
                reasons,
            ),
            scope_id=scope_id,
            page_number=page_number,
            route_kind=RouteKind.TEXT_TABLE,
            reason_codes=reasons,
            failure_domains=domains,
            steps=_steps(
                [
                    (
                        FallbackAction.ODL_HYBRID,
                        ODL_HYBRID_MAX_ATTEMPTS,
                        EngineName.OPENDATALOADER_HYBRID,
                        "local_page_failed",
                    ),
                    (
                        FallbackAction.REVERIFY,
                        1,
                        None,
                        "after_odl_hybrid",
                    ),
                    (
                        FallbackAction.MINERU_PAGE,
                        MINERU_MAX_ATTEMPTS,
                        EngineName.MINERU,
                        "odl_hybrid_reverify_failed",
                    ),
                    (
                        FallbackAction.REVERIFY,
                        1,
                        None,
                        "after_mineru_page",
                    ),
                    (
                        FallbackAction.EXTRACTION_QUARANTINE,
                        0,
                        None,
                        "mineru_reverify_failed",
                    ),
                ]
            ),
            fallback_required=True,
            external_model_required=False,
        )

    @staticmethod
    def _provenance_route(
        source_sha256: str,
        scope_id: str,
        page_number: int | None,
        issues: list,
    ) -> ScopeRoutePlan:
        reasons = sorted({issue.reason_code for issue in issues})
        return ScopeRoutePlan(
            route_id=_route_id(
                source_sha256,
                scope_id,
                RouteKind.PROVENANCE,
                reasons,
            ),
            scope_id=scope_id,
            page_number=page_number,
            route_kind=RouteKind.PROVENANCE,
            reason_codes=reasons,
            failure_domains=[FailureDomain.PROVENANCE],
            steps=_steps(
                [
                    (
                        FallbackAction.PROVENANCE_REPAIR,
                        MAX_PROVENANCE_REPAIR_ATTEMPTS,
                        None,
                        "source_anchor_validation_failed",
                    ),
                    (
                        FallbackAction.PROVENANCE_VERIFY,
                        1,
                        None,
                        "after_provenance_repair",
                    ),
                    (
                        FallbackAction.EXTRACTION_QUARANTINE,
                        0,
                        None,
                        "source_anchor_mapping_failed",
                    ),
                ]
            ),
            fallback_required=True,
            external_model_required=False,
        )

    @staticmethod
    def _graph_route(
        source_sha256: str,
        scope_id: str,
        page_number: int | None,
        issues: list,
    ) -> ScopeRoutePlan:
        reasons = sorted({issue.reason_code for issue in issues})
        return ScopeRoutePlan(
            route_id=_route_id(
                source_sha256,
                scope_id,
                RouteKind.GRAPH_TOPOLOGY,
                reasons,
            ),
            scope_id=scope_id,
            page_number=page_number,
            route_kind=RouteKind.GRAPH_TOPOLOGY,
            reason_codes=reasons,
            failure_domains=[FailureDomain.VISUAL],
            steps=_steps(
                [
                    (
                        FallbackAction.QWEN_FLOWCHART,
                        QWEN_MAX_ATTEMPTS,
                        EngineName.QWEN3_VL,
                        "graph_topology_invalid",
                    ),
                    (
                        FallbackAction.GRAPH_VERIFY,
                        1,
                        None,
                        "after_qwen_flowchart",
                    ),
                    (
                        FallbackAction.EXTRACTION_QUARANTINE,
                        0,
                        None,
                        "qwen_retry_cap_exhausted",
                    ),
                ]
            ),
            fallback_required=True,
            external_model_required=True,
        )

    @staticmethod
    def _visual_route(
        source_sha256: str,
        scope_id: str,
        page_number: int | None,
        narrative: NarrativeCoverageResult,
    ) -> ScopeRoutePlan:
        if narrative.sufficient:
            kind = RouteKind.VISUAL_NARRATIVE
            reasons = ["visual_narrative_coverage_sufficient"]
            steps = _steps(
                [
                    (
                        FallbackAction.NARRATIVE_SUBSTITUTION,
                        1,
                        None,
                        "narrative_coverage_verified",
                    ),
                    (
                        FallbackAction.PROVENANCE_VERIFY,
                        1,
                        None,
                        "after_narrative_substitution",
                    ),
                    (
                        FallbackAction.PASS,
                        0,
                        None,
                        "substitution_provenance_valid",
                    ),
                ]
            )
            fallback_required = False
            external_model_required = False
        else:
            kind = RouteKind.VISUAL_QWEN
            reasons = ["visual_narrative_coverage_insufficient"]
            steps = _steps(
                [
                    (
                        FallbackAction.QWEN_FLOWCHART,
                        QWEN_MAX_ATTEMPTS,
                        EngineName.QWEN3_VL,
                        "narrative_coverage_insufficient",
                    ),
                    (
                        FallbackAction.GRAPH_VERIFY,
                        1,
                        None,
                        "after_qwen_flowchart",
                    ),
                    (
                        FallbackAction.PROVENANCE_VERIFY,
                        1,
                        None,
                        "graph_topology_valid",
                    ),
                    (
                        FallbackAction.EXTRACTION_QUARANTINE,
                        0,
                        None,
                        "graph_or_provenance_invalid_after_retry",
                    ),
                ]
            )
            fallback_required = True
            external_model_required = True

        return ScopeRoutePlan(
            route_id=_route_id(
                source_sha256,
                scope_id,
                kind,
                reasons,
            ),
            scope_id=scope_id,
            page_number=page_number,
            route_kind=kind,
            reason_codes=reasons,
            failure_domains=[FailureDomain.VISUAL],
            steps=steps,
            fallback_required=fallback_required,
            external_model_required=external_model_required,
        )

    @staticmethod
    def _count_action(
        routes: list[ScopeRoutePlan],
        action: FallbackAction,
    ) -> int:
        return sum(
            any(step.action == action for step in route.steps)
            for route in routes
        )
