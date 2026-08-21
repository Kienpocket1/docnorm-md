from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from rag_pipeline.contracts import EngineName
from rag_pipeline.extractors.opendataloader_hybrid import (
    OpenDataLoaderHybridPageRun,
)
from rag_pipeline.failure import (
    ODL_HYBRID_MAX_ATTEMPTS,
    ExtractionRoutingPlan,
    FallbackAction,
    RouteKind,
)


HYBRID_FALLBACK_STAGE_VERSION = "2026-08-18-6.1B.1"


class HybridPageExtractor(Protocol):
    def extract(
        self,
        input_pdf: Path | str,
        *,
        page_number: int,
        reason_codes: list[str] | tuple[str, ...],
        run_id: str | None = None,
    ) -> OpenDataLoaderHybridPageRun: ...


@dataclass(frozen=True, slots=True)
class HybridFallbackPageAttempt:
    page_number: int
    route_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    run: OpenDataLoaderHybridPageRun


@dataclass(frozen=True, slots=True)
class HybridFallbackExecution:
    execution_id: str
    routing_plan_router_version: str
    attempts: tuple[HybridFallbackPageAttempt, ...]
    next_action: str

    @property
    def page_numbers(self) -> tuple[int, ...]:
        return tuple(attempt.page_number for attempt in self.attempts)


def utc_execution_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    return f"stage6_1b_{timestamp}"


class OpenDataLoaderHybridFallbackStage:
    """Execute only the ODL Hybrid step from a Stage 6.1A plan."""

    def __init__(self, extractor: HybridPageExtractor) -> None:
        self.extractor = extractor

    def execute(
        self,
        input_pdf: Path | str,
        routing_plan: ExtractionRoutingPlan,
        *,
        execution_id: str | None = None,
    ) -> HybridFallbackExecution:
        active_execution_id = execution_id or utc_execution_id()
        pages = self._select_pages(routing_plan)
        attempts: list[HybridFallbackPageAttempt] = []

        for page_number, route_ids, reason_codes in pages:
            run = self.extractor.extract(
                input_pdf,
                page_number=page_number,
                reason_codes=reason_codes,
                run_id=(
                    f"{active_execution_id}_page_{page_number:04d}"
                ),
            )
            attempts.append(
                HybridFallbackPageAttempt(
                    page_number=page_number,
                    route_ids=tuple(route_ids),
                    reason_codes=tuple(reason_codes),
                    run=run,
                )
            )

        return HybridFallbackExecution(
            execution_id=active_execution_id,
            routing_plan_router_version=routing_plan.router_version,
            attempts=tuple(attempts),
            next_action=("REVERIFY" if attempts else "NO_HYBRID_REQUIRED"),
        )

    @staticmethod
    def _select_pages(
        routing_plan: ExtractionRoutingPlan,
    ) -> list[tuple[int, list[str], list[str]]]:
        pages: dict[int, dict[str, set[str]]] = {}

        for route in routing_plan.routes:
            hybrid_steps = [
                step
                for step in route.steps
                if step.action == FallbackAction.ODL_HYBRID
            ]

            if not hybrid_steps:
                continue

            if not route.fallback_required:
                raise ValueError(
                    "ODL Hybrid route must require fallback"
                )

            if route.route_kind != RouteKind.TEXT_TABLE:
                raise ValueError(
                    "ODL Hybrid may only execute for TEXT_TABLE routes"
                )

            if route.page_number is None:
                raise ValueError(
                    "ODL Hybrid route requires a PDF page number"
                )

            if len(hybrid_steps) != 1:
                raise ValueError(
                    "Each route must contain exactly one ODL Hybrid step"
                )

            hybrid_step = hybrid_steps[0]

            if hybrid_step.max_attempts != ODL_HYBRID_MAX_ATTEMPTS:
                raise ValueError("ODL Hybrid retry cap must equal one")

            if hybrid_step.engine != EngineName.OPENDATALOADER_HYBRID:
                raise ValueError("ODL Hybrid route engine is inconsistent")

            page = pages.setdefault(
                route.page_number,
                {"route_ids": set(), "reason_codes": set()},
            )
            page["route_ids"].add(route.route_id)
            page["reason_codes"].update(route.reason_codes)

        return [
            (
                page_number,
                sorted(values["route_ids"]),
                sorted(values["reason_codes"]),
            )
            for page_number, values in sorted(pages.items())
        ]
