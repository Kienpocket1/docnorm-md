from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from pathlib import Path

from rag_pipeline.stores import (
    load_models_jsonl,
    write_models_jsonl,
)

from rag_pipeline.contracts import (
    CatalogEntry,
    EngineName,
    FailureDomain,
    FileFormat,
    QuarantineRecord,
    QuarantineType,
    ResumeFrom,
    RouteDecision,
    RouteType,
    VersionAction,
    VersionAssignment,
)


class RouterError(ValueError):
    """Raised when an assignment cannot be routed."""


@dataclass(frozen=True, slots=True)
class RoutingBatchResult:
    routes_to_extract: list[RouteDecision]
    unsupported_routes: list[RouteDecision]
    reused_assignments: list[VersionAssignment]
    routing_quarantine: list[QuarantineRecord]

    @property
    def all_routes(self) -> list[RouteDecision]:
        return [
            *self.routes_to_extract,
            *self.unsupported_routes,
        ]


def _create_route_id(
    entry: CatalogEntry,
    assignment: VersionAssignment,
    route_type: RouteType,
) -> str:
    material = "|".join(
        [
            entry.catalog_id,
            assignment.version_id,
            route_type.value,
        ]
    )

    return f"route-{uuid5(NAMESPACE_URL, material).hex}"


def route_catalog_entry(
    entry: CatalogEntry,
    assignment: VersionAssignment,
) -> RouteDecision:
    if assignment.catalog_id != entry.catalog_id:
        raise RouterError(
            "Assignment/catalog ID mismatch"
        )

    if assignment.source_sha256 != entry.source_sha256:
        raise RouterError(
            "Assignment/catalog SHA256 mismatch"
        )

    route_mapping: dict[
        FileFormat,
        tuple[
            RouteType,
            EngineName,
            list[EngineName],
            str,
        ],
    ] = {
        FileFormat.PDF: (
            RouteType.PDF_OPENDATALOADER,
            EngineName.OPENDATALOADER_LOCAL,
            [
                EngineName.OPENDATALOADER_HYBRID,
                EngineName.MINERU,
            ],
            "pdf_local_first",
        ),
        FileFormat.DOCX: (
            RouteType.DOCX_NATIVE,
            EngineName.DOCX_NATIVE,
            [],
            "docx_native_extraction",
        ),
        FileFormat.DOC: (
            RouteType.DOC_CONVERT,
            EngineName.LIBREOFFICE,
            [EngineName.MANUAL],
            "legacy_doc_local_conversion",
        ),
        FileFormat.IMAGE: (
            RouteType.IMAGE_VISUAL,
            EngineName.QWEN3_VL,
            [EngineName.MANUAL],
            "standalone_image_visual_extraction",
        ),
        FileFormat.UNKNOWN: (
            RouteType.UNSUPPORTED,
            EngineName.MANUAL,
            [],
            "unsupported_file_format",
        ),
    }

    (
        route_type,
        primary_engine,
        fallback_engines,
        reason_code,
    ) = route_mapping[entry.file_format]

    return RouteDecision(
        route_id=_create_route_id(
            entry=entry,
            assignment=assignment,
            route_type=route_type,
        ),
        catalog_id=entry.catalog_id,
        version_id=assignment.version_id,
        file_format=entry.file_format,
        route_type=route_type,
        primary_engine=primary_engine,
        fallback_engines=fallback_engines,
        reason_code=reason_code,
    )


def create_routing_quarantine(
    route: RouteDecision,
    assignment: VersionAssignment,
) -> QuarantineRecord:
    if route.route_type != RouteType.UNSUPPORTED:
        raise RouterError(
            "Only unsupported routes enter "
            "routing quarantine"
        )

    material = f"routing-quarantine:{route.route_id}"

    quarantine_id = (
        f"qr-{uuid5(NAMESPACE_URL, material).hex}"
    )

    return QuarantineRecord(
        quarantine_id=quarantine_id,
        queue=QuarantineType.EXTRACTION,
        failure_domain=FailureDomain.ROUTING,
        root_cause_stage=ResumeFrom.ROUTE,
        reason_code=route.reason_code,
        resume_from=ResumeFrom.ROUTE,
        attempt_count=0,
        max_attempts=0,
        catalog_id=route.catalog_id,
        family_id=assignment.family_id,
        version_id=assignment.version_id,
        details={
            "route_id": route.route_id,
            "file_format": route.file_format.value,
        },
    )


def route_assignments(
    catalog_entries: list[CatalogEntry],
    assignments: list[VersionAssignment],
) -> RoutingBatchResult:
    catalog_by_id: dict[str, CatalogEntry] = {}

    for entry in catalog_entries:
        if entry.catalog_id in catalog_by_id:
            raise RouterError(
                f"Duplicate catalog_id: {entry.catalog_id}"
            )

        catalog_by_id[entry.catalog_id] = entry

    assignment_catalog_ids: set[str] = set()

    routes_to_extract: list[RouteDecision] = []
    unsupported_routes: list[RouteDecision] = []
    reused_assignments: list[VersionAssignment] = []
    routing_quarantine: list[QuarantineRecord] = []

    for assignment in assignments:
        if assignment.catalog_id in assignment_catalog_ids:
            raise RouterError(
                "Multiple assignments for catalog_id: "
                f"{assignment.catalog_id}"
            )

        assignment_catalog_ids.add(
            assignment.catalog_id
        )

        entry = catalog_by_id.get(
            assignment.catalog_id
        )

        if entry is None:
            raise RouterError(
                "Assignment references unknown catalog_id: "
                f"{assignment.catalog_id}"
            )

        if (
            assignment.action
            == VersionAction.REUSE_VERSION
        ):
            reused_assignments.append(assignment)
            continue

        route = route_catalog_entry(
            entry=entry,
            assignment=assignment,
        )

        if route.route_type == RouteType.UNSUPPORTED:
            unsupported_routes.append(route)
            routing_quarantine.append(
                create_routing_quarantine(
                    route=route,
                    assignment=assignment,
                )
            )
            continue

        routes_to_extract.append(route)

    return RoutingBatchResult(
        routes_to_extract=routes_to_extract,
        unsupported_routes=unsupported_routes,
        reused_assignments=reused_assignments,
        routing_quarantine=routing_quarantine,
    )

def write_route_decisions_manifest(
    records: list[RouteDecision],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        records,
        output_path,
    )


def load_route_decisions_manifest(
    input_path: str | Path,
) -> list[RouteDecision]:
    return load_models_jsonl(
        input_path,
        RouteDecision,
    )


def write_routing_quarantine_manifest(
    records: list[QuarantineRecord],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        records,
        output_path,
    )


def load_routing_quarantine_manifest(
    input_path: str | Path,
) -> list[QuarantineRecord]:
    return load_models_jsonl(
        input_path,
        QuarantineRecord,
    )