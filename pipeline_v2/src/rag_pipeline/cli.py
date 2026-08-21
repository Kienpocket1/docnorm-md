from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from rag_pipeline.stages.catalog import (
    CatalogError,
    load_catalog_manifest,
    scan_data_root,
    write_catalog_manifest,
)

from rag_pipeline.stages.resolver import (
    DEFAULT_FUZZY_REVIEW_THRESHOLD,
    apply_resolver_reviews,
    load_resolution_manifest,
    load_resolver_index_manifest,
    load_resolver_review_manifest,
    partition_resolution_decisions,
    resolve_catalog_entries,
    write_resolution_manifest,
    write_resolver_quarantine_manifest,
    load_resolution_manifest,
)

from rag_pipeline.stages.version_manager import (
    assign_versions,
    build_resolver_index_candidate,
    load_family_registry_manifest,
    load_resolver_index_candidate_manifest,
    load_version_registry_manifest,
    write_family_registry_manifest,
    write_resolver_index_candidate_manifest,
    write_version_assignments_manifest,
    write_version_registry_manifest,
    load_version_assignments_manifest,
    write_version_assignments_manifest,
)

from rag_pipeline.stages.router import (
    load_route_decisions_manifest,
    route_assignments,
    write_route_decisions_manifest,
    write_routing_quarantine_manifest,
)

from rag_pipeline.stores import JsonlStoreError


app = typer.Typer(
    name="rag-pipeline",
    help="Local document preparation pipeline.",
    no_args_is_help=True,
    add_completion=False,
)

@app.callback()
def main() -> None:
    """Internal RAG document preparation commands."""

@app.command("catalog")
def catalog_command(
    data_root: Annotated[
        Path,
        typer.Option(
            "--data-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Canonical root containing raw data.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
            help="Destination catalog JSONL manifest.",
        ),
    ],
    scan_root: Annotated[
        Path | None,
        typer.Option(
            "--scan-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            help="Optional subfolder to scan.",
        ),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            "--recursive/--no-recursive",
            help="Scan subfolders recursively.",
        ),
    ] = True,
) -> None:
    try:
        entries = scan_data_root(
            data_root=data_root,
            scan_root=scan_root,
            recursive=recursive,
        )

        if not entries:
            raise CatalogError(
                "No supported source files were found"
            )

        manifest_path = write_catalog_manifest(
            entries=entries,
            output_path=output,
        )
    except CatalogError as error:
        typer.secho(
            f"Catalog failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    format_counts = Counter(
        entry.file_format.value
        for entry in entries
    )

    typer.secho(
        "Catalog completed.",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Documents: {len(entries)}")

    for file_format, count in sorted(
        format_counts.items()
    ):
        typer.echo(f"{file_format}: {count}")

    typer.echo(f"Manifest: {manifest_path}")

@app.command("resolve")
def resolve_command(
    catalog_manifest: Annotated[
        Path,
        typer.Option(
            "--catalog-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Catalog JSONL produced by Catalog stage.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory for Resolver artifacts.",
        ),
    ],
    resolver_index: Annotated[
        Path | None,
        typer.Option(
            "--resolver-index",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional index of previously known versions.",
        ),
    ] = None,
    fuzzy_review_threshold: Annotated[
        float,
        typer.Option(
            "--fuzzy-review-threshold",
            min=0,
            max=100,
            help=(
                "Fuzzy score that requires manual review."
            ),
        ),
    ] = DEFAULT_FUZZY_REVIEW_THRESHOLD,
) -> None:
    try:
        catalog_entries = load_catalog_manifest(
            catalog_manifest
        )

        if not catalog_entries:
            raise ValueError(
                "Catalog manifest contains no documents"
            )

        references = (
            load_resolver_index_manifest(
                resolver_index
            )
            if resolver_index is not None
            else []
        )

        decisions = resolve_catalog_entries(
            entries=catalog_entries,
            references=references,
            fuzzy_review_threshold=(
                fuzzy_review_threshold
            ),
        )

        routing = partition_resolution_decisions(
            decisions
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        decisions_path = (
            output_dir / "resolver_decisions.jsonl"
        )
        continue_path = (
            output_dir / "continue_to_version.jsonl"
        )
        quarantine_path = (
            output_dir / "resolver_quarantine.jsonl"
        )
        rejected_path = (
            output_dir / "terminal_rejected.jsonl"
        )

        write_resolution_manifest(
            decisions,
            decisions_path,
        )
        write_resolution_manifest(
            routing.continue_to_version,
            continue_path,
        )
        write_resolver_quarantine_manifest(
            routing.resolver_quarantine,
            quarantine_path,
        )
        write_resolution_manifest(
            routing.terminal_rejected,
            rejected_path,
        )
    except (
        CatalogError,
        JsonlStoreError,
        ValueError,
    ) as error:
        typer.secho(
            f"Resolver failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    outcome_counts = Counter(
        decision.outcome.value
        for decision in decisions
    )

    typer.secho(
        "Resolver completed.",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Catalog documents: {len(catalog_entries)}"
    )
    typer.echo(
        f"Known index entries: {len(references)}"
    )

    for outcome, count in sorted(
        outcome_counts.items()
    ):
        typer.echo(f"{outcome}: {count}")

    typer.echo(
        "Continue to Version Manager: "
        f"{len(routing.continue_to_version)}"
    )
    typer.echo(
        "Resolver quarantine: "
        f"{len(routing.resolver_quarantine)}"
    )
    typer.echo(
        "Terminal rejected: "
        f"{len(routing.terminal_rejected)}"
    )
    typer.echo(f"Output directory: {output_dir}")

@app.command("resolve-review")
def resolve_review_command(
    decisions_manifest: Annotated[
        Path,
        typer.Option(
            "--decisions-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Original Resolver decisions JSONL.",
        ),
    ],
    reviews_manifest: Annotated[
        Path,
        typer.Option(
            "--reviews-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Human Resolver reviews JSONL.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory for reviewed Resolver output.",
        ),
    ],
) -> None:
    try:
        original_decisions = load_resolution_manifest(
            decisions_manifest
        )
        reviews = load_resolver_review_manifest(
            reviews_manifest
        )

        if not original_decisions:
            raise ValueError(
                "Decision manifest is empty"
            )

        if not reviews:
            raise ValueError(
                "Review manifest is empty"
            )

        final_decisions = apply_resolver_reviews(
            decisions=original_decisions,
            reviews=reviews,
        )

        routing = partition_resolution_decisions(
            final_decisions
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_resolution_manifest(
            final_decisions,
            output_dir / "reviewed_decisions.jsonl",
        )
        write_resolution_manifest(
            routing.continue_to_version,
            output_dir / "continue_to_version.jsonl",
        )
        write_resolver_quarantine_manifest(
            routing.resolver_quarantine,
            output_dir / "resolver_quarantine.jsonl",
        )
        write_resolution_manifest(
            routing.terminal_rejected,
            output_dir / "terminal_rejected.jsonl",
        )
    except (
        JsonlStoreError,
        ValueError,
    ) as error:
        typer.secho(
            f"Resolver review failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    outcome_counts = Counter(
        decision.outcome.value
        for decision in final_decisions
    )

    typer.secho(
        "Resolver review completed.",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Original decisions: {len(original_decisions)}"
    )
    typer.echo(f"Reviews applied: {len(reviews)}")

    for outcome, count in sorted(
        outcome_counts.items()
    ):
        typer.echo(f"{outcome}: {count}")

    typer.echo(
        "Continue to Version Manager: "
        f"{len(routing.continue_to_version)}"
    )
    typer.echo(
        "Remaining Resolver quarantine: "
        f"{len(routing.resolver_quarantine)}"
    )
    typer.echo(
        "Terminal rejected: "
        f"{len(routing.terminal_rejected)}"
    )

@app.command("version")
def version_command(
    catalog_manifest: Annotated[
        Path,
        typer.Option(
            "--catalog-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Catalog JSONL manifest.",
        ),
    ],
    decisions_manifest: Annotated[
        Path,
        typer.Option(
            "--decisions-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help=(
                "Resolver continue_to_version JSONL."
            ),
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory for staged version state.",
        ),
    ],
    family_registry: Annotated[
        Path | None,
        typer.Option(
            "--family-registry",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional committed family registry.",
        ),
    ] = None,
    version_registry: Annotated[
        Path | None,
        typer.Option(
            "--version-registry",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional committed version registry.",
        ),
    ] = None,
    resolver_index: Annotated[
        Path | None,
        typer.Option(
            "--resolver-index",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional committed Resolver Index.",
        ),
    ] = None,
) -> None:
    try:
        catalog_entries = load_catalog_manifest(
            catalog_manifest
        )
        decisions = load_resolution_manifest(
            decisions_manifest
        )

        if not catalog_entries:
            raise ValueError(
                "Catalog manifest is empty"
            )

        if not decisions:
            raise ValueError(
                "Version input decision manifest is empty"
            )

        existing_families = (
            load_family_registry_manifest(
                family_registry
            )
            if family_registry is not None
            else []
        )

        existing_versions = (
            load_version_registry_manifest(
                version_registry
            )
            if version_registry is not None
            else []
        )

        existing_resolver_index = (
            load_resolver_index_candidate_manifest(
                resolver_index
            )
            if resolver_index is not None
            else []
        )

        result = assign_versions(
            catalog_entries=catalog_entries,
            decisions=decisions,
            existing_families=existing_families,
            existing_versions=existing_versions,
        )

        resolver_index_candidate = (
            build_resolver_index_candidate(
                catalog_entries=catalog_entries,
                assignments=result.assignments,
                existing_index=(
                    existing_resolver_index
                ),
            )
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_family_registry_manifest(
            result.families,
            output_dir
            / "family_registry_candidate.jsonl",
        )
        write_version_registry_manifest(
            result.versions,
            output_dir
            / "version_registry_candidate.jsonl",
        )
        write_version_assignments_manifest(
            result.assignments,
            output_dir
            / "version_assignments.jsonl",
        )
        write_family_registry_manifest(
            result.created_families,
            output_dir
            / "created_families.jsonl",
        )
        write_version_registry_manifest(
            result.created_versions,
            output_dir
            / "created_versions.jsonl",
        )
        write_resolver_index_candidate_manifest(
            resolver_index_candidate,
            output_dir
            / "resolver_index_candidate.jsonl",
        )
    except (
        CatalogError,
        JsonlStoreError,
        ValueError,
    ) as error:
        typer.secho(
            f"Version Manager failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    action_counts = Counter(
        assignment.action.value
        for assignment in result.assignments
    )

    typer.secho(
        "Version Manager completed.",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Catalog documents: {len(catalog_entries)}"
    )
    typer.echo(
        f"Assignments: {len(result.assignments)}"
    )
    typer.echo(
        f"Families in candidate: {len(result.families)}"
    )
    typer.echo(
        f"Versions in candidate: {len(result.versions)}"
    )
    typer.echo(
        f"Created families: "
        f"{len(result.created_families)}"
    )
    typer.echo(
        f"Created versions: "
        f"{len(result.created_versions)}"
    )

    for action, count in sorted(
        action_counts.items()
    ):
        typer.echo(f"{action}: {count}")

    typer.echo(
        "State: CANDIDATE_ONLY_NOT_COMMITTED"
    )
    typer.echo(f"Output directory: {output_dir}")

@app.command("route")
def route_command(
    catalog_manifest: Annotated[
        Path,
        typer.Option(
            "--catalog-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Catalog JSONL manifest.",
        ),
    ],
    assignments_manifest: Annotated[
        Path,
        typer.Option(
            "--assignments-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Version assignments JSONL.",
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory for Router artifacts.",
        ),
    ],
) -> None:
    try:
        catalog_entries = load_catalog_manifest(
            catalog_manifest
        )
        assignments = load_version_assignments_manifest(
            assignments_manifest
        )

        if not catalog_entries:
            raise ValueError(
                "Catalog manifest is empty"
            )

        if not assignments:
            raise ValueError(
                "Version assignments manifest is empty"
            )

        routing = route_assignments(
            catalog_entries=catalog_entries,
            assignments=assignments,
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_route_decisions_manifest(
            routing.all_routes,
            output_dir / "route_decisions.jsonl",
        )
        write_route_decisions_manifest(
            routing.routes_to_extract,
            output_dir / "routes_to_extract.jsonl",
        )
        write_route_decisions_manifest(
            routing.unsupported_routes,
            output_dir / "unsupported_routes.jsonl",
        )
        write_version_assignments_manifest(
            routing.reused_assignments,
            output_dir / "reused_assignments.jsonl",
        )
        write_routing_quarantine_manifest(
            routing.routing_quarantine,
            output_dir / "routing_quarantine.jsonl",
        )
    except (
        CatalogError,
        JsonlStoreError,
        ValueError,
    ) as error:
        typer.secho(
            f"Router failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    route_counts = Counter(
        route.route_type.value
        for route in routing.all_routes
    )

    typer.secho(
        "Router completed.",
        fg=typer.colors.GREEN,
    )
    typer.echo(
        f"Assignments: {len(assignments)}"
    )
    typer.echo(
        "Routes to extract: "
        f"{len(routing.routes_to_extract)}"
    )
    typer.echo(
        "Reused versions: "
        f"{len(routing.reused_assignments)}"
    )
    typer.echo(
        "Unsupported routes: "
        f"{len(routing.unsupported_routes)}"
    )
    typer.echo(
        "Routing quarantine: "
        f"{len(routing.routing_quarantine)}"
    )

    for route_type, count in sorted(
        route_counts.items()
    ):
        typer.echo(f"{route_type}: {count}")

    typer.echo(f"Output directory: {output_dir}")
