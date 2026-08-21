from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from rag_pipeline.contracts import (
    CatalogEntry,
    FamilyRecord,
    PublicationState,
    ResolutionDecision,
    ResolutionOutcome,
    ResolverIndexEntry,
    VersionAction,
    VersionAssignment,
    VersionRecord,
)

from rag_pipeline.stages.resolver import (
    extract_document_code,
    normalize_document_name,
)

from rag_pipeline.stores import (
    load_models_jsonl,
    write_models_jsonl,
)

class VersionManagerError(ValueError):
    """Raised when version state is inconsistent."""


@dataclass(frozen=True, slots=True)
class VersionManagerResult:
    families: list[FamilyRecord]
    versions: list[VersionRecord]
    assignments: list[VersionAssignment]
    created_families: list[FamilyRecord]
    created_versions: list[VersionRecord]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_version_label(
    version_number: int,
) -> str:
    if version_number < 1:
        raise ValueError(
            "version_number must be greater than zero"
        )

    return f"v{version_number:03d}"


def create_version_id(
    family_id: str,
    version_number: int,
    source_sha256: str,
) -> str:
    version_label = create_version_label(
        version_number
    )

    return (
        f"{family_id}-"
        f"{version_label}-"
        f"{source_sha256[:12]}"
    )


def _create_assignment(
    *,
    decision: ResolutionDecision,
    entry: CatalogEntry,
    version: VersionRecord,
    action: VersionAction,
) -> VersionAssignment:
    material = "|".join(
        [
            decision.decision_id,
            version.version_id,
            action.value,
        ]
    )

    assignment_id = (
        f"va-{uuid5(NAMESPACE_URL, material).hex}"
    )

    return VersionAssignment(
        assignment_id=assignment_id,
        decision_id=decision.decision_id,
        catalog_id=entry.catalog_id,
        family_id=version.family_id,
        version_id=version.version_id,
        version_number=version.version_number,
        version_label=version.version_label,
        source_sha256=entry.source_sha256,
        action=action,
    )


def _validate_active_invariant(
    versions: list[VersionRecord],
) -> None:
    active_by_family: dict[
        str,
        list[VersionRecord],
    ] = {}

    for version in versions:
        if (
            version.publication_state
            == PublicationState.ACTIVE
        ):
            active_by_family.setdefault(
                version.family_id,
                [],
            ).append(version)

    invalid_families = {
        family_id: records
        for family_id, records
        in active_by_family.items()
        if len(records) > 1
    }

    if invalid_families:
        family_ids = ", ".join(
            sorted(invalid_families)
        )

        raise VersionManagerError(
            "Multiple ACTIVE versions detected for: "
            f"{family_ids}"
        )


def assign_versions(
    catalog_entries: list[CatalogEntry],
    decisions: list[ResolutionDecision],
    existing_families: list[FamilyRecord] | None = None,
    existing_versions: list[VersionRecord] | None = None,
) -> VersionManagerResult:
    families = list(existing_families or [])
    versions = list(existing_versions or [])

    catalog_by_id: dict[str, CatalogEntry] = {}

    for entry in catalog_entries:
        if entry.catalog_id in catalog_by_id:
            raise VersionManagerError(
                "Duplicate catalog_id: "
                f"{entry.catalog_id}"
            )

        catalog_by_id[entry.catalog_id] = entry

    decision_by_catalog: dict[
        str,
        ResolutionDecision,
    ] = {}

    for decision in decisions:
        if decision.catalog_id in decision_by_catalog:
            raise VersionManagerError(
                "Multiple decisions for catalog_id: "
                f"{decision.catalog_id}"
            )

        decision_by_catalog[
            decision.catalog_id
        ] = decision

    if set(catalog_by_id) != set(decision_by_catalog):
        missing_decisions = (
            set(catalog_by_id) - set(decision_by_catalog)
        )
        unknown_decisions = (
            set(decision_by_catalog) - set(catalog_by_id)
        )

        raise VersionManagerError(
            "Catalog/decision mismatch. "
            f"Missing={sorted(missing_decisions)}, "
            f"Unknown={sorted(unknown_decisions)}"
        )

    family_by_id: dict[str, FamilyRecord] = {}

    for family in families:
        if family.family_id in family_by_id:
            raise VersionManagerError(
                "Duplicate family_id: "
                f"{family.family_id}"
            )

        family_by_id[family.family_id] = family

    version_by_id: dict[str, VersionRecord] = {}
    version_by_sha: dict[str, VersionRecord] = {}
    versions_by_family: dict[
        str,
        list[VersionRecord],
    ] = {}

    for version in versions:
        if version.version_id in version_by_id:
            raise VersionManagerError(
                "Duplicate version_id: "
                f"{version.version_id}"
            )

        if version.family_id not in family_by_id:
            raise VersionManagerError(
                "Version references unknown family: "
                f"{version.family_id}"
            )

        existing_sha_version = version_by_sha.get(
            version.source_sha256
        )

        if existing_sha_version is not None:
            raise VersionManagerError(
                "Duplicate SHA256 in version registry: "
                f"{version.source_sha256}"
            )

        version_by_id[version.version_id] = version
        version_by_sha[version.source_sha256] = version
        versions_by_family.setdefault(
            version.family_id,
            [],
        ).append(version)

    _validate_active_invariant(versions)

    created_families: list[FamilyRecord] = []
    created_versions: list[VersionRecord] = []
    assignments: list[VersionAssignment] = []
    new_family_ids: set[str] = set()

    allowed_outcomes = {
        ResolutionOutcome.SAME_VERSION,
        ResolutionOutcome.NEW_VERSION,
        ResolutionOutcome.UNRELATED,
    }

    for decision in decisions:
        if decision.outcome not in allowed_outcomes:
            raise VersionManagerError(
                "Decision outcome cannot enter "
                f"Version Manager: {decision.outcome.value}"
            )

        entry = catalog_by_id[decision.catalog_id]

        if not decision.family_id:
            raise VersionManagerError(
                "Decision does not contain family_id"
            )

        family_id = decision.family_id

        # SAME_VERSION phải trỏ đúng version và SHA.
        if (
            decision.outcome
            == ResolutionOutcome.SAME_VERSION
        ):
            matched_version = version_by_id.get(
                decision.matched_version_id or ""
            )

            if matched_version is None:
                raise VersionManagerError(
                    "SAME_VERSION references unknown version"
                )

            if matched_version.family_id != family_id:
                raise VersionManagerError(
                    "SAME_VERSION family mismatch"
                )

            if (
                matched_version.source_sha256
                != entry.source_sha256
            ):
                raise VersionManagerError(
                    "SAME_VERSION SHA256 mismatch"
                )

            assignments.append(
                _create_assignment(
                    decision=decision,
                    entry=entry,
                    version=matched_version,
                    action=VersionAction.REUSE_VERSION,
                )
            )
            continue

        # Idempotency: SHA đã tồn tại trong cùng family.
        known_sha_version = version_by_sha.get(
            entry.source_sha256
        )

        if known_sha_version is not None:
            if known_sha_version.family_id != family_id:
                raise VersionManagerError(
                    "Identical SHA256 assigned to "
                    "different families"
                )

            assignments.append(
                _create_assignment(
                    decision=decision,
                    entry=entry,
                    version=known_sha_version,
                    action=VersionAction.REUSE_VERSION,
                )
            )
            continue

        family = family_by_id.get(family_id)
        created_family_now = False

        if decision.outcome == ResolutionOutcome.UNRELATED:
            if family is None:
                family = FamilyRecord(
                    family_id=family_id,
                    canonical_name=Path(
                        entry.source_name
                    ).stem,
                    document_code=(
                        extract_document_code(
                            entry.source_name
                        )
                    ),
                    aliases=[entry.source_name],
                )

                family_by_id[family_id] = family
                families.append(family)
                created_families.append(family)
                new_family_ids.add(family_id)
                created_family_now = True
            elif family_id not in new_family_ids:
                raise VersionManagerError(
                    "UNRELATED decision proposes an "
                    "already existing family"
                )

        elif decision.outcome == (
            ResolutionOutcome.NEW_VERSION
        ):
            if family is None:
                raise VersionManagerError(
                    "NEW_VERSION references unknown family"
                )

        if family is None:
            raise VersionManagerError(
                f"Unable to resolve family: {family_id}"
            )

        family_versions = versions_by_family.get(
            family_id,
            [],
        )

        if (
            decision.outcome
            == ResolutionOutcome.NEW_VERSION
            and not family_versions
        ):
            raise VersionManagerError(
                "NEW_VERSION family has no existing version"
            )

        next_version_number = (
            max(
                (
                    version.version_number
                    for version in family_versions
                ),
                default=0,
            )
            + 1
        )

        active_versions = [
            version
            for version in family_versions
            if (
                version.publication_state
                == PublicationState.ACTIVE
            )
        ]

        supersedes_version_id = (
            active_versions[0].version_id
            if active_versions
            else None
        )

        version_label = create_version_label(
            next_version_number
        )
        version_id = create_version_id(
            family_id=family_id,
            version_number=next_version_number,
            source_sha256=entry.source_sha256,
        )

        new_version = VersionRecord(
            family_id=family_id,
            version_id=version_id,
            catalog_id=entry.catalog_id,
            source_sha256=entry.source_sha256,
            version_number=next_version_number,
            version_label=version_label,
            publication_state=PublicationState.DRAFT,
            supersedes_version_id=(
                supersedes_version_id
            ),
        )

        versions.append(new_version)
        created_versions.append(new_version)
        version_by_id[version_id] = new_version
        version_by_sha[
            entry.source_sha256
        ] = new_version
        versions_by_family.setdefault(
            family_id,
            [],
        ).append(new_version)

        action = (
            VersionAction.CREATE_FAMILY_AND_VERSION
            if created_family_now
            else VersionAction.CREATE_VERSION
        )

        assignments.append(
            _create_assignment(
                decision=decision,
                entry=entry,
                version=new_version,
                action=action,
            )
        )

        if entry.source_name not in family.aliases:
            updated_family = family.model_copy(
                update={
                    "aliases": [
                        *family.aliases,
                        entry.source_name,
                    ],
                    "updated_at": _utc_now(),
                }
            )

            family_by_id[family_id] = updated_family

            families = [
                (
                    updated_family
                    if existing.family_id == family_id
                    else existing
                )
                for existing in families
            ]

    return VersionManagerResult(
        families=sorted(
            families,
            key=lambda family: family.family_id,
        ),
        versions=sorted(
            versions,
            key=lambda version: (
                version.family_id,
                version.version_number,
            ),
        ),
        assignments=assignments,
        created_families=created_families,
        created_versions=created_versions,
    )

def build_resolver_index_candidate(
    catalog_entries: list[CatalogEntry],
    assignments: list[VersionAssignment],
    existing_index: list[
        ResolverIndexEntry
    ] | None = None,
) -> list[ResolverIndexEntry]:
    catalog_by_id: dict[str, CatalogEntry] = {}

    for entry in catalog_entries:
        if entry.catalog_id in catalog_by_id:
            raise VersionManagerError(
                "Duplicate catalog_id while building "
                f"Resolver Index: {entry.catalog_id}"
            )

        catalog_by_id[entry.catalog_id] = entry

    index_by_version: dict[
        str,
        ResolverIndexEntry,
    ] = {}

    version_by_sha: dict[str, str] = {}

    for reference in existing_index or []:
        if reference.version_id in index_by_version:
            raise VersionManagerError(
                "Duplicate version_id in Resolver Index: "
                f"{reference.version_id}"
            )

        existing_version_id = version_by_sha.get(
            reference.source_sha256
        )

        if existing_version_id is not None:
            raise VersionManagerError(
                "Duplicate SHA256 in Resolver Index: "
                f"{reference.source_sha256}"
            )

        index_by_version[
            reference.version_id
        ] = reference

        version_by_sha[
            reference.source_sha256
        ] = reference.version_id

    for assignment in assignments:
        entry = catalog_by_id.get(
            assignment.catalog_id
        )

        if entry is None:
            raise VersionManagerError(
                "Assignment references unknown catalog_id: "
                f"{assignment.catalog_id}"
            )

        if (
            entry.source_sha256
            != assignment.source_sha256
        ):
            raise VersionManagerError(
                "Assignment/catalog SHA256 mismatch: "
                f"{assignment.catalog_id}"
            )

        known_reference = index_by_version.get(
            assignment.version_id
        )

        if known_reference is not None:
            if (
                known_reference.family_id
                != assignment.family_id
                or known_reference.source_sha256
                != assignment.source_sha256
            ):
                raise VersionManagerError(
                    "Existing Resolver Index entry "
                    "conflicts with assignment"
                )

            continue

        sha_version_id = version_by_sha.get(
            assignment.source_sha256
        )

        if (
            sha_version_id is not None
            and sha_version_id
            != assignment.version_id
        ):
            raise VersionManagerError(
                "One SHA256 cannot map to multiple "
                "version_id values"
            )

        reference = ResolverIndexEntry(
            family_id=assignment.family_id,
            version_id=assignment.version_id,
            source_sha256=assignment.source_sha256,
            source_name=entry.source_name,
            normalized_name=normalize_document_name(
                entry.source_name
            ),
            document_code=extract_document_code(
                entry.source_name
            ),
        )

        index_by_version[
            assignment.version_id
        ] = reference

        version_by_sha[
            assignment.source_sha256
        ] = assignment.version_id

    return sorted(
        index_by_version.values(),
        key=lambda reference: (
            reference.family_id,
            reference.version_id,
        ),
    )

def write_family_registry_manifest(
    records: list[FamilyRecord],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        records,
        output_path,
    )


def load_family_registry_manifest(
    input_path: str | Path,
) -> list[FamilyRecord]:
    return load_models_jsonl(
        input_path,
        FamilyRecord,
    )


def write_version_registry_manifest(
    records: list[VersionRecord],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        records,
        output_path,
    )


def load_version_registry_manifest(
    input_path: str | Path,
) -> list[VersionRecord]:
    return load_models_jsonl(
        input_path,
        VersionRecord,
    )


def write_version_assignments_manifest(
    records: list[VersionAssignment],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        records,
        output_path,
    )


def load_version_assignments_manifest(
    input_path: str | Path,
) -> list[VersionAssignment]:
    return load_models_jsonl(
        input_path,
        VersionAssignment,
    )


def write_resolver_index_candidate_manifest(
    records: list[ResolverIndexEntry],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        records,
        output_path,
    )


def load_resolver_index_candidate_manifest(
    input_path: str | Path,
) -> list[ResolverIndexEntry]:
    return load_models_jsonl(
        input_path,
        ResolverIndexEntry,
    )