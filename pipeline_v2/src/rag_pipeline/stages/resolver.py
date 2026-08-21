from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from rapidfuzz.fuzz import token_set_ratio

from rag_pipeline.contracts import (
    CatalogEntry,
    FailureDomain,
    QuarantineRecord,
    QuarantineType,
    ResolutionDecision,
    ResolutionOutcome,
    ResolverIndexEntry,
    ResolverReviewRecord,
    ResumeFrom,
)

from rag_pipeline.stores import (
    load_models_jsonl,
    write_models_jsonl,
)


DEFAULT_FUZZY_REVIEW_THRESHOLD = 90.0


def _ascii_fold(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", value)

    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


def normalize_document_name(source_name: str) -> str:
    stem = Path(source_name).stem
    folded = _ascii_fold(stem).casefold()
    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        folded,
    )

    return " ".join(normalized.split())


def extract_document_code(
    source_name: str,
) -> str | None:
    stem = Path(source_name).stem
    folded = _ascii_fold(stem).upper()

    # Ví dụ: QT.TCKT.01
    qt_match = re.search(
        r"(?<![A-Z0-9])"
        r"QT[\s._-]+"
        r"([A-Z][A-Z0-9]{1,15})"
        r"[\s._-]+"
        r"(\d{1,4})(?!\d)",
        folded,
    )

    if qt_match:
        department = qt_match.group(1).casefold()
        number = qt_match.group(2)

        return f"qt-{department}-{number}"

    # Ví dụ: ID627, kể cả khi dính vào chuỗi số phía trước.
    id_match = re.search(
        r"ID[\s._-]*(\d{1,10})(?!\d)",
        folded,
    )

    if id_match:
        return f"id{id_match.group(1)}"

    # Ví dụ: 34 HDQT
    hdqt_match = re.search(
        r"(?<!\d)(\d{1,6})[\s._-]*HDQT(?![A-Z])",
        folded,
    )

    if hdqt_match:
        return f"{hdqt_match.group(1)}-hdqt"

    # Fallback cho mã quyết định dạng QD 123...
    qd_match = re.search(
        r"(?<![A-Z0-9])"
        r"QD[\s._-]*(\d{1,30})(?!\d)",
        folded,
    )

    if qd_match:
        return f"qd-{qd_match.group(1)}"

    return None


def propose_family_id(entry: CatalogEntry) -> str:
    document_code = extract_document_code(
        entry.source_name
    )

    if document_code:
        return document_code

    normalized_name = normalize_document_name(
        entry.source_name
    )

    slug = normalized_name.replace(" ", "-")
    slug = slug[:40].strip("-") or "document"

    name_digest = hashlib.sha256(
        normalized_name.encode("utf-8")
    ).hexdigest()[:8]

    return f"doc-{slug}-{name_digest}"


def _create_decision_id(
    entry: CatalogEntry,
    outcome: ResolutionOutcome,
    reason_code: str,
    family_id: str | None,
    candidate_family_ids: list[str],
) -> str:
    material = "|".join(
        [
            entry.catalog_id,
            outcome.value,
            reason_code,
            family_id or "",
            ",".join(sorted(candidate_family_ids)),
        ]
    )

    return f"res-{uuid5(NAMESPACE_URL, material).hex}"


def _create_decision(
    *,
    entry: CatalogEntry,
    outcome: ResolutionOutcome,
    confidence: float,
    reason_code: str,
    family_id: str | None = None,
    matched_version_id: str | None = None,
    candidate_family_ids: list[str] | None = None,
    requires_human_review: bool = False,
    evidence: dict | None = None,
) -> ResolutionDecision:
    candidates = sorted(
        set(candidate_family_ids or [])
    )

    return ResolutionDecision(
        decision_id=_create_decision_id(
            entry=entry,
            outcome=outcome,
            reason_code=reason_code,
            family_id=family_id,
            candidate_family_ids=candidates,
        ),
        catalog_id=entry.catalog_id,
        outcome=outcome,
        family_id=family_id,
        matched_version_id=matched_version_id,
        candidate_family_ids=candidates,
        confidence=confidence,
        reason_code=reason_code,
        requires_human_review=requires_human_review,
        evidence=evidence or {},
    )


def resolve_catalog_entry(
    entry: CatalogEntry,
    references: list[ResolverIndexEntry],
    fuzzy_review_threshold: float = (
        DEFAULT_FUZZY_REVIEW_THRESHOLD
    ),
) -> ResolutionDecision:
    if not 0 <= fuzzy_review_threshold <= 100:
        raise ValueError(
            "fuzzy_review_threshold must be between 0 and 100"
        )

    normalized_name = normalize_document_name(
        entry.source_name
    )
    document_code = extract_document_code(
        entry.source_name
    )

    base_evidence = {
        "normalized_name": normalized_name,
        "document_code": document_code,
    }

    # Rule 1: SHA256 giống hoàn toàn.
    exact_sha_matches = [
        reference
        for reference in references
        if reference.source_sha256
        == entry.source_sha256
    ]

    if exact_sha_matches:
        family_ids = sorted(
            {
                reference.family_id
                for reference in exact_sha_matches
            }
        )

        if len(family_ids) > 1:
            return _create_decision(
                entry=entry,
                outcome=ResolutionOutcome.CONFLICT,
                confidence=1.0,
                reason_code=(
                    "same_sha256_in_multiple_families"
                ),
                candidate_family_ids=family_ids,
                requires_human_review=True,
                evidence={
                    **base_evidence,
                    "matching_rule": "EXACT_SHA256",
                },
            )

        selected_reference = sorted(
            exact_sha_matches,
            key=lambda reference: (
                reference.version_id
            ),
        )[0]

        return _create_decision(
            entry=entry,
            outcome=ResolutionOutcome.SAME_VERSION,
            confidence=1.0,
            reason_code="exact_sha256_match",
            family_id=selected_reference.family_id,
            matched_version_id=(
                selected_reference.version_id
            ),
            evidence={
                **base_evidence,
                "matching_rule": "EXACT_SHA256",
            },
        )

    # Rule 2: Cùng mã tài liệu nhưng SHA khác.
    if document_code:
        code_matches = [
            reference
            for reference in references
            if (
                reference.document_code
                and reference.document_code.casefold()
                == document_code.casefold()
            )
        ]

        if code_matches:
            family_ids = sorted(
                {
                    reference.family_id
                    for reference in code_matches
                }
            )

            if len(family_ids) == 1:
                return _create_decision(
                    entry=entry,
                    outcome=(
                        ResolutionOutcome.NEW_VERSION
                    ),
                    confidence=0.98,
                    reason_code="same_document_code",
                    family_id=family_ids[0],
                    evidence={
                        **base_evidence,
                        "matching_rule": "DOCUMENT_CODE",
                    },
                )

            return _create_decision(
                entry=entry,
                outcome=ResolutionOutcome.CONFLICT,
                confidence=0.98,
                reason_code=(
                    "document_code_in_multiple_families"
                ),
                candidate_family_ids=family_ids,
                requires_human_review=True,
                evidence={
                    **base_evidence,
                    "matching_rule": "DOCUMENT_CODE",
                },
            )

    # Rule 3: Tên chuẩn hóa giống hoàn toàn.
    name_matches = [
        reference
        for reference in references
        if (
            reference.normalized_name.casefold()
            == normalized_name.casefold()
        )
    ]

    if name_matches:
        family_ids = sorted(
            {
                reference.family_id
                for reference in name_matches
            }
        )

        if len(family_ids) == 1:
            return _create_decision(
                entry=entry,
                outcome=ResolutionOutcome.NEW_VERSION,
                confidence=0.93,
                reason_code="exact_normalized_name",
                family_id=family_ids[0],
                evidence={
                    **base_evidence,
                    "matching_rule": (
                        "EXACT_NORMALIZED_NAME"
                    ),
                },
            )

        return _create_decision(
            entry=entry,
            outcome=ResolutionOutcome.CONFLICT,
            confidence=0.93,
            reason_code=(
                "normalized_name_in_multiple_families"
            ),
            candidate_family_ids=family_ids,
            requires_human_review=True,
            evidence={
                **base_evidence,
                "matching_rule": (
                    "EXACT_NORMALIZED_NAME"
                ),
            },
        )

    # Rule 4: Fuzzy chỉ đề nghị review, không tự gộp.
    family_scores: dict[str, float] = {}

    for reference in references:
        score = float(
            token_set_ratio(
                normalized_name,
                reference.normalized_name,
            )
        )

        previous_score = family_scores.get(
            reference.family_id,
            0.0,
        )

        family_scores[reference.family_id] = max(
            score,
            previous_score,
        )

    if family_scores:
        best_score = max(family_scores.values())

        if best_score >= fuzzy_review_threshold:
            score_cutoff = max(
                fuzzy_review_threshold,
                best_score - 3.0,
            )

            candidates = sorted(
                family_id
                for family_id, score
                in family_scores.items()
                if score >= score_cutoff
            )

            return _create_decision(
                entry=entry,
                outcome=ResolutionOutcome.CONFLICT,
                confidence=best_score / 100.0,
                reason_code=(
                    "fuzzy_name_match_requires_review"
                ),
                candidate_family_ids=candidates,
                requires_human_review=True,
                evidence={
                    **base_evidence,
                    "matching_rule": "FUZZY_NAME",
                    "candidate_scores": {
                        family_id: family_scores[
                            family_id
                        ]
                        for family_id in candidates
                    },
                },
            )

    # Rule 5: Không khớp family nào đã biết.
    proposed_family = propose_family_id(entry)

    return _create_decision(
        entry=entry,
        outcome=ResolutionOutcome.UNRELATED,
        confidence=1.0 if document_code else 0.80,
        reason_code="no_known_family_match",
        family_id=proposed_family,
        evidence={
            **base_evidence,
            "matching_rule": "NEW_FAMILY_PROPOSAL",
        },
    )


def resolve_catalog_entries(
    entries: list[CatalogEntry],
    references: list[ResolverIndexEntry],
    fuzzy_review_threshold: float = (
        DEFAULT_FUZZY_REVIEW_THRESHOLD
    ),
) -> list[ResolutionDecision]:
    catalog_ids = [
        entry.catalog_id
        for entry in entries
    ]

    if len(catalog_ids) != len(set(catalog_ids)):
        raise ValueError(
            "Catalog contains duplicate catalog_id values"
        )

    return [
        resolve_catalog_entry(
            entry=entry,
            references=references,
            fuzzy_review_threshold=(
                fuzzy_review_threshold
            ),
        )
        for entry in entries
    ]

@dataclass(frozen=True, slots=True)
class ResolutionRoutingResult:
    continue_to_version: list[ResolutionDecision]
    resolver_quarantine: list[QuarantineRecord]
    terminal_rejected: list[ResolutionDecision]


def create_resolver_quarantine(
    decision: ResolutionDecision,
) -> QuarantineRecord:
    if decision.outcome != ResolutionOutcome.CONFLICT:
        raise ValueError(
            "Only CONFLICT decisions enter "
            "Resolver quarantine"
        )

    material = (
        f"resolver-quarantine:"
        f"{decision.decision_id}"
    )

    quarantine_id = (
        f"qr-{uuid5(NAMESPACE_URL, material).hex}"
    )

    return QuarantineRecord(
        quarantine_id=quarantine_id,
        queue=QuarantineType.RESOLVER,
        failure_domain=FailureDomain.RESOLUTION,
        root_cause_stage=ResumeFrom.RESOLVE,
        reason_code=decision.reason_code,
        resume_from=ResumeFrom.RESOLVE,
        attempt_count=0,
        max_attempts=0,
        catalog_id=decision.catalog_id,
        details={
            "decision_id": decision.decision_id,
            "candidate_family_ids": (
                decision.candidate_family_ids
            ),
            "confidence": decision.confidence,
        },
    )


def partition_resolution_decisions(
    decisions: list[ResolutionDecision],
) -> ResolutionRoutingResult:
    continue_to_version: list[
        ResolutionDecision
    ] = []
    resolver_quarantine: list[
        QuarantineRecord
    ] = []
    terminal_rejected: list[
        ResolutionDecision
    ] = []

    for decision in decisions:
        if decision.outcome == ResolutionOutcome.CONFLICT:
            resolver_quarantine.append(
                create_resolver_quarantine(decision)
            )
            continue

        if decision.outcome == ResolutionOutcome.REJECTED:
            terminal_rejected.append(decision)
            continue

        continue_to_version.append(decision)

    return ResolutionRoutingResult(
        continue_to_version=continue_to_version,
        resolver_quarantine=resolver_quarantine,
        terminal_rejected=terminal_rejected,
    )


def apply_resolver_review(
    original: ResolutionDecision,
    review: ResolverReviewRecord,
) -> ResolutionDecision:
    if original.outcome != ResolutionOutcome.CONFLICT:
        raise ValueError(
            "Manual Resolver review can only be applied "
            "to a CONFLICT decision"
        )

    if original.decision_id != review.original_decision_id:
        raise ValueError(
            "Review does not reference the original decision"
        )

    if original.catalog_id != review.catalog_id:
        raise ValueError(
            "Review catalog_id does not match decision"
        )

    material = "|".join(
        [
            original.decision_id,
            review.review_id,
            review.outcome.value,
        ]
    )

    reviewed_decision_id = (
        f"res-{uuid5(NAMESPACE_URL, material).hex}"
    )

    return ResolutionDecision(
        decision_id=reviewed_decision_id,
        catalog_id=original.catalog_id,
        outcome=review.outcome,
        family_id=review.family_id,
        matched_version_id=(
            review.matched_version_id
        ),
        candidate_family_ids=[],
        confidence=1.0,
        reason_code=review.reason_code,
        requires_human_review=False,
        evidence={
            **original.evidence,
            "manual_review": {
                "review_id": review.review_id,
                "reviewer": review.reviewer,
                "original_decision_id": (
                    original.decision_id
                ),
                "original_candidate_family_ids": (
                    original.candidate_family_ids
                ),
                "reviewed_at": (
                    review.reviewed_at.isoformat()
                ),
            },
        },
        decided_at=review.reviewed_at,
    )


def write_resolution_manifest(
    decisions: list[ResolutionDecision],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        decisions,
        output_path,
    )


def load_resolution_manifest(
    input_path: str | Path,
) -> list[ResolutionDecision]:
    return load_models_jsonl(
        input_path,
        ResolutionDecision,
    )


def write_resolver_index_manifest(
    references: list[ResolverIndexEntry],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        references,
        output_path,
    )


def load_resolver_index_manifest(
    input_path: str | Path,
) -> list[ResolverIndexEntry]:
    return load_models_jsonl(
        input_path,
        ResolverIndexEntry,
    )


def write_resolver_quarantine_manifest(
    records: list[QuarantineRecord],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        records,
        output_path,
    )


def load_resolver_quarantine_manifest(
    input_path: str | Path,
) -> list[QuarantineRecord]:
    return load_models_jsonl(
        input_path,
        QuarantineRecord,
    )


def write_resolver_review_manifest(
    reviews: list[ResolverReviewRecord],
    output_path: str | Path,
) -> Path:
    return write_models_jsonl(
        reviews,
        output_path,
    )


def load_resolver_review_manifest(
    input_path: str | Path,
) -> list[ResolverReviewRecord]:
    return load_models_jsonl(
        input_path,
        ResolverReviewRecord,
    )

def apply_resolver_reviews(
    decisions: list[ResolutionDecision],
    reviews: list[ResolverReviewRecord],
) -> list[ResolutionDecision]:
    decision_by_id: dict[
        str,
        ResolutionDecision,
    ] = {}

    for decision in decisions:
        if decision.decision_id in decision_by_id:
            raise ValueError(
                "Duplicate decision_id in decision manifest: "
                f"{decision.decision_id}"
            )

        decision_by_id[decision.decision_id] = decision

    review_by_decision_id: dict[
        str,
        ResolverReviewRecord,
    ] = {}

    for review in reviews:
        decision_id = review.original_decision_id

        if decision_id not in decision_by_id:
            raise ValueError(
                "Review references unknown decision: "
                f"{decision_id}"
            )

        if decision_id in review_by_decision_id:
            raise ValueError(
                "Multiple reviews reference one decision: "
                f"{decision_id}"
            )

        review_by_decision_id[decision_id] = review

    final_decisions: list[ResolutionDecision] = []

    for decision in decisions:
        review = review_by_decision_id.get(
            decision.decision_id
        )

        if review is None:
            final_decisions.append(decision)
            continue

        final_decisions.append(
            apply_resolver_review(
                original=decision,
                review=review,
            )
        )

    return final_decisions