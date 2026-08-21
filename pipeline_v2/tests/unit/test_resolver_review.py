import pytest
from pydantic import ValidationError

from rag_pipeline.contracts import (
    FailureDomain,
    QuarantineType,
    ResolutionDecision,
    ResolutionOutcome,
    ResolverIndexEntry,
    ResolverReviewRecord,
    ResumeFrom,
)
from rag_pipeline.stages.resolver import (
    apply_resolver_reviews,
    apply_resolver_review,
    load_resolution_manifest,
    load_resolver_index_manifest,
    load_resolver_quarantine_manifest,
    load_resolver_review_manifest,
    partition_resolution_decisions,
    write_resolution_manifest,
    write_resolver_index_manifest,
    write_resolver_quarantine_manifest,
    write_resolver_review_manifest,
)


SHA_A = "a" * 64


def make_conflict() -> ResolutionDecision:
    return ResolutionDecision(
        decision_id="res-conflict-001",
        catalog_id="cat-001",
        outcome=ResolutionOutcome.CONFLICT,
        candidate_family_ids=[
            "family-a",
            "family-b",
        ],
        confidence=0.95,
        reason_code="family_conflict",
        requires_human_review=True,
    )


def make_unrelated() -> ResolutionDecision:
    return ResolutionDecision(
        decision_id="res-unrelated-001",
        catalog_id="cat-002",
        outcome=ResolutionOutcome.UNRELATED,
        family_id="family-new",
        confidence=1.0,
        reason_code="no_known_family_match",
    )


def make_rejected() -> ResolutionDecision:
    return ResolutionDecision(
        decision_id="res-rejected-001",
        catalog_id="cat-003",
        outcome=ResolutionOutcome.REJECTED,
        confidence=1.0,
        reason_code="manual_rejection",
    )


def test_conflict_enters_resolver_quarantine() -> None:
    routing = partition_resolution_decisions(
        [make_conflict()]
    )

    assert not routing.continue_to_version
    assert not routing.terminal_rejected
    assert len(routing.resolver_quarantine) == 1

    record = routing.resolver_quarantine[0]

    assert record.queue == QuarantineType.RESOLVER
    assert (
        record.failure_domain
        == FailureDomain.RESOLUTION
    )
    assert record.root_cause_stage == ResumeFrom.RESOLVE
    assert record.resume_from == ResumeFrom.RESOLVE


def test_unrelated_continues_to_version_manager() -> None:
    routing = partition_resolution_decisions(
        [make_unrelated()]
    )

    assert len(routing.continue_to_version) == 1
    assert not routing.resolver_quarantine
    assert not routing.terminal_rejected


def test_rejected_is_terminal() -> None:
    routing = partition_resolution_decisions(
        [make_rejected()]
    )

    assert not routing.continue_to_version
    assert not routing.resolver_quarantine
    assert len(routing.terminal_rejected) == 1


def test_manual_rejected_review_remains_terminal() -> None:
    original = make_conflict()

    review = ResolverReviewRecord(
        review_id="review-001",
        original_decision_id=original.decision_id,
        catalog_id=original.catalog_id,
        outcome=ResolutionOutcome.REJECTED,
        reviewer="manual-reviewer",
        reason_code="invalid_internal_document",
    )

    final_decision = apply_resolver_review(
        original,
        review,
    )

    routing = partition_resolution_decisions(
        [final_decision]
    )

    assert not routing.continue_to_version
    assert len(routing.terminal_rejected) == 1


def test_manual_new_version_review_continues() -> None:
    original = make_conflict()

    review = ResolverReviewRecord(
        review_id="review-002",
        original_decision_id=original.decision_id,
        catalog_id=original.catalog_id,
        outcome=ResolutionOutcome.NEW_VERSION,
        family_id="family-a",
        reviewer="manual-reviewer",
        reason_code="confirmed_new_version",
    )

    final_decision = apply_resolver_review(
        original,
        review,
    )

    routing = partition_resolution_decisions(
        [final_decision]
    )

    assert len(routing.continue_to_version) == 1
    assert (
        routing.continue_to_version[0].family_id
        == "family-a"
    )


def test_review_cannot_apply_to_non_conflict() -> None:
    original = make_unrelated()

    review = ResolverReviewRecord(
        review_id="review-003",
        original_decision_id=original.decision_id,
        catalog_id=original.catalog_id,
        outcome=ResolutionOutcome.UNRELATED,
        family_id="family-new",
        reviewer="manual-reviewer",
        reason_code="confirmed_unrelated",
    )

    with pytest.raises(
        ValueError,
        match="only be applied",
    ):
        apply_resolver_review(
            original,
            review,
        )


def test_review_record_cannot_return_conflict() -> None:
    with pytest.raises(ValidationError):
        ResolverReviewRecord(
            review_id="review-004",
            original_decision_id="res-conflict-001",
            catalog_id="cat-001",
            outcome=ResolutionOutcome.CONFLICT,
            reviewer="manual-reviewer",
            reason_code="still_uncertain",
        )


def test_resolver_artifacts_round_trip(
    tmp_path,
) -> None:
    decision = make_conflict()

    routing = partition_resolution_decisions(
        [decision]
    )

    review = ResolverReviewRecord(
        review_id="review-005",
        original_decision_id=decision.decision_id,
        catalog_id=decision.catalog_id,
        outcome=ResolutionOutcome.REJECTED,
        reviewer="manual-reviewer",
        reason_code="rejected",
    )

    index_entry = ResolverIndexEntry(
        family_id="family-a",
        version_id="family-a-v01",
        source_sha256=SHA_A,
        source_name="Document A.pdf",
        normalized_name="document a",
        document_code=None,
    )

    decision_path = tmp_path / "decisions.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    review_path = tmp_path / "reviews.jsonl"
    index_path = tmp_path / "resolver_index.jsonl"

    write_resolution_manifest(
        [decision],
        decision_path,
    )
    write_resolver_quarantine_manifest(
        routing.resolver_quarantine,
        quarantine_path,
    )
    write_resolver_review_manifest(
        [review],
        review_path,
    )
    write_resolver_index_manifest(
        [index_entry],
        index_path,
    )

    assert load_resolution_manifest(
        decision_path
    ) == [decision]

    assert load_resolver_quarantine_manifest(
        quarantine_path
    ) == routing.resolver_quarantine

    assert load_resolver_review_manifest(
        review_path
    ) == [review]

    assert load_resolver_index_manifest(
        index_path
    ) == [index_entry]

def test_batch_review_preserves_unreviewed_conflict() -> None:
    first = make_conflict()

    second = ResolutionDecision(
        decision_id="res-conflict-002",
        catalog_id="cat-002",
        outcome=ResolutionOutcome.CONFLICT,
        candidate_family_ids=["family-c"],
        confidence=0.91,
        reason_code="second_conflict",
        requires_human_review=True,
    )

    review = ResolverReviewRecord(
        review_id="review-batch-001",
        original_decision_id=first.decision_id,
        catalog_id=first.catalog_id,
        outcome=ResolutionOutcome.REJECTED,
        reviewer="manual-reviewer",
        reason_code="rejected_by_reviewer",
    )

    final_decisions = apply_resolver_reviews(
        decisions=[first, second],
        reviews=[review],
    )

    assert final_decisions[0].outcome == (
        ResolutionOutcome.REJECTED
    )
    assert final_decisions[1].outcome == (
        ResolutionOutcome.CONFLICT
    )


def test_batch_review_rejects_duplicate_reviews() -> None:
    original = make_conflict()

    first_review = ResolverReviewRecord(
        review_id="review-duplicate-001",
        original_decision_id=original.decision_id,
        catalog_id=original.catalog_id,
        outcome=ResolutionOutcome.REJECTED,
        reviewer="reviewer-a",
        reason_code="rejected",
    )

    second_review = ResolverReviewRecord(
        review_id="review-duplicate-002",
        original_decision_id=original.decision_id,
        catalog_id=original.catalog_id,
        outcome=ResolutionOutcome.NEW_VERSION,
        family_id="family-a",
        reviewer="reviewer-b",
        reason_code="new_version",
    )

    with pytest.raises(
        ValueError,
        match="Multiple reviews",
    ):
        apply_resolver_reviews(
            decisions=[original],
            reviews=[
                first_review,
                second_review,
            ],
        )


def test_batch_review_rejects_unknown_decision() -> None:
    original = make_conflict()

    review = ResolverReviewRecord(
        review_id="review-unknown-001",
        original_decision_id="res-does-not-exist",
        catalog_id="cat-unknown",
        outcome=ResolutionOutcome.REJECTED,
        reviewer="manual-reviewer",
        reason_code="rejected",
    )

    with pytest.raises(
        ValueError,
        match="unknown decision",
    ):
        apply_resolver_reviews(
            decisions=[original],
            reviews=[review],
        )