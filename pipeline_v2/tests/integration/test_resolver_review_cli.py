from typer.testing import CliRunner

from rag_pipeline.cli import app
from rag_pipeline.contracts import (
    ResolutionDecision,
    ResolutionOutcome,
    ResolverReviewRecord,
)
from rag_pipeline.stages.resolver import (
    load_resolution_manifest,
    load_resolver_quarantine_manifest,
    write_resolution_manifest,
    write_resolver_review_manifest,
)


runner = CliRunner()


def test_resolve_review_cli_routes_rejected_terminal(
    tmp_path,
) -> None:
    decisions_path = tmp_path / "decisions.jsonl"
    reviews_path = tmp_path / "reviews.jsonl"
    output_dir = tmp_path / "reviewed"

    rejected_source = ResolutionDecision(
        decision_id="res-conflict-reject",
        catalog_id="cat-reject",
        outcome=ResolutionOutcome.CONFLICT,
        candidate_family_ids=["family-a"],
        confidence=0.95,
        reason_code="requires_review",
        requires_human_review=True,
    )

    accepted_source = ResolutionDecision(
        decision_id="res-conflict-accept",
        catalog_id="cat-accept",
        outcome=ResolutionOutcome.CONFLICT,
        candidate_family_ids=["family-b"],
        confidence=0.94,
        reason_code="requires_review",
        requires_human_review=True,
    )

    rejected_review = ResolverReviewRecord(
        review_id="review-reject",
        original_decision_id=(
            rejected_source.decision_id
        ),
        catalog_id=rejected_source.catalog_id,
        outcome=ResolutionOutcome.REJECTED,
        reviewer="manual-reviewer",
        reason_code="invalid_document",
    )

    accepted_review = ResolverReviewRecord(
        review_id="review-accept",
        original_decision_id=(
            accepted_source.decision_id
        ),
        catalog_id=accepted_source.catalog_id,
        outcome=ResolutionOutcome.NEW_VERSION,
        family_id="family-b",
        reviewer="manual-reviewer",
        reason_code="confirmed_new_version",
    )

    write_resolution_manifest(
        [
            rejected_source,
            accepted_source,
        ],
        decisions_path,
    )
    write_resolver_review_manifest(
        [
            rejected_review,
            accepted_review,
        ],
        reviews_path,
    )

    result = runner.invoke(
        app,
        [
            "resolve-review",
            "--decisions-manifest",
            str(decisions_path),
            "--reviews-manifest",
            str(reviews_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output

    continuing = load_resolution_manifest(
        output_dir / "continue_to_version.jsonl"
    )
    quarantined = (
        load_resolver_quarantine_manifest(
            output_dir / "resolver_quarantine.jsonl"
        )
    )
    rejected = load_resolution_manifest(
        output_dir / "terminal_rejected.jsonl"
    )

    assert len(continuing) == 1
    assert continuing[0].outcome == (
        ResolutionOutcome.NEW_VERSION
    )

    assert not quarantined

    assert len(rejected) == 1
    assert rejected[0].outcome == (
        ResolutionOutcome.REJECTED
    )