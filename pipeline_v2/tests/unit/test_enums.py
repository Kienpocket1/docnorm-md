from rag_pipeline.contracts.enums import (
    AssetClassification,
    AssetRole,
    FailureDomain,
    QualityDisposition,
    QualitySeverity,
    QuarantineType,
    ResolutionOutcome,
    ResumeFrom,
)


def test_asset_taxonomy_is_frozen() -> None:
    assert [item.value for item in AssetClassification] == [
        "INFORMATIVE",
        "DECORATIVE",
        "UNKNOWN",
    ]
    assert {item.value for item in AssetRole} == {
        "BRAND_LOGO",
        "FLOWCHART",
        "TABLE_IMAGE",
        "FIGURE",
        "SIGNATURE",
        "PAGE_DECORATION",
        "UNKNOWN",
    }


def test_extraction_quality_taxonomy_is_frozen() -> None:
    assert [item.value for item in QualitySeverity] == [
        "INFO",
        "WARNING",
        "ERROR",
    ]
    assert [item.value for item in QualityDisposition] == [
        "PASS",
        "PASS_WITH_WARNINGS",
        "FALLBACK_REQUIRED",
        "QUARANTINE_REQUIRED",
    ]


def test_quarantine_taxonomy_is_frozen() -> None:
    assert {item.value for item in QuarantineType} == {
        "RESOLVER",
        "EXTRACTION",
        "INDEXING",
    }


def test_resume_from_taxonomy_is_frozen() -> None:
    assert [item.value for item in ResumeFrom] == [
        "CATALOG",
        "RESOLVE",
        "VERSION",
        "ROUTE",
        "EXTRACT",
        "EXTRACTION_QUALITY",
        "NORMALIZE",
        "CHUNK",
        "INDEX_VALIDATE",
        "PUBLISH",
    ]


def test_internal_validators_are_not_resume_stages() -> None:
    forbidden_values = {
        "PROVENANCE_VERIFY",
        "GRAPH_REVERIFY",
        "VISUAL",
    }

    resume_values = {item.value for item in ResumeFrom}

    assert forbidden_values.isdisjoint(resume_values)


def test_failure_domain_does_not_extend_quarantine_queue() -> None:
    quarantine_values = {item.value for item in QuarantineType}

    assert FailureDomain.VISUAL.value not in quarantine_values
    assert FailureDomain.PROVENANCE.value not in quarantine_values


def test_resolution_outcomes_include_rejected_terminal() -> None:
    assert ResolutionOutcome.REJECTED.value == "REJECTED"
