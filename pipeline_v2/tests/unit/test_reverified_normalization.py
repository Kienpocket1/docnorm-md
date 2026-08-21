from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag_pipeline.contracts import EngineName, ResumeFrom, StageStatus
from rag_pipeline.normalization.contract_native import (
    build_normalization,
    load_normalization_commit,
)
from rag_pipeline.normalization.reverified_handoff import (
    REVERIFIED_NORMALIZATION_VERSION,
    ReverifiedNormalizationConfig,
    ReverifiedNormalizationInputError,
    ReverifiedNormalizationOrchestrator,
)
from rag_pipeline.stages.hybrid_reverify import (
    HybridReverificationConfig,
    HybridReverifyMergeStage,
)
from test_hybrid_reverify import _base_document, _hybrid_run, _routing_plan


def _inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _reverified_commit(tmp_path: Path, *, valid: bool = True):
    stage = HybridReverifyMergeStage(
        HybridReverificationConfig(output_root=tmp_path / "stage6c")
    )
    run = _hybrid_run(tmp_path, include_bbox=valid)
    return stage.execute(
        _base_document(),
        _routing_plan(),
        [run],
        input_extraction_run_id="stage5-input",
        input_extraction_commit_directory=tmp_path / "stage5-input",
        input_extraction_manifest_sha256="b" * 64,
        run_id="stage6c-input",
    )


def test_normalizes_accepted_reverified_commit_and_is_chunk_ready(
    tmp_path: Path,
) -> None:
    source = _reverified_commit(tmp_path)
    before = _inventory(source.commit_directory)
    orchestrator = ReverifiedNormalizationOrchestrator(
        ReverifiedNormalizationConfig(
            handoff_output_root=tmp_path / "handoff",
            normalization_output_root=tmp_path / "normalization",
        )
    )
    result = orchestrator.normalize_reverified_commit(
        source.commit_directory,
        run_id="normalized-output",
        handoff_run_id="handoff-output",
    )
    after = _inventory(source.commit_directory)

    assert before == after
    assert result.extraction_handoff.document == source.document
    assert result.extraction_handoff.manifest.orchestrator_version == (
        REVERIFIED_NORMALIZATION_VERSION
    )
    assert result.extraction_handoff.manifest.next_action == "NORMALIZE"
    assert result.normalization_commit.manifest.next_action == "CHUNK"
    assert result.normalization_commit.stage_result.stage == ResumeFrom.NORMALIZE
    assert result.normalization_commit.stage_result.status == StageStatus.PASS
    assert result.normalization_commit.report.input_source_element_count == 2
    assert result.normalization_commit.report.selected_source_element_count == 2
    assert result.normalization_commit.report.filtered_source_element_count == 0
    assert result.normalization_commit.report.normalized_block_count == 2
    assert result.normalization_commit.report.provenance_coverage == 1.0
    assert EngineName.OPENDATALOADER_HYBRID in source.document.engine_chain


def test_normalization_rebuild_is_deterministic(tmp_path: Path) -> None:
    source = _reverified_commit(tmp_path)
    orchestrator = ReverifiedNormalizationOrchestrator(
        ReverifiedNormalizationConfig(
            handoff_output_root=tmp_path / "handoff",
            normalization_output_root=tmp_path / "normalization",
        )
    )
    result = orchestrator.normalize_reverified_commit(
        source.commit_directory,
        run_id="normalized-output",
        handoff_run_id="handoff-output",
    )
    expected = build_normalization(result.extraction_handoff)
    loaded = load_normalization_commit(
        result.normalization_commit.commit_directory
    )

    assert expected.document == loaded.document
    assert expected.markdown == loaded.markdown
    assert expected.report == loaded.report


def test_rejects_reverified_commit_with_pending_mineru_page(
    tmp_path: Path,
) -> None:
    source = _reverified_commit(tmp_path, valid=False)
    orchestrator = ReverifiedNormalizationOrchestrator(
        ReverifiedNormalizationConfig(
            handoff_output_root=tmp_path / "handoff",
            normalization_output_root=tmp_path / "normalization",
        )
    )

    with pytest.raises(
        ReverifiedNormalizationInputError,
        match="not routed to NORMALIZE",
    ):
        orchestrator.normalize_reverified_commit(source.commit_directory)

    assert not (tmp_path / "handoff").exists()
    assert not (tmp_path / "normalization").exists()


def test_commits_have_closed_hash_inventories(tmp_path: Path) -> None:
    source = _reverified_commit(tmp_path)
    orchestrator = ReverifiedNormalizationOrchestrator(
        ReverifiedNormalizationConfig(
            handoff_output_root=tmp_path / "handoff",
            normalization_output_root=tmp_path / "normalization",
        )
    )
    result = orchestrator.normalize_reverified_commit(
        source.commit_directory,
        run_id="normalized-output",
        handoff_run_id="handoff-output",
    )
    handoff_files = {
        path.name
        for path in result.extraction_handoff.commit_directory.iterdir()
        if path.is_file()
    }
    normalization_files = {
        path.name
        for path in result.normalization_commit.commit_directory.iterdir()
        if path.is_file()
    }

    assert handoff_files == {
        "commit_manifest.json",
        "extracted_document.json",
        "extraction_quality_report.json",
        "indexing_selection.json",
        "stage_result.json",
    }
    assert normalization_files == {
        "commit_manifest.json",
        "document.normalized.md",
        "normalization_report.json",
        "normalized_document.json",
        "stage_result.json",
    }
