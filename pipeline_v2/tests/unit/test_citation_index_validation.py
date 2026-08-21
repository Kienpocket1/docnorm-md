from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from rag_pipeline.indexing import (
    CitationIndexCommitExistsError,
    CitationIndexCommitIntegrityError,
    CitationIndexPolicy,
    CitationIndexStructureError,
    CitationIndexValidator,
    CitationIndexValidatorConfig,
    IndexValidationInputs,
    build_citation_index_validation,
    load_citation_index_commit,
    load_index_validation_inputs,
)
from rag_pipeline.contracts import ResumeFrom, StageStatus
from test_semantic_chunking import (
    build_normalization_commit,
    make_chunker,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_inventory(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            sha256_file(path),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def build_chunking_commit(
    tmp_path: Path,
    *,
    table_content: str | None = None,
):
    _, normalization = build_normalization_commit(
        tmp_path,
        table_content=table_content,
    )
    chunking = make_chunker(tmp_path).chunk_commit(
        normalization.commit_directory,
        run_id="chunking-input",
    )
    return normalization, chunking


def make_validator(
    tmp_path: Path,
    *,
    max_index_tokens: int = 2048,
) -> CitationIndexValidator:
    return CitationIndexValidator(
        CitationIndexValidatorConfig(
            output_root=tmp_path / "citation-index-commits",
            max_index_tokens=max_index_tokens,
        )
    )


def test_build_resolves_every_chunk_range_and_anchor(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    inputs = load_index_validation_inputs(chunking.commit_directory)
    build = build_citation_index_validation(inputs)

    assert build.report.input_chunk_count == 3
    assert build.report.citation_count == 3
    assert build.report.index_record_count == 3
    assert build.report.source_segment_count == 4
    assert build.report.resolved_source_range_count == 4
    assert build.report.source_anchor_reference_count == 4
    assert build.report.unique_source_anchor_count == 4
    assert build.report.citation_coverage == 1.0
    assert build.report.index_record_coverage == 1.0
    assert build.report.source_range_hash_coverage == 1.0
    assert build.report.page_provenance_coverage == 1.0
    assert build.report.bbox_provenance_coverage == 1.0
    assert build.report.filtered_element_leakage_count == 0
    assert build.report.external_call_count == 0
    assert build.report.embedding_invocation_count == 0

    blocks_by_id = {
        block.block_id: block
        for block in inputs.normalization.document.blocks
    }

    for citation, index_record, chunk in zip(
        build.citations,
        build.index_records,
        inputs.chunking.chunks,
    ):
        assert citation.chunk_id == chunk.chunk_id
        assert index_record.chunk_id == chunk.chunk_id
        assert index_record.citation_id == citation.citation_id
        assert index_record.text == chunk.text
        assert index_record.page_numbers == citation.page_numbers
        assert index_record.source_anchor_ids == [
            anchor.anchor_id for anchor in citation.source_anchors
        ]

        for segment in citation.source_segments:
            block = blocks_by_id[segment.normalized_block_id]
            raw = block.markdown[
                segment.char_start : segment.char_end
            ]
            assert hashlib.sha256(
                raw.encode("utf-8")
            ).hexdigest() == segment.raw_sha256


def test_validation_is_deterministic_for_same_chunk_commit(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    inputs = load_index_validation_inputs(chunking.commit_directory)
    first = build_citation_index_validation(inputs)
    second = build_citation_index_validation(inputs)

    assert first == second
    assert first.report.generated_at == chunking.manifest.committed_at


def test_citation_and_index_ids_are_stable_and_unique(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    build = build_citation_index_validation(
        load_index_validation_inputs(chunking.commit_directory)
    )
    citation_ids = [item.citation_id for item in build.citations]
    index_ids = [item.index_record_id for item in build.index_records]

    assert len(citation_ids) == len(set(citation_ids))
    assert len(index_ids) == len(set(index_ids))
    assert all(item.startswith("citation-") for item in citation_ids)
    assert all(item.startswith("index-record-") for item in index_ids)


def test_atomic_commit_is_resumable_and_input_is_immutable(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    input_before = file_inventory(chunking.commit_directory)
    commit = make_validator(tmp_path).validate_commit(
        chunking.commit_directory,
        run_id="validated-index",
    )
    input_after = file_inventory(chunking.commit_directory)

    assert input_before == input_after
    assert len(list(commit.commit_directory.iterdir())) == 5
    assert commit.manifest.artifact_count == 4
    assert commit.manifest.next_action == "EMBED_AND_STAGE_INDEX"
    assert commit.manifest.external_call_count == 0
    assert commit.stage_result.stage == ResumeFrom.INDEX_VALIDATE
    assert commit.stage_result.status == StageStatus.PASS

    loaded = load_citation_index_commit(commit.commit_directory)
    assert loaded.citations == commit.citations
    assert loaded.index_records == commit.index_records
    assert loaded.report == commit.report


def test_existing_validation_commit_is_not_overwritten(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    validator = make_validator(tmp_path)
    first = validator.validate_commit(
        chunking.commit_directory,
        run_id="immutable-index-validation",
    )
    manifest_path = first.commit_directory / "commit_manifest.json"
    original_bytes = manifest_path.read_bytes()

    with pytest.raises(CitationIndexCommitExistsError):
        validator.validate_commit(
            chunking.commit_directory,
            run_id="immutable-index-validation",
        )

    assert manifest_path.read_bytes() == original_bytes


def test_reader_rejects_index_jsonl_tampering(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    commit = make_validator(tmp_path).validate_commit(
        chunking.commit_directory,
        run_id="tampered-index-validation",
    )
    index_path = commit.commit_directory / "index_records.jsonl"
    index_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(CitationIndexCommitIntegrityError):
        load_citation_index_commit(commit.commit_directory)


def test_source_range_hash_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    inputs = load_index_validation_inputs(chunking.commit_directory)
    first_chunk = inputs.chunking.chunks[0]
    metadata = dict(first_chunk.metadata)
    source_segments = [
        dict(segment) for segment in metadata["source_segments"]
    ]
    source_segments[0]["raw_sha256"] = "0" * 64
    metadata["source_segments"] = source_segments
    bad_chunk = first_chunk.model_copy(update={"metadata": metadata})
    bad_chunking = replace(
        inputs.chunking,
        chunks=(bad_chunk, *inputs.chunking.chunks[1:]),
    )
    bad_inputs = IndexValidationInputs(
        chunking=bad_chunking,
        normalization=inputs.normalization,
        extraction=inputs.extraction,
    )

    with pytest.raises(CitationIndexStructureError):
        build_citation_index_validation(bad_inputs)


def test_index_token_limit_rejects_oversized_record(
    tmp_path: Path,
) -> None:
    long_cell = " ".join(f"word{index}" for index in range(80))
    table = (
        "| Key | Value |\n"
        "| --- | --- |\n"
        f"| {long_cell} | retained |"
    )
    _, chunking = build_chunking_commit(
        tmp_path,
        table_content=table,
    )
    inputs = load_index_validation_inputs(chunking.commit_directory)

    with pytest.raises(CitationIndexStructureError):
        build_citation_index_validation(
            inputs,
            CitationIndexPolicy(max_index_tokens=32),
        )


def test_missing_bbox_provenance_is_rejected(
    tmp_path: Path,
) -> None:
    _, chunking = build_chunking_commit(tmp_path)
    inputs = load_index_validation_inputs(chunking.commit_directory)
    first_chunk = inputs.chunking.chunks[0]
    first_anchor = first_chunk.source_anchors[0]
    bad_anchor = first_anchor.model_copy(update={"bbox": None})
    bad_chunk = first_chunk.model_copy(
        update={
            "source_anchors": [
                bad_anchor,
                *first_chunk.source_anchors[1:],
            ]
        }
    )
    bad_chunking = replace(
        inputs.chunking,
        chunks=(bad_chunk, *inputs.chunking.chunks[1:]),
    )
    bad_inputs = IndexValidationInputs(
        chunking=bad_chunking,
        normalization=inputs.normalization,
        extraction=inputs.extraction,
    )

    with pytest.raises(CitationIndexStructureError):
        build_citation_index_validation(bad_inputs)
