from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag_pipeline.chunking import (
    ChunkingCommitExistsError,
    ChunkingCommitIntegrityError,
    ChunkingPolicy,
    ChunkingStructureError,
    SemanticChunker,
    SemanticChunkerConfig,
    build_chunking,
    count_tokens,
    load_chunking_commit,
    load_chunking_inputs,
    split_semantic_ranges,
)
from rag_pipeline.contracts import ResumeFrom, StageStatus
from test_contract_native_normalization import (
    build_extraction_commit,
    make_normalizer,
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


def build_normalization_commit(
    tmp_path: Path,
    *,
    table_content: str | None = None,
):
    extraction = build_extraction_commit(
        tmp_path,
        table_content=table_content,
    )
    normalization = make_normalizer(tmp_path).normalize_commit(
        extraction.commit_directory,
        run_id="normalization-input",
    )
    return extraction, normalization


def make_chunker(
    tmp_path: Path,
    *,
    target_tokens: int = 32,
    max_tokens: int = 64,
    structural_max_tokens: int = 256,
) -> SemanticChunker:
    return SemanticChunker(
        SemanticChunkerConfig(
            output_root=tmp_path / "chunking-commits",
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            structural_max_tokens=structural_max_tokens,
        )
    )


def test_unicode_lexical_tokenizer_is_deterministic() -> None:
    text = "Café, vật tư! Owner's record."

    assert count_tokens(text) == 8
    assert count_tokens(text) == count_tokens(text)


def test_semantic_split_ranges_are_complete_and_bounded() -> None:
    text = (
        "Câu thứ nhất kết thúc. "
        "Câu thứ hai có thêm nội dung để kiểm tra. "
        "Câu thứ ba kết thúc! "
        "Câu cuối cùng hoàn tất."
    )
    ranges = split_semantic_ranges(text, max_tokens=12)

    assert len(ranges) > 1
    assert ranges[0].char_start == 0
    assert ranges[-1].char_end == len(text)
    assert all(
        current.char_end == following.char_start
        for current, following in zip(ranges, ranges[1:])
    )
    assert all(item.token_count <= 12 for item in ranges)
    assert "".join(
        text[item.char_start : item.char_end]
        for item in ranges
    ) == text


def test_build_preserves_structures_order_and_provenance(
    tmp_path: Path,
) -> None:
    _, normalization = build_normalization_commit(tmp_path)
    inputs = load_chunking_inputs(normalization.commit_directory)
    build = build_chunking(
        inputs,
        ChunkingPolicy(
            target_tokens=32,
            max_tokens=64,
            structural_max_tokens=256,
        ),
    )

    assert build.report.input_normalized_block_count == 4
    assert build.report.represented_normalized_block_count == 4
    assert build.report.structural_block_count == 2
    assert build.report.structural_chunk_count == 2
    assert build.report.table_block_count == 1
    assert build.report.list_block_count == 1
    assert build.report.block_coverage == 1.0
    assert build.report.text_range_coverage == 1.0
    assert build.report.provenance_coverage == 1.0
    assert build.report.ocr_invocation_count == 0
    assert [chunk.ordinal for chunk in build.chunks] == list(
        range(1, len(build.chunks) + 1)
    )

    chunks_by_type = {
        chunk.metadata["element_types"][0]: chunk
        for chunk in build.chunks
        if chunk.metadata["contains_structural_block"]
    }
    table_block = normalization.document.blocks[3]
    list_block = normalization.document.blocks[2]

    assert chunks_by_type["TABLE"].text == table_block.markdown
    assert chunks_by_type["LIST"].text == list_block.markdown
    assert chunks_by_type["TABLE"].source_anchors == [
        table_block.source_spans[0].source_anchor
    ]
    assert chunks_by_type["LIST"].source_anchors == [
        list_block.source_spans[0].source_anchor
    ]


def test_chunking_is_deterministic_for_same_input(
    tmp_path: Path,
) -> None:
    _, normalization = build_normalization_commit(tmp_path)
    inputs = load_chunking_inputs(normalization.commit_directory)
    policy = ChunkingPolicy(
        target_tokens=32,
        max_tokens=64,
        structural_max_tokens=256,
    )

    first = build_chunking(inputs, policy)
    second = build_chunking(inputs, policy)

    assert first == second
    assert first.report.generated_at == normalization.manifest.committed_at


def test_atomic_commit_is_resumable_and_input_is_immutable(
    tmp_path: Path,
) -> None:
    _, normalization = build_normalization_commit(tmp_path)
    input_before = file_inventory(normalization.commit_directory)
    commit = make_chunker(tmp_path).chunk_commit(
        normalization.commit_directory,
        run_id="semantic-chunks",
    )
    input_after = file_inventory(normalization.commit_directory)

    assert input_before == input_after
    assert len(list(commit.commit_directory.iterdir())) == 4
    assert commit.manifest.artifact_count == 3
    assert commit.manifest.next_action == "INDEX_VALIDATE"
    assert commit.manifest.ocr_invocation_count == 0
    assert commit.stage_result.stage == ResumeFrom.CHUNK
    assert commit.stage_result.status == StageStatus.PASS

    loaded = load_chunking_commit(commit.commit_directory)
    assert loaded.chunks == commit.chunks
    assert loaded.jsonl == commit.jsonl
    assert loaded.report == commit.report


def test_existing_chunk_commit_is_not_overwritten(
    tmp_path: Path,
) -> None:
    _, normalization = build_normalization_commit(tmp_path)
    chunker = make_chunker(tmp_path)
    first = chunker.chunk_commit(
        normalization.commit_directory,
        run_id="immutable-chunks",
    )
    manifest_path = first.commit_directory / "commit_manifest.json"
    original_bytes = manifest_path.read_bytes()

    with pytest.raises(ChunkingCommitExistsError):
        chunker.chunk_commit(
            normalization.commit_directory,
            run_id="immutable-chunks",
        )

    assert manifest_path.read_bytes() == original_bytes


def test_reader_rejects_chunk_jsonl_tampering(
    tmp_path: Path,
) -> None:
    _, normalization = build_normalization_commit(tmp_path)
    commit = make_chunker(tmp_path).chunk_commit(
        normalization.commit_directory,
        run_id="tampered-chunks",
    )
    chunks_path = commit.commit_directory / "chunks.jsonl"
    chunks_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ChunkingCommitIntegrityError):
        load_chunking_commit(commit.commit_directory)


def test_oversized_structural_block_fails_without_partial_commit(
    tmp_path: Path,
) -> None:
    long_cell = " ".join(f"word{index}" for index in range(80))
    table = (
        "| Key | Value |\n"
        "| --- | --- |\n"
        f"| {long_cell} | retained |"
    )
    _, normalization = build_normalization_commit(
        tmp_path,
        table_content=table,
    )
    chunker = make_chunker(
        tmp_path,
        target_tokens=32,
        max_tokens=32,
        structural_max_tokens=32,
    )

    with pytest.raises(ChunkingStructureError):
        chunker.chunk_commit(
            normalization.commit_directory,
            run_id="oversized-structure",
        )

    assert not (
        tmp_path / "chunking-commits" / "oversized-structure"
    ).exists()


def test_chunk_ids_and_instances_are_stable_and_unique(
    tmp_path: Path,
) -> None:
    _, normalization = build_normalization_commit(tmp_path)
    build = build_chunking(
        load_chunking_inputs(normalization.commit_directory),
        ChunkingPolicy(
            target_tokens=32,
            max_tokens=64,
            structural_max_tokens=256,
        ),
    )
    chunk_ids = [chunk.chunk_id for chunk in build.chunks]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert all(
        chunk.chunk_id.startswith(f"{chunk.family_id}#semantic-")
        for chunk in build.chunks
    )
    assert all(
        chunk.chunk_instance_id
        == f"{chunk.version_id}::{chunk.chunk_id}"
        for chunk in build.chunks
    )
