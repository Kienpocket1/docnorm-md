from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.contracts import (
    ChunkRecord,
    ElementType,
    ResumeFrom,
    SourceAnchor,
    StageResult,
    StageStatus,
)
from rag_pipeline.normalization import (
    NormalizationCommitBundle,
    load_normalization_commit,
)
from rag_pipeline.orchestration import (
    CommitArtifactRecord,
    ExtractionCommitBundle,
    load_extraction_commit,
)
from rag_pipeline.orchestration.extraction import (
    artifact_record,
    model_json_bytes,
    sha256_file,
    utc_run_id,
    validate_run_id,
    write_payload,
)


SEMANTIC_CHUNKER_VERSION = "2026-08-17-5.2B.1"
TOKENIZER_VERSION = "unicode-lexical-v1"
CHUNKING_COMMIT_SCHEMA_VERSION = "1.0"

CHUNKS_FILENAME = "chunks.jsonl"
CHUNKING_REPORT_FILENAME = "chunking_report.json"
STAGE_RESULT_FILENAME = "stage_result.json"
COMMIT_MANIFEST_FILENAME = "commit_manifest.json"

TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
STRUCTURAL_TYPES = {
    ElementType.TABLE,
    ElementType.LIST,
    ElementType.FLOWCHART,
    ElementType.FORMULA,
}
HEADING_TYPES = {
    ElementType.TITLE,
    ElementType.HEADING,
}


class ChunkingManifestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class ChunkingPolicy(ChunkingManifestModel):
    target_tokens: int = Field(default=350, ge=32, le=4096)
    max_tokens: int = Field(default=512, ge=32, le=8192)
    structural_max_tokens: int = Field(
        default=2048,
        ge=32,
        le=16384,
    )

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.target_tokens > self.max_tokens:
            raise ValueError("target_tokens must not exceed max_tokens")

        if self.max_tokens > self.structural_max_tokens:
            raise ValueError(
                "max_tokens must not exceed structural_max_tokens"
            )

        return self


class ChunkingReport(ChunkingManifestModel):
    chunker_version: str = Field(min_length=1)
    tokenizer_version: str = Field(min_length=1)
    input_normalization_run_id: str = Field(min_length=1)
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    structural_max_tokens: int = Field(ge=1)
    input_normalized_block_count: int = Field(ge=1)
    represented_normalized_block_count: int = Field(ge=1)
    source_segment_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    total_token_count: int = Field(ge=1)
    min_chunk_token_count: int = Field(ge=1)
    max_chunk_token_count: int = Field(ge=1)
    table_block_count: int = Field(ge=0)
    list_block_count: int = Field(ge=0)
    flowchart_block_count: int = Field(ge=0)
    formula_block_count: int = Field(ge=0)
    structural_block_count: int = Field(ge=0)
    structural_chunk_count: int = Field(ge=0)
    oversized_atomic_chunk_count: int = Field(ge=0)
    source_anchor_count: int = Field(ge=1)
    unique_source_anchor_count: int = Field(ge=1)
    page_numbers: list[int]
    normalized_block_ids: list[str] = Field(min_length=1)
    source_element_ids: list[str] = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)
    block_coverage: Literal[1.0] = 1.0
    text_range_coverage: Literal[1.0] = 1.0
    provenance_coverage: Literal[1.0] = 1.0
    duplicate_chunk_id_count: Literal[0] = 0
    empty_chunk_count: Literal[0] = 0
    hard_limit_violation_count: Literal[0] = 0
    ocr_invocation_count: Literal[0] = 0
    passed: Literal[True] = True
    generated_at: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if (
            self.input_normalized_block_count
            != self.represented_normalized_block_count
        ):
            raise ValueError("Every normalized block must be represented")

        if len(self.normalized_block_ids) != len(
            set(self.normalized_block_ids)
        ):
            raise ValueError("normalized_block_ids must be unique")

        if len(self.normalized_block_ids) != (
            self.represented_normalized_block_count
        ):
            raise ValueError(
                "normalized_block_ids length must equal block count"
            )

        if len(self.source_element_ids) != len(
            set(self.source_element_ids)
        ):
            raise ValueError("source_element_ids must be unique")

        if len(self.source_element_ids) != (
            self.represented_normalized_block_count
        ):
            raise ValueError(
                "source_element_ids length must equal block count"
            )

        if len(self.chunk_ids) != self.chunk_count:
            raise ValueError("chunk_ids length must equal chunk_count")

        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("chunk_ids must be unique")

        structural_count = (
            self.table_block_count
            + self.list_block_count
            + self.flowchart_block_count
            + self.formula_block_count
        )

        if structural_count != self.structural_block_count:
            raise ValueError(
                "Structural type counts must cover structural blocks"
            )

        if self.structural_chunk_count != self.structural_block_count:
            raise ValueError(
                "Every structural block must have one isolated chunk"
            )

        if self.page_numbers != sorted(set(self.page_numbers)):
            raise ValueError("page_numbers must be sorted and unique")

        if self.min_chunk_token_count > self.max_chunk_token_count:
            raise ValueError("Chunk token range is invalid")

        return self


class ChunkingCommitManifest(ChunkingManifestModel):
    schema_version: Literal["1.0"]
    chunker_version: str = Field(min_length=1)
    tokenizer_version: str = Field(min_length=1)
    run_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    input_normalization_run_id: str = Field(min_length=1)
    input_normalization_commit_directory: str = Field(min_length=1)
    input_normalization_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    structural_max_tokens: int = Field(ge=1)
    input_normalized_block_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    next_action: Literal["INDEX_VALIDATE"] = "INDEX_VALIDATE"
    ocr_invocation_count: Literal[0] = 0
    artifact_count: int = Field(ge=3)
    artifacts: list[CommitArtifactRecord] = Field(min_length=3)
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        paths = [artifact.relative_path for artifact in self.artifacts]

        if len(paths) != len(set(paths)):
            raise ValueError("Chunking artifact paths must be unique")

        if self.artifact_count != len(self.artifacts):
            raise ValueError("artifact_count must equal artifacts length")

        ChunkingPolicy(
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            structural_max_tokens=self.structural_max_tokens,
        )
        return self


@dataclass(frozen=True, slots=True)
class SemanticChunkerConfig:
    output_root: Path
    target_tokens: int = 350
    max_tokens: int = 512
    structural_max_tokens: int = 2048

    @property
    def policy(self) -> ChunkingPolicy:
        return ChunkingPolicy(
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            structural_max_tokens=self.structural_max_tokens,
        )


@dataclass(frozen=True, slots=True)
class TextRange:
    char_start: int
    char_end: int
    text: str
    token_count: int


@dataclass(frozen=True, slots=True)
class SourceSegment:
    block_id: str
    source_element_id: str
    element_type: ElementType
    heading_path: tuple[str, ...]
    source_anchor: SourceAnchor
    normalized_span_id: str
    char_start: int
    char_end: int
    text: str
    token_count: int
    raw_sha256: str
    structural: bool
    heading: bool


@dataclass(frozen=True, slots=True)
class ChunkingInputs:
    normalization: NormalizationCommitBundle
    extraction: ExtractionCommitBundle


@dataclass(frozen=True, slots=True)
class ChunkingBuildResult:
    chunks: tuple[ChunkRecord, ...]
    jsonl: str
    report: ChunkingReport


@dataclass(frozen=True, slots=True)
class ChunkingCommitBundle:
    commit_directory: Path
    manifest: ChunkingCommitManifest
    chunks: tuple[ChunkRecord, ...]
    jsonl: str
    report: ChunkingReport
    stage_result: StageResult


class ChunkingError(RuntimeError):
    """Base exception for semantic chunking."""


class ChunkingInputError(ChunkingError):
    pass


class ChunkingStructureError(ChunkingError):
    pass


class ChunkingCommitExistsError(ChunkingError):
    pass


class ChunkingCommitIntegrityError(ChunkingError):
    pass


def count_tokens(text: str) -> int:
    return sum(1 for _ in TOKEN_PATTERN.finditer(text))


def split_semantic_ranges(
    text: str,
    max_tokens: int,
) -> tuple[TextRange, ...]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    matches = list(TOKEN_PATTERN.finditer(text))

    if not matches:
        raise ChunkingStructureError("Text does not contain indexable tokens")

    ranges: list[TextRange] = []
    cursor = 0
    token_index = 0

    while token_index < len(matches):
        remaining = len(matches) - token_index

        if remaining <= max_tokens:
            char_end = len(text)
            consumed = remaining
        else:
            window = matches[token_index : token_index + max_tokens]
            preferred_start = max(1, int(max_tokens * 0.6))
            preferred: list[tuple[int, int]] = []

            for local_index, match in enumerate(window, start=1):
                if local_index < preferred_start:
                    continue

                next_start = (
                    matches[token_index + local_index].start()
                    if token_index + local_index < len(matches)
                    else len(text)
                )
                gap = text[match.end() : next_start]

                if match.group(0) in {".", "!", "?", ";", ":"}:
                    preferred.append((local_index, match.end()))
                elif "\n" in gap:
                    preferred.append((local_index, match.end()))

            if preferred:
                consumed, char_end = preferred[-1]
            else:
                consumed = max_tokens
                char_end = window[-1].end()

        raw_text = text[cursor:char_end]
        normalized_text = raw_text.strip()
        token_count = count_tokens(normalized_text)

        if not normalized_text or token_count < 1:
            raise ChunkingStructureError(
                "Semantic split produced an empty text range"
            )

        if token_count > max_tokens:
            raise ChunkingStructureError(
                "Semantic split exceeded the configured token limit"
            )

        ranges.append(
            TextRange(
                char_start=cursor,
                char_end=char_end,
                text=normalized_text,
                token_count=token_count,
            )
        )
        cursor = char_end
        token_index += consumed

    if cursor < len(text):
        final = ranges[-1]
        raw_text = text[final.char_start :]
        normalized_text = raw_text.strip()
        ranges[-1] = TextRange(
            char_start=final.char_start,
            char_end=len(text),
            text=normalized_text,
            token_count=count_tokens(normalized_text),
        )

    expected_start = 0

    for text_range in ranges:
        if text_range.char_start != expected_start:
            raise ChunkingStructureError(
                "Semantic text ranges are not contiguous"
            )
        expected_start = text_range.char_end

    if expected_start != len(text):
        raise ChunkingStructureError(
            "Semantic text ranges do not cover the complete block"
        )

    return tuple(ranges)


def load_chunking_inputs(
    normalization_commit_directory: Path | str,
) -> ChunkingInputs:
    normalization = load_normalization_commit(
        normalization_commit_directory
    )
    extraction = load_extraction_commit(
        normalization.manifest.input_extraction_commit_directory
    )

    if normalization.manifest.input_extraction_run_id != (
        extraction.manifest.run_id
    ):
        raise ChunkingInputError(
            "Normalization references a different extraction run"
        )

    return ChunkingInputs(
        normalization=normalization,
        extraction=extraction,
    )


def validate_chunking_inputs(inputs: ChunkingInputs) -> None:
    normalization = inputs.normalization
    extraction = inputs.extraction

    if normalization.manifest.next_action != "CHUNK":
        raise ChunkingInputError(
            "Normalization commit is not routed to CHUNK"
        )

    if normalization.stage_result.status != StageStatus.PASS:
        raise ChunkingInputError(
            "Normalization stage must have PASS status"
        )

    normalized_document = normalization.document
    extracted_document = extraction.document
    normalized_identity = (
        normalized_document.catalog_id,
        normalized_document.family_id,
        normalized_document.version_id,
        normalized_document.source_sha256,
    )
    extracted_identity = (
        extracted_document.catalog_id,
        extracted_document.family_id,
        extracted_document.version_id,
        extracted_document.source_sha256,
    )

    if normalized_identity != extracted_identity:
        raise ChunkingInputError(
            "Normalization and extraction identities differ"
        )

    source_element_ids = [
        block.source_spans[0].source_element_id
        for block in normalized_document.blocks
    ]

    if source_element_ids != normalization.report.source_element_ids:
        raise ChunkingInputError(
            "Normalized block order differs from normalization report"
        )

    if source_element_ids != extraction.selection.indexable_element_ids:
        raise ChunkingInputError(
            "Normalized blocks differ from extraction selection"
        )

    extracted_ids = {
        element.element_id
        for element in extracted_document.elements
    }

    if not set(source_element_ids) <= extracted_ids:
        raise ChunkingInputError(
            "A normalized source element is absent from extraction"
        )


def _segment_payload(segment: SourceSegment) -> dict[str, object]:
    return {
        "block_id": segment.block_id,
        "source_element_id": segment.source_element_id,
        "element_type": segment.element_type.value,
        "normalized_span_id": segment.normalized_span_id,
        "source_anchor_id": segment.source_anchor.anchor_id,
        "char_start": segment.char_start,
        "char_end": segment.char_end,
        "raw_sha256": segment.raw_sha256,
    }


def _build_segments(
    inputs: ChunkingInputs,
    policy: ChunkingPolicy,
) -> tuple[SourceSegment, ...]:
    normalized_document = inputs.normalization.document
    element_types = {
        element.element_id: element.element_type
        for element in inputs.extraction.document.elements
    }
    segments: list[SourceSegment] = []

    for block in normalized_document.blocks:
        span = block.source_spans[0]
        element_type = element_types[span.source_element_id]
        structural = element_type in STRUCTURAL_TYPES
        block_token_count = count_tokens(block.markdown)

        if block_token_count < 1:
            raise ChunkingStructureError(
                f"Normalized block has no tokens: {block.block_id}"
            )

        if structural and block_token_count > policy.structural_max_tokens:
            raise ChunkingStructureError(
                "Structural block exceeds structural_max_tokens: "
                f"{block.block_id}"
            )

        if structural or block_token_count <= policy.max_tokens:
            ranges = (
                TextRange(
                    char_start=0,
                    char_end=len(block.markdown),
                    text=block.markdown,
                    token_count=block_token_count,
                ),
            )
        else:
            ranges = split_semantic_ranges(
                block.markdown,
                policy.max_tokens,
            )

        for text_range in ranges:
            raw_text = block.markdown[
                text_range.char_start : text_range.char_end
            ]
            segments.append(
                SourceSegment(
                    block_id=block.block_id,
                    source_element_id=span.source_element_id,
                    element_type=element_type,
                    heading_path=tuple(block.heading_path),
                    source_anchor=span.source_anchor,
                    normalized_span_id=span.span_id,
                    char_start=text_range.char_start,
                    char_end=text_range.char_end,
                    text=text_range.text,
                    token_count=text_range.token_count,
                    raw_sha256=hashlib.sha256(
                        raw_text.encode("utf-8")
                    ).hexdigest(),
                    structural=structural,
                    heading=element_type in HEADING_TYPES,
                )
            )

    return tuple(segments)


def _group_segments(
    segments: tuple[SourceSegment, ...],
    policy: ChunkingPolicy,
) -> tuple[tuple[SourceSegment, ...], ...]:
    groups: list[tuple[SourceSegment, ...]] = []
    pending: list[SourceSegment] = []

    def flush() -> None:
        if pending:
            groups.append(tuple(pending))
            pending.clear()

    for segment in segments:
        if segment.structural:
            flush()
            groups.append((segment,))
            continue

        if segment.heading:
            flush()

        if pending and pending[0].heading_path != segment.heading_path:
            flush()

        candidate = "\n\n".join(
            item.text for item in [*pending, segment]
        )

        if pending and count_tokens(candidate) > policy.target_tokens:
            flush()

        pending.append(segment)

        if count_tokens(segment.text) > policy.target_tokens:
            flush()

    flush()
    return tuple(groups)


def _deduplicated_anchors(
    group: tuple[SourceSegment, ...],
) -> list[SourceAnchor]:
    anchors: list[SourceAnchor] = []
    seen: set[str] = set()

    for segment in group:
        anchor = segment.source_anchor

        if anchor.anchor_id not in seen:
            seen.add(anchor.anchor_id)
            anchors.append(anchor)

    return anchors


def _stable_chunk_id(
    family_id: str,
    source_sha256: str,
    heading_path: tuple[str, ...],
    text: str,
    group: tuple[SourceSegment, ...],
) -> str:
    payload = {
        "source_sha256": source_sha256,
        "heading_path": list(heading_path),
        "text": text,
        "segments": [_segment_payload(segment) for segment in group],
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(serialized).hexdigest()[:32]
    return f"{family_id}#semantic-{digest}"


def _build_chunk_records(
    inputs: ChunkingInputs,
    groups: tuple[tuple[SourceSegment, ...], ...],
    policy: ChunkingPolicy,
) -> tuple[ChunkRecord, ...]:
    document = inputs.normalization.document
    chunks: list[ChunkRecord] = []

    for ordinal, group in enumerate(groups, start=1):
        text = "\n\n".join(segment.text for segment in group)
        token_count = count_tokens(text)
        heading_path = group[0].heading_path
        chunk_id = _stable_chunk_id(
            document.family_id,
            document.source_sha256,
            heading_path,
            text,
            group,
        )
        structural = any(segment.structural for segment in group)
        chunks.append(
            ChunkRecord(
                family_id=document.family_id,
                version_id=document.version_id,
                chunk_id=chunk_id,
                chunk_instance_id=(
                    f"{document.version_id}::{chunk_id}"
                ),
                source_sha256=document.source_sha256,
                ordinal=ordinal,
                heading_path=list(heading_path),
                text=text,
                token_count=token_count,
                source_anchors=_deduplicated_anchors(group),
                metadata={
                    "chunker_version": SEMANTIC_CHUNKER_VERSION,
                    "tokenizer_version": TOKENIZER_VERSION,
                    "target_tokens": policy.target_tokens,
                    "max_tokens": policy.max_tokens,
                    "structural_max_tokens": (
                        policy.structural_max_tokens
                    ),
                    "contains_structural_block": structural,
                    "oversized_atomic": (
                        structural and token_count > policy.max_tokens
                    ),
                    "block_ids": list(
                        dict.fromkeys(
                            segment.block_id for segment in group
                        )
                    ),
                    "source_element_ids": list(
                        dict.fromkeys(
                            segment.source_element_id
                            for segment in group
                        )
                    ),
                    "element_types": list(
                        dict.fromkeys(
                            segment.element_type.value
                            for segment in group
                        )
                    ),
                    "source_segments": [
                        _segment_payload(segment) for segment in group
                    ],
                },
            )
        )

    return tuple(chunks)


def _validate_text_range_coverage(
    inputs: ChunkingInputs,
    segments: tuple[SourceSegment, ...],
) -> None:
    segments_by_block: dict[str, list[SourceSegment]] = {}

    for segment in segments:
        segments_by_block.setdefault(segment.block_id, []).append(segment)

    for block in inputs.normalization.document.blocks:
        block_segments = segments_by_block.get(block.block_id, [])

        if not block_segments:
            raise ChunkingStructureError(
                f"Normalized block is not represented: {block.block_id}"
            )

        expected_start = 0

        for segment in block_segments:
            if segment.char_start != expected_start:
                raise ChunkingStructureError(
                    f"Non-contiguous source ranges: {block.block_id}"
                )
            expected_start = segment.char_end

        if expected_start != len(block.markdown):
            raise ChunkingStructureError(
                f"Incomplete source range coverage: {block.block_id}"
            )


def _chunks_jsonl(chunks: tuple[ChunkRecord, ...]) -> str:
    return "".join(
        chunk.model_dump_json() + "\n"
        for chunk in chunks
    )


def build_chunking(
    inputs: ChunkingInputs,
    policy: ChunkingPolicy | None = None,
) -> ChunkingBuildResult:
    active_policy = policy or ChunkingPolicy()
    validate_chunking_inputs(inputs)
    segments = _build_segments(inputs, active_policy)
    _validate_text_range_coverage(inputs, segments)
    groups = _group_segments(segments, active_policy)
    chunks = _build_chunk_records(inputs, groups, active_policy)

    if not chunks:
        raise ChunkingStructureError("Semantic chunker produced no chunks")

    token_counts = [chunk.token_count for chunk in chunks]
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    normalized_blocks = inputs.normalization.document.blocks
    normalized_block_ids = [block.block_id for block in normalized_blocks]
    source_element_ids = [
        block.source_spans[0].source_element_id
        for block in normalized_blocks
    ]
    element_types = {
        element.element_id: element.element_type
        for element in inputs.extraction.document.elements
    }
    type_counter = Counter(
        element_types[source_element_id]
        for source_element_id in source_element_ids
    )
    structural_chunks = [
        chunk
        for chunk in chunks
        if chunk.metadata["contains_structural_block"]
    ]
    oversized_atomic = [
        chunk
        for chunk in chunks
        if chunk.metadata["oversized_atomic"]
    ]
    anchors = [
        anchor
        for chunk in chunks
        for anchor in chunk.source_anchors
    ]
    unique_anchor_ids = {
        anchor.anchor_id for anchor in anchors
    }
    page_numbers = sorted(
        {
            anchor.page_number
            for anchor in anchors
            if anchor.page_number is not None
        }
    )
    structural_block_count = sum(
        type_counter[element_type]
        for element_type in STRUCTURAL_TYPES
    )
    report = ChunkingReport(
        chunker_version=SEMANTIC_CHUNKER_VERSION,
        tokenizer_version=TOKENIZER_VERSION,
        input_normalization_run_id=(
            inputs.normalization.manifest.run_id
        ),
        catalog_id=inputs.normalization.document.catalog_id,
        family_id=inputs.normalization.document.family_id,
        version_id=inputs.normalization.document.version_id,
        source_sha256=inputs.normalization.document.source_sha256,
        target_tokens=active_policy.target_tokens,
        max_tokens=active_policy.max_tokens,
        structural_max_tokens=active_policy.structural_max_tokens,
        input_normalized_block_count=len(normalized_blocks),
        represented_normalized_block_count=len(normalized_block_ids),
        source_segment_count=len(segments),
        chunk_count=len(chunks),
        total_token_count=sum(token_counts),
        min_chunk_token_count=min(token_counts),
        max_chunk_token_count=max(token_counts),
        table_block_count=type_counter[ElementType.TABLE],
        list_block_count=type_counter[ElementType.LIST],
        flowchart_block_count=type_counter[ElementType.FLOWCHART],
        formula_block_count=type_counter[ElementType.FORMULA],
        structural_block_count=structural_block_count,
        structural_chunk_count=len(structural_chunks),
        oversized_atomic_chunk_count=len(oversized_atomic),
        source_anchor_count=len(anchors),
        unique_source_anchor_count=len(unique_anchor_ids),
        page_numbers=page_numbers,
        normalized_block_ids=normalized_block_ids,
        source_element_ids=source_element_ids,
        chunk_ids=chunk_ids,
        generated_at=inputs.normalization.manifest.committed_at,
    )
    return ChunkingBuildResult(
        chunks=chunks,
        jsonl=_chunks_jsonl(chunks),
        report=report,
    )


class SemanticChunker:
    def __init__(self, config: SemanticChunkerConfig) -> None:
        self.config = config
        self.policy = config.policy

    def chunk_commit(
        self,
        normalization_commit_directory: Path | str,
        *,
        run_id: str | None = None,
    ) -> ChunkingCommitBundle:
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        inputs = load_chunking_inputs(normalization_commit_directory)
        active_run_id = run_id or utc_run_id().replace(
            "extraction-",
            "chunking-",
            1,
        )
        validate_run_id(active_run_id)
        self._ensure_target_is_available(active_run_id)
        build = build_chunking(inputs, self.policy)
        report = build.report
        stage_result = StageResult(
            stage=ResumeFrom.CHUNK,
            status=StageStatus.PASS,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_seconds=time.monotonic() - started_monotonic,
            artifact_paths=[
                CHUNKS_FILENAME,
                CHUNKING_REPORT_FILENAME,
            ],
            metrics={
                "input_normalized_block_count": (
                    report.input_normalized_block_count
                ),
                "chunk_count": report.chunk_count,
                "total_token_count": report.total_token_count,
                "structural_chunk_count": (
                    report.structural_chunk_count
                ),
                "oversized_atomic_chunk_count": (
                    report.oversized_atomic_chunk_count
                ),
                "ocr_invocation_count": 0,
                "next_action": "INDEX_VALIDATE",
            },
            message="Semantic structure-aware chunking passed",
        )
        return self._commit(
            run_id=active_run_id,
            inputs=inputs,
            build=build,
            stage_result=stage_result,
        )

    def _ensure_target_is_available(self, run_id: str) -> None:
        output_root = self.config.output_root.expanduser().resolve()

        if (output_root / run_id).exists():
            raise ChunkingCommitExistsError(
                f"Chunking commit already exists: {run_id}"
            )

    def _commit(
        self,
        *,
        run_id: str,
        inputs: ChunkingInputs,
        build: ChunkingBuildResult,
        stage_result: StageResult,
    ) -> ChunkingCommitBundle:
        output_root = self.config.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        final_directory = output_root / run_id

        if final_directory.exists():
            raise ChunkingCommitExistsError(
                f"Chunking commit already exists: {run_id}"
            )

        staging_directory = output_root / (
            f".{run_id}.staging-{uuid.uuid4().hex}"
        )
        staging_directory.mkdir(parents=False, exist_ok=False)

        try:
            payloads = {
                CHUNKS_FILENAME: build.jsonl.encode("utf-8"),
                CHUNKING_REPORT_FILENAME: model_json_bytes(
                    build.report
                ),
                STAGE_RESULT_FILENAME: model_json_bytes(stage_result),
            }
            artifacts: list[CommitArtifactRecord] = []

            for filename, payload in payloads.items():
                path = staging_directory / filename
                write_payload(path, payload)
                artifacts.append(
                    artifact_record(path, staging_directory)
                )

            normalization_manifest_path = (
                inputs.normalization.commit_directory
                / COMMIT_MANIFEST_FILENAME
            )
            report = build.report
            manifest = ChunkingCommitManifest(
                schema_version=CHUNKING_COMMIT_SCHEMA_VERSION,
                chunker_version=SEMANTIC_CHUNKER_VERSION,
                tokenizer_version=TOKENIZER_VERSION,
                run_id=run_id,
                input_normalization_run_id=(
                    inputs.normalization.manifest.run_id
                ),
                input_normalization_commit_directory=str(
                    inputs.normalization.commit_directory
                ),
                input_normalization_manifest_sha256=sha256_file(
                    normalization_manifest_path
                ),
                catalog_id=report.catalog_id,
                family_id=report.family_id,
                version_id=report.version_id,
                source_sha256=report.source_sha256,
                target_tokens=report.target_tokens,
                max_tokens=report.max_tokens,
                structural_max_tokens=report.structural_max_tokens,
                input_normalized_block_count=(
                    report.input_normalized_block_count
                ),
                chunk_count=report.chunk_count,
                artifact_count=len(artifacts),
                artifacts=artifacts,
            )
            write_payload(
                staging_directory / COMMIT_MANIFEST_FILENAME,
                model_json_bytes(manifest),
            )
            load_chunking_commit(staging_directory)

            if final_directory.exists():
                raise ChunkingCommitExistsError(
                    f"Chunking commit already exists: {run_id}"
                )

            staging_directory.rename(final_directory)

        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
            raise

        return load_chunking_commit(final_directory)


def _parse_chunks_jsonl(raw_jsonl: str) -> tuple[ChunkRecord, ...]:
    if not raw_jsonl or not raw_jsonl.endswith("\n"):
        raise ChunkingCommitIntegrityError(
            "Chunk JSONL must be non-empty and newline terminated"
        )

    lines = raw_jsonl.splitlines()

    if not lines or any(not line.strip() for line in lines):
        raise ChunkingCommitIntegrityError(
            "Chunk JSONL contains an empty record"
        )

    try:
        return tuple(
            ChunkRecord.model_validate_json(line)
            for line in lines
        )
    except Exception as error:
        raise ChunkingCommitIntegrityError(
            "Chunk JSONL contains an invalid ChunkRecord"
        ) from error


def load_chunking_commit(
    commit_directory: Path | str,
    *,
    verify_input_commit: bool = True,
) -> ChunkingCommitBundle:
    root = Path(commit_directory).expanduser().resolve()

    if not root.is_dir():
        raise ChunkingCommitIntegrityError(
            f"Chunking commit directory not found: {root}"
        )

    try:
        manifest = ChunkingCommitManifest.model_validate_json(
            (root / COMMIT_MANIFEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise ChunkingCommitIntegrityError(
            "Chunking manifest is missing or invalid"
        ) from error

    recorded_paths = {
        artifact.relative_path for artifact in manifest.artifacts
    }
    required_paths = {
        CHUNKS_FILENAME,
        CHUNKING_REPORT_FILENAME,
        STAGE_RESULT_FILENAME,
    }

    if recorded_paths != required_paths:
        raise ChunkingCommitIntegrityError(
            "Chunking commit has unexpected typed artifacts"
        )

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }

    if actual_paths != required_paths | {COMMIT_MANIFEST_FILENAME}:
        raise ChunkingCommitIntegrityError(
            "Chunking file inventory differs from manifest"
        )

    for artifact in manifest.artifacts:
        path = (root / artifact.relative_path).resolve()

        if root not in path.parents:
            raise ChunkingCommitIntegrityError(
                "Chunking artifact escapes commit directory"
            )

        if path.stat().st_size != artifact.size_bytes:
            raise ChunkingCommitIntegrityError(
                f"Chunking size mismatch: {artifact.relative_path}"
            )

        if sha256_file(path) != artifact.sha256:
            raise ChunkingCommitIntegrityError(
                f"Chunking SHA256 mismatch: {artifact.relative_path}"
            )

    try:
        raw_jsonl = (root / CHUNKS_FILENAME).read_text(
            encoding="utf-8"
        )
        chunks = _parse_chunks_jsonl(raw_jsonl)
        report = ChunkingReport.model_validate_json(
            (root / CHUNKING_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        stage_result = StageResult.model_validate_json(
            (root / STAGE_RESULT_FILENAME).read_text(encoding="utf-8")
        )
    except ChunkingCommitIntegrityError:
        raise
    except Exception as error:
        raise ChunkingCommitIntegrityError(
            "Committed chunking artifact is invalid"
        ) from error

    _validate_chunking_commit(
        manifest=manifest,
        chunks=chunks,
        raw_jsonl=raw_jsonl,
        report=report,
        stage_result=stage_result,
    )

    if verify_input_commit:
        try:
            inputs = load_chunking_inputs(
                manifest.input_normalization_commit_directory
            )
        except Exception as error:
            raise ChunkingCommitIntegrityError(
                "Input normalization commit is invalid"
            ) from error

        normalization_manifest_path = (
            inputs.normalization.commit_directory
            / COMMIT_MANIFEST_FILENAME
        )

        if sha256_file(normalization_manifest_path) != (
            manifest.input_normalization_manifest_sha256
        ):
            raise ChunkingCommitIntegrityError(
                "Input normalization manifest SHA256 changed"
            )

        if inputs.normalization.manifest.run_id != (
            manifest.input_normalization_run_id
        ):
            raise ChunkingCommitIntegrityError(
                "Input normalization run ID changed"
            )

        policy = ChunkingPolicy(
            target_tokens=manifest.target_tokens,
            max_tokens=manifest.max_tokens,
            structural_max_tokens=manifest.structural_max_tokens,
        )
        expected = build_chunking(inputs, policy)

        if (
            expected.chunks != chunks
            or expected.jsonl != raw_jsonl
            or expected.report != report
        ):
            raise ChunkingCommitIntegrityError(
                "Chunking is not reproducible from normalization commit"
            )

    return ChunkingCommitBundle(
        commit_directory=root,
        manifest=manifest,
        chunks=chunks,
        jsonl=raw_jsonl,
        report=report,
        stage_result=stage_result,
    )


def _validate_chunking_commit(
    *,
    manifest: ChunkingCommitManifest,
    chunks: tuple[ChunkRecord, ...],
    raw_jsonl: str,
    report: ChunkingReport,
    stage_result: StageResult,
) -> None:
    if (
        manifest.chunker_version != SEMANTIC_CHUNKER_VERSION
        or report.chunker_version != SEMANTIC_CHUNKER_VERSION
    ):
        raise ChunkingCommitIntegrityError(
            "Chunker version differs from installed chunker"
        )

    if (
        manifest.tokenizer_version != TOKENIZER_VERSION
        or report.tokenizer_version != TOKENIZER_VERSION
    ):
        raise ChunkingCommitIntegrityError(
            "Tokenizer version differs from installed tokenizer"
        )

    identity = (
        manifest.catalog_id,
        manifest.family_id,
        manifest.version_id,
        manifest.source_sha256,
    )

    if identity != (
        report.catalog_id,
        report.family_id,
        report.version_id,
        report.source_sha256,
    ):
        raise ChunkingCommitIntegrityError(
            "Chunking report identity differs from manifest"
        )

    if any(
        (
            chunk.family_id,
            chunk.version_id,
            chunk.source_sha256,
        )
        != (
            manifest.family_id,
            manifest.version_id,
            manifest.source_sha256,
        )
        for chunk in chunks
    ):
        raise ChunkingCommitIntegrityError(
            "A chunk identity differs from manifest"
        )

    expected_ordinals = list(range(1, len(chunks) + 1))

    if [chunk.ordinal for chunk in chunks] != expected_ordinals:
        raise ChunkingCommitIntegrityError(
            "Chunk ordinals must be contiguous and one-based"
        )

    if [chunk.chunk_id for chunk in chunks] != report.chunk_ids:
        raise ChunkingCommitIntegrityError(
            "Chunk order differs from report"
        )

    if len(chunks) != manifest.chunk_count:
        raise ChunkingCommitIntegrityError(
            "Chunk count differs from manifest"
        )

    if manifest.chunk_count != report.chunk_count:
        raise ChunkingCommitIntegrityError(
            "Chunk count differs between manifest and report"
        )

    if manifest.input_normalized_block_count != (
        report.input_normalized_block_count
    ):
        raise ChunkingCommitIntegrityError(
            "Input block count differs between manifest and report"
        )

    policy_tuple = (
        manifest.target_tokens,
        manifest.max_tokens,
        manifest.structural_max_tokens,
    )

    if policy_tuple != (
        report.target_tokens,
        report.max_tokens,
        report.structural_max_tokens,
    ):
        raise ChunkingCommitIntegrityError(
            "Chunking policy differs between manifest and report"
        )

    if _chunks_jsonl(chunks) != raw_jsonl:
        raise ChunkingCommitIntegrityError(
            "Chunk JSONL is not canonical"
        )

    metadata_block_ids: list[str] = []
    metadata_source_element_ids: list[str] = []
    metadata_source_segment_count = 0
    source_anchors: list[SourceAnchor] = []
    structural_chunk_count = 0
    oversized_atomic_chunk_count = 0

    for chunk in chunks:
        if count_tokens(chunk.text) != chunk.token_count:
            raise ChunkingCommitIntegrityError(
                f"Token count differs for chunk: {chunk.chunk_id}"
            )

        metadata = chunk.metadata

        if metadata.get("chunker_version") != SEMANTIC_CHUNKER_VERSION:
            raise ChunkingCommitIntegrityError(
                "Chunk metadata has an unexpected chunker version"
            )

        if metadata.get("tokenizer_version") != TOKENIZER_VERSION:
            raise ChunkingCommitIntegrityError(
                "Chunk metadata has an unexpected tokenizer version"
            )

        source_segments = metadata.get("source_segments")

        if not isinstance(source_segments, list) or not source_segments:
            raise ChunkingCommitIntegrityError(
                "Chunk metadata must contain source segments"
            )

        block_ids = metadata.get("block_ids")
        source_element_ids = metadata.get("source_element_ids")

        if not isinstance(block_ids, list) or not block_ids:
            raise ChunkingCommitIntegrityError(
                "Chunk metadata must contain normalized block IDs"
            )

        if not isinstance(source_element_ids, list) or not source_element_ids:
            raise ChunkingCommitIntegrityError(
                "Chunk metadata must contain source element IDs"
            )

        for block_id in block_ids:
            if block_id not in metadata_block_ids:
                metadata_block_ids.append(block_id)

        for source_element_id in source_element_ids:
            if source_element_id not in metadata_source_element_ids:
                metadata_source_element_ids.append(source_element_id)

        metadata_source_segment_count += len(source_segments)
        source_anchors.extend(chunk.source_anchors)
        contains_structural = bool(
            metadata.get("contains_structural_block")
        )
        oversized_atomic = bool(metadata.get("oversized_atomic"))

        if contains_structural:
            structural_chunk_count += 1

        if oversized_atomic:
            oversized_atomic_chunk_count += 1

        if not contains_structural:
            if chunk.token_count > manifest.max_tokens:
                raise ChunkingCommitIntegrityError(
                    "Non-structural chunk exceeds max_tokens"
                )
        elif chunk.token_count > manifest.structural_max_tokens:
            raise ChunkingCommitIntegrityError(
                "Structural chunk exceeds structural_max_tokens"
            )

        if oversized_atomic != (
            contains_structural
            and chunk.token_count > manifest.max_tokens
        ):
            raise ChunkingCommitIntegrityError(
                "Chunk oversized_atomic flag is inconsistent"
            )

    if sum(chunk.token_count for chunk in chunks) != (
        report.total_token_count
    ):
        raise ChunkingCommitIntegrityError(
            "Total token count differs from report"
        )

    token_counts = [chunk.token_count for chunk in chunks]

    if (
        min(token_counts) != report.min_chunk_token_count
        or max(token_counts) != report.max_chunk_token_count
    ):
        raise ChunkingCommitIntegrityError(
            "Chunk token range differs from report"
        )

    if metadata_block_ids != report.normalized_block_ids:
        raise ChunkingCommitIntegrityError(
            "Normalized block order differs from report"
        )

    if metadata_source_element_ids != report.source_element_ids:
        raise ChunkingCommitIntegrityError(
            "Source element order differs from report"
        )

    if metadata_source_segment_count != report.source_segment_count:
        raise ChunkingCommitIntegrityError(
            "Source segment count differs from report"
        )

    if len(source_anchors) != report.source_anchor_count:
        raise ChunkingCommitIntegrityError(
            "Source anchor count differs from report"
        )

    if len({anchor.anchor_id for anchor in source_anchors}) != (
        report.unique_source_anchor_count
    ):
        raise ChunkingCommitIntegrityError(
            "Unique source anchor count differs from report"
        )

    page_numbers = sorted(
        {
            anchor.page_number
            for anchor in source_anchors
            if anchor.page_number is not None
        }
    )

    if page_numbers != report.page_numbers:
        raise ChunkingCommitIntegrityError(
            "Provenance page inventory differs from report"
        )

    if structural_chunk_count != report.structural_chunk_count:
        raise ChunkingCommitIntegrityError(
            "Structural chunk count differs from report"
        )

    if oversized_atomic_chunk_count != (
        report.oversized_atomic_chunk_count
    ):
        raise ChunkingCommitIntegrityError(
            "Oversized atomic chunk count differs from report"
        )

    if manifest.input_normalization_run_id != (
        report.input_normalization_run_id
    ):
        raise ChunkingCommitIntegrityError(
            "Input normalization run differs from report"
        )

    if (
        stage_result.stage != ResumeFrom.CHUNK
        or stage_result.status != StageStatus.PASS
        or stage_result.metrics.get("next_action")
        != "INDEX_VALIDATE"
    ):
        raise ChunkingCommitIntegrityError(
            "Chunking stage result is inconsistent"
        )

    if stage_result.artifact_paths != [
        CHUNKS_FILENAME,
        CHUNKING_REPORT_FILENAME,
    ]:
        raise ChunkingCommitIntegrityError(
            "Chunking stage artifact paths are inconsistent"
        )

    expected_stage_metrics = {
        "input_normalized_block_count": (
            report.input_normalized_block_count
        ),
        "chunk_count": report.chunk_count,
        "total_token_count": report.total_token_count,
        "structural_chunk_count": report.structural_chunk_count,
        "oversized_atomic_chunk_count": (
            report.oversized_atomic_chunk_count
        ),
        "ocr_invocation_count": 0,
        "next_action": "INDEX_VALIDATE",
    }

    if stage_result.metrics != expected_stage_metrics:
        raise ChunkingCommitIntegrityError(
            "Chunking stage metrics differ from report"
        )
