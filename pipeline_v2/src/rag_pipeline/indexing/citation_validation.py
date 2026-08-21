from __future__ import annotations

import hashlib
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.chunking import (
    ChunkingCommitBundle,
    count_tokens,
    load_chunking_commit,
    load_chunking_inputs,
)
from rag_pipeline.chunking.semantic import ChunkingInputs
from rag_pipeline.contracts import (
    ChunkRecord,
    ElementType,
    ResumeFrom,
    SourceAnchor,
    StageResult,
    StageStatus,
)
from rag_pipeline.normalization import NormalizationCommitBundle
from rag_pipeline.orchestration import (
    CommitArtifactRecord,
    ExtractionCommitBundle,
)
from rag_pipeline.orchestration.extraction import (
    artifact_record,
    model_json_bytes,
    sha256_file,
    stable_identifier,
    utc_run_id,
    validate_run_id,
    write_payload,
)


CITATION_INDEX_VALIDATOR_VERSION = "2026-08-18-5.2C.1"
CITATION_INDEX_COMMIT_SCHEMA_VERSION = "1.0"

CITATIONS_FILENAME = "citations.jsonl"
INDEX_RECORDS_FILENAME = "index_records.jsonl"
VALIDATION_REPORT_FILENAME = "citation_index_report.json"
STAGE_RESULT_FILENAME = "stage_result.json"
COMMIT_MANIFEST_FILENAME = "commit_manifest.json"


class CitationIndexModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class CitationIndexPolicy(CitationIndexModel):
    require_page_numbers: Literal[True] = True
    require_bounding_boxes: Literal[True] = True
    max_index_tokens: int = Field(default=2048, ge=32, le=16384)


class CitationSegment(CitationIndexModel):
    normalized_block_id: str = Field(min_length=1)
    source_element_id: str = Field(min_length=1)
    element_type: ElementType
    normalized_span_id: str = Field(min_length=1)
    source_anchor_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("Citation segment range must be positive")

        return self


class CitationRecord(CitationIndexModel):
    schema_version: Literal["1.0"] = "1.0"
    citation_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_id: str = Field(min_length=1)
    chunk_instance_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    heading_path: list[str]
    display_locator: str = Field(min_length=1)
    page_numbers: list[int] = Field(min_length=1)
    source_anchors: list[SourceAnchor] = Field(min_length=1)
    source_segments: list[CitationSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citation(self) -> Self:
        anchor_ids = [anchor.anchor_id for anchor in self.source_anchors]

        if len(anchor_ids) != len(set(anchor_ids)):
            raise ValueError("Citation source anchors must be unique")

        expected_pages = sorted(
            {
                anchor.page_number
                for anchor in self.source_anchors
                if anchor.page_number is not None
            }
        )

        if self.page_numbers != expected_pages:
            raise ValueError(
                "Citation page_numbers must derive from source anchors"
            )

        if any(
            anchor.source_sha256 != self.source_sha256
            for anchor in self.source_anchors
        ):
            raise ValueError("Citation anchor source SHA256 differs")

        if not {
            segment.source_anchor_id
            for segment in self.source_segments
        } <= set(anchor_ids):
            raise ValueError(
                "Citation segment references an absent source anchor"
            )

        return self


class IndexReadyRecord(CitationIndexModel):
    schema_version: Literal["1.0"] = "1.0"
    index_record_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunk_id: str = Field(min_length=1)
    chunk_instance_id: str = Field(min_length=1)
    ordinal: int = Field(ge=1)
    text: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: int = Field(ge=1)
    heading_path: list[str]
    citation_id: str = Field(min_length=1)
    page_numbers: list[int] = Field(min_length=1)
    source_anchor_ids: list[str] = Field(min_length=1)
    normalized_block_ids: list[str] = Field(min_length=1)
    source_element_ids: list[str] = Field(min_length=1)
    element_types: list[ElementType] = Field(min_length=1)
    contains_structural_block: bool
    oversized_atomic: bool

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        expected_hash = hashlib.sha256(
            self.text.encode("utf-8")
        ).hexdigest()

        if self.content_sha256 != expected_hash:
            raise ValueError("Index content SHA256 differs from text")

        if self.chunk_instance_id != (
            f"{self.version_id}::{self.chunk_id}"
        ):
            raise ValueError("Index chunk instance ID is invalid")

        for values, name in (
            (self.page_numbers, "page_numbers"),
            (self.source_anchor_ids, "source_anchor_ids"),
            (self.normalized_block_ids, "normalized_block_ids"),
            (self.source_element_ids, "source_element_ids"),
            (self.element_types, "element_types"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"Index {name} must be unique")

        if self.page_numbers != sorted(self.page_numbers):
            raise ValueError("Index page_numbers must be sorted")

        return self


class CitationIndexReport(CitationIndexModel):
    validator_version: str = Field(min_length=1)
    input_chunking_run_id: str = Field(min_length=1)
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_chunk_count: int = Field(ge=1)
    citation_count: int = Field(ge=1)
    index_record_count: int = Field(ge=1)
    source_segment_count: int = Field(ge=1)
    source_anchor_reference_count: int = Field(ge=1)
    unique_source_anchor_count: int = Field(ge=1)
    resolved_source_range_count: int = Field(ge=1)
    page_anchor_count: int = Field(ge=1)
    bbox_anchor_count: int = Field(ge=1)
    citations_with_pages_count: int = Field(ge=1)
    citations_with_bboxes_count: int = Field(ge=1)
    structural_chunk_count: int = Field(ge=0)
    total_token_count: int = Field(ge=1)
    min_record_token_count: int = Field(ge=1)
    max_record_token_count: int = Field(ge=1)
    duplicate_content_hash_group_count: int = Field(ge=0)
    page_numbers: list[int] = Field(min_length=1)
    chunk_ids: list[str] = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)
    index_record_ids: list[str] = Field(min_length=1)
    citation_coverage: Literal[1.0] = 1.0
    index_record_coverage: Literal[1.0] = 1.0
    source_range_hash_coverage: Literal[1.0] = 1.0
    page_provenance_coverage: Literal[1.0] = 1.0
    bbox_provenance_coverage: Literal[1.0] = 1.0
    orphan_citation_count: Literal[0] = 0
    orphan_index_record_count: Literal[0] = 0
    duplicate_citation_id_count: Literal[0] = 0
    duplicate_index_record_id_count: Literal[0] = 0
    invalid_source_range_count: Literal[0] = 0
    source_hash_mismatch_count: Literal[0] = 0
    token_count_mismatch_count: Literal[0] = 0
    filtered_element_leakage_count: Literal[0] = 0
    external_call_count: Literal[0] = 0
    ocr_invocation_count: Literal[0] = 0
    embedding_invocation_count: Literal[0] = 0
    passed: Literal[True] = True
    generated_at: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if not (
            self.input_chunk_count
            == self.citation_count
            == self.index_record_count
        ):
            raise ValueError(
                "Every chunk must have one citation and index record"
            )

        list_counts = (
            (self.chunk_ids, self.input_chunk_count, "chunk_ids"),
            (self.citation_ids, self.citation_count, "citation_ids"),
            (
                self.index_record_ids,
                self.index_record_count,
                "index_record_ids",
            ),
        )

        for values, expected_count, name in list_counts:
            if len(values) != expected_count:
                raise ValueError(f"{name} length differs from count")

            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")

        if self.page_numbers != sorted(set(self.page_numbers)):
            raise ValueError("page_numbers must be sorted and unique")

        if self.min_record_token_count > self.max_record_token_count:
            raise ValueError("Index record token range is invalid")

        if self.resolved_source_range_count != self.source_segment_count:
            raise ValueError("Every source segment must resolve")

        if self.citations_with_pages_count != self.citation_count:
            raise ValueError("Every citation must contain a page")

        if self.citations_with_bboxes_count != self.citation_count:
            raise ValueError("Every citation must contain a bbox")

        return self


class CitationIndexCommitManifest(CitationIndexModel):
    schema_version: Literal["1.0"]
    validator_version: str = Field(min_length=1)
    run_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    input_chunking_run_id: str = Field(min_length=1)
    input_chunking_commit_directory: str = Field(min_length=1)
    input_chunking_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    require_page_numbers: Literal[True]
    require_bounding_boxes: Literal[True]
    max_index_tokens: int = Field(ge=1)
    input_chunk_count: int = Field(ge=1)
    citation_count: int = Field(ge=1)
    index_record_count: int = Field(ge=1)
    next_action: Literal["EMBED_AND_STAGE_INDEX"] = (
        "EMBED_AND_STAGE_INDEX"
    )
    external_call_count: Literal[0] = 0
    artifact_count: int = Field(ge=4)
    artifacts: list[CommitArtifactRecord] = Field(min_length=4)
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_inventory(self) -> Self:
        paths = [artifact.relative_path for artifact in self.artifacts]

        if len(paths) != len(set(paths)):
            raise ValueError("Citation/index artifact paths must be unique")

        if self.artifact_count != len(self.artifacts):
            raise ValueError("artifact_count must equal artifacts length")

        if not (
            self.input_chunk_count
            == self.citation_count
            == self.index_record_count
        ):
            raise ValueError("Citation/index manifest counts differ")

        CitationIndexPolicy(
            require_page_numbers=self.require_page_numbers,
            require_bounding_boxes=self.require_bounding_boxes,
            max_index_tokens=self.max_index_tokens,
        )
        return self


@dataclass(frozen=True, slots=True)
class CitationIndexValidatorConfig:
    output_root: Path
    require_page_numbers: bool = True
    require_bounding_boxes: bool = True
    max_index_tokens: int = 2048

    @property
    def policy(self) -> CitationIndexPolicy:
        return CitationIndexPolicy(
            require_page_numbers=self.require_page_numbers,
            require_bounding_boxes=self.require_bounding_boxes,
            max_index_tokens=self.max_index_tokens,
        )


@dataclass(frozen=True, slots=True)
class IndexValidationInputs:
    chunking: ChunkingCommitBundle
    normalization: NormalizationCommitBundle
    extraction: ExtractionCommitBundle


@dataclass(frozen=True, slots=True)
class CitationIndexBuildResult:
    citations: tuple[CitationRecord, ...]
    citation_jsonl: str
    index_records: tuple[IndexReadyRecord, ...]
    index_jsonl: str
    report: CitationIndexReport


@dataclass(frozen=True, slots=True)
class CitationIndexCommitBundle:
    commit_directory: Path
    manifest: CitationIndexCommitManifest
    citations: tuple[CitationRecord, ...]
    citation_jsonl: str
    index_records: tuple[IndexReadyRecord, ...]
    index_jsonl: str
    report: CitationIndexReport
    stage_result: StageResult


class CitationIndexError(RuntimeError):
    """Base exception for citation and index validation."""


class CitationIndexInputError(CitationIndexError):
    pass


class CitationIndexStructureError(CitationIndexError):
    pass


class CitationIndexCommitExistsError(CitationIndexError):
    pass


class CitationIndexCommitIntegrityError(CitationIndexError):
    pass


def load_index_validation_inputs(
    chunking_commit_directory: Path | str,
) -> IndexValidationInputs:
    chunking = load_chunking_commit(chunking_commit_directory)
    chunking_inputs: ChunkingInputs = load_chunking_inputs(
        chunking.manifest.input_normalization_commit_directory
    )

    if chunking.manifest.input_normalization_run_id != (
        chunking_inputs.normalization.manifest.run_id
    ):
        raise CitationIndexInputError(
            "Chunking references a different normalization run"
        )

    return IndexValidationInputs(
        chunking=chunking,
        normalization=chunking_inputs.normalization,
        extraction=chunking_inputs.extraction,
    )


def validate_index_inputs(inputs: IndexValidationInputs) -> None:
    chunking = inputs.chunking

    if chunking.manifest.next_action != "INDEX_VALIDATE":
        raise CitationIndexInputError(
            "Chunking commit is not routed to INDEX_VALIDATE"
        )

    if chunking.stage_result.status != StageStatus.PASS:
        raise CitationIndexInputError("Chunking stage must have PASS status")

    identity = (
        chunking.manifest.catalog_id,
        chunking.manifest.family_id,
        chunking.manifest.version_id,
        chunking.manifest.source_sha256,
    )
    normalization_identity = (
        inputs.normalization.document.catalog_id,
        inputs.normalization.document.family_id,
        inputs.normalization.document.version_id,
        inputs.normalization.document.source_sha256,
    )
    extraction_identity = (
        inputs.extraction.document.catalog_id,
        inputs.extraction.document.family_id,
        inputs.extraction.document.version_id,
        inputs.extraction.document.source_sha256,
    )

    if identity != normalization_identity or identity != extraction_identity:
        raise CitationIndexInputError(
            "Chunking, normalization, and extraction identities differ"
        )

    if len(chunking.chunks) != chunking.report.chunk_count:
        raise CitationIndexInputError(
            "Chunking records differ from chunking report"
        )


def _parse_segments(chunk: ChunkRecord) -> tuple[CitationSegment, ...]:
    payloads = chunk.metadata.get("source_segments")

    if not isinstance(payloads, list) or not payloads:
        raise CitationIndexStructureError(
            f"Chunk has no source segments: {chunk.chunk_id}"
        )

    try:
        return tuple(
            CitationSegment(
                normalized_block_id=payload["block_id"],
                source_element_id=payload["source_element_id"],
                element_type=payload["element_type"],
                normalized_span_id=payload["normalized_span_id"],
                source_anchor_id=payload["source_anchor_id"],
                char_start=payload["char_start"],
                char_end=payload["char_end"],
                raw_sha256=payload["raw_sha256"],
            )
            for payload in payloads
            if isinstance(payload, dict)
        )
    except Exception as error:
        raise CitationIndexStructureError(
            f"Invalid source segment metadata: {chunk.chunk_id}"
        ) from error


def _display_locator(page_numbers: list[int]) -> str:
    if len(page_numbers) == 1:
        return f"p. {page_numbers[0]}"

    return "pp. " + ", ".join(str(page) for page in page_numbers)


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _deduplicate_types(
    values: list[ElementType],
) -> list[ElementType]:
    return list(dict.fromkeys(values))


def _validate_and_resolve_chunk(
    chunk: ChunkRecord,
    inputs: IndexValidationInputs,
    policy: CitationIndexPolicy,
) -> tuple[CitationRecord, IndexReadyRecord]:
    blocks_by_id = {
        block.block_id: block
        for block in inputs.normalization.document.blocks
    }
    elements_by_id = {
        element.element_id: element
        for element in inputs.extraction.document.elements
    }
    filtered_ids = set(inputs.extraction.selection.filtered_element_ids)
    segments = _parse_segments(chunk)

    if len(segments) != len(chunk.metadata["source_segments"]):
        raise CitationIndexStructureError(
            f"Chunk contains a non-object source segment: {chunk.chunk_id}"
        )

    expected_anchors: list[SourceAnchor] = []
    seen_anchor_ids: set[str] = set()
    segment_texts: list[str] = []

    for segment in segments:
        block = blocks_by_id.get(segment.normalized_block_id)

        if block is None:
            raise CitationIndexStructureError(
                f"Citation block is absent: {segment.normalized_block_id}"
            )

        span = block.source_spans[0]
        element = elements_by_id.get(segment.source_element_id)

        if element is None:
            raise CitationIndexStructureError(
                f"Citation source element is absent: "
                f"{segment.source_element_id}"
            )

        if segment.source_element_id in filtered_ids:
            raise CitationIndexStructureError(
                "Filtered source element leaked into an index record"
            )

        if span.source_element_id != segment.source_element_id:
            raise CitationIndexStructureError(
                "Citation segment source element differs from block"
            )

        if span.span_id != segment.normalized_span_id:
            raise CitationIndexStructureError(
                "Citation normalized span ID differs from block"
            )

        if span.source_anchor.anchor_id != segment.source_anchor_id:
            raise CitationIndexStructureError(
                "Citation source anchor ID differs from block"
            )

        if element.element_type != segment.element_type:
            raise CitationIndexStructureError(
                "Citation element type differs from extraction"
            )

        if segment.char_end > len(block.markdown):
            raise CitationIndexStructureError(
                "Citation source range exceeds normalized block"
            )

        raw_text = block.markdown[
            segment.char_start : segment.char_end
        ]
        raw_sha256 = hashlib.sha256(
            raw_text.encode("utf-8")
        ).hexdigest()

        if raw_sha256 != segment.raw_sha256:
            raise CitationIndexStructureError(
                "Citation source range SHA256 differs from normalized text"
            )

        normalized_segment_text = raw_text.strip()

        if not normalized_segment_text:
            raise CitationIndexStructureError(
                "Citation source range resolves to empty text"
            )

        segment_texts.append(normalized_segment_text)
        anchor = span.source_anchor

        if anchor.anchor_id not in seen_anchor_ids:
            seen_anchor_ids.add(anchor.anchor_id)
            expected_anchors.append(anchor)

    if "\n\n".join(segment_texts) != chunk.text:
        raise CitationIndexStructureError(
            f"Chunk text differs from source ranges: {chunk.chunk_id}"
        )

    if expected_anchors != chunk.source_anchors:
        raise CitationIndexStructureError(
            f"Chunk anchors differ from source ranges: {chunk.chunk_id}"
        )

    if any(
        blocks_by_id[segment.normalized_block_id].heading_path
        != chunk.heading_path
        for segment in segments
    ):
        raise CitationIndexStructureError(
            f"Chunk heading path differs from source block: {chunk.chunk_id}"
        )

    if count_tokens(chunk.text) != chunk.token_count:
        raise CitationIndexStructureError(
            f"Chunk token count is invalid: {chunk.chunk_id}"
        )

    if chunk.token_count > policy.max_index_tokens:
        raise CitationIndexStructureError(
            f"Chunk exceeds max_index_tokens: {chunk.chunk_id}"
        )

    page_numbers = sorted(
        {
            anchor.page_number
            for anchor in expected_anchors
            if anchor.page_number is not None
        }
    )

    if policy.require_page_numbers and not page_numbers:
        raise CitationIndexStructureError(
            f"Chunk has no page provenance: {chunk.chunk_id}"
        )

    if policy.require_bounding_boxes and any(
        anchor.bbox is None for anchor in expected_anchors
    ):
        raise CitationIndexStructureError(
            f"Chunk has incomplete bbox provenance: {chunk.chunk_id}"
        )

    page_count = inputs.extraction.document.page_count

    if page_count is not None and any(
        page < 1 or page > page_count for page in page_numbers
    ):
        raise CitationIndexStructureError(
            f"Chunk page provenance exceeds document: {chunk.chunk_id}"
        )

    citation_id = stable_identifier(
        "citation",
        chunk.chunk_instance_id,
        ",".join(anchor.anchor_id for anchor in expected_anchors),
        ",".join(segment.raw_sha256 for segment in segments),
    )
    citation = CitationRecord(
        citation_id=citation_id,
        family_id=chunk.family_id,
        version_id=chunk.version_id,
        source_sha256=chunk.source_sha256,
        chunk_id=chunk.chunk_id,
        chunk_instance_id=chunk.chunk_instance_id,
        ordinal=chunk.ordinal,
        heading_path=chunk.heading_path,
        display_locator=_display_locator(page_numbers),
        page_numbers=page_numbers,
        source_anchors=expected_anchors,
        source_segments=list(segments),
    )
    content_sha256 = hashlib.sha256(
        chunk.text.encode("utf-8")
    ).hexdigest()
    index_record_id = stable_identifier(
        "index-record",
        chunk.chunk_instance_id,
        content_sha256,
        citation_id,
    )
    index_record = IndexReadyRecord(
        index_record_id=index_record_id,
        family_id=chunk.family_id,
        version_id=chunk.version_id,
        source_sha256=chunk.source_sha256,
        chunk_id=chunk.chunk_id,
        chunk_instance_id=chunk.chunk_instance_id,
        ordinal=chunk.ordinal,
        text=chunk.text,
        content_sha256=content_sha256,
        token_count=chunk.token_count,
        heading_path=chunk.heading_path,
        citation_id=citation_id,
        page_numbers=page_numbers,
        source_anchor_ids=[
            anchor.anchor_id for anchor in expected_anchors
        ],
        normalized_block_ids=_deduplicate(
            [segment.normalized_block_id for segment in segments]
        ),
        source_element_ids=_deduplicate(
            [segment.source_element_id for segment in segments]
        ),
        element_types=_deduplicate_types(
            [segment.element_type for segment in segments]
        ),
        contains_structural_block=bool(
            chunk.metadata["contains_structural_block"]
        ),
        oversized_atomic=bool(chunk.metadata["oversized_atomic"]),
    )
    return citation, index_record


def _validate_complete_source_ranges(
    inputs: IndexValidationInputs,
    citations: tuple[CitationRecord, ...],
) -> None:
    segments_by_block: dict[str, list[CitationSegment]] = {}

    for citation in citations:
        for segment in citation.source_segments:
            segments_by_block.setdefault(
                segment.normalized_block_id,
                [],
            ).append(segment)

    for block in inputs.normalization.document.blocks:
        segments = segments_by_block.get(block.block_id, [])

        if not segments:
            raise CitationIndexStructureError(
                f"Normalized block has no citation: {block.block_id}"
            )

        expected_start = 0

        for segment in segments:
            if segment.char_start != expected_start:
                raise CitationIndexStructureError(
                    f"Citation ranges are not contiguous: {block.block_id}"
                )
            expected_start = segment.char_end

        if expected_start != len(block.markdown):
            raise CitationIndexStructureError(
                f"Citation ranges do not cover block: {block.block_id}"
            )


def _records_jsonl(records: tuple[BaseModel, ...]) -> str:
    return "".join(record.model_dump_json() + "\n" for record in records)


def build_citation_index_validation(
    inputs: IndexValidationInputs,
    policy: CitationIndexPolicy | None = None,
) -> CitationIndexBuildResult:
    active_policy = policy or CitationIndexPolicy()
    validate_index_inputs(inputs)
    citations: list[CitationRecord] = []
    index_records: list[IndexReadyRecord] = []

    for chunk in inputs.chunking.chunks:
        citation, index_record = _validate_and_resolve_chunk(
            chunk,
            inputs,
            active_policy,
        )
        citations.append(citation)
        index_records.append(index_record)

    citation_tuple = tuple(citations)
    index_tuple = tuple(index_records)
    _validate_complete_source_ranges(inputs, citation_tuple)
    chunk_ids = [chunk.chunk_id for chunk in inputs.chunking.chunks]
    citation_ids = [citation.citation_id for citation in citation_tuple]
    index_record_ids = [
        record.index_record_id for record in index_tuple
    ]

    if len(citation_ids) != len(set(citation_ids)):
        raise CitationIndexStructureError("Citation IDs are not unique")

    if len(index_record_ids) != len(set(index_record_ids)):
        raise CitationIndexStructureError("Index record IDs are not unique")

    all_anchors = [
        anchor
        for citation in citation_tuple
        for anchor in citation.source_anchors
    ]
    unique_anchor_ids = {anchor.anchor_id for anchor in all_anchors}
    source_segment_count = sum(
        len(citation.source_segments) for citation in citation_tuple
    )
    token_counts = [record.token_count for record in index_tuple]
    page_numbers = sorted(
        {
            page
            for citation in citation_tuple
            for page in citation.page_numbers
        }
    )
    content_hash_counts = Counter(
        record.content_sha256 for record in index_tuple
    )
    duplicate_content_groups = sum(
        1 for count in content_hash_counts.values() if count > 1
    )
    report = CitationIndexReport(
        validator_version=CITATION_INDEX_VALIDATOR_VERSION,
        input_chunking_run_id=inputs.chunking.manifest.run_id,
        catalog_id=inputs.chunking.manifest.catalog_id,
        family_id=inputs.chunking.manifest.family_id,
        version_id=inputs.chunking.manifest.version_id,
        source_sha256=inputs.chunking.manifest.source_sha256,
        input_chunk_count=len(inputs.chunking.chunks),
        citation_count=len(citation_tuple),
        index_record_count=len(index_tuple),
        source_segment_count=source_segment_count,
        source_anchor_reference_count=len(all_anchors),
        unique_source_anchor_count=len(unique_anchor_ids),
        resolved_source_range_count=source_segment_count,
        page_anchor_count=sum(
            1 for anchor in all_anchors if anchor.page_number is not None
        ),
        bbox_anchor_count=sum(
            1 for anchor in all_anchors if anchor.bbox is not None
        ),
        citations_with_pages_count=sum(
            1 for citation in citation_tuple if citation.page_numbers
        ),
        citations_with_bboxes_count=sum(
            1
            for citation in citation_tuple
            if all(anchor.bbox is not None for anchor in citation.source_anchors)
        ),
        structural_chunk_count=sum(
            1
            for record in index_tuple
            if record.contains_structural_block
        ),
        total_token_count=sum(token_counts),
        min_record_token_count=min(token_counts),
        max_record_token_count=max(token_counts),
        duplicate_content_hash_group_count=duplicate_content_groups,
        page_numbers=page_numbers,
        chunk_ids=chunk_ids,
        citation_ids=citation_ids,
        index_record_ids=index_record_ids,
        generated_at=inputs.chunking.manifest.committed_at,
    )
    return CitationIndexBuildResult(
        citations=citation_tuple,
        citation_jsonl=_records_jsonl(citation_tuple),
        index_records=index_tuple,
        index_jsonl=_records_jsonl(index_tuple),
        report=report,
    )


class CitationIndexValidator:
    def __init__(self, config: CitationIndexValidatorConfig) -> None:
        self.config = config
        self.policy = config.policy

    def validate_commit(
        self,
        chunking_commit_directory: Path | str,
        *,
        run_id: str | None = None,
    ) -> CitationIndexCommitBundle:
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        inputs = load_index_validation_inputs(chunking_commit_directory)
        active_run_id = run_id or utc_run_id().replace(
            "extraction-",
            "citation-index-",
            1,
        )
        validate_run_id(active_run_id)
        self._ensure_target_is_available(active_run_id)
        build = build_citation_index_validation(inputs, self.policy)
        report = build.report
        stage_result = StageResult(
            stage=ResumeFrom.INDEX_VALIDATE,
            status=StageStatus.PASS,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_seconds=time.monotonic() - started_monotonic,
            artifact_paths=[
                CITATIONS_FILENAME,
                INDEX_RECORDS_FILENAME,
                VALIDATION_REPORT_FILENAME,
            ],
            metrics={
                "input_chunk_count": report.input_chunk_count,
                "citation_count": report.citation_count,
                "index_record_count": report.index_record_count,
                "source_segment_count": report.source_segment_count,
                "unique_source_anchor_count": (
                    report.unique_source_anchor_count
                ),
                "external_call_count": 0,
                "next_action": "EMBED_AND_STAGE_INDEX",
            },
            message="Citation and index readiness validation passed",
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
            raise CitationIndexCommitExistsError(
                f"Citation/index commit already exists: {run_id}"
            )

    def _commit(
        self,
        *,
        run_id: str,
        inputs: IndexValidationInputs,
        build: CitationIndexBuildResult,
        stage_result: StageResult,
    ) -> CitationIndexCommitBundle:
        output_root = self.config.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        final_directory = output_root / run_id

        if final_directory.exists():
            raise CitationIndexCommitExistsError(
                f"Citation/index commit already exists: {run_id}"
            )

        staging_directory = output_root / (
            f".{run_id}.staging-{uuid.uuid4().hex}"
        )
        staging_directory.mkdir(parents=False, exist_ok=False)

        try:
            payloads = {
                CITATIONS_FILENAME: build.citation_jsonl.encode("utf-8"),
                INDEX_RECORDS_FILENAME: build.index_jsonl.encode("utf-8"),
                VALIDATION_REPORT_FILENAME: model_json_bytes(build.report),
                STAGE_RESULT_FILENAME: model_json_bytes(stage_result),
            }
            artifacts: list[CommitArtifactRecord] = []

            for filename, payload in payloads.items():
                path = staging_directory / filename
                write_payload(path, payload)
                artifacts.append(
                    artifact_record(path, staging_directory)
                )

            chunking_manifest_path = (
                inputs.chunking.commit_directory / COMMIT_MANIFEST_FILENAME
            )
            report = build.report
            manifest = CitationIndexCommitManifest(
                schema_version=CITATION_INDEX_COMMIT_SCHEMA_VERSION,
                validator_version=CITATION_INDEX_VALIDATOR_VERSION,
                run_id=run_id,
                input_chunking_run_id=inputs.chunking.manifest.run_id,
                input_chunking_commit_directory=str(
                    inputs.chunking.commit_directory
                ),
                input_chunking_manifest_sha256=sha256_file(
                    chunking_manifest_path
                ),
                catalog_id=report.catalog_id,
                family_id=report.family_id,
                version_id=report.version_id,
                source_sha256=report.source_sha256,
                require_page_numbers=self.policy.require_page_numbers,
                require_bounding_boxes=self.policy.require_bounding_boxes,
                max_index_tokens=self.policy.max_index_tokens,
                input_chunk_count=report.input_chunk_count,
                citation_count=report.citation_count,
                index_record_count=report.index_record_count,
                artifact_count=len(artifacts),
                artifacts=artifacts,
            )
            write_payload(
                staging_directory / COMMIT_MANIFEST_FILENAME,
                model_json_bytes(manifest),
            )
            load_citation_index_commit(staging_directory)

            if final_directory.exists():
                raise CitationIndexCommitExistsError(
                    f"Citation/index commit already exists: {run_id}"
                )

            staging_directory.rename(final_directory)

        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
            raise

        return load_citation_index_commit(final_directory)


def _parse_jsonl(
    raw_jsonl: str,
    model_type: type[CitationRecord] | type[IndexReadyRecord],
    label: str,
) -> tuple[CitationRecord, ...] | tuple[IndexReadyRecord, ...]:
    if not raw_jsonl or not raw_jsonl.endswith("\n"):
        raise CitationIndexCommitIntegrityError(
            f"{label} JSONL must be non-empty and newline terminated"
        )

    lines = raw_jsonl.splitlines()

    if not lines or any(not line.strip() for line in lines):
        raise CitationIndexCommitIntegrityError(
            f"{label} JSONL contains an empty record"
        )

    try:
        return tuple(
            model_type.model_validate_json(line) for line in lines
        )
    except Exception as error:
        raise CitationIndexCommitIntegrityError(
            f"{label} JSONL contains an invalid record"
        ) from error


def load_citation_index_commit(
    commit_directory: Path | str,
    *,
    verify_input_commit: bool = True,
) -> CitationIndexCommitBundle:
    root = Path(commit_directory).expanduser().resolve()

    if not root.is_dir():
        raise CitationIndexCommitIntegrityError(
            f"Citation/index commit directory not found: {root}"
        )

    try:
        manifest = CitationIndexCommitManifest.model_validate_json(
            (root / COMMIT_MANIFEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise CitationIndexCommitIntegrityError(
            "Citation/index manifest is missing or invalid"
        ) from error

    recorded_paths = {
        artifact.relative_path for artifact in manifest.artifacts
    }
    required_paths = {
        CITATIONS_FILENAME,
        INDEX_RECORDS_FILENAME,
        VALIDATION_REPORT_FILENAME,
        STAGE_RESULT_FILENAME,
    }

    if recorded_paths != required_paths:
        raise CitationIndexCommitIntegrityError(
            "Citation/index commit has unexpected typed artifacts"
        )

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }

    if actual_paths != required_paths | {COMMIT_MANIFEST_FILENAME}:
        raise CitationIndexCommitIntegrityError(
            "Citation/index file inventory differs from manifest"
        )

    for artifact in manifest.artifacts:
        path = (root / artifact.relative_path).resolve()

        if root not in path.parents:
            raise CitationIndexCommitIntegrityError(
                "Citation/index artifact escapes commit directory"
            )

        if path.stat().st_size != artifact.size_bytes:
            raise CitationIndexCommitIntegrityError(
                f"Citation/index size mismatch: {artifact.relative_path}"
            )

        if sha256_file(path) != artifact.sha256:
            raise CitationIndexCommitIntegrityError(
                f"Citation/index SHA256 mismatch: {artifact.relative_path}"
            )

    try:
        citation_jsonl = (root / CITATIONS_FILENAME).read_text(
            encoding="utf-8"
        )
        index_jsonl = (root / INDEX_RECORDS_FILENAME).read_text(
            encoding="utf-8"
        )
        parsed_citations = _parse_jsonl(
            citation_jsonl,
            CitationRecord,
            "Citation",
        )
        parsed_index_records = _parse_jsonl(
            index_jsonl,
            IndexReadyRecord,
            "Index",
        )
        citations = tuple(parsed_citations)
        index_records = tuple(parsed_index_records)
        report = CitationIndexReport.model_validate_json(
            (root / VALIDATION_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        stage_result = StageResult.model_validate_json(
            (root / STAGE_RESULT_FILENAME).read_text(encoding="utf-8")
        )
    except CitationIndexCommitIntegrityError:
        raise
    except Exception as error:
        raise CitationIndexCommitIntegrityError(
            "Committed citation/index artifact is invalid"
        ) from error

    _validate_citation_index_commit(
        manifest=manifest,
        citations=citations,
        citation_jsonl=citation_jsonl,
        index_records=index_records,
        index_jsonl=index_jsonl,
        report=report,
        stage_result=stage_result,
    )

    if verify_input_commit:
        try:
            inputs = load_index_validation_inputs(
                manifest.input_chunking_commit_directory
            )
        except Exception as error:
            raise CitationIndexCommitIntegrityError(
                "Input chunking commit is invalid"
            ) from error

        chunking_manifest_path = (
            inputs.chunking.commit_directory / COMMIT_MANIFEST_FILENAME
        )

        if sha256_file(chunking_manifest_path) != (
            manifest.input_chunking_manifest_sha256
        ):
            raise CitationIndexCommitIntegrityError(
                "Input chunking manifest SHA256 changed"
            )

        if inputs.chunking.manifest.run_id != (
            manifest.input_chunking_run_id
        ):
            raise CitationIndexCommitIntegrityError(
                "Input chunking run ID changed"
            )

        policy = CitationIndexPolicy(
            require_page_numbers=manifest.require_page_numbers,
            require_bounding_boxes=manifest.require_bounding_boxes,
            max_index_tokens=manifest.max_index_tokens,
        )
        expected = build_citation_index_validation(inputs, policy)

        if (
            expected.citations != citations
            or expected.citation_jsonl != citation_jsonl
            or expected.index_records != index_records
            or expected.index_jsonl != index_jsonl
            or expected.report != report
        ):
            raise CitationIndexCommitIntegrityError(
                "Citation/index output is not reproducible from chunks"
            )

    return CitationIndexCommitBundle(
        commit_directory=root,
        manifest=manifest,
        citations=citations,
        citation_jsonl=citation_jsonl,
        index_records=index_records,
        index_jsonl=index_jsonl,
        report=report,
        stage_result=stage_result,
    )


def _validate_citation_index_commit(
    *,
    manifest: CitationIndexCommitManifest,
    citations: tuple[CitationRecord, ...],
    citation_jsonl: str,
    index_records: tuple[IndexReadyRecord, ...],
    index_jsonl: str,
    report: CitationIndexReport,
    stage_result: StageResult,
) -> None:
    if (
        manifest.validator_version != CITATION_INDEX_VALIDATOR_VERSION
        or report.validator_version != CITATION_INDEX_VALIDATOR_VERSION
    ):
        raise CitationIndexCommitIntegrityError(
            "Validator version differs from installed validator"
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
        raise CitationIndexCommitIntegrityError(
            "Citation/index report identity differs from manifest"
        )

    if any(
        (
            manifest.catalog_id,
            citation.family_id,
            citation.version_id,
            citation.source_sha256,
        )
        != identity
        for citation in citations
    ):
        raise CitationIndexCommitIntegrityError(
            "A citation identity differs from manifest"
        )

    if any(
        (
            manifest.catalog_id,
            record.family_id,
            record.version_id,
            record.source_sha256,
        )
        != identity
        for record in index_records
    ):
        raise CitationIndexCommitIntegrityError(
            "An index record identity differs from manifest"
        )

    if _records_jsonl(citations) != citation_jsonl:
        raise CitationIndexCommitIntegrityError(
            "Citation JSONL is not canonical"
        )

    if _records_jsonl(index_records) != index_jsonl:
        raise CitationIndexCommitIntegrityError(
            "Index JSONL is not canonical"
        )

    counts = (
        len(citations),
        len(index_records),
        report.input_chunk_count,
        manifest.input_chunk_count,
    )

    if len(set(counts)) != 1:
        raise CitationIndexCommitIntegrityError(
            "Citation/index record counts are inconsistent"
        )

    if [citation.citation_id for citation in citations] != (
        report.citation_ids
    ):
        raise CitationIndexCommitIntegrityError(
            "Citation order differs from report"
        )

    if [record.index_record_id for record in index_records] != (
        report.index_record_ids
    ):
        raise CitationIndexCommitIntegrityError(
            "Index record order differs from report"
        )

    citation_chunk_ids = [
        citation.chunk_id for citation in citations
    ]
    index_chunk_ids = [record.chunk_id for record in index_records]

    if (
        citation_chunk_ids != report.chunk_ids
        or index_chunk_ids != report.chunk_ids
    ):
        raise CitationIndexCommitIntegrityError(
            "Chunk order differs between citations, index, and report"
        )

    expected_ordinals = list(range(1, len(citations) + 1))

    if [citation.ordinal for citation in citations] != expected_ordinals:
        raise CitationIndexCommitIntegrityError(
            "Citation ordinals must be contiguous and one-based"
        )

    if [record.ordinal for record in index_records] != expected_ordinals:
        raise CitationIndexCommitIntegrityError(
            "Index ordinals must be contiguous and one-based"
        )

    citation_by_id = {
        citation.citation_id: citation for citation in citations
    }

    for citation, record in zip(citations, index_records):
        if citation.ordinal != record.ordinal:
            raise CitationIndexCommitIntegrityError(
                "Citation and index ordinals differ"
            )

        if citation.chunk_id != record.chunk_id:
            raise CitationIndexCommitIntegrityError(
                "Citation and index chunk IDs differ"
            )

        if record.citation_id not in citation_by_id:
            raise CitationIndexCommitIntegrityError(
                "Index record references an absent citation"
            )

        if citation.citation_id != record.citation_id:
            raise CitationIndexCommitIntegrityError(
                "Citation and index record linkage differs"
            )

        if citation.page_numbers != record.page_numbers:
            raise CitationIndexCommitIntegrityError(
                "Citation and index page provenance differs"
            )

        if [
            anchor.anchor_id for anchor in citation.source_anchors
        ] != record.source_anchor_ids:
            raise CitationIndexCommitIntegrityError(
                "Citation and index source anchors differ"
            )

        if count_tokens(record.text) != record.token_count:
            raise CitationIndexCommitIntegrityError(
                "Index token count differs from deterministic tokenizer"
            )

        if record.token_count > manifest.max_index_tokens:
            raise CitationIndexCommitIntegrityError(
                "Index record exceeds max_index_tokens"
            )

    token_counts = [record.token_count for record in index_records]

    if sum(token_counts) != report.total_token_count:
        raise CitationIndexCommitIntegrityError(
            "Total index token count differs from report"
        )

    if (
        min(token_counts) != report.min_record_token_count
        or max(token_counts) != report.max_record_token_count
    ):
        raise CitationIndexCommitIntegrityError(
            "Index token range differs from report"
        )

    source_segments = [
        segment
        for citation in citations
        for segment in citation.source_segments
    ]
    source_anchors = [
        anchor
        for citation in citations
        for anchor in citation.source_anchors
    ]

    if len(source_segments) != report.source_segment_count:
        raise CitationIndexCommitIntegrityError(
            "Citation source segment count differs from report"
        )

    if len(source_anchors) != report.source_anchor_reference_count:
        raise CitationIndexCommitIntegrityError(
            "Citation source anchor count differs from report"
        )

    if len({anchor.anchor_id for anchor in source_anchors}) != (
        report.unique_source_anchor_count
    ):
        raise CitationIndexCommitIntegrityError(
            "Unique source anchor count differs from report"
        )

    page_anchor_count = sum(
        1 for anchor in source_anchors if anchor.page_number is not None
    )
    bbox_anchor_count = sum(
        1 for anchor in source_anchors if anchor.bbox is not None
    )

    if page_anchor_count != report.page_anchor_count:
        raise CitationIndexCommitIntegrityError(
            "Page anchor count differs from report"
        )

    if bbox_anchor_count != report.bbox_anchor_count:
        raise CitationIndexCommitIntegrityError(
            "Bounding-box anchor count differs from report"
        )

    page_numbers = sorted(
        {
            page
            for citation in citations
            for page in citation.page_numbers
        }
    )

    if page_numbers != report.page_numbers:
        raise CitationIndexCommitIntegrityError(
            "Citation page inventory differs from report"
        )

    citations_with_pages = sum(
        1 for citation in citations if citation.page_numbers
    )
    citations_with_bboxes = sum(
        1
        for citation in citations
        if all(anchor.bbox is not None for anchor in citation.source_anchors)
    )

    if citations_with_pages != report.citations_with_pages_count:
        raise CitationIndexCommitIntegrityError(
            "Citation page coverage differs from report"
        )

    if citations_with_bboxes != report.citations_with_bboxes_count:
        raise CitationIndexCommitIntegrityError(
            "Citation bbox coverage differs from report"
        )

    structural_chunk_count = sum(
        1 for record in index_records if record.contains_structural_block
    )

    if structural_chunk_count != report.structural_chunk_count:
        raise CitationIndexCommitIntegrityError(
            "Structural index record count differs from report"
        )

    content_hash_counts = Counter(
        record.content_sha256 for record in index_records
    )
    duplicate_content_groups = sum(
        1 for count in content_hash_counts.values() if count > 1
    )

    if duplicate_content_groups != (
        report.duplicate_content_hash_group_count
    ):
        raise CitationIndexCommitIntegrityError(
            "Duplicate content hash count differs from report"
        )

    if manifest.input_chunking_run_id != report.input_chunking_run_id:
        raise CitationIndexCommitIntegrityError(
            "Input chunking run differs from report"
        )

    if (
        stage_result.stage != ResumeFrom.INDEX_VALIDATE
        or stage_result.status != StageStatus.PASS
        or stage_result.metrics.get("next_action")
        != "EMBED_AND_STAGE_INDEX"
    ):
        raise CitationIndexCommitIntegrityError(
            "Citation/index stage result is inconsistent"
        )

    if stage_result.artifact_paths != [
        CITATIONS_FILENAME,
        INDEX_RECORDS_FILENAME,
        VALIDATION_REPORT_FILENAME,
    ]:
        raise CitationIndexCommitIntegrityError(
            "Citation/index stage artifact paths are inconsistent"
        )

    expected_stage_metrics = {
        "input_chunk_count": report.input_chunk_count,
        "citation_count": report.citation_count,
        "index_record_count": report.index_record_count,
        "source_segment_count": report.source_segment_count,
        "unique_source_anchor_count": report.unique_source_anchor_count,
        "external_call_count": 0,
        "next_action": "EMBED_AND_STAGE_INDEX",
    }

    if stage_result.metrics != expected_stage_metrics:
        raise CitationIndexCommitIntegrityError(
            "Citation/index stage metrics differ from report"
        )
