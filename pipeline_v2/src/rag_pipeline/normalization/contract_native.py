from __future__ import annotations

import hashlib
import re
import shutil
import time
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.contracts import (
    ElementType,
    NormalizedBlock,
    NormalizedDocument,
    NormalizedSpan,
    ResumeFrom,
    SourceElement,
    StageResult,
    StageStatus,
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
    stable_identifier,
    utc_run_id,
    validate_run_id,
    write_payload,
)


CONTRACT_NATIVE_NORMALIZER_VERSION = "2026-08-17-5.2A.1"
NORMALIZATION_COMMIT_SCHEMA_VERSION = "1.0"

NORMALIZED_DOCUMENT_FILENAME = "normalized_document.json"
NORMALIZED_MARKDOWN_FILENAME = "document.normalized.md"
NORMALIZATION_REPORT_FILENAME = "normalization_report.json"
STAGE_RESULT_FILENAME = "stage_result.json"
COMMIT_MANIFEST_FILENAME = "commit_manifest.json"


class NormalizationManifestModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class NormalizationReport(NormalizationManifestModel):
    normalizer_version: str = Field(min_length=1)
    input_extraction_run_id: str = Field(min_length=1)
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_source_element_count: int = Field(ge=1)
    selected_source_element_count: int = Field(ge=1)
    filtered_source_element_count: int = Field(ge=0)
    normalized_block_count: int = Field(ge=1)
    source_span_count: int = Field(ge=1)
    markdown_character_count: int = Field(ge=1)
    broken_character_count: int = Field(ge=0)
    title_count: int = Field(ge=0)
    heading_count: int = Field(ge=0)
    table_count: int = Field(ge=0)
    list_count: int = Field(ge=0)
    flowchart_count: int = Field(ge=0)
    valid_gfm_table_count: int = Field(ge=0)
    invalid_gfm_table_count: Literal[0] = 0
    type_counts: dict[str, int]
    page_numbers: list[int]
    source_element_ids: list[str] = Field(min_length=1)
    provenance_coverage: Literal[1.0] = 1.0
    structure_preserved: Literal[True] = True
    ocr_invocation_count: Literal[0] = 0
    passed: Literal[True] = True
    generated_at: datetime

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if (
            self.selected_source_element_count
            + self.filtered_source_element_count
            != self.input_source_element_count
        ):
            raise ValueError(
                "Selected and filtered counts must cover source elements"
            )

        if (
            self.normalized_block_count
            != self.selected_source_element_count
        ):
            raise ValueError(
                "Each selected source element must produce one block"
            )

        if self.source_span_count != self.normalized_block_count:
            raise ValueError("Each normalized block must have one span")

        if len(self.source_element_ids) != len(
            set(self.source_element_ids)
        ):
            raise ValueError("source_element_ids must be unique")

        if len(self.source_element_ids) != self.normalized_block_count:
            raise ValueError(
                "source_element_ids length must equal block count"
            )

        if self.table_count != self.valid_gfm_table_count:
            raise ValueError("Every normalized table must be valid GFM")

        if self.broken_character_count != 0:
            raise ValueError("Normalized Markdown contains broken text")

        if self.page_numbers != sorted(set(self.page_numbers)):
            raise ValueError("page_numbers must be sorted and unique")

        if sum(self.type_counts.values()) != self.normalized_block_count:
            raise ValueError("type_counts must cover every block")

        return self


class NormalizationCommitManifest(NormalizationManifestModel):
    schema_version: Literal["1.0"]
    normalizer_version: str = Field(min_length=1)
    run_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    input_extraction_run_id: str = Field(min_length=1)
    input_extraction_commit_directory: str = Field(min_length=1)
    input_extraction_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    catalog_id: str = Field(min_length=1)
    family_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_source_element_count: int = Field(ge=1)
    selected_source_element_count: int = Field(ge=1)
    filtered_source_element_count: int = Field(ge=0)
    normalized_block_count: int = Field(ge=1)
    next_action: Literal["CHUNK"] = "CHUNK"
    ocr_invocation_count: Literal[0] = 0
    artifact_count: int = Field(ge=4)
    artifacts: list[CommitArtifactRecord] = Field(min_length=4)
    committed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @model_validator(mode="after")
    def validate_inventory_and_counts(self) -> Self:
        paths = [
            artifact.relative_path
            for artifact in self.artifacts
        ]

        if len(paths) != len(set(paths)):
            raise ValueError("Normalization artifact paths must be unique")

        if self.artifact_count != len(self.artifacts):
            raise ValueError("artifact_count must equal artifacts length")

        if (
            self.selected_source_element_count
            + self.filtered_source_element_count
            != self.input_source_element_count
        ):
            raise ValueError(
                "Normalization selection counts are inconsistent"
            )

        if (
            self.normalized_block_count
            != self.selected_source_element_count
        ):
            raise ValueError(
                "Normalized block count must equal selected count"
            )

        return self


@dataclass(frozen=True, slots=True)
class NormalizationBuildResult:
    document: NormalizedDocument
    markdown: str
    report: NormalizationReport


@dataclass(frozen=True, slots=True)
class NormalizerConfig:
    output_root: Path


@dataclass(frozen=True, slots=True)
class NormalizationCommitBundle:
    commit_directory: Path
    manifest: NormalizationCommitManifest
    document: NormalizedDocument
    markdown: str
    report: NormalizationReport
    stage_result: StageResult


class NormalizationError(RuntimeError):
    """Base exception for contract-native normalization."""


class NormalizationInputError(NormalizationError):
    pass


class NormalizationStructureError(NormalizationError):
    pass


class NormalizationCommitExistsError(NormalizationError):
    pass


class NormalizationCommitIntegrityError(NormalizationError):
    pass


def normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n")


def normalize_plain_text(value: str) -> str:
    return " ".join(normalize_unicode(value).split()).strip()


def normalize_structured_text(value: str) -> str:
    lines = [
        normalize_unicode(line).expandtabs(2).rstrip()
        for line in value.splitlines()
    ]

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def split_gfm_row(line: str) -> list[str] | None:
    stripped = line.strip()

    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None

    body = stripped[1:-1]
    cells: list[str] = []
    buffer: list[str] = []
    escaped = False

    for character in body:
        if character == "|" and not escaped:
            cells.append("".join(buffer).strip())
            buffer = []
            continue

        buffer.append(character)

        if character == "\\" and not escaped:
            escaped = True
        else:
            escaped = False

    cells.append("".join(buffer).strip())
    return cells


def is_valid_gfm_table(markdown: str) -> bool:
    lines = markdown.splitlines()

    if len(lines) < 2 or any(not line.strip() for line in lines):
        return False

    rows = [split_gfm_row(line) for line in lines]

    if any(row is None for row in rows):
        return False

    typed_rows = [row for row in rows if row is not None]
    column_count = len(typed_rows[0])

    if column_count < 1:
        return False

    if any(len(row) != column_count for row in typed_rows):
        return False

    separator_pattern = re.compile(r"^:?-{3,}:?$")
    return all(
        separator_pattern.fullmatch(cell) is not None
        for cell in typed_rows[1]
    )


def is_valid_list(markdown: str) -> bool:
    nonempty_lines = [
        line for line in markdown.splitlines() if line.strip()
    ]

    if not nonempty_lines:
        return False

    marker = re.compile(
        r"^\s*(?:[-+*]|\d+\.|[A-Za-z]+\.)\s+\S"
    )
    return marker.match(nonempty_lines[0]) is not None


def render_element(element: SourceElement) -> str:
    element_type = element.element_type

    if element_type == ElementType.TABLE:
        markdown = normalize_structured_text(element.content)

        if not is_valid_gfm_table(markdown):
            raise NormalizationStructureError(
                f"Invalid GFM table: {element.element_id}"
            )

        return markdown

    if element_type == ElementType.LIST:
        markdown = normalize_structured_text(element.content)

        if not is_valid_list(markdown):
            raise NormalizationStructureError(
                f"Invalid Markdown list: {element.element_id}"
            )

        return markdown

    if element_type == ElementType.FLOWCHART:
        markdown = normalize_structured_text(element.content)

        if not markdown:
            raise NormalizationStructureError(
                f"Selected flowchart has no semantics: {element.element_id}"
            )

        return markdown

    text = normalize_plain_text(element.content)

    if not text:
        raise NormalizationStructureError(
            f"Selected element has no text: {element.element_id}"
        )

    if element_type == ElementType.TITLE:
        text = re.sub(r"^#{1,6}\s+", "", text)
        return f"# {text}"

    if element_type == ElementType.HEADING:
        text = re.sub(r"^#{1,6}\s+", "", text)
        return f"## {text}"

    return text


def normalized_heading_path(element: SourceElement) -> list[str]:
    return [
        normalized
        for item in element.source_anchor.section_path
        if (normalized := normalize_plain_text(item))
    ]


def validate_extraction_input(
    extraction: ExtractionCommitBundle,
) -> None:
    manifest = extraction.manifest

    if not manifest.local_result_accepted:
        raise NormalizationInputError(
            "Normalization requires an accepted Local extraction"
        )

    if manifest.fallback_required:
        raise NormalizationInputError(
            "Normalization cannot consume a pending fallback"
        )

    if manifest.next_action != "NORMALIZE":
        raise NormalizationInputError(
            "Extraction commit is not routed to NORMALIZE"
        )

    if extraction.stage_result.status != StageStatus.PASS:
        raise NormalizationInputError(
            "Extraction quality stage must have PASS status"
        )

    document = extraction.document
    selection = extraction.selection
    element_ids = [
        element.element_id
        for element in document.elements
    ]
    indexable_set = set(selection.indexable_element_ids)
    filtered_set = set(selection.filtered_element_ids)
    expected_indexable_order = [
        element_id
        for element_id in element_ids
        if element_id in indexable_set
    ]

    if selection.indexable_element_ids != expected_indexable_order:
        raise NormalizationInputError(
            "Indexable selection must preserve document order"
        )

    if set(element_ids) != indexable_set | filtered_set:
        raise NormalizationInputError(
            "Selection does not cover every extracted element"
        )


def build_normalization(
    extraction: ExtractionCommitBundle,
) -> NormalizationBuildResult:
    validate_extraction_input(extraction)
    source_document = extraction.document
    selection = extraction.selection
    elements_by_id = {
        element.element_id: element
        for element in source_document.elements
    }
    blocks: list[NormalizedBlock] = []
    type_counter: Counter[str] = Counter()
    pages: set[int] = set()

    for source_element_id in selection.indexable_element_ids:
        element = elements_by_id[source_element_id]
        markdown = render_element(element)
        block_id = stable_identifier(
            "normalized-block",
            source_document.source_sha256,
            source_document.version_id,
            source_element_id,
        )
        span = NormalizedSpan(
            span_id=stable_identifier(
                "normalized-span",
                source_document.source_sha256,
                source_document.version_id,
                source_element_id,
            ),
            source_element_id=source_element_id,
            char_start=0,
            char_end=len(markdown),
            source_anchor=element.source_anchor,
        )
        blocks.append(
            NormalizedBlock(
                block_id=block_id,
                heading_path=normalized_heading_path(element),
                markdown=markdown,
                source_spans=[span],
            )
        )
        type_counter[element.element_type.value] += 1

        if element.source_anchor.page_number is not None:
            pages.add(element.source_anchor.page_number)

        lineage_anchors = element.metadata.get("merged_source_anchors", [])

        if isinstance(lineage_anchors, list):
            for lineage_anchor in lineage_anchors:
                if not isinstance(lineage_anchor, dict):
                    continue

                lineage_page = lineage_anchor.get("page_number")

                if isinstance(lineage_page, int) and lineage_page >= 1:
                    pages.add(lineage_page)

    normalized_document = NormalizedDocument(
        catalog_id=source_document.catalog_id,
        family_id=source_document.family_id,
        version_id=source_document.version_id,
        source_sha256=source_document.source_sha256,
        blocks=blocks,
        normalized_at=extraction.manifest.committed_at,
    )
    markdown = "\n\n".join(
        block.markdown
        for block in blocks
    ) + "\n"
    broken_character_count = markdown.count("\ufffd")
    report = NormalizationReport(
        normalizer_version=CONTRACT_NATIVE_NORMALIZER_VERSION,
        input_extraction_run_id=extraction.manifest.run_id,
        catalog_id=source_document.catalog_id,
        family_id=source_document.family_id,
        version_id=source_document.version_id,
        source_sha256=source_document.source_sha256,
        input_source_element_count=len(source_document.elements),
        selected_source_element_count=len(
            selection.indexable_element_ids
        ),
        filtered_source_element_count=len(
            selection.filtered_element_ids
        ),
        normalized_block_count=len(blocks),
        source_span_count=sum(
            len(block.source_spans)
            for block in blocks
        ),
        markdown_character_count=len(markdown),
        broken_character_count=broken_character_count,
        title_count=type_counter[ElementType.TITLE.value],
        heading_count=type_counter[ElementType.HEADING.value],
        table_count=type_counter[ElementType.TABLE.value],
        list_count=type_counter[ElementType.LIST.value],
        flowchart_count=type_counter[ElementType.FLOWCHART.value],
        valid_gfm_table_count=type_counter[
            ElementType.TABLE.value
        ],
        type_counts=dict(sorted(type_counter.items())),
        page_numbers=sorted(pages),
        source_element_ids=list(selection.indexable_element_ids),
        generated_at=extraction.manifest.committed_at,
    )
    return NormalizationBuildResult(
        document=normalized_document,
        markdown=markdown,
        report=report,
    )


class ContractNativeNormalizer:
    def __init__(self, config: NormalizerConfig) -> None:
        self.config = config

    def normalize_commit(
        self,
        extraction_commit_directory: Path | str,
        *,
        run_id: str | None = None,
    ) -> NormalizationCommitBundle:
        started_at = datetime.now(timezone.utc)
        started_monotonic = time.monotonic()
        extraction = load_extraction_commit(
            extraction_commit_directory
        )
        active_run_id = run_id or utc_run_id().replace(
            "extraction-",
            "normalization-",
            1,
        )
        validate_run_id(active_run_id)
        self._ensure_target_is_available(active_run_id)
        build = build_normalization(extraction)
        stage_result = StageResult(
            stage=ResumeFrom.NORMALIZE,
            status=StageStatus.PASS,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            duration_seconds=time.monotonic() - started_monotonic,
            artifact_paths=[
                NORMALIZED_DOCUMENT_FILENAME,
                NORMALIZED_MARKDOWN_FILENAME,
                NORMALIZATION_REPORT_FILENAME,
            ],
            metrics={
                "input_source_element_count": (
                    build.report.input_source_element_count
                ),
                "normalized_block_count": (
                    build.report.normalized_block_count
                ),
                "filtered_source_element_count": (
                    build.report.filtered_source_element_count
                ),
                "valid_gfm_table_count": (
                    build.report.valid_gfm_table_count
                ),
                "ocr_invocation_count": 0,
                "next_action": "CHUNK",
            },
            message="Contract-native normalization passed",
        )
        return self._commit(
            run_id=active_run_id,
            extraction=extraction,
            build=build,
            stage_result=stage_result,
        )

    def _ensure_target_is_available(self, run_id: str) -> None:
        output_root = self.config.output_root.expanduser().resolve()

        if (output_root / run_id).exists():
            raise NormalizationCommitExistsError(
                f"Normalization commit already exists: {run_id}"
            )

    def _commit(
        self,
        *,
        run_id: str,
        extraction: ExtractionCommitBundle,
        build: NormalizationBuildResult,
        stage_result: StageResult,
    ) -> NormalizationCommitBundle:
        output_root = self.config.output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        final_directory = output_root / run_id

        if final_directory.exists():
            raise NormalizationCommitExistsError(
                f"Normalization commit already exists: {run_id}"
            )

        staging_directory = output_root / (
            f".{run_id}.staging-{uuid.uuid4().hex}"
        )
        staging_directory.mkdir(parents=False, exist_ok=False)

        try:
            payloads = {
                NORMALIZED_DOCUMENT_FILENAME: model_json_bytes(
                    build.document
                ),
                NORMALIZED_MARKDOWN_FILENAME: build.markdown.encode(
                    "utf-8"
                ),
                NORMALIZATION_REPORT_FILENAME: model_json_bytes(
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

            extraction_manifest_path = (
                extraction.commit_directory / "commit_manifest.json"
            )
            report = build.report
            manifest = NormalizationCommitManifest(
                schema_version=NORMALIZATION_COMMIT_SCHEMA_VERSION,
                normalizer_version=CONTRACT_NATIVE_NORMALIZER_VERSION,
                run_id=run_id,
                input_extraction_run_id=extraction.manifest.run_id,
                input_extraction_commit_directory=str(
                    extraction.commit_directory
                ),
                input_extraction_manifest_sha256=sha256_file(
                    extraction_manifest_path
                ),
                catalog_id=report.catalog_id,
                family_id=report.family_id,
                version_id=report.version_id,
                source_sha256=report.source_sha256,
                input_source_element_count=(
                    report.input_source_element_count
                ),
                selected_source_element_count=(
                    report.selected_source_element_count
                ),
                filtered_source_element_count=(
                    report.filtered_source_element_count
                ),
                normalized_block_count=report.normalized_block_count,
                artifact_count=len(artifacts),
                artifacts=artifacts,
            )
            write_payload(
                staging_directory / COMMIT_MANIFEST_FILENAME,
                model_json_bytes(manifest),
            )
            load_normalization_commit(staging_directory)

            if final_directory.exists():
                raise NormalizationCommitExistsError(
                    f"Normalization commit already exists: {run_id}"
                )

            staging_directory.rename(final_directory)

        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
            raise

        return load_normalization_commit(final_directory)


def load_normalization_commit(
    commit_directory: Path | str,
    *,
    verify_input_commit: bool = True,
) -> NormalizationCommitBundle:
    root = Path(commit_directory).expanduser().resolve()

    if not root.is_dir():
        raise NormalizationCommitIntegrityError(
            f"Normalization commit directory not found: {root}"
        )

    try:
        manifest = NormalizationCommitManifest.model_validate_json(
            (root / COMMIT_MANIFEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
    except Exception as error:
        raise NormalizationCommitIntegrityError(
            "Normalization manifest is missing or invalid"
        ) from error

    recorded_paths = {
        artifact.relative_path
        for artifact in manifest.artifacts
    }
    required_paths = {
        NORMALIZED_DOCUMENT_FILENAME,
        NORMALIZED_MARKDOWN_FILENAME,
        NORMALIZATION_REPORT_FILENAME,
        STAGE_RESULT_FILENAME,
    }

    if recorded_paths != required_paths:
        raise NormalizationCommitIntegrityError(
            "Normalization commit has unexpected typed artifacts"
        )

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }

    if actual_paths != required_paths | {COMMIT_MANIFEST_FILENAME}:
        raise NormalizationCommitIntegrityError(
            "Normalization file inventory differs from manifest"
        )

    for artifact in manifest.artifacts:
        path = (root / artifact.relative_path).resolve()

        if root not in path.parents:
            raise NormalizationCommitIntegrityError(
                "Normalization artifact escapes commit directory"
            )

        if path.stat().st_size != artifact.size_bytes:
            raise NormalizationCommitIntegrityError(
                f"Normalization size mismatch: {artifact.relative_path}"
            )

        if sha256_file(path) != artifact.sha256:
            raise NormalizationCommitIntegrityError(
                f"Normalization SHA256 mismatch: {artifact.relative_path}"
            )

    try:
        document = NormalizedDocument.model_validate_json(
            (root / NORMALIZED_DOCUMENT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        markdown = (root / NORMALIZED_MARKDOWN_FILENAME).read_text(
            encoding="utf-8"
        )
        report = NormalizationReport.model_validate_json(
            (root / NORMALIZATION_REPORT_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        stage_result = StageResult.model_validate_json(
            (root / STAGE_RESULT_FILENAME).read_text(encoding="utf-8")
        )
    except Exception as error:
        raise NormalizationCommitIntegrityError(
            "Committed normalization artifact is invalid"
        ) from error

    _validate_normalization_commit(
        manifest=manifest,
        document=document,
        markdown=markdown,
        report=report,
        stage_result=stage_result,
    )

    if verify_input_commit:
        extraction_directory = Path(
            manifest.input_extraction_commit_directory
        )
        extraction = load_extraction_commit(extraction_directory)
        extraction_manifest_path = (
            extraction.commit_directory / "commit_manifest.json"
        )

        if sha256_file(extraction_manifest_path) != (
            manifest.input_extraction_manifest_sha256
        ):
            raise NormalizationCommitIntegrityError(
                "Input extraction manifest SHA256 changed"
            )

        expected = build_normalization(extraction)

        if (
            expected.document != document
            or expected.markdown != markdown
            or expected.report != report
        ):
            raise NormalizationCommitIntegrityError(
                "Normalization is not reproducible from extraction commit"
            )

    return NormalizationCommitBundle(
        commit_directory=root,
        manifest=manifest,
        document=document,
        markdown=markdown,
        report=report,
        stage_result=stage_result,
    )


def _validate_normalization_commit(
    *,
    manifest: NormalizationCommitManifest,
    document: NormalizedDocument,
    markdown: str,
    report: NormalizationReport,
    stage_result: StageResult,
) -> None:
    if (
        manifest.normalizer_version
        != CONTRACT_NATIVE_NORMALIZER_VERSION
        or report.normalizer_version
        != CONTRACT_NATIVE_NORMALIZER_VERSION
    ):
        raise NormalizationCommitIntegrityError(
            "Normalization version differs from the installed normalizer"
        )

    if (
        manifest.input_extraction_run_id
        != report.input_extraction_run_id
    ):
        raise NormalizationCommitIntegrityError(
            "Input extraction run differs between manifest and report"
        )

    identity = (
        manifest.catalog_id,
        manifest.family_id,
        manifest.version_id,
        manifest.source_sha256,
    )

    if identity != (
        document.catalog_id,
        document.family_id,
        document.version_id,
        document.source_sha256,
    ):
        raise NormalizationCommitIntegrityError(
            "Normalized document identity differs from manifest"
        )

    if identity != (
        report.catalog_id,
        report.family_id,
        report.version_id,
        report.source_sha256,
    ):
        raise NormalizationCommitIntegrityError(
            "Normalization report identity differs from manifest"
        )

    reconstructed_markdown = "\n\n".join(
        block.markdown
        for block in document.blocks
    ) + "\n"

    if reconstructed_markdown != markdown:
        raise NormalizationCommitIntegrityError(
            "Normalized Markdown differs from typed blocks"
        )

    if len(markdown) != report.markdown_character_count:
        raise NormalizationCommitIntegrityError(
            "Markdown character count differs from report"
        )

    if markdown.count("\ufffd") != report.broken_character_count:
        raise NormalizationCommitIntegrityError(
            "Broken-character count differs from report"
        )

    manifest_counts = (
        manifest.input_source_element_count,
        manifest.selected_source_element_count,
        manifest.filtered_source_element_count,
        manifest.normalized_block_count,
    )
    report_counts = (
        report.input_source_element_count,
        report.selected_source_element_count,
        report.filtered_source_element_count,
        report.normalized_block_count,
    )

    if manifest_counts != report_counts:
        raise NormalizationCommitIntegrityError(
            "Normalization counts differ between manifest and report"
        )

    if len(document.blocks) != report.normalized_block_count:
        raise NormalizationCommitIntegrityError(
            "Normalized block count differs from report"
        )

    source_element_ids = [
        block.source_spans[0].source_element_id
        for block in document.blocks
    ]

    if source_element_ids != report.source_element_ids:
        raise NormalizationCommitIntegrityError(
            "Normalized source order differs from report"
        )

    source_span_count = 0
    page_numbers: set[int] = set()

    for block in document.blocks:
        if len(block.source_spans) != 1:
            raise NormalizationCommitIntegrityError(
                "Every normalized block must contain one source span"
            )

        span = block.source_spans[0]
        source_span_count += 1

        if span.source_anchor.source_sha256 != document.source_sha256:
            raise NormalizationCommitIntegrityError(
                "Normalized source anchor has a different source SHA256"
            )

        if span.source_anchor.page_number is not None:
            page_numbers.add(span.source_anchor.page_number)

        if span.char_start != 0 or span.char_end != len(block.markdown):
            raise NormalizationCommitIntegrityError(
                "Normalized span does not cover its complete block"
            )

    if source_span_count != report.source_span_count:
        raise NormalizationCommitIntegrityError(
            "Source span count differs from report"
        )

    if sorted(page_numbers) != report.page_numbers:
        raise NormalizationCommitIntegrityError(
            "Provenance page inventory differs from report"
        )

    if document.normalized_at != report.generated_at:
        raise NormalizationCommitIntegrityError(
            "Normalization timestamps differ between document and report"
        )

    if (
        stage_result.stage != ResumeFrom.NORMALIZE
        or stage_result.status != StageStatus.PASS
        or stage_result.metrics.get("next_action") != "CHUNK"
    ):
        raise NormalizationCommitIntegrityError(
            "Normalization stage result is inconsistent"
        )

    if stage_result.artifact_paths != [
        NORMALIZED_DOCUMENT_FILENAME,
        NORMALIZED_MARKDOWN_FILENAME,
        NORMALIZATION_REPORT_FILENAME,
    ]:
        raise NormalizationCommitIntegrityError(
            "Normalization stage artifact paths are inconsistent"
        )

    expected_stage_metrics = {
        "input_source_element_count": (
            report.input_source_element_count
        ),
        "normalized_block_count": report.normalized_block_count,
        "filtered_source_element_count": (
            report.filtered_source_element_count
        ),
        "valid_gfm_table_count": report.valid_gfm_table_count,
        "ocr_invocation_count": 0,
        "next_action": "CHUNK",
    }

    if stage_result.metrics != expected_stage_metrics:
        raise NormalizationCommitIntegrityError(
            "Normalization stage metrics differ from report"
        )
