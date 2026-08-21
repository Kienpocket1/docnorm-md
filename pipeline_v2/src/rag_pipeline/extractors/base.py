from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from rag_pipeline.contracts import (
    EngineName,
    ExtractedDocument,
    FileFormat,
    RouteDecision,
)


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_EXPECTED_SUFFIXES = {
    FileFormat.PDF: {".pdf"},
    FileFormat.DOCX: {".docx"},
    FileFormat.DOC: {".doc"},
}


class ExtractionContractError(ValueError):
    """Raised when an extractor violates the shared Stage 5 contract."""


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """Engine-neutral input for every Stage 5 extractor."""

    source_path: Path
    catalog_id: str
    family_id: str
    version_id: str
    route: RouteDecision
    run_id: str | None = None

    def __post_init__(self) -> None:
        resolved_source = Path(self.source_path).expanduser().resolve()
        object.__setattr__(self, "source_path", resolved_source)

        if not resolved_source.is_file():
            raise FileNotFoundError(
                f"Extraction source not found: {resolved_source}"
            )

        if resolved_source.stat().st_size < 1:
            raise ExtractionContractError("Extraction source is empty")

        for field_name in ("catalog_id", "family_id", "version_id"):
            if not getattr(self, field_name).strip():
                raise ExtractionContractError(
                    f"{field_name} must not be empty"
                )

        if self.route.catalog_id != self.catalog_id:
            raise ExtractionContractError(
                "Route catalog_id does not match extraction request"
            )

        if self.route.version_id != self.version_id:
            raise ExtractionContractError(
                "Route version_id does not match extraction request"
            )

        expected_suffixes = _EXPECTED_SUFFIXES.get(self.route.file_format)

        if (
            expected_suffixes is not None
            and resolved_source.suffix.casefold() not in expected_suffixes
        ):
            expected = ", ".join(sorted(expected_suffixes))
            raise ExtractionContractError(
                f"{self.route.file_format.value} route requires {expected}"
            )

        if self.run_id is not None and not _RUN_ID_PATTERN.fullmatch(
            self.run_id
        ):
            raise ExtractionContractError(
                "run_id contains unsupported characters"
            )


@dataclass(frozen=True, slots=True)
class ExtractionOutput:
    """Unified successful output emitted by PDF and DOCX extractors."""

    document: ExtractedDocument
    engine_version: str
    elapsed_seconds: float
    artifact_paths: tuple[Path, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    raw_handle: Any | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.engine_version.strip():
            raise ExtractionContractError(
                "engine_version must not be empty"
            )

        if self.elapsed_seconds < 0:
            raise ExtractionContractError(
                "elapsed_seconds must be non-negative"
            )


@runtime_checkable
class UnifiedDocumentExtractor(Protocol):
    engine: EngineName

    def extract(self, request: ExtractionRequest) -> ExtractionOutput: ...


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def stable_identifier(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{suffix}"


def validate_unified_document(
    request: ExtractionRequest,
    document: ExtractedDocument,
    *,
    expected_engine: EngineName,
) -> None:
    """Enforce the common boundary before Stage 6 receives a document."""

    expected_identity = (
        request.catalog_id,
        request.family_id,
        request.version_id,
    )
    actual_identity = (
        document.catalog_id,
        document.family_id,
        document.version_id,
    )

    if actual_identity != expected_identity:
        raise ExtractionContractError(
            "ExtractedDocument identity does not match request"
        )

    source_sha256 = sha256_file(request.source_path)

    if document.source_sha256 != source_sha256:
        raise ExtractionContractError(
            "ExtractedDocument source SHA256 does not match source file"
        )

    if expected_engine not in document.engine_chain:
        raise ExtractionContractError(
            "ExtractedDocument engine_chain does not contain route engine"
        )

    if not document.elements:
        raise ExtractionContractError(
            "ExtractedDocument must contain at least one element"
        )

    anchors = [element.source_anchor for element in document.elements]

    if any(anchor.source_sha256 != source_sha256 for anchor in anchors):
        raise ExtractionContractError(
            "Every source anchor must reference the source SHA256"
        )


def provenance_coverage(document: ExtractedDocument) -> float:
    if not document.elements:
        return 0.0

    covered = sum(
        1
        for element in document.elements
        if element.source_anchor.anchor_id
        and element.source_anchor.source_sha256
        and element.source_anchor.block_index >= 0
    )
    return covered / len(document.elements)
