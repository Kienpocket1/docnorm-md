from __future__ import annotations

import hashlib
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import docx
from docx import Document as open_docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.table import Table
from docx.text.paragraph import Paragraph

from rag_pipeline.contracts import (
    AssetClassification,
    AssetReference,
    AssetRole,
    ElementType,
    EngineName,
    ExtractedAsset,
    ExtractedDocument,
    RouteType,
    SourceAnchor,
    SourceElement,
)
from rag_pipeline.extractors.base import (
    ExtractionContractError,
    ExtractionOutput,
    ExtractionRequest,
    provenance_coverage,
    sha256_file,
    stable_identifier,
    validate_unified_document,
)


DOCX_NATIVE_EXTRACTOR_VERSION = "2026-08-18-5.0.1"
_HEADING_PATTERN = re.compile(r"^(?:heading|tiêu đề)\s*(\d+)$", re.I)
_LIST_STYLE_PATTERN = re.compile(r"^(?:list|danh sách)", re.I)


class DocxNativeExtractionError(RuntimeError):
    """Raised when a DOCX package cannot be extracted losslessly enough."""


@dataclass(frozen=True, slots=True)
class DocxNativeConfig:
    include_headers_footers: bool = True
    include_embedded_images: bool = True


@dataclass(slots=True)
class _AssetAccumulator:
    asset_id: str
    content_sha256: str
    media_type: str
    size_bytes: int
    physical_paths: list[str]
    references: list[AssetReference]


def _iter_body_blocks(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """Yield top-level paragraphs and tables in document reading order."""

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _style_name(paragraph: Paragraph) -> str:
    try:
        return paragraph.style.name or ""
    except (AttributeError, KeyError):
        return ""


def _heading_level(paragraph: Paragraph) -> int | None:
    match = _HEADING_PATTERN.match(_style_name(paragraph).strip())

    if match:
        return max(1, int(match.group(1)))

    properties = paragraph._p.pPr
    outline_level = (
        properties.outlineLvl
        if properties is not None
        else None
    )

    if outline_level is not None and outline_level.val is not None:
        return int(outline_level.val) + 1

    return None


def _list_info(paragraph: Paragraph) -> tuple[int, bool] | None:
    properties = paragraph._p.pPr
    numbering = properties.numPr if properties is not None else None

    if numbering is not None and numbering.numId is not None:
        level = (
            int(numbering.ilvl.val)
            if numbering.ilvl is not None
            and numbering.ilvl.val is not None
            else 0
        )
        style_name = _style_name(paragraph).casefold()
        numbered = "number" in style_name or "số" in style_name
        return level, numbered

    style_name = _style_name(paragraph).strip()

    if _LIST_STYLE_PATTERN.match(style_name):
        numbered = "number" in style_name.casefold()
        return 0, numbered

    return None


def _has_section_break(paragraph: Paragraph) -> bool:
    properties = paragraph._p.pPr
    return properties is not None and properties.sectPr is not None


def _clean_paragraph_text(paragraph: Paragraph) -> str:
    return paragraph.text.replace("\r", "\n").strip()


def _escape_gfm_cell(value: str) -> str:
    return (
        value.replace("\r", "")
        .replace("\n", "<br>")
        .replace("|", "\\|")
        .strip()
    )


def _render_table(table: Table) -> tuple[str, dict[str, Any]]:
    rows: list[list[str]] = []
    seen_cells: set[int] = set()
    merged_slot_count = 0

    for row in table.rows:
        rendered_row: list[str] = []

        for cell in row.cells:
            cell_identity = id(cell._tc)

            if cell_identity in seen_cells:
                rendered_row.append("")
                merged_slot_count += 1
                continue

            seen_cells.add(cell_identity)
            rendered_row.append(_escape_gfm_cell(cell.text))

        rows.append(rendered_row)

    column_count = max((len(row) for row in rows), default=1)

    if not rows:
        rows = [[""]]

    for row in rows:
        row.extend([""] * (column_count - len(row)))

    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row) + " |"
        for row in body
    )

    return "\n".join(lines), {
        "row_count": len(rows),
        "column_count": column_count,
        "merged_slot_count": merged_slot_count,
        "gfm_table": True,
    }


def _image_relationship_ids(paragraph: Paragraph) -> list[str]:
    try:
        values = paragraph._p.xpath(".//a:blip/@r:embed")
    except (AttributeError, KeyError, TypeError):
        return []

    return [str(value) for value in values if value]


def _update_heading_stack(
    stack: list[str],
    level: int,
    text: str,
) -> list[str]:
    position = max(0, level - 1)
    updated = stack[:position]

    while len(updated) < position:
        updated.append("")

    updated.append(text)
    return updated


class DocxNativeExtractor:
    engine = EngineName.DOCX_NATIVE

    def __init__(self, config: DocxNativeConfig | None = None) -> None:
        self.config = config or DocxNativeConfig()

    def extract(self, request: ExtractionRequest) -> ExtractionOutput:
        if request.route.route_type != RouteType.DOCX_NATIVE:
            raise ExtractionContractError(
                "DocxNativeExtractor requires a DOCX_NATIVE route"
            )

        if request.route.primary_engine != EngineName.DOCX_NATIVE:
            raise ExtractionContractError(
                "DOCX native route must use DOCX_NATIVE engine"
            )

        if not zipfile.is_zipfile(request.source_path):
            raise DocxNativeExtractionError(
                "Source is not a valid OPC/DOCX package"
            )

        started = time.perf_counter()
        source_sha256 = sha256_file(request.source_path)

        try:
            document = open_docx(str(request.source_path))
        except Exception as error:
            raise DocxNativeExtractionError(
                f"Unable to open DOCX package: {error}"
            ) from error

        elements: list[SourceElement] = []
        assets_by_hash: dict[str, _AssetAccumulator] = {}
        title_path: list[str] = []
        heading_stack: list[str] = []
        section_index = 0
        table_count = 0
        list_count = 0
        embedded_image_count = 0
        header_footer_count = 0
        pending_list: list[tuple[str, int, bool]] = []
        pending_list_path: list[str] = []
        pending_list_section = 0

        def current_section_path() -> list[str]:
            return [
                value
                for value in [*title_path, *heading_stack]
                if value
            ]

        def append_element(
            element_type: ElementType,
            content: str,
            *,
            section_path: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
        ) -> SourceElement:
            block_index = len(elements)
            active_path = list(
                current_section_path()
                if section_path is None
                else section_path
            )
            element_id = stable_identifier(
                "element",
                source_sha256,
                block_index,
                element_type.value,
            )
            anchor = SourceAnchor(
                anchor_id=stable_identifier(
                    "anchor",
                    source_sha256,
                    block_index,
                    element_type.value,
                ),
                source_sha256=source_sha256,
                page_number=None,
                block_index=block_index,
                bbox=None,
                section_path=active_path,
            )
            element = SourceElement(
                element_id=element_id,
                element_type=element_type,
                content=content,
                source_anchor=anchor,
                engine=EngineName.DOCX_NATIVE,
                metadata=metadata or {},
            )
            elements.append(element)
            return element

        def flush_list() -> None:
            nonlocal list_count

            if not pending_list:
                return

            lines: list[str] = []
            item_metadata: list[dict[str, Any]] = []

            for ordinal, (text, level, numbered) in enumerate(
                pending_list,
                start=1,
            ):
                marker = f"{ordinal}." if numbered else "-"
                lines.append(f"{'  ' * level}{marker} {text}")
                item_metadata.append(
                    {
                        "ordinal": ordinal,
                        "level": level,
                        "numbered": numbered,
                    }
                )

            append_element(
                ElementType.LIST,
                "\n".join(lines),
                section_path=pending_list_path,
                metadata={
                    "indexable": True,
                    "section_index": pending_list_section,
                    "item_count": len(pending_list),
                    "items": item_metadata,
                },
            )
            pending_list.clear()
            list_count += 1

        def append_embedded_images(paragraph: Paragraph) -> None:
            nonlocal embedded_image_count

            if not self.config.include_embedded_images:
                return

            for relationship_id in _image_relationship_ids(paragraph):
                relationship = paragraph.part.rels.get(relationship_id)

                if (
                    relationship is None
                    or relationship.reltype != RT.IMAGE
                ):
                    continue

                target_part = relationship.target_part
                blob = bytes(target_part.blob)
                content_sha256 = hashlib.sha256(blob).hexdigest()
                internal_path = str(target_part.partname).lstrip("/")
                asset_id = stable_identifier(
                    "asset",
                    source_sha256,
                    content_sha256,
                )
                image_element = append_element(
                    ElementType.IMAGE,
                    "",
                    metadata={
                        "indexable": False,
                        "asset_id": asset_id,
                        "embedded_path": internal_path,
                        "relationship_id": relationship_id,
                        "content_sha256": content_sha256,
                    },
                )
                reference = AssetReference(
                    reference_id=stable_identifier(
                        "asset-reference",
                        source_sha256,
                        image_element.source_anchor.block_index,
                        relationship_id,
                    ),
                    source_path=internal_path,
                    source_anchor=image_element.source_anchor,
                    referenced_in_structure=True,
                    metadata={"container": "docx"},
                )
                accumulator = assets_by_hash.get(content_sha256)

                if accumulator is None:
                    accumulator = _AssetAccumulator(
                        asset_id=asset_id,
                        content_sha256=content_sha256,
                        media_type=target_part.content_type,
                        size_bytes=len(blob),
                        physical_paths=[],
                        references=[],
                    )
                    assets_by_hash[content_sha256] = accumulator

                if internal_path not in accumulator.physical_paths:
                    accumulator.physical_paths.append(internal_path)

                accumulator.references.append(reference)
                embedded_image_count += 1

        for block in _iter_body_blocks(document):
            if isinstance(block, Paragraph):
                text = _clean_paragraph_text(block)
                list_info = _list_info(block)
                section_break = _has_section_break(block)

                if list_info is not None and text:
                    if not pending_list:
                        pending_list_path = current_section_path()
                        pending_list_section = section_index

                    pending_list.append(
                        (text, list_info[0], list_info[1])
                    )
                    append_embedded_images(block)

                    if section_break:
                        flush_list()
                        section_index += 1

                    continue

                flush_list()
                style_name = _style_name(block)
                heading_level = _heading_level(block)

                if text:
                    if style_name.casefold() in {
                        "title",
                        "document title",
                        "tiêu đề",
                    }:
                        title_path = [text]
                        heading_stack = []
                        element_type = ElementType.TITLE
                    elif heading_level is not None:
                        heading_stack = _update_heading_stack(
                            heading_stack,
                            heading_level,
                            text,
                        )
                        element_type = ElementType.HEADING
                    elif "caption" in style_name.casefold():
                        element_type = ElementType.CAPTION
                    else:
                        element_type = ElementType.PARAGRAPH

                    append_element(
                        element_type,
                        text,
                        metadata={
                            "indexable": True,
                            "style_name": style_name,
                            "heading_level": heading_level,
                            "section_index": section_index,
                            "section_break_after": section_break,
                        },
                    )

                append_embedded_images(block)

                if section_break:
                    section_index += 1

            elif isinstance(block, Table):
                flush_list()
                table_markdown, table_metrics = _render_table(block)
                append_element(
                    ElementType.TABLE,
                    table_markdown,
                    metadata={
                        "indexable": True,
                        "section_index": section_index,
                        **table_metrics,
                    },
                )
                table_count += 1

        flush_list()

        if self.config.include_headers_footers:
            header_footer_count = self._append_headers_footers(
                document,
                append_element,
            )

        if not elements:
            raise DocxNativeExtractionError(
                "DOCX contains no extractable content"
            )

        assets = [
            ExtractedAsset(
                asset_id=asset.asset_id,
                source_sha256=source_sha256,
                content_sha256=asset.content_sha256,
                media_type=asset.media_type,
                size_bytes=asset.size_bytes,
                classification=AssetClassification.UNKNOWN,
                role=AssetRole.UNKNOWN,
                physical_paths=asset.physical_paths,
                references=asset.references,
                retain_raw=True,
                indexable=False,
                metadata={
                    "container": "docx",
                    "occurrence_count": len(asset.references),
                },
            )
            for asset in sorted(
                assets_by_hash.values(),
                key=lambda item: item.asset_id,
            )
        ]
        extracted_document = ExtractedDocument(
            catalog_id=request.catalog_id,
            family_id=request.family_id,
            version_id=request.version_id,
            source_sha256=source_sha256,
            page_count=None,
            engine_chain=[EngineName.DOCX_NATIVE],
            elements=elements,
            assets=assets,
        )
        validate_unified_document(
            request,
            extracted_document,
            expected_engine=EngineName.DOCX_NATIVE,
        )

        if sha256_file(request.source_path) != source_sha256:
            raise DocxNativeExtractionError(
                "DOCX source changed while extraction was running"
            )

        elapsed = time.perf_counter() - started
        metrics = {
            "extractor_version": DOCX_NATIVE_EXTRACTOR_VERSION,
            "document_format": "DOCX",
            "source_element_count": len(elements),
            "title_count": sum(
                element.element_type == ElementType.TITLE
                for element in elements
            ),
            "heading_count": sum(
                element.element_type == ElementType.HEADING
                for element in elements
            ),
            "paragraph_count": sum(
                element.element_type == ElementType.PARAGRAPH
                for element in elements
            ),
            "table_count": table_count,
            "list_count": list_count,
            "caption_count": sum(
                element.element_type == ElementType.CAPTION
                for element in elements
            ),
            "header_footer_count": header_footer_count,
            "section_count": len(document.sections),
            "embedded_image_occurrence_count": embedded_image_count,
            "canonical_asset_count": len(assets),
            "provenance_coverage": provenance_coverage(
                extracted_document
            ),
            "page_count_known": False,
            "external_calls": 0,
            "ocr_invocations": 0,
        }
        return ExtractionOutput(
            document=extracted_document,
            engine_version=(
                f"python-docx/{docx.__version__};"
                f"extractor/{DOCX_NATIVE_EXTRACTOR_VERSION}"
            ),
            elapsed_seconds=elapsed,
            artifact_paths=(),
            metrics=metrics,
        )

    @staticmethod
    def _append_headers_footers(
        document: DocxDocument,
        append_element: Any,
    ) -> int:
        seen: set[tuple[str, str, str]] = set()
        count = 0

        for section_index, section in enumerate(document.sections):
            containers = (
                ("header", section.header, ElementType.HEADER),
                ("first_page_header", section.first_page_header,
                 ElementType.HEADER),
                ("even_page_header", section.even_page_header,
                 ElementType.HEADER),
                ("footer", section.footer, ElementType.FOOTER),
                ("first_page_footer", section.first_page_footer,
                 ElementType.FOOTER),
                ("even_page_footer", section.even_page_footer,
                 ElementType.FOOTER),
            )

            for role, container, element_type in containers:
                if section_index > 0 and container.is_linked_to_previous:
                    continue

                part_name = str(container.part.partname)

                for paragraph in container.paragraphs:
                    text = _clean_paragraph_text(paragraph)

                    if not text:
                        continue

                    identity = (role, part_name, text)

                    if identity in seen:
                        continue

                    seen.add(identity)
                    append_element(
                        element_type,
                        text,
                        section_path=[],
                        metadata={
                            "indexable": False,
                            "section_index": section_index,
                            "header_footer_role": role,
                            "part_name": part_name,
                            "style_name": _style_name(paragraph),
                        },
                    )
                    count += 1

        return count
