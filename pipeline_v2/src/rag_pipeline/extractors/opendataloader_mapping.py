from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rag_pipeline.contracts import (
    AssetClassification,
    AssetReference,
    AssetRole,
    BoundingBox,
    ElementType,
    EngineName,
    ExtractedAsset,
    ExtractedDocument,
    ExtractionQualityIssue,
    FailureDomain,
    QualitySeverity,
    SourceAnchor,
    SourceElement,
)
from rag_pipeline.extractors.opendataloader_local import (
    OpenDataLoaderLocalRun,
    WorkerArtifactRecord,
)


PETROLIMEX_AVIATION_LOGO_SHA256 = (
    "edb02a3290131f7d1ff7316935219287"
    "033b35dab9499242a2abbfcad5607330"
)
OPENDATALOADER_MAPPING_VERSION = "2026-08-17-5.1D.1"


@dataclass(frozen=True, slots=True)
class DocumentIdentity:
    catalog_id: str
    family_id: str
    version_id: str


@dataclass(frozen=True, slots=True)
class AssetPolicy:
    classification: AssetClassification
    role: AssetRole
    indexable: bool
    reason_code: str


@dataclass(frozen=True, slots=True)
class TableRenderResult:
    markdown: str
    declared_rows: int
    declared_columns: int
    actual_rows: int
    actual_cells: int
    merged_cells: int
    invalid_positions: int
    collisions: int
    missing_slots: int


@dataclass(frozen=True, slots=True)
class OpenDataLoaderMappingResult:
    document: ExtractedDocument
    mapping_issues: tuple[ExtractionQualityIssue, ...]
    indexable_element_ids: tuple[str, ...]
    filtered_element_ids: tuple[str, ...]
    metrics: dict[str, Any]


DEFAULT_ASSET_POLICIES: dict[str, AssetPolicy] = {
    PETROLIMEX_AVIATION_LOGO_SHA256: AssetPolicy(
        classification=AssetClassification.DECORATIVE,
        role=AssetRole.BRAND_LOGO,
        indexable=False,
        reason_code="known_petrolimex_aviation_brand_logo",
    )
}


def normalize_key(value: object) -> str:
    return re.sub(r"[\s_-]+", "", str(value)).casefold()


def clean_text(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(text.split()).strip()


def get_value(mapping: Mapping[str, Any], *expected_names: str) -> Any:
    expected = {normalize_key(name) for name in expected_names}

    for key, value in mapping.items():
        if normalize_key(key) in expected:
            return value

    return None


def node_type(node: Mapping[str, Any]) -> str | None:
    value = get_value(node, "type", "element type")

    if isinstance(value, str) and value.strip():
        return value.strip().casefold()

    return None


def safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not number.is_integer():
        return None

    return int(number)


def stable_id(prefix: str, *parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def source_basename(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None

    return Path(value.replace("\\", "/")).name


def iter_typed_nodes(value: Any):
    if isinstance(value, dict):
        if node_type(value) is not None:
            yield value

        for child in value.values():
            if isinstance(child, (dict, list)):
                yield from iter_typed_nodes(child)

    elif isinstance(value, list):
        for child in value:
            yield from iter_typed_nodes(child)


def unique_text_parts(parts: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()

    for part in parts:
        text = part.strip()
        key = clean_text(text).casefold()

        if text and key not in seen:
            seen.add(key)
            output.append(text)

    return output


def render_generic_text(node: Mapping[str, Any]) -> str:
    parts: list[str] = []
    direct_content = get_value(node, "content", "text", "value")

    if isinstance(direct_content, str):
        content = clean_text(direct_content)

        if content:
            parts.append(content)

    children = get_value(node, "kids", "children")

    if isinstance(children, list):
        for child in children:
            if not isinstance(child, dict):
                continue

            child_kind = node_type(child)

            if child_kind == "list":
                rendered = render_list(child)
            elif child_kind == "table":
                rendered = render_table(child).markdown
            else:
                rendered = render_generic_text(child)

            if rendered:
                parts.append(rendered)

    return "\n".join(unique_text_parts(parts))


def alphabetic_ordinal(ordinal: int) -> str:
    letters: list[str] = []
    value = ordinal

    while value > 0:
        value, remainder = divmod(value - 1, 26)
        letters.append(chr(ord("a") + remainder))

    return "".join(reversed(letters)) or "a"


def roman_ordinal(ordinal: int) -> str:
    values = (
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    )
    output: list[str] = []
    remainder = ordinal

    for value, numeral in values:
        count, remainder = divmod(remainder, value)
        output.extend(numeral for _ in range(count))

    return "".join(output) or "i"


def list_marker(numbering_style: object, ordinal: int) -> str:
    style = clean_text(numbering_style).casefold()

    if "unordered" in style or not style:
        return "-"

    if "english letters" in style:
        marker = alphabetic_ordinal(ordinal)

        if "upper case" in style:
            marker = marker.upper()

        return f"{marker}."

    if "roman" in style:
        return f"{roman_ordinal(ordinal)}."

    return f"{ordinal}."


def render_list(node: Mapping[str, Any]) -> str:
    items = get_value(node, "list items", "items")

    if not isinstance(items, list):
        return ""

    numbering_style = get_value(node, "numbering style")
    lines: list[str] = []

    for ordinal, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        marker = list_marker(numbering_style, ordinal)
        direct_content = get_value(item, "content", "text", "value")
        item_parts: list[str] = []

        if isinstance(direct_content, str):
            content = clean_text(direct_content)

            if content:
                item_parts.append(content)

        children = get_value(item, "kids", "children")
        nested_blocks: list[str] = []

        if isinstance(children, list):
            for child in children:
                if not isinstance(child, dict):
                    continue

                if node_type(child) == "list":
                    rendered = render_list(child)
                    if rendered:
                        nested_blocks.append(rendered)
                else:
                    rendered = render_generic_text(child)
                    if rendered:
                        item_parts.append(rendered)

        item_parts = unique_text_parts(item_parts)
        primary_text = item_parts[0] if item_parts else ""
        lines.append(f"{marker} {primary_text}".rstrip())

        for continuation in item_parts[1:]:
            lines.extend(
                f"  {line}"
                for line in continuation.splitlines()
            )

        for nested_block in nested_blocks:
            lines.extend(
                f"  {line}"
                for line in nested_block.splitlines()
            )

    return "\n".join(lines).strip()


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>").strip()


def render_table(node: Mapping[str, Any]) -> TableRenderResult:
    declared_rows = safe_int(get_value(node, "number of rows")) or 0
    declared_columns = (
        safe_int(get_value(node, "number of columns")) or 0
    )
    rows = get_value(node, "rows")
    rows = rows if isinstance(rows, list) else []

    cells: list[Mapping[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        row_cells = get_value(row, "cells")

        if isinstance(row_cells, list):
            cells.extend(
                cell
                for cell in row_cells
                if isinstance(cell, dict)
            )

    row_numbers = [
        value
        for cell in cells
        if (value := safe_int(get_value(cell, "row number")))
        is not None
    ]
    column_numbers = [
        value
        for cell in cells
        if (value := safe_int(get_value(cell, "column number")))
        is not None
    ]
    row_base = 0 if row_numbers and min(row_numbers) == 0 else 1
    column_base = (
        0 if column_numbers and min(column_numbers) == 0 else 1
    )

    matrix: list[list[str | None]] = [
        [None for _ in range(declared_columns)]
        for _ in range(declared_rows)
    ]
    merged_cells = 0
    invalid_positions = 0
    collisions = 0

    for cell in cells:
        raw_row = safe_int(get_value(cell, "row number"))
        raw_column = safe_int(get_value(cell, "column number"))

        if raw_row is None or raw_column is None:
            invalid_positions += 1
            continue

        row_index = raw_row - row_base
        column_index = raw_column - column_base
        row_span = safe_int(get_value(cell, "row span")) or 1
        column_span = safe_int(get_value(cell, "column span")) or 1

        if row_span > 1 or column_span > 1:
            merged_cells += 1

        if (
            row_index < 0
            or column_index < 0
            or row_index + row_span > declared_rows
            or column_index + column_span > declared_columns
        ):
            invalid_positions += 1
            continue

        content = render_generic_text(cell)

        for row_offset in range(row_span):
            for column_offset in range(column_span):
                target_row = row_index + row_offset
                target_column = column_index + column_offset

                if matrix[target_row][target_column] is not None:
                    collisions += 1
                    continue

                if row_offset == 0 and column_offset == 0:
                    matrix[target_row][target_column] = content
                else:
                    matrix[target_row][target_column] = ""

    missing_slots = sum(
        1
        for row in matrix
        for value in row
        if value is None
    )

    normalized_matrix = [
        [escape_table_cell(value or "") for value in row]
        for row in matrix
    ]
    markdown_lines: list[str] = []

    if normalized_matrix and declared_columns > 0:
        markdown_lines.append(
            "| " + " | ".join(normalized_matrix[0]) + " |"
        )
        markdown_lines.append(
            "| " + " | ".join("---" for _ in range(declared_columns))
            + " |"
        )

        for row in normalized_matrix[1:]:
            markdown_lines.append(
                "| " + " | ".join(row) + " |"
            )

    return TableRenderResult(
        markdown="\n".join(markdown_lines),
        declared_rows=declared_rows,
        declared_columns=declared_columns,
        actual_rows=len(rows),
        actual_cells=len(cells),
        merged_cells=merged_cells,
        invalid_positions=invalid_positions,
        collisions=collisions,
        missing_slots=missing_slots,
    )


class OpenDataLoaderJsonMapper:
    def __init__(
        self,
        known_asset_policies: Mapping[str, AssetPolicy] | None = None,
    ) -> None:
        policies = (
            DEFAULT_ASSET_POLICIES
            if known_asset_policies is None
            else known_asset_policies
        )
        self.known_asset_policies = dict(policies)

    def map_run(
        self,
        run: OpenDataLoaderLocalRun,
        identity: DocumentIdentity,
    ) -> OpenDataLoaderMappingResult:
        manifest = run.manifest
        source_sha256 = manifest.input.sha256
        structured_json_path = manifest.output.structured_json.path
        markdown_path = manifest.output.markdown.path

        structured_payload = json.loads(
            structured_json_path.read_text(encoding="utf-8-sig")
        )

        if not isinstance(structured_payload, dict):
            raise TypeError("OpenDataLoader JSON root must be an object")

        markdown = markdown_path.read_text(encoding="utf-8")
        mapping_issues: list[ExtractionQualityIssue] = []

        assets, asset_by_path, asset_issues = self._build_assets(
            structured_payload=structured_payload,
            structured_json_path=structured_json_path,
            markdown=markdown,
            run=run,
        )
        mapping_issues.extend(asset_issues)

        root_children = get_value(
            structured_payload,
            "kids",
            "children",
        )

        if not isinstance(root_children, list):
            raise ValueError(
                "OpenDataLoader JSON root does not contain kids list"
            )

        elements: list[SourceElement] = []
        indexable_element_ids: list[str] = []
        filtered_element_ids: list[str] = []
        current_heading_path: list[str] = []
        mapped_table_count = 0
        mapped_list_count = 0
        merged_cell_count = 0
        invalid_table_slot_count = 0
        skipped_node_count = 0
        bbox_element_count = 0

        for block_index, raw_node in enumerate(root_children):
            if not isinstance(raw_node, dict):
                skipped_node_count += 1
                continue

            kind = node_type(raw_node) or "unknown"
            page_number = safe_int(
                get_value(raw_node, "page number", "page")
            )
            bbox, bbox_valid = self._bounding_box(raw_node)

            if not bbox_valid:
                mapping_issues.append(
                    self._issue(
                        source_sha256=source_sha256,
                        reason_code="invalid_source_bbox",
                        severity=QualitySeverity.ERROR,
                        failure_domain=FailureDomain.PROVENANCE,
                        message="Source block has an invalid bounding box",
                        page_numbers=(
                            [page_number]
                            if page_number is not None
                            else []
                        ),
                        triggers_fallback=True,
                    )
                )

            element_type = self._element_type(raw_node)
            metadata: dict[str, Any] = {
                "odl_type": kind,
                "odl_id": get_value(raw_node, "id"),
                "indexable": True,
            }

            if kind == "table":
                table_result = render_table(raw_node)
                content = table_result.markdown
                mapped_table_count += 1
                merged_cell_count += table_result.merged_cells
                invalid_table_slot_count += (
                    table_result.invalid_positions
                    + table_result.collisions
                    + table_result.missing_slots
                )
                metadata.update(
                    {
                        "declared_rows": table_result.declared_rows,
                        "declared_columns": table_result.declared_columns,
                        "actual_rows": table_result.actual_rows,
                        "actual_cells": table_result.actual_cells,
                        "merged_cells": table_result.merged_cells,
                        "invalid_positions": (
                            table_result.invalid_positions
                        ),
                        "collisions": table_result.collisions,
                        "missing_slots": table_result.missing_slots,
                    }
                )

                if (
                    table_result.actual_rows
                    != table_result.declared_rows
                    or table_result.invalid_positions
                    or table_result.collisions
                    or table_result.missing_slots
                ):
                    mapping_issues.append(
                        self._issue(
                            source_sha256=source_sha256,
                            reason_code="table_grid_mapping_failed",
                            severity=QualitySeverity.ERROR,
                            failure_domain=FailureDomain.TABLE,
                            message=(
                                "Table grid could not be mapped without "
                                "loss"
                            ),
                            page_numbers=(
                                [page_number]
                                if page_number is not None
                                else []
                            ),
                            triggers_fallback=True,
                            details={
                                "declared_rows": (
                                    table_result.declared_rows
                                ),
                                "declared_columns": (
                                    table_result.declared_columns
                                ),
                                "actual_rows": table_result.actual_rows,
                                "actual_cells": table_result.actual_cells,
                                "invalid_positions": (
                                    table_result.invalid_positions
                                ),
                                "collisions": table_result.collisions,
                                "missing_slots": table_result.missing_slots,
                            },
                        )
                    )

            elif kind == "list":
                content = render_list(raw_node)
                mapped_list_count += 1
                metadata.update(
                    {
                        "numbering_style": get_value(
                            raw_node,
                            "numbering style",
                        ),
                        "declared_items": get_value(
                            raw_node,
                            "number of list items",
                        ),
                    }
                )

            elif kind == "image":
                content = ""
                source = get_value(raw_node, "source")
                image_record = self._match_image_record(
                    source=source,
                    structured_json_path=structured_json_path,
                    image_records=manifest.output.images,
                )
                asset = (
                    asset_by_path.get(image_record.relative_path)
                    if image_record is not None
                    else None
                )

                if asset is not None:
                    metadata.update(
                        {
                            "asset_id": asset.asset_id,
                            "asset_classification": (
                                asset.classification.value
                            ),
                            "asset_role": asset.role.value,
                            "asset_content_sha256": (
                                asset.content_sha256
                            ),
                            "indexable": asset.indexable,
                        }
                    )
                else:
                    metadata["indexable"] = False

            else:
                content = render_generic_text(raw_node)

            if kind == "heading":
                heading_text = content.strip()
                heading_level = clean_text(
                    get_value(raw_node, "heading level", "level")
                )
                metadata["heading_level_raw"] = heading_level

                if heading_level.casefold() == "doctitle":
                    current_heading_path = [heading_text]
                elif current_heading_path:
                    current_heading_path = [
                        current_heading_path[0],
                        heading_text,
                    ]
                else:
                    current_heading_path = [heading_text]

            if not content and element_type not in {
                ElementType.IMAGE,
                ElementType.FLOWCHART,
            }:
                skipped_node_count += 1
                mapping_issues.append(
                    self._issue(
                        source_sha256=source_sha256,
                        reason_code="empty_mapped_source_block",
                        severity=QualitySeverity.ERROR,
                        failure_domain=FailureDomain.TEXT,
                        message=(
                            "Non-visual source block mapped to empty text"
                        ),
                        page_numbers=(
                            [page_number]
                            if page_number is not None
                            else []
                        ),
                        triggers_fallback=True,
                    )
                )
                continue

            element_id = stable_id(
                "element",
                source_sha256,
                block_index,
                page_number,
                kind,
            )
            anchor = SourceAnchor(
                anchor_id=stable_id(
                    "anchor",
                    source_sha256,
                    block_index,
                    page_number,
                    kind,
                ),
                source_sha256=source_sha256,
                page_number=page_number,
                block_index=block_index,
                bbox=bbox,
                section_path=list(current_heading_path),
            )
            element = SourceElement(
                element_id=element_id,
                element_type=element_type,
                content=content,
                source_anchor=anchor,
                engine=EngineName.OPENDATALOADER_LOCAL,
                metadata=metadata,
            )
            elements.append(element)

            if bbox is not None:
                bbox_element_count += 1

            if bool(metadata.get("indexable")):
                indexable_element_ids.append(element_id)
            else:
                filtered_element_ids.append(element_id)

        document = ExtractedDocument(
            catalog_id=identity.catalog_id,
            family_id=identity.family_id,
            version_id=identity.version_id,
            source_sha256=source_sha256,
            page_count=manifest.output.page_count,
            engine_chain=[EngineName.OPENDATALOADER_LOCAL],
            elements=elements,
            assets=assets,
        )
        pages_with_elements = sorted(
            {
                element.source_anchor.page_number
                for element in elements
                if element.source_anchor.page_number is not None
            }
        )
        metrics = {
            "mapping_version": OPENDATALOADER_MAPPING_VERSION,
            "root_node_count": len(root_children),
            "mapped_element_count": len(elements),
            "indexable_element_count": len(indexable_element_ids),
            "filtered_element_count": len(filtered_element_ids),
            "skipped_node_count": skipped_node_count,
            "bbox_element_count": bbox_element_count,
            "bbox_coverage": (
                bbox_element_count / len(elements)
                if elements
                else 0.0
            ),
            "pages_with_elements": pages_with_elements,
            "mapped_table_count": mapped_table_count,
            "mapped_list_count": mapped_list_count,
            "merged_cell_count": merged_cell_count,
            "invalid_table_slot_count": invalid_table_slot_count,
            "asset_count": len(assets),
            "physical_image_file_count": sum(
                len(asset.physical_paths)
                for asset in assets
            ),
            "asset_reference_count": sum(
                len(asset.references)
                for asset in assets
            ),
        }

        return OpenDataLoaderMappingResult(
            document=document,
            mapping_issues=tuple(mapping_issues),
            indexable_element_ids=tuple(indexable_element_ids),
            filtered_element_ids=tuple(filtered_element_ids),
            metrics=metrics,
        )

    def _build_assets(
        self,
        *,
        structured_payload: dict[str, Any],
        structured_json_path: Path,
        markdown: str,
        run: OpenDataLoaderLocalRun,
    ) -> tuple[
        list[ExtractedAsset],
        dict[str, ExtractedAsset],
        list[ExtractionQualityIssue],
    ]:
        manifest = run.manifest
        source_sha256 = manifest.input.sha256
        image_records = manifest.output.images
        image_by_relative_path = {
            image.relative_path: image
            for image in image_records
        }
        markdown_image_names = {
            name.casefold()
            for name in re.findall(
                r"imageFile\d+\.[A-Za-z0-9]+",
                markdown,
                flags=re.IGNORECASE,
            )
        }
        structured_references: dict[
            str,
            list[tuple[int, Mapping[str, Any]]],
        ] = {}
        issues: list[ExtractionQualityIssue] = []

        image_nodes = [
            node
            for node in iter_typed_nodes(structured_payload)
            if node_type(node) == "image"
        ]

        for image_index, image_node in enumerate(image_nodes):
            image_record = self._match_image_record(
                source=get_value(image_node, "source"),
                structured_json_path=structured_json_path,
                image_records=image_records,
            )

            if image_record is None:
                page_number = safe_int(
                    get_value(image_node, "page number", "page")
                )
                issues.append(
                    self._issue(
                        source_sha256=source_sha256,
                        reason_code="unresolved_image_reference",
                        severity=QualitySeverity.ERROR,
                        failure_domain=FailureDomain.PROVENANCE,
                        message=(
                            "Structured image reference does not match "
                            "a physical artifact"
                        ),
                        page_numbers=(
                            [page_number]
                            if page_number is not None
                            else []
                        ),
                        triggers_fallback=True,
                    )
                )
                continue

            structured_references.setdefault(
                image_record.relative_path,
                [],
            ).append((image_index, image_node))

        assets: list[ExtractedAsset] = []
        asset_by_relative_path: dict[str, ExtractedAsset] = {}

        for group in manifest.output.image_hash_groups:
            policy = self.known_asset_policies.get(
                group.content_sha256,
                AssetPolicy(
                    classification=AssetClassification.UNKNOWN,
                    role=AssetRole.UNKNOWN,
                    indexable=False,
                    reason_code="unclassified_extracted_asset",
                ),
            )
            references: list[AssetReference] = []

            for relative_path in group.physical_paths:
                record = image_by_relative_path[relative_path]
                basename = Path(relative_path).name.casefold()

                for image_index, image_node in structured_references.get(
                    relative_path,
                    [],
                ):
                    page_number = safe_int(
                        get_value(image_node, "page number", "page")
                    )
                    bbox, _ = self._bounding_box(image_node)
                    anchor = SourceAnchor(
                        anchor_id=stable_id(
                            "asset-anchor",
                            source_sha256,
                            relative_path,
                            image_index,
                        ),
                        source_sha256=source_sha256,
                        page_number=page_number,
                        block_index=image_index,
                        bbox=bbox,
                    )
                    references.append(
                        AssetReference(
                            reference_id=stable_id(
                                "asset-reference",
                                source_sha256,
                                relative_path,
                                image_index,
                            ),
                            source_path=relative_path,
                            source_anchor=anchor,
                            referenced_in_markdown=(
                                basename in markdown_image_names
                            ),
                            referenced_in_structure=True,
                        )
                    )

                if (
                    basename in markdown_image_names
                    and relative_path not in structured_references
                ):
                    references.append(
                        AssetReference(
                            reference_id=stable_id(
                                "asset-reference",
                                source_sha256,
                                relative_path,
                                "markdown-only",
                            ),
                            source_path=relative_path,
                            source_anchor=None,
                            referenced_in_markdown=True,
                            referenced_in_structure=False,
                        )
                    )

            first_record = image_by_relative_path[
                group.physical_paths[0]
            ]
            media_type = (
                mimetypes.guess_type(first_record.path.name)[0]
                or "application/octet-stream"
            )
            asset = ExtractedAsset(
                asset_id=stable_id(
                    "asset",
                    source_sha256,
                    group.content_sha256,
                ),
                source_sha256=source_sha256,
                content_sha256=group.content_sha256,
                media_type=media_type,
                size_bytes=first_record.size_bytes,
                classification=policy.classification,
                role=policy.role,
                physical_paths=sorted(group.physical_paths),
                references=references,
                retain_raw=True,
                indexable=policy.indexable,
                metadata={
                    "classification_reason": policy.reason_code,
                    "physical_file_count": len(group.physical_paths),
                    "reference_count": len(references),
                },
            )
            assets.append(asset)

            for relative_path in group.physical_paths:
                asset_by_relative_path[relative_path] = asset

            if policy.classification == AssetClassification.UNKNOWN:
                issues.append(
                    self._issue(
                        source_sha256=source_sha256,
                        reason_code="unclassified_visual_asset",
                        severity=QualitySeverity.WARNING,
                        failure_domain=FailureDomain.VISUAL,
                        message=(
                            "Extracted visual asset is not classified as "
                            "decorative or informative"
                        ),
                        asset_ids=[asset.asset_id],
                        triggers_fallback=True,
                        details={
                            "content_sha256": asset.content_sha256,
                            "physical_file_count": len(
                                asset.physical_paths
                            ),
                        },
                    )
                )

        return assets, asset_by_relative_path, issues

    @staticmethod
    def _match_image_record(
        *,
        source: object,
        structured_json_path: Path,
        image_records: list[WorkerArtifactRecord],
    ) -> WorkerArtifactRecord | None:
        if not isinstance(source, str) or not source.strip():
            return None

        source_path = Path(source)

        if not source_path.is_absolute():
            source_path = structured_json_path.parent / source_path

        resolved_source = source_path.resolve()
        exact_matches = [
            record
            for record in image_records
            if record.path.resolve() == resolved_source
        ]

        if len(exact_matches) == 1:
            return exact_matches[0]

        basename = source_basename(source)

        if basename is None:
            return None

        basename_matches = [
            record
            for record in image_records
            if record.path.name.casefold() == basename.casefold()
        ]

        if len(basename_matches) == 1:
            return basename_matches[0]

        return None

    @staticmethod
    def _bounding_box(
        node: Mapping[str, Any],
    ) -> tuple[BoundingBox | None, bool]:
        raw_bbox = get_value(node, "bounding box", "bbox")

        if raw_bbox is None:
            return None, True

        if (
            not isinstance(raw_bbox, list)
            or len(raw_bbox) != 4
            or not all(
                isinstance(value, (int, float))
                for value in raw_bbox
            )
        ):
            return None, False

        x0, y0, x1, y1 = [float(value) for value in raw_bbox]

        try:
            return (
                BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                True,
            )
        except ValueError:
            return None, False

    @staticmethod
    def _element_type(node: Mapping[str, Any]) -> ElementType:
        kind = node_type(node)

        if kind == "heading":
            heading_level = clean_text(
                get_value(node, "heading level", "level")
            ).casefold()

            if heading_level == "doctitle":
                return ElementType.TITLE

            return ElementType.HEADING

        mapping = {
            "paragraph": ElementType.PARAGRAPH,
            "text block": ElementType.PARAGRAPH,
            "list": ElementType.LIST,
            "table": ElementType.TABLE,
            "image": ElementType.IMAGE,
            "caption": ElementType.CAPTION,
        }
        return mapping.get(kind or "", ElementType.UNKNOWN)

    @staticmethod
    def _issue(
        *,
        source_sha256: str,
        reason_code: str,
        severity: QualitySeverity,
        failure_domain: FailureDomain,
        message: str,
        page_numbers: list[int] | None = None,
        element_ids: list[str] | None = None,
        asset_ids: list[str] | None = None,
        triggers_fallback: bool,
        details: dict[str, Any] | None = None,
    ) -> ExtractionQualityIssue:
        pages = sorted(set(page_numbers or []))
        elements = sorted(set(element_ids or []))
        assets = sorted(set(asset_ids or []))

        return ExtractionQualityIssue(
            issue_id=stable_id(
                "quality-issue",
                source_sha256,
                reason_code,
                ",".join(map(str, pages)),
                ",".join(elements),
                ",".join(assets),
            ),
            reason_code=reason_code,
            severity=severity,
            failure_domain=failure_domain,
            message=message,
            page_numbers=pages,
            element_ids=elements,
            asset_ids=assets,
            triggers_fallback=triggers_fallback,
            details=details or {},
        )
