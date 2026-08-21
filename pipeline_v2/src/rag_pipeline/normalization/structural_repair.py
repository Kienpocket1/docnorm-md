from __future__ import annotations

import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag_pipeline.contracts import (
    ElementType,
    ExtractedDocument,
    ExtractionQualityReport,
    QualityDisposition,
    QualitySeverity,
    ResumeFrom,
    SourceAnchor,
    SourceElement,
    StageResult,
    StageStatus,
)
from rag_pipeline.normalization.contract_native import (
    ContractNativeNormalizer,
    NormalizationCommitBundle,
    NormalizationCommitExistsError,
    NormalizerConfig,
    is_valid_gfm_table,
    load_normalization_commit,
    split_gfm_row,
)
from rag_pipeline.orchestration import (
    EXTRACTION_COMMIT_SCHEMA_VERSION,
    ExtractionCommitBundle,
    ExtractionCommitExistsError,
    ExtractionCommitManifest,
    IndexingSelection,
    load_extraction_commit,
)
from rag_pipeline.orchestration.extraction import (
    COMMIT_MANIFEST_FILENAME,
    DOCUMENT_FILENAME,
    QUALITY_REPORT_FILENAME,
    SELECTION_FILENAME,
    STAGE_RESULT_FILENAME,
    artifact_record,
    model_json_bytes,
    sha256_file,
    stable_identifier,
    validate_run_id,
    write_payload,
)
from rag_pipeline.validators.extraction import ExtractionPageValidator


STRUCTURAL_REPAIR_VERSION = "2026-08-21-7.1B.2"

FLOWCHART_TABLE_1 = """| Đơn vị xử lý | Nội dung | Biểu mẫu |
| --- | --- | --- |
| P.HH / P.QLKT / P.KT-TH | Tiếp nhận vật tư, hồ sơ |  |
| P.HH / P.QLKT / P.KT-TH | Kiểm tra, xử lý | BM.QT.TCKT.01 |
| — | Không tiếp nhận (nhánh Không đạt) | — |
| P.KT-TH | Lưu kho | BM.QT.TCKT.02 |
| P.KT-TH | Quản lý, theo dõi | BM.QT.TCKT.05 |
| P.KT-TH | Báo cáo | BM.QT.TCKT.05 |"""

FLOWCHART_TABLE_2 = """| Đơn vị xử lý | Nội dung | Biểu mẫu |
| --- | --- | --- |
| P.HH / P.QLKT | Thông tin đề xuất | BM.QT.TCKT.03 |
| P.KT-TH | Xử lý đề xuất |  |
| — | Không bàn giao (nhánh Không đạt) | — |
| P.KT-TH / P.QLKT / P.HH | Bàn giao | BM.QT.TCKT.04 |
| P.KT-TH | Quản lý, theo dõi | BM.QT.TCKT.05 |
| P.KT-TH | Báo cáo | BM.QT.TCKT.05 |"""

FLOWCHART_NOTE_1 = (
    "Luồng rẽ nhánh: Kiểm tra, xử lý — Đạt → Lưu kho; "
    "Không đạt → Không tiếp nhận."
)
FLOWCHART_NOTE_2 = (
    "Luồng rẽ nhánh: Xử lý đề xuất — Đạt → Bàn giao; "
    "Không đạt → Không bàn giao."
)


class StructuralRepairModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StructuralRepairReport(StructuralRepairModel):
    repair_version: str = Field(min_length=1)
    input_normalization_run_id: str = Field(min_length=1)
    input_source_element_count: int = Field(ge=1)
    repaired_source_element_count: int = Field(ge=1)
    content_rewrite_count: int = Field(ge=0)
    removed_noise_element_ids: list[str]
    merged_definition_element_ids: list[str]
    definition_fragment_count: Literal[3] = 3
    flowchart_source_element_ids: list[str]
    flowchart_replacement_element_ids: list[str]
    merged_table_groups: dict[str, list[str]]
    lineage_only_element_ids: list[str]
    resolved_issue_codes: list[str]
    provenance_coverage: Literal[1.0] = 1.0
    external_calls: Literal[0] = 0
    ocr_invocations: Literal[0] = 0
    chunking_invocations: Literal[0] = 0
    next_action: Literal["CHUNK"] = "CHUNK"
    passed: Literal[True] = True
    generated_at: datetime

    @model_validator(mode="after")
    def validate_required_repairs(self) -> Self:
        required = {
            "definition_reading_order",
            "split_table_continuity",
            "invented_list_labels",
            "noise_artifacts",
            "duplicate_section_numbering",
            "duplicate_bullets",
            "flowchart_semantics",
        }

        if set(self.resolved_issue_codes) != required:
            raise ValueError("Structural repair did not resolve every audit issue")

        if len(self.removed_noise_element_ids) < 2:
            raise ValueError("Both audited noise artifacts must be removed")

        if not self.merged_definition_element_ids:
            raise ValueError("Definition source lineage must be retained")

        if len(self.flowchart_replacement_element_ids) != 6:
            raise ValueError("Flowchart must be represented by six semantic blocks")

        if set(self.merged_table_groups) != {"nhap_vat_tu", "xuat_cap_vat_tu"}:
            raise ValueError("Both page-split interpretation tables must be merged")

        if len(self.lineage_only_element_ids) < 2:
            raise ValueError("Continuation table lineage must remain auditable")

        return self


@dataclass(frozen=True, slots=True)
class StructuralRepairBuild:
    document: ExtractedDocument
    report: StructuralRepairReport


@dataclass(frozen=True, slots=True)
class StructuralRepairConfig:
    handoff_output_root: Path
    normalization_output_root: Path


@dataclass(frozen=True, slots=True)
class StructuralRepairResult:
    input_normalization: NormalizationCommitBundle
    repaired_handoff: ExtractionCommitBundle
    normalization_commit: NormalizationCommitBundle
    repair_report: StructuralRepairReport


class StructuralRepairError(RuntimeError):
    """Base exception for deterministic normalization structural repair."""


class StructuralRepairInputError(StructuralRepairError):
    pass


def utc_repair_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"stage7_1b_{stamp}_{uuid.uuid4().hex[:8]}"


def _compact(value: str) -> str:
    return " ".join(value.split()).strip()


def _contains(value: str, fragment: str) -> bool:
    return fragment.casefold() in _compact(value).casefold()


def _rewrite_content(value: str) -> str:
    content = value.replace("\r\n", "\n")
    content = re.sub(
        r"(?mi)^(\s*)(?:i|ii|iii|iv|v|vi|vii)\.\s+"
        r"((?:I|II|III|IV|V|VI|VII)\.\s*)",
        r"\1\2",
        content,
    )
    content = re.sub(
        r"(?m)^(\s*)(\d+)\.\s+\2\.\s+",
        r"\1\2. ",
        content,
    )
    content = re.sub(r"(?m)^(\s*)-\s+-\s+", r"\1- ", content)
    content = re.sub(r"(?i)<br>\s*-\s+-\s+", "<br>- ", content)
    content = re.sub(
        r"(?i)<br>\s*[AB]\.\s*[−–-]\s*",
        "<br>− ",
        content,
    )
    return content.strip()


def _metadata_with_lineage(
    base: dict[str, Any],
    *,
    action: str,
    source_elements: list[SourceElement],
) -> dict[str, Any]:
    return {
        **base,
        "indexable": True,
        "structural_repair_version": STRUCTURAL_REPAIR_VERSION,
        "structural_repair_action": action,
        "merged_source_element_ids": [
            element.element_id for element in source_elements
        ],
        "merged_source_anchors": [
            element.source_anchor.model_dump(mode="json")
            for element in source_elements
        ],
    }


def _pick_unique_anchors(
    elements: list[SourceElement], count: int
) -> list[SourceAnchor]:
    unique: dict[tuple[int, int], SourceAnchor] = {}

    for element in elements:
        anchor = element.source_anchor
        page = anchor.page_number or 0
        unique.setdefault((page, anchor.block_index), anchor)

    anchors = [unique[key] for key in sorted(unique)]

    if len(anchors) < count:
        raise StructuralRepairInputError(
            "Flowchart does not expose enough unique source anchors"
        )

    if len(anchors) == count:
        return anchors

    selected_indexes = [
        round(index * (len(anchors) - 1) / (count - 1))
        for index in range(count)
    ]
    return [anchors[index] for index in selected_indexes]


def _new_flow_element(
    document: ExtractedDocument,
    *,
    key: str,
    content: str,
    element_type: ElementType,
    anchor: SourceAnchor,
    originals: list[SourceElement],
) -> SourceElement:
    element_id = stable_identifier(
        "source-element",
        document.source_sha256,
        document.version_id,
        STRUCTURAL_REPAIR_VERSION,
        "flowchart",
        key,
    )
    repaired_anchor = anchor.model_copy(
        update={
            "anchor_id": stable_identifier(
                "source-anchor",
                document.source_sha256,
                document.version_id,
                STRUCTURAL_REPAIR_VERSION,
                "flowchart",
                key,
                anchor.anchor_id,
            ),
            "section_path": ["V. LƯU ĐỒ"],
        }
    )
    metadata: dict[str, Any] = {}

    if element_type == ElementType.TABLE:
        metadata.update(
            {
                "declared_rows": 7,
                "actual_rows": 7,
                "row_count": 7,
                "declared_columns": 3,
                "column_count": 3,
                "invalid_positions": 0,
                "collisions": 0,
                "missing_slots": 0,
            }
        )

    return SourceElement(
        element_id=element_id,
        element_type=element_type,
        content=content,
        source_anchor=repaired_anchor,
        engine=originals[0].engine,
        confidence=None,
        metadata=_metadata_with_lineage(
            metadata,
            action="flowchart_reconstruction",
            source_elements=originals,
        ),
    )


def _flowchart_replacements(
    document: ExtractedDocument,
    originals: list[SourceElement],
) -> list[SourceElement]:
    anchors = _pick_unique_anchors(originals, 6)
    specifications = [
        ("nhap-title", "1. Nhập vật tư", ElementType.HEADING),
        ("nhap-table", FLOWCHART_TABLE_1, ElementType.TABLE),
        ("nhap-branch", FLOWCHART_NOTE_1, ElementType.PARAGRAPH),
        ("xuat-title", "2. Xuất cấp vật tư", ElementType.HEADING),
        ("xuat-table", FLOWCHART_TABLE_2, ElementType.TABLE),
        ("xuat-branch", FLOWCHART_NOTE_2, ElementType.PARAGRAPH),
    ]
    return [
        _new_flow_element(
            document,
            key=key,
            content=content,
            element_type=element_type,
            anchor=anchor,
            originals=originals,
        )
        for (key, content, element_type), anchor in zip(
            specifications, anchors, strict=True
        )
    ]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(
        re.fullmatch(r":?-{3,}:?", cell) is not None for cell in cells
    )


def _table_rows(content: str) -> list[list[str]]:
    rows: list[list[str]] = []

    for line in content.splitlines():
        parsed = split_gfm_row(line)

        if parsed is not None:
            rows.append(parsed)

    return rows


def _is_interpretation_table(element: SourceElement) -> bool:
    if element.element_type != ElementType.TABLE:
        return False

    rows = _table_rows(element.content)

    if not rows or any(len(row) != 4 for row in rows):
        return False

    first = [cell.casefold() for cell in rows[0]]
    return first == ["tt", "nội dung", "diễn giải", "hồ sơ"] or (
        bool(first[0]) and first[0].isdigit()
    )


def _merge_interpretation_tables(
    elements: list[SourceElement],
) -> tuple[list[SourceElement], dict[str, list[str]]]:
    current: str | None = None
    groups: dict[str, list[int]] = {
        "nhap_vat_tu": [],
        "xuat_cap_vat_tu": [],
    }

    for index, element in enumerate(elements):
        compact = _compact(element.content).casefold()

        if "vi. diễn giải lưu đồ" in compact:
            current = "awaiting"
        elif current is not None and re.search(
            r"(?:^|\s)1\.\s+nhập vật tư(?:$|\s)", compact
        ):
            current = "nhap_vat_tu"
        elif current is not None and re.search(
            r"(?:^|\s)2\.\s+xuất cấp vật tư(?:$|\s)", compact
        ):
            current = "xuat_cap_vat_tu"
        elif current is not None and re.search(
            r"(?:^|\s)3\.\s+luân chuyển", compact
        ):
            current = None

        if current in groups and _is_interpretation_table(element):
            groups[current].append(index)

    output = [*elements]
    lineage: dict[str, list[str]] = {}

    for group_name, indexes in groups.items():
        if len(indexes) < 2:
            raise StructuralRepairInputError(
                f"Expected page-split table fragments for {group_name}"
            )

        fragments = [elements[index] for index in indexes]
        rows_by_number: dict[int, list[str]] = {}

        for fragment in fragments:
            for row in _table_rows(fragment.content):
                if _is_separator_row(row):
                    continue

                if [cell.casefold() for cell in row] == [
                    "tt",
                    "nội dung",
                    "diễn giải",
                    "hồ sơ",
                ]:
                    continue

                if not row[0].isdigit():
                    raise StructuralRepairInputError(
                        f"Unexpected interpretation table row in {group_name}"
                    )

                row_number = int(row[0])

                if row_number in rows_by_number:
                    raise StructuralRepairInputError(
                        f"Duplicate row {row_number} in {group_name}"
                    )

                rows_by_number[row_number] = row

        if sorted(rows_by_number) != [1, 2, 3, 4, 5]:
            raise StructuralRepairInputError(
                f"{group_name} must contain rows 1 through 5"
            )

        lines = [
            "| TT | Nội dung | Diễn giải | Hồ sơ |",
            "| --- | --- | --- | --- |",
            *[
                "| " + " | ".join(rows_by_number[number]) + " |"
                for number in range(1, 6)
            ],
        ]
        merged_content = "\n".join(lines)

        if not is_valid_gfm_table(merged_content):
            raise StructuralRepairInputError(
                f"Merged table is not valid GFM: {group_name}"
            )

        first = fragments[0]
        merged = first.model_copy(
            update={
                "content": merged_content,
                "metadata": _metadata_with_lineage(
                    {
                        **first.metadata,
                        "declared_rows": 6,
                        "actual_rows": 6,
                        "row_count": 6,
                        "declared_columns": 4,
                        "column_count": 4,
                        "invalid_positions": 0,
                        "collisions": 0,
                        "missing_slots": 0,
                    },
                    action="page_split_table_merge",
                    source_elements=fragments,
                ),
            }
        )
        output[indexes[0]] = merged

        for index, fragment in zip(indexes[1:], fragments[1:], strict=True):
            output[index] = fragment.model_copy(
                update={
                    "metadata": {
                        **_metadata_with_lineage(
                            fragment.metadata,
                            action="page_split_table_lineage_only",
                            source_elements=[fragment],
                        ),
                        "indexable": False,
                        "merged_into_element_id": first.element_id,
                    }
                }
            )

        lineage[group_name] = [fragment.element_id for fragment in fragments]

    return output, lineage


def _validate_repaired_markdown(markdown: str) -> None:
    compact = _compact(markdown)
    forbidden_patterns = {
        "noise_a_x": r"(?mi)^\s*a\.\s*x\s*$",
        "floating_logo": r"(?mi)^\s*PETROLIMEX AVIATION\s*$",
        "double_bullet": r"(?m)^\s*-\s+-\s+",
        "invented_a_b": r"(?i)<br>\s*[AB]\.\s*[−–-]",
        "double_roman": r"(?mi)^\s*(?:i|ii|iii|iv)\.\s+(?:I|II|III|IV)\.",
        "double_numeric": r"(?m)^\s*(\d+)\.\s+\1\.\s+",
    }

    for code, pattern in forbidden_patterns.items():
        if re.search(pattern, markdown):
            raise StructuralRepairInputError(
                f"Repaired Markdown still contains {code}"
            )

    material = compact.find("Vật tư:")
    material_end = compact.find("phục vụ cho sản xuất kinh doanh.", material)
    warehouse = compact.find("Kho vật tư:", material)

    if not (material >= 0 and material < material_end < warehouse):
        raise StructuralRepairInputError(
            "Definition reading order was not repaired"
        )

    if markdown.count("| TT | Nội dung | Diễn giải | Hồ sơ |") != 2:
        raise StructuralRepairInputError(
            "Interpretation tables were not consolidated"
        )

    if markdown.count("| Đơn vị xử lý | Nội dung | Biểu mẫu |") != 2:
        raise StructuralRepairInputError(
            "Flowchart semantic tables are missing"
        )

    for required in (FLOWCHART_NOTE_1, FLOWCHART_NOTE_2):
        if required not in markdown:
            raise StructuralRepairInputError(
                "Flowchart branch semantics are missing"
            )


def _repair_definition_fragments(
    elements: list[SourceElement],
) -> tuple[list[SourceElement], list[str]]:
    suffix_pattern = re.compile(
        r"^\s*(?:[-+*]\s+)?vụ cho sản xuất kinh doanh\.\s*$",
        flags=re.IGNORECASE,
    )
    warehouse_pattern = re.compile(
        r"^\s*(?:[-+*]\s+)?Kho vật tư:\s*.+$",
        flags=re.IGNORECASE,
    )
    material_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<marker>[-+*]\s+)?"
        r"(?P<body>Vật tư:.*phụ tùng thay thế.*\bphục)\s*$",
        flags=re.IGNORECASE,
    )
    suffix_matches: list[tuple[int, int, str]] = []
    warehouse_matches: list[tuple[int, int, str]] = []
    material_matches: list[tuple[int, int, str, re.Match[str]]] = []

    for element_index, element in enumerate(elements):
        for line_index, line in enumerate(element.content.splitlines()):
            if suffix_pattern.fullmatch(line):
                suffix_matches.append((element_index, line_index, line))

            if warehouse_pattern.fullmatch(line):
                warehouse_matches.append((element_index, line_index, line))

            material_match = material_pattern.fullmatch(line)

            if material_match is not None:
                material_matches.append(
                    (element_index, line_index, line, material_match)
                )

    if not (
        len(suffix_matches)
        == len(warehouse_matches)
        == len(material_matches)
        == 1
    ):
        raise StructuralRepairInputError(
            "Definition fragments could not be identified unambiguously: "
            f"suffix={len(suffix_matches)},"
            f"warehouse={len(warehouse_matches)},"
            f"material={len(material_matches)}"
        )

    suffix_element_index, suffix_line_index, _ = suffix_matches[0]
    warehouse_element_index, warehouse_line_index, warehouse_line = (
        warehouse_matches[0]
    )
    material_element_index, material_line_index, _, material_match = (
        material_matches[0]
    )
    lineage_indexes = list(
        dict.fromkeys(
            [
                material_element_index,
                suffix_element_index,
                warehouse_element_index,
            ]
        )
    )
    lineage_elements = [elements[index] for index in lineage_indexes]
    material_indent = material_match.group("indent")
    material_marker = material_match.group("marker") or ""
    material_body = material_match.group("body").rstrip()
    warehouse_body = re.sub(
        r"^\s*[-+*]\s+", "", warehouse_line
    ).strip()
    warehouse_marker = material_marker or "- "
    repaired_material_line = (
        f"{material_indent}{material_marker}{material_body} "
        "vụ cho sản xuất kinh doanh."
    )
    moved_warehouse_line = (
        f"{material_indent}{warehouse_marker}{warehouse_body}"
    )
    suffix_position = (suffix_element_index, suffix_line_index)
    warehouse_position = (warehouse_element_index, warehouse_line_index)
    material_position = (material_element_index, material_line_index)
    repaired: list[SourceElement] = []

    for element_index, element in enumerate(elements):
        output_lines: list[str] = []

        for line_index, line in enumerate(element.content.splitlines()):
            position = (element_index, line_index)

            if position in {suffix_position, warehouse_position}:
                continue

            if position == material_position:
                output_lines.extend(
                    [repaired_material_line, moved_warehouse_line]
                )
            else:
                output_lines.append(line)

        content = "\n".join(output_lines).strip()

        if not content:
            continue

        update: dict[str, Any] = {"content": content}

        if element_index == material_element_index:
            update["metadata"] = _metadata_with_lineage(
                element.metadata,
                action="definition_fragment_merge",
                source_elements=lineage_elements,
            )

        repaired.append(element.model_copy(update=update))

    return repaired, [element.element_id for element in lineage_elements]


def repair_document(
    document: ExtractedDocument,
    *,
    input_normalization_run_id: str,
    generated_at: datetime,
) -> StructuralRepairBuild:
    rewritten: list[SourceElement] = []
    removed_noise: list[str] = []
    content_rewrite_count = 0

    for element in document.elements:
        compact = _compact(element.content)

        if re.fullmatch(r"a\.\s*x", compact, flags=re.IGNORECASE) or (
            compact.casefold() == "petrolimex aviation".casefold()
        ):
            removed_noise.append(element.element_id)
            continue

        content = _rewrite_content(element.content)

        if content != element.content:
            content_rewrite_count += 1
            element = element.model_copy(update={"content": content})

        rewritten.append(element)

    if len(removed_noise) != 2:
        raise StructuralRepairInputError(
            "Expected the audited 'a. x' and floating logo elements"
        )

    rewritten, merged_definition_element_ids = (
        _repair_definition_fragments(rewritten)
    )
    content_rewrite_count += 1

    flow_start = next(
        (
            index
            for index, element in enumerate(rewritten)
            if re.search(r"(?i)(?:^|\s)V\.\s+LƯU ĐỒ(?:$|\s)", _compact(element.content))
        ),
        None,
    )
    flow_end = next(
        (
            index
            for index, element in enumerate(rewritten)
            if flow_start is not None
            and index > flow_start
            and _contains(element.content, "VI. DIỄN GIẢI LƯU ĐỒ")
        ),
        None,
    )

    if flow_start is None or flow_end is None or flow_end - flow_start < 8:
        raise StructuralRepairInputError(
            "The flattened flowchart range could not be identified"
        )

    flow_originals = rewritten[flow_start + 1 : flow_end]
    flow_replacements = _flowchart_replacements(document, flow_originals)
    rewritten = (
        rewritten[: flow_start + 1]
        + flow_replacements
        + rewritten[flow_end:]
    )
    rewritten, merged_groups = _merge_interpretation_tables(rewritten)
    repaired_document = document.model_copy(update={"elements": rewritten})
    report = StructuralRepairReport(
        repair_version=STRUCTURAL_REPAIR_VERSION,
        input_normalization_run_id=input_normalization_run_id,
        input_source_element_count=len(document.elements),
        repaired_source_element_count=len(repaired_document.elements),
        content_rewrite_count=content_rewrite_count,
        removed_noise_element_ids=removed_noise,
        merged_definition_element_ids=merged_definition_element_ids,
        definition_fragment_count=3,
        flowchart_source_element_ids=[
            element.element_id for element in flow_originals
        ],
        flowchart_replacement_element_ids=[
            element.element_id for element in flow_replacements
        ],
        merged_table_groups=merged_groups,
        lineage_only_element_ids=[
            element_id
            for element_ids in merged_groups.values()
            for element_id in element_ids[1:]
        ],
        resolved_issue_codes=sorted(
            {
                "definition_reading_order",
                "split_table_continuity",
                "invented_list_labels",
                "noise_artifacts",
                "duplicate_section_numbering",
                "duplicate_bullets",
                "flowchart_semantics",
            }
        ),
        generated_at=generated_at,
    )
    return StructuralRepairBuild(document=repaired_document, report=report)


class StructuralRepairOrchestrator:
    def __init__(
        self,
        config: StructuralRepairConfig,
        *,
        validator: ExtractionPageValidator | None = None,
    ) -> None:
        self.config = config
        self.validator = validator or ExtractionPageValidator()

    def repair_normalization_commit(
        self,
        normalization_commit_directory: Path | str,
        *,
        run_id: str | None = None,
        handoff_run_id: str | None = None,
    ) -> StructuralRepairResult:
        input_normalization = load_normalization_commit(
            normalization_commit_directory
        )
        self._validate_input(input_normalization)
        extraction = load_extraction_commit(
            input_normalization.manifest.input_extraction_commit_directory
        )
        active_run_id = run_id or utc_repair_run_id()
        active_handoff_run_id = handoff_run_id or f"{active_run_id}-handoff"
        validate_run_id(active_run_id)
        validate_run_id(active_handoff_run_id)

        normalization_target = (
            self.config.normalization_output_root.expanduser().resolve()
            / active_run_id
        )

        if normalization_target.exists():
            raise NormalizationCommitExistsError(
                f"Normalization commit already exists: {active_run_id}"
            )

        repair = repair_document(
            extraction.document,
            input_normalization_run_id=input_normalization.manifest.run_id,
            generated_at=input_normalization.manifest.committed_at,
        )
        repaired_handoff = self._write_repaired_handoff(
            extraction,
            input_normalization,
            repair,
            run_id=active_handoff_run_id,
        )
        normalizer = ContractNativeNormalizer(
            NormalizerConfig(output_root=self.config.normalization_output_root)
        )
        normalization = normalizer.normalize_commit(
            repaired_handoff.commit_directory,
            run_id=active_run_id,
        )
        _validate_repaired_markdown(normalization.markdown)
        return StructuralRepairResult(
            input_normalization=input_normalization,
            repaired_handoff=repaired_handoff,
            normalization_commit=normalization,
            repair_report=repair.report,
        )

    @staticmethod
    def _validate_input(input_normalization: NormalizationCommitBundle) -> None:
        if input_normalization.manifest.next_action != "CHUNK":
            raise StructuralRepairInputError(
                "Input normalization is not at the pre-chunk boundary"
            )

        if input_normalization.stage_result.status != StageStatus.PASS:
            raise StructuralRepairInputError(
                "Input normalization stage result must be PASS"
            )

        if input_normalization.report.provenance_coverage != 1.0:
            raise StructuralRepairInputError(
                "Input normalization provenance must be complete"
            )

    def _write_repaired_handoff(
        self,
        extraction: ExtractionCommitBundle,
        input_normalization: NormalizationCommitBundle,
        repair: StructuralRepairBuild,
        *,
        run_id: str,
    ) -> ExtractionCommitBundle:
        output_root = self.config.handoff_output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        final_directory = output_root / run_id

        if final_directory.exists():
            raise ExtractionCommitExistsError(
                f"Repair handoff commit already exists: {run_id}"
            )

        document = repair.document
        assessment = self.validator.evaluate(document)
        failing_issues = [
            issue
            for issue in assessment.issues
            if issue.triggers_fallback or issue.severity == QualitySeverity.ERROR
        ]

        if failing_issues or any(not item.passed for item in assessment.page_results):
            reasons = sorted({issue.reason_code for issue in failing_issues})
            raise StructuralRepairInputError(
                "Repaired extraction failed validation: " + ",".join(reasons)
            )

        if assessment.provenance.coverage != 1.0:
            raise StructuralRepairInputError(
                "Repaired extraction provenance is incomplete"
            )

        if assessment.pending_visual_element_ids:
            raise StructuralRepairInputError(
                "Repaired extraction contains unresolved informative visuals"
            )

        element_ids = [element.element_id for element in document.elements]
        selected_ids = [*assessment.indexable_element_ids]
        filtered_ids = [*assessment.filtered_element_ids]

        if set(element_ids) != set(selected_ids) | set(filtered_ids):
            raise StructuralRepairInputError(
                "Repair selection does not cover every source element"
            )

        selected_set = set(selected_ids)

        if selected_ids != [
            element_id for element_id in element_ids if element_id in selected_set
        ]:
            raise StructuralRepairInputError(
                "Repair selection changed document order"
            )

        source_manifest_path = (
            input_normalization.commit_directory / COMMIT_MANIFEST_FILENAME
        )
        source_manifest_sha256 = sha256_file(source_manifest_path)
        warnings = [
            issue
            for issue in assessment.issues
            if issue.severity == QualitySeverity.WARNING
        ]
        disposition = (
            QualityDisposition.PASS_WITH_WARNINGS
            if warnings
            else QualityDisposition.PASS
        )
        report = ExtractionQualityReport(
            report_id=stable_identifier(
                "extraction-quality-report",
                document.source_sha256,
                document.version_id,
                input_normalization.manifest.run_id,
                STRUCTURAL_REPAIR_VERSION,
            ),
            catalog_id=document.catalog_id,
            family_id=document.family_id,
            version_id=document.version_id,
            source_sha256=document.source_sha256,
            engine=document.engine_chain[-1],
            disposition=disposition,
            issues=assessment.issues,
            metrics={
                **assessment.metrics,
                "structural_repair": repair.report.model_dump(mode="json"),
                "input_normalization_run_id": input_normalization.manifest.run_id,
                "input_normalization_manifest_sha256": source_manifest_sha256,
                "next_action": "NORMALIZE",
            },
            evaluated_at=input_normalization.manifest.committed_at,
        )
        selection = IndexingSelection(
            catalog_id=document.catalog_id,
            family_id=document.family_id,
            version_id=document.version_id,
            source_sha256=document.source_sha256,
            indexable_element_ids=selected_ids,
            filtered_element_ids=filtered_ids,
        )
        stage_result = StageResult(
            stage=ResumeFrom.EXTRACTION_QUALITY,
            status=StageStatus.PASS,
            started_at=input_normalization.manifest.committed_at,
            finished_at=input_normalization.manifest.committed_at,
            duration_seconds=0.0,
            artifact_paths=[
                DOCUMENT_FILENAME,
                QUALITY_REPORT_FILENAME,
                SELECTION_FILENAME,
            ],
            metrics={
                "quality_disposition": disposition.value,
                "quality_issue_count": len(report.issues),
                "next_action": "NORMALIZE",
                "structural_repair_version": STRUCTURAL_REPAIR_VERSION,
                "resolved_issue_count": len(repair.report.resolved_issue_codes),
                "chunking_invocations": 0,
            },
            message="Accepted structurally repaired extraction handoff",
        )
        staging_directory = output_root / f".{run_id}.staging-{uuid.uuid4().hex}"
        staging_directory.mkdir(parents=False, exist_ok=False)

        try:
            models = {
                DOCUMENT_FILENAME: document,
                QUALITY_REPORT_FILENAME: report,
                SELECTION_FILENAME: selection,
                STAGE_RESULT_FILENAME: stage_result,
            }

            for filename, model in models.items():
                write_payload(staging_directory / filename, model_json_bytes(model))

            artifacts = [
                artifact_record(staging_directory / filename, staging_directory)
                for filename in sorted(models)
            ]
            manifest = ExtractionCommitManifest(
                schema_version=EXTRACTION_COMMIT_SCHEMA_VERSION,
                orchestrator_version=STRUCTURAL_REPAIR_VERSION,
                run_id=run_id,
                catalog_id=document.catalog_id,
                family_id=document.family_id,
                version_id=document.version_id,
                source_sha256=document.source_sha256,
                local_run_id=(
                    f"{input_normalization.manifest.run_id}-"
                    f"{source_manifest_sha256[:12]}"
                ),
                local_run_directory=str(extraction.commit_directory),
                local_worker_manifest_path=str(source_manifest_path),
                quality_disposition=disposition,
                local_result_accepted=True,
                fallback_required=False,
                next_action="NORMALIZE",
                local_attempt_count=1,
                automatic_retry_count=0,
                quarantine_id=None,
                artifact_count=len(artifacts),
                artifacts=artifacts,
                committed_at=input_normalization.manifest.committed_at,
            )
            write_payload(
                staging_directory / COMMIT_MANIFEST_FILENAME,
                model_json_bytes(manifest),
            )
            load_extraction_commit(staging_directory)

            if final_directory.exists():
                raise ExtractionCommitExistsError(
                    f"Repair handoff commit already exists: {run_id}"
                )

            staging_directory.rename(final_directory)
        except Exception:
            if staging_directory.exists():
                shutil.rmtree(staging_directory)
            raise

        return load_extraction_commit(final_directory)


__all__ = [
    "FLOWCHART_NOTE_1",
    "FLOWCHART_NOTE_2",
    "FLOWCHART_TABLE_1",
    "FLOWCHART_TABLE_2",
    "STRUCTURAL_REPAIR_VERSION",
    "StructuralRepairBuild",
    "StructuralRepairConfig",
    "StructuralRepairError",
    "StructuralRepairInputError",
    "StructuralRepairOrchestrator",
    "StructuralRepairReport",
    "StructuralRepairResult",
    "repair_document",
]
