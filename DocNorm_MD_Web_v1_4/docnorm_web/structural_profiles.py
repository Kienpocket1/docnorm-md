from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .math_exam import MATH_REPAIR_VERSION, MATH_RESOLVED_ISSUE_CODES
from .models import RepairProfile
from .normalizer import ConversionError, clean_markdown

PETROLIMEX_MATERIALS_REPAIR_VERSION = "2026-08-21-1.1.0"

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
    "Luồng rẽ nhánh: Kiểm tra, xử lý — Đạt → Lưu kho; Không đạt → Không tiếp nhận."
)
FLOWCHART_NOTE_2 = (
    "Luồng rẽ nhánh: Xử lý đề xuất — Đạt → Bàn giao; Không đạt → Không bàn giao."
)

RESOLVED_ISSUE_CODES = (
    "definition_reading_order",
    "duplicate_bullets",
    "duplicate_section_numbering",
    "flowchart_semantics",
    "invented_list_labels",
    "noise_artifacts",
    "repeated_page_headers",
    "signature_footer_semantics",
    "split_table_continuity",
)

_HARD_FINGERPRINTS = (
    re.compile(r"(?mi)^\s*#{0,6}\s*QUY TRÌNH QUẢN LÝ VẬT TƯ\s*$"),
    re.compile(r"\(QT\.TCKT\.01\)", re.IGNORECASE),
    re.compile(r"BM\.QT\.TCKT\.01", re.IGNORECASE),
    re.compile(r"BM\.QT\.TCKT\.05", re.IGNORECASE),
    re.compile(r"(?i)\bV\.\s*LƯU ĐỒ\b"),
    re.compile(r"(?i)\bVI\.\s*DIỄN GIẢI LƯU ĐỒ\b"),
)
_SOFT_FINGERPRINTS = (
    re.compile(r"CÔNG TY CỔ PHẦN NHIÊN LIỆU BAY PETROLIMEX", re.IGNORECASE),
    re.compile(r"(?mi)^\s*(?:[-+*]\s+)?Vật tư:", re.IGNORECASE),
    re.compile(r"(?mi)^\s*(?:[-+*]\s+)?Kho vật tư:", re.IGNORECASE),
    re.compile(r"BM\.QT\.TCKT\.03", re.IGNORECASE),
    re.compile(r"BM\.QT\.TCKT\.06", re.IGNORECASE),
    re.compile(r"P\.KT-TH", re.IGNORECASE),
)


class StructuralProfileError(ConversionError):
    code = "STRUCTURAL_PROFILE_FAILED"


@dataclass(frozen=True, slots=True)
class ProfileFingerprint:
    matched: bool
    score: float
    hard_matches: int
    soft_matches: int


@dataclass(frozen=True, slots=True)
class ProfileRepairResult:
    markdown: str
    requested: RepairProfile
    applied: RepairProfile
    repair_version: str | None
    fingerprint_score: float
    resolved_issue_codes: tuple[str, ...]


def fingerprint_petrolimex_materials(markdown: str) -> ProfileFingerprint:
    text = unicodedata.normalize("NFC", markdown)
    hard_matches = sum(
        pattern.search(text) is not None for pattern in _HARD_FINGERPRINTS
    )
    soft_matches = sum(
        pattern.search(text) is not None for pattern in _SOFT_FINGERPRINTS
    )
    total = len(_HARD_FINGERPRINTS) + len(_SOFT_FINGERPRINTS)
    score = (hard_matches + soft_matches) / total
    matched = hard_matches == len(_HARD_FINGERPRINTS) and soft_matches >= 5
    return ProfileFingerprint(
        matched=matched,
        score=round(score, 4),
        hard_matches=hard_matches,
        soft_matches=soft_matches,
    )


def apply_structural_profile(
    markdown: str,
    requested: RepairProfile,
) -> ProfileRepairResult:
    fingerprint = fingerprint_petrolimex_materials(markdown)
    if requested == RepairProfile.MATH_EXAM:
        return ProfileRepairResult(
            markdown=markdown,
            requested=requested,
            applied=RepairProfile.MATH_EXAM,
            repair_version=MATH_REPAIR_VERSION,
            fingerprint_score=1.0,
            resolved_issue_codes=MATH_RESOLVED_ISSUE_CODES,
        )
    if requested == RepairProfile.GENERIC:
        return ProfileRepairResult(
            markdown=markdown,
            requested=requested,
            applied=RepairProfile.GENERIC,
            repair_version=None,
            fingerprint_score=fingerprint.score,
            resolved_issue_codes=(),
        )
    if requested == RepairProfile.AUTO and not fingerprint.matched:
        return ProfileRepairResult(
            markdown=markdown,
            requested=requested,
            applied=RepairProfile.GENERIC,
            repair_version=None,
            fingerprint_score=fingerprint.score,
            resolved_issue_codes=(),
        )
    if not fingerprint.matched:
        raise StructuralProfileError(
            "Tài liệu không khớp fingerprint Quy trình quản lý vật tư; "
            "profile chuyên biệt đã dừng để tránh sửa nhầm nội dung."
        )

    repaired = _rewrite_markers(markdown)
    repaired = _repair_definition_fragments(repaired)
    repaired = _merge_interpretation_table(
        repaired,
        section_title="1. Nhập vật tư",
        next_section_title="2. Xuất cấp vật tư",
    )
    repaired = _merge_interpretation_table(
        repaired,
        section_title="2. Xuất cấp vật tư",
        next_section_title="3. Luân chuyển",
    )
    repaired = _repair_flowcharts(repaired)
    repaired = _remove_noise_and_reclassify_headers(repaired)
    repaired = _remove_invented_form_numbering(repaired)
    repaired = clean_markdown(repaired)
    validate_petrolimex_materials_markdown(repaired)
    return ProfileRepairResult(
        markdown=repaired,
        requested=requested,
        applied=RepairProfile.PETROLIMEX_MATERIALS,
        repair_version=PETROLIMEX_MATERIALS_REPAIR_VERSION,
        fingerprint_score=fingerprint.score,
        resolved_issue_codes=RESOLVED_ISSUE_CODES,
    )


def _rewrite_markers(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(
        r"(?mi)^(\s*)(?:i|ii|iii|iv|v|vi|vii)\.\s+"
        r"((?:I|II|III|IV|V|VI|VII)\.\s*)",
        r"\1\2",
        text,
    )
    text = re.sub(r"(?m)^(\s*)(\d+)\.\s+\2\.\s+", r"\1\2. ", text)
    text = re.sub(r"(?m)^(\s*)[-+*]\s+[-+*]\s+", r"\1- ", text)
    text = re.sub(r"(?i)<br>\s*[-+*]\s+[-+*]\s+", "<br>- ", text)
    text = re.sub(r"(?i)<br>\s*[AB]\.\s*[−–-]\s*", "<br>− ", text)
    return text


def _repair_definition_fragments(markdown: str) -> str:
    lines = markdown.splitlines()
    suffix_pattern = re.compile(
        r"^\s*(?:[-+*]\s+)?vụ cho sản xuất kinh doanh\.\s*$",
        re.IGNORECASE,
    )
    warehouse_pattern = re.compile(
        r"^\s*(?:[-+*]\s+)?Kho vật tư:\s*.+$",
        re.IGNORECASE,
    )
    material_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<marker>[-+*]\s+)?"
        r"(?P<body>Vật tư:.*phụ tùng thay thế.*\bphục)\s*$",
        re.IGNORECASE,
    )
    suffixes = [
        index for index, line in enumerate(lines) if suffix_pattern.fullmatch(line)
    ]
    warehouses = [
        index for index, line in enumerate(lines) if warehouse_pattern.fullmatch(line)
    ]
    materials = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := material_pattern.fullmatch(line)) is not None
    ]

    if not suffixes and not materials:
        compact = " ".join(markdown.split())
        material = compact.casefold().find("vật tư:")
        material_end = compact.casefold().find(
            "phục vụ cho sản xuất kinh doanh.", material
        )
        warehouse = compact.casefold().find("kho vật tư:", material)
        if material >= 0 and material < material_end < warehouse:
            return markdown
    if not (len(suffixes) == len(warehouses) == len(materials) == 1):
        raise StructuralProfileError(
            "Không nhận diện duy nhất ba mảnh định nghĩa Vật tư/Kho vật tư."
        )

    suffix_index = suffixes[0]
    warehouse_index = warehouses[0]
    material_index, match = materials[0]
    marker = match.group("marker") or "- "
    indent = match.group("indent")
    material_line = (
        f"{indent}{marker}{match.group('body').rstrip()} vụ cho sản xuất kinh doanh."
    )
    warehouse_body = re.sub(r"^\s*[-+*]\s+", "", lines[warehouse_index]).strip()
    warehouse_line = f"{indent}{marker}{warehouse_body}"
    output: list[str] = []
    for index, line in enumerate(lines):
        if index in {suffix_index, warehouse_index}:
            continue
        if index == material_index:
            output.extend([material_line, warehouse_line])
        else:
            output.append(line)
    return "\n".join(output)


def _plain_line(line: str) -> str:
    return re.sub(r"^\s*#{1,6}\s*", "", line).strip()


def _find_section(lines: list[str], title: str, start: int) -> int | None:
    expected = title.casefold()
    for index in range(start, len(lines)):
        plain = _plain_line(lines[index]).casefold()
        if plain == expected or plain.startswith(expected):
            return index
    return None


def _split_gfm_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped[1:-1])]


def _separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _merge_interpretation_table(
    markdown: str,
    *,
    section_title: str,
    next_section_title: str,
) -> str:
    lines = markdown.splitlines()
    explanation = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(
                r"VI\.\s*DIỄN GIẢI LƯU ĐỒ:?",
                _plain_line(line),
                flags=re.IGNORECASE,
            )
        ),
        None,
    )
    if explanation is None:
        raise StructuralProfileError("Không tìm thấy mục VI. DIỄN GIẢI LƯU ĐỒ.")
    start = _find_section(lines, section_title, explanation + 1)
    if start is None:
        raise StructuralProfileError(f"Không tìm thấy mục {section_title}.")
    end = _find_section(lines, next_section_title, start + 1)
    if end is None:
        raise StructuralProfileError(f"Không tìm thấy mốc sau {section_title}.")

    table_indexes = [
        index
        for index in range(start + 1, end)
        if _split_gfm_row(lines[index]) is not None
    ]
    if not table_indexes:
        raise StructuralProfileError(f"Không tìm thấy bảng tại {section_title}.")
    first, last = table_indexes[0], table_indexes[-1]
    if any(
        lines[index].strip()
        for index in range(first, last + 1)
        if index not in table_indexes
    ):
        raise StructuralProfileError(
            f"Bảng {section_title} chứa nội dung xen kẽ mơ hồ."
        )

    rows_by_number: dict[int, list[str]] = {}
    for index in table_indexes:
        cells = _split_gfm_row(lines[index])
        if cells is None or len(cells) != 4 or _separator(cells):
            continue
        folded = [cell.casefold() for cell in cells]
        if folded == ["tt", "nội dung", "diễn giải", "hồ sơ"]:
            continue
        if not cells[0].isdigit():
            raise StructuralProfileError(f"Dòng bảng không hợp lệ tại {section_title}.")
        number = int(cells[0])
        if number in rows_by_number:
            raise StructuralProfileError(f"Trùng dòng {number} tại {section_title}.")
        rows_by_number[number] = cells
    if sorted(rows_by_number) != [1, 2, 3, 4, 5]:
        raise StructuralProfileError(f"Bảng {section_title} không đủ dòng 1–5.")

    merged = [
        "| TT | Nội dung | Diễn giải | Hồ sơ |",
        "| --- | --- | --- | --- |",
        *["| " + " | ".join(rows_by_number[number]) + " |" for number in range(1, 6)],
    ]
    return "\n".join([*lines[:first], *merged, *lines[last + 1 :]])


def _repair_flowcharts(markdown: str) -> str:
    lines = markdown.splitlines()
    start = next(
        (
            index
            for index, line in enumerate(lines)
            if re.fullmatch(r"V\.\s*LƯU ĐỒ", _plain_line(line), re.IGNORECASE)
        ),
        None,
    )
    end = next(
        (
            index
            for index, line in enumerate(lines)
            if start is not None
            and index > start
            and re.fullmatch(
                r"VI\.\s*DIỄN GIẢI LƯU ĐỒ:?",
                _plain_line(line),
                re.IGNORECASE,
            )
        ),
        None,
    )
    if start is None or end is None or end - start < 4:
        raise StructuralProfileError("Không khoanh vùng được hai lưu đồ gốc.")
    replacement = [
        "## V. LƯU ĐỒ",
        "",
        "### 1. Nhập vật tư",
        "",
        *FLOWCHART_TABLE_1.splitlines(),
        "",
        FLOWCHART_NOTE_1,
        "",
        "### 2. Xuất cấp vật tư",
        "",
        *FLOWCHART_TABLE_2.splitlines(),
        "",
        FLOWCHART_NOTE_2,
        "",
    ]
    return "\n".join([*lines[:start], *replacement, *lines[end:]])


def _remove_noise_and_reclassify_headers(markdown: str) -> str:
    title = "QUY TRÌNH QUẢN LÝ VẬT TƯ"
    title_seen = False
    boilerplate_seen: set[str] = set()
    output: list[str] = []
    for line in markdown.splitlines():
        plain = _plain_line(line)
        folded = plain.casefold()
        if re.fullmatch(r"a\.\s*x", plain, re.IGNORECASE):
            continue
        if folded == "petrolimex aviation":
            continue
        if folded == title.casefold():
            if title_seen:
                continue
            title_seen = True
            output.append(f"# {title}")
            continue
        boilerplate_key = None
        if re.fullmatch(r"\(QT\.TCKT\.01\)", plain, re.IGNORECASE):
            boilerplate_key = "document_code"
        elif re.fullmatch(r"Ver\.01\s*[–-]\s*10/04/2019", plain, re.IGNORECASE):
            boilerplate_key = "document_version"
        if boilerplate_key is not None:
            if boilerplate_key in boilerplate_seen:
                continue
            boilerplate_seen.add(boilerplate_key)
        is_heading = re.match(r"^\s*#{1,6}\s+", line) is not None
        company_form_header = folded.startswith(
            "công ty cổ phần nhiên liệu bay petrolimex chi nhánh"
        )
        signature_footer = (
            folded.startswith("đại diện bên giao đại diện bên nhận")
            or (
                "duyệt" in folded
                and any(
                    marker in folded
                    for marker in ("người đề nghị", "người nhận", "quản lý kho")
                )
            )
            or (
                "ngày tháng năm" in folded
                and any(
                    marker in folded for marker in ("quản lý kho", "người bàn giao")
                )
            )
        )
        if is_heading and (company_form_header or signature_footer):
            output.append(plain)
        else:
            output.append(line)
    return "\n".join(output)


def _remove_invented_form_numbering(markdown: str) -> str:
    return re.sub(
        r"(?i)1\.\s*(BM\.QT\.TCKT\.01)\s*,?\s*<br>\s*2\.\s*"
        r"(BM\.QT\.TCKT\.02)",
        r"\1<br>\2",
        markdown,
    )


def _validate_gfm_tables(markdown: str) -> None:
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        if _split_gfm_row(lines[index]) is None:
            index += 1
            continue
        block: list[list[str]] = []
        while index < len(lines):
            cells = _split_gfm_row(lines[index])
            if cells is None:
                break
            block.append(cells)
            index += 1
        if len(block) < 2:
            raise StructuralProfileError("Phát hiện bảng Markdown chỉ có một dòng.")
        widths = {len(row) for row in block}
        if len(widths) != 1 or not _separator(block[1]):
            raise StructuralProfileError("Phát hiện bảng Markdown GFM bị gãy.")
        if any(_separator(row) for row in block[2:]):
            raise StructuralProfileError("Separator của bảng nằm sau dòng dữ liệu.")


def validate_petrolimex_materials_markdown(markdown: str) -> None:
    forbidden = {
        "rác a. x": r"(?mi)^\s*a\.\s*x\s*$",
        "logo chữ trôi nổi": r"(?mi)^\s*PETROLIMEX AVIATION\s*$",
        "đánh số La Mã kép": r"(?mi)^\s*(?:i|ii|iii|iv|v|vi|vii)\.\s+(?:I|II|III|IV|V|VI|VII)\.",
        "đánh số số học kép": r"(?m)^\s*(\d+)\.\s+\1\.\s+",
        "gạch đầu dòng kép": r"(?m)^\s*[-+*]\s+[-+*]\s+",
        "ký hiệu A/B tự sinh": r"(?i)<br>\s*[AB]\.\s*[−–-]",
        "mã biểu mẫu tự đánh số": r"(?i)1\.\s*BM\.QT\.TCKT\.01\s*,?\s*<br>\s*2\.",
        "footer chữ ký thành heading": r"(?mi)^\s*#{1,6}\s+.*(?:Đại diện bên giao Đại diện bên nhận|DUYỆT.*(?:NGƯỜI|QUẢN LÝ KHO))\s*$",
    }
    for label, pattern in forbidden.items():
        if re.search(pattern, markdown):
            raise StructuralProfileError(f"Hậu kiểm còn lỗi: {label}.")

    compact = " ".join(markdown.split()).casefold()
    material = compact.find("vật tư:")
    material_end = compact.find("phục vụ cho sản xuất kinh doanh.", material)
    warehouse = compact.find("kho vật tư:", material)
    if not (material >= 0 and material < material_end < warehouse):
        raise StructuralProfileError("Hậu kiểm thứ tự định nghĩa không đạt.")
    if markdown.count("| TT | Nội dung | Diễn giải | Hồ sơ |") != 2:
        raise StructuralProfileError("Hai bảng diễn giải chưa được hợp nhất.")
    if markdown.count("| Đơn vị xử lý | Nội dung | Biểu mẫu |") != 2:
        raise StructuralProfileError("Hai lưu đồ chưa có bảng ngữ nghĩa.")
    if FLOWCHART_NOTE_1 not in markdown or FLOWCHART_NOTE_2 not in markdown:
        raise StructuralProfileError("Lưu đồ chưa có mô tả nhánh Đạt/Không đạt.")
    title_count = sum(
        _plain_line(line).casefold() == "quy trình quản lý vật tư"
        for line in markdown.splitlines()
    )
    if title_count != 1:
        raise StructuralProfileError("Tiêu đề trang lặp chưa được khử nhất quán.")
    for label, pattern in (
        ("mã tài liệu", r"(?mi)^\s*#{0,6}\s*\(QT\.TCKT\.01\)\s*$"),
        ("phiên bản tài liệu", r"(?mi)^\s*Ver\.01\s*[–-]\s*10/04/2019\s*$"),
    ):
        if len(re.findall(pattern, markdown)) > 1:
            raise StructuralProfileError(f"Boilerplate {label} vẫn bị lặp.")
    _validate_gfm_tables(markdown)


def count_gfm_tables(markdown: str) -> int:
    lines = markdown.splitlines()
    count = 0
    for index in range(len(lines) - 1):
        header = _split_gfm_row(lines[index])
        separator = _split_gfm_row(lines[index + 1])
        if header is not None and separator is not None and _separator(separator):
            count += 1
    return count
