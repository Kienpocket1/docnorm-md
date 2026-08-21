from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable
from typing import Any

GEOMETRY_OCR_VERSION = "2026-08-21-1.4.0"

_SIGNATURE_LABEL_RE = re.compile(
    r"^(?:KT\.\s*)?(?:TỔNG\s+GIÁM\s+ĐỐC|GIÁM\s+ĐỐC|PHÓ\s+GIÁM\s+ĐỐC|"
    r"CHỦ\s+TỊCH|NGƯỜI\s+(?:LẬP|DUYỆT|KIỂM\s+TRA)|ĐẠI\s+DIỆN\b.*|"
    r"BÊN\s+(?:GIAO|NHẬN|MUA|BÁN))$",
    re.IGNORECASE,
)


def clean_text(value: Any) -> str:
    return " ".join(
        unicodedata.normalize("NFC", str(value or ""))
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    ).strip()


def bbox_from_polygon(polygon: Any) -> list[int] | None:
    if not isinstance(polygon, (list, tuple)):
        return None
    points: list[tuple[float, float]] = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    if not points:
        return None
    return [
        round(min(point[0] for point in points)),
        round(min(point[1] for point in points)),
        round(max(point[0] for point in points)),
        round(max(point[1] for point in points)),
    ]


def normalize_detections(raw_results: Iterable[Any]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for item in raw_results:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        bbox = bbox_from_polygon(item[0])
        text = clean_text(item[1])
        try:
            confidence = float(item[2])
        except (TypeError, ValueError):
            continue
        if bbox is None or not text or confidence < 0.18:
            continue
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            continue
        detections.append(
            {
                "text": text,
                "confidence": max(0.0, min(1.0, confidence)),
                "bbox": bbox,
            }
        )
    detections.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return detections


def _overlap_ratio(first: list[int], second: list[int]) -> float:
    overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    return overlap / max(1, min(first[3] - first[1], second[3] - second[1]))


def cluster_rows(detections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group OCR boxes into visual rows, then order every row left-to-right."""

    ordered = sorted(
        (dict(item) for item in detections),
        key=lambda item: (
            (item["bbox"][1] + item["bbox"][3]) / 2,
            item["bbox"][0],
        ),
    )
    rows: list[dict[str, Any]] = []
    for item in ordered:
        bbox = item["bbox"]
        center = (bbox[1] + bbox[3]) / 2
        best: dict[str, Any] | None = None
        best_distance = float("inf")
        for row in rows[-4:]:
            distance = abs(center - row["center_y"])
            tolerance = max(4.0, 0.58 * max(row["height"], bbox[3] - bbox[1]))
            if (
                distance <= tolerance or _overlap_ratio(row["bbox"], bbox) >= 0.45
            ) and distance < best_distance:
                best = row
                best_distance = distance
        if best is None:
            rows.append(
                {
                    "items": [item],
                    "bbox": list(bbox),
                    "center_y": center,
                    "height": bbox[3] - bbox[1],
                }
            )
            continue
        best["items"].append(item)
        best["bbox"] = [
            min(best["bbox"][0], bbox[0]),
            min(best["bbox"][1], bbox[1]),
            max(best["bbox"][2], bbox[2]),
            max(best["bbox"][3], bbox[3]),
        ]
        best["center_y"] = statistics.median(
            (entry["bbox"][1] + entry["bbox"][3]) / 2 for entry in best["items"]
        )
        best["height"] = statistics.median(
            entry["bbox"][3] - entry["bbox"][1] for entry in best["items"]
        )

    for row in rows:
        row["items"].sort(key=lambda item: item["bbox"][0])
        row["text"] = " ".join(item["text"] for item in row["items"])
        row["confidence"] = sum(
            item["confidence"] * max(1, len(item["text"])) for item in row["items"]
        ) / max(1, sum(len(item["text"]) for item in row["items"]))
    rows.sort(key=lambda row: (row["center_y"], row["bbox"][0]))
    return rows


def _group_projection(indices: Any, *, maximum_gap: int = 3) -> list[int]:
    values = [int(value) for value in indices]
    if not values:
        return []
    groups: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if value - groups[-1][-1] <= maximum_gap:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [round(statistics.median(group)) for group in groups]


def detect_table_grids(rgb_image: Any) -> list[dict[str, Any]]:
    """Detect ruled tables with OpenCV; return cell boundary coordinates."""

    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    image = np.asarray(rgb_image)
    if image.ndim != 3 or image.shape[0] < 80 or image.shape[1] < 80:
        return []
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, width // 28), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(24, height // 35))),
    )
    grid_mask = cv2.bitwise_or(horizontal, vertical)
    contours, _ = cv2.findContours(
        grid_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates: list[dict[str, Any]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < width * 0.28 or h < height * 0.045 or w * h < width * height * 0.02:
            continue
        region_h = horizontal[y : y + h, x : x + w]
        region_v = vertical[y : y + h, x : x + w]
        y_strength = (region_h > 0).sum(axis=1)
        x_strength = (region_v > 0).sum(axis=0)
        ys = _group_projection((y_strength >= max(20, int(w * 0.22))).nonzero()[0])
        xs = _group_projection((x_strength >= max(20, int(h * 0.22))).nonzero()[0])
        if len(xs) < 3 or len(ys) < 2:
            continue
        absolute_xs = sorted({x + value for value in xs})
        absolute_ys = sorted({y + value for value in ys})
        if absolute_xs[-1] - absolute_xs[0] < width * 0.25:
            continue
        candidates.append(
            {
                "bbox": [
                    absolute_xs[0],
                    absolute_ys[0],
                    absolute_xs[-1],
                    absolute_ys[-1],
                ],
                "x_lines": absolute_xs,
                "y_lines": absolute_ys,
            }
        )
    candidates.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return candidates


def _inside(center_x: float, center_y: float, bbox: list[int], margin: int = 2) -> bool:
    return (
        bbox[0] - margin <= center_x <= bbox[2] + margin
        and bbox[1] - margin <= center_y <= bbox[3] + margin
    )


def render_grid_table(
    grid: dict[str, Any],
    detections: Iterable[dict[str, Any]],
) -> str:
    xs = list(grid["x_lines"])
    ys = list(grid["y_lines"])
    width = len(xs) - 1
    height = len(ys) - 1
    cells: list[list[list[dict[str, Any]]]] = [
        [[] for _ in range(width)] for _ in range(height)
    ]
    for item in detections:
        x0, y0, x1, y1 = item["bbox"]
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        if not _inside(center_x, center_y, grid["bbox"]):
            continue
        column = next(
            (index for index in range(width) if xs[index] <= center_x <= xs[index + 1]),
            None,
        )
        row = next(
            (
                index
                for index in range(height)
                if ys[index] <= center_y <= ys[index + 1]
            ),
            None,
        )
        if row is not None and column is not None:
            cells[row][column].append(dict(item))

    def cell_text(items: list[dict[str, Any]]) -> str:
        rows = cluster_rows(items)
        text = "<br>".join(row["text"] for row in rows if row["text"])
        return text.replace("|", "\\|").strip()

    rendered_rows = [
        [cell_text(cells[row][column]) for column in range(width)]
        for row in range(height)
    ]
    if not any(any(cell for cell in row) for row in rendered_rows):
        return ""
    # GFM requires a header row. Reuse the first physical row rather than
    # inventing labels such as "Cột 1", which would add content absent from
    # the source document.
    header = rendered_rows[0]
    body = rendered_rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _uppercase_ratio(text: str) -> float:
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return 0.0
    return sum(character.isupper() for character in letters) / len(letters)


def _row_to_markdown(row: dict[str, Any], page_width: int, median_height: float) -> str:
    text = clean_text(row.get("text"))
    if not text:
        return ""
    if re.match(r"^(?:[-+•·]|\d+[.)]|[a-z][.)])\s+", text, re.IGNORECASE):
        return re.sub(r"^[•·]\s*", "- ", text)
    if _SIGNATURE_LABEL_RE.fullmatch(text):
        return text
    bbox = row["bbox"]
    centered = abs(((bbox[0] + bbox[2]) / 2) - page_width / 2) <= page_width * 0.14
    looks_heading = len(text) <= 150 and (
        (_uppercase_ratio(text) >= 0.76 and len(text) >= 6)
        or (centered and row["height"] >= median_height * 1.22)
        or re.match(r"^(?:Chương|Điều)\s+[IVXLCDM\d]+\b", text, re.IGNORECASE)
    )
    if looks_heading:
        return f"## {text}"
    return text


def render_page_markdown(
    detections: list[dict[str, Any]],
    *,
    page_width: int,
    page_height: int,
    table_grids: list[dict[str, Any]] | None = None,
    boilerplate: set[str] | None = None,
) -> tuple[str, dict[str, int]]:
    grids = table_grids or []
    boilerplate = boilerplate or set()
    outside: list[dict[str, Any]] = []
    for item in detections:
        x0, y0, x1, y1 = item["bbox"]
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        if any(_inside(center_x, center_y, grid["bbox"]) for grid in grids):
            continue
        outside.append(item)
    rows = cluster_rows(outside)
    heights = [row["height"] for row in rows if row["height"] > 0]
    median_height = statistics.median(heights) if heights else 18.0

    objects: list[tuple[float, str, str]] = []
    for row in rows:
        folded = clean_text(row["text"]).casefold()
        if folded in boilerplate:
            continue
        rendered = _row_to_markdown(row, page_width, median_height)
        if rendered:
            objects.append((row["bbox"][1], "text", rendered))
    tables = 0
    for index, grid in enumerate(grids, start=1):
        table = render_grid_table(grid, detections)
        if table:
            tables += 1
            objects.append(
                (
                    grid["bbox"][1],
                    "table",
                    f"<!-- detected-table: {index} -->\n\n{table}",
                )
            )
    objects.sort(key=lambda item: item[0])

    sections: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            sections.append(" ".join(paragraph))
            paragraph.clear()

    previous_y: float | None = None
    for y_value, kind, content in objects:
        if (
            kind == "table"
            or content.startswith("## ")
            or re.match(r"^(?:[-+]|\d+[.)]|[a-z][.)])\s+", content, re.IGNORECASE)
        ):
            flush()
            sections.append(content)
        else:
            if previous_y is not None and y_value - previous_y > median_height * 1.9:
                flush()
            paragraph.append(content)
        previous_y = y_value
    flush()
    markdown = "\n\n".join(section for section in sections if section.strip()).strip()
    return markdown + "\n" if markdown else "", {"tables": tables, "rows": len(rows)}


def repeated_margin_boilerplate(
    page_detections: dict[int, list[dict[str, Any]]],
    *,
    page_height: int,
) -> set[str]:
    if len(page_detections) < 4:
        return set()
    counts: Counter[str] = Counter()
    for detections in page_detections.values():
        seen: set[str] = set()
        for row in cluster_rows(detections):
            y_center = row["center_y"]
            if page_height * 0.11 < y_center < page_height * 0.89:
                continue
            text = clean_text(row["text"]).casefold()
            if 3 <= len(text) <= 160:
                seen.add(text)
        counts.update(seen)
    threshold = max(3, round(len(page_detections) * 0.60))
    return {text for text, count in counts.items() if count >= threshold}
