from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORKER_VERSION = "2026-08-21-1.4.0"
PRODUCT_ROOT = Path(__file__).resolve().parents[1]
if str(PRODUCT_ROOT) not in sys.path:
    sys.path.insert(0, str(PRODUCT_ROOT))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometry-preserving EasyOCR worker for scanned PDF documents."
    )
    parser.add_argument("--input-pdf", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--progress-json")
    parser.add_argument("--target-long-side", type=int, default=2200)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_gpu(device: str) -> bool:
    import torch

    if device == "cpu":
        return False
    available = bool(torch.cuda.is_available())
    if device == "cuda" and not available:
        raise RuntimeError("EasyOCR được yêu cầu chạy CUDA nhưng torch không thấy GPU")
    return available


def load_reader(device: str) -> tuple[Any, str]:
    import easyocr

    use_gpu = resolve_gpu(device)
    try:
        reader = easyocr.Reader(
            ["vi", "en"],
            gpu=use_gpu,
            verbose=False,
            download_enabled=False,
        )
        return reader, "cuda" if use_gpu else "cpu"
    except Exception:
        if device != "auto" or not use_gpu:
            raise
        import torch

        torch.cuda.empty_cache()
        reader = easyocr.Reader(
            ["vi", "en"],
            gpu=False,
            verbose=False,
            download_enabled=False,
        )
        return reader, "cpu-fallback"


def render_page(page: Any, target_long_side: int) -> Any:
    import pymupdf
    from PIL import Image

    longest = max(float(page.rect.width), float(page.rect.height), 1.0)
    scale = max(1.0, float(target_long_side) / longest)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return image


def main() -> int:
    arguments = parse_arguments()
    source = Path(arguments.input_pdf).expanduser().resolve()
    output = Path(arguments.output_json).expanduser().resolve()
    progress = (
        Path(arguments.progress_json).expanduser().resolve()
        if arguments.progress_json
        else output.with_name("worker_progress.json")
    )
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    report: dict[str, Any] = {
        "schema_version": "docnorm-easyocr-geometry-v1",
        "worker_version": WORKER_VERSION,
        "status": "FAILED",
        "input_pdf": str(source),
        "input_sha256": None,
        "device": None,
        "page_count": 0,
        "pages": [],
        "elapsed_seconds": 0.0,
        "error": None,
    }
    started = time.perf_counter()
    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        source_digest = sha256_file(source)
        report["input_sha256"] = source_digest

        import numpy as np
        import pymupdf

        from docnorm_web.geometry_ocr import (
            detect_table_grids,
            normalize_detections,
            render_page_markdown,
            repeated_margin_boilerplate,
        )

        document = pymupdf.open(source)
        try:
            if document.page_count < 1:
                raise RuntimeError("PDF không có trang")
            report["page_count"] = document.page_count
            reader, actual_device = load_reader(arguments.device)
            report["device"] = actual_device
            page_evidence: dict[int, dict[str, Any]] = {}

            for page_number in range(1, document.page_count + 1):
                page_started = time.perf_counter()
                page = document.load_page(page_number - 1)
                image = render_page(page, arguments.target_long_side)
                raw = reader.readtext(
                    np.asarray(image),
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                    batch_size=1,
                    workers=0,
                )
                detections = normalize_detections(raw)
                grids = detect_table_grids(image)
                page_evidence[page_number] = {
                    "detections": detections,
                    "grids": grids,
                    "width": image.width,
                    "height": image.height,
                    "elapsed_seconds": round(time.perf_counter() - page_started, 3),
                }
                write_json_atomic(
                    progress,
                    {
                        "schema_version": "docnorm-worker-progress-v1",
                        "worker": "EASYOCR_GEOMETRY",
                        "completed_pages": page_number,
                        "total_pages": document.page_count,
                        "current_page": page_number,
                    },
                )

            common_height = round(
                sum(item["height"] for item in page_evidence.values())
                / len(page_evidence)
            )
            boilerplate = repeated_margin_boilerplate(
                {page: item["detections"] for page, item in page_evidence.items()},
                page_height=common_height,
            )
            for page_number, evidence in page_evidence.items():
                markdown, structure = render_page_markdown(
                    evidence["detections"],
                    page_width=evidence["width"],
                    page_height=evidence["height"],
                    table_grids=evidence["grids"],
                    boilerplate=boilerplate,
                )
                confidences = [
                    float(item["confidence"]) for item in evidence["detections"]
                ]
                mean_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0.0
                )
                alphanumeric = sum(character.isalnum() for character in markdown)
                status = (
                    "PASS"
                    if alphanumeric >= 20 and mean_confidence >= 0.30
                    else "FAILED_VALIDATION"
                )
                report["pages"].append(
                    {
                        "page_number": page_number,
                        "status": status,
                        "markdown": markdown if status == "PASS" else "",
                        "detection_count": len(evidence["detections"]),
                        "mean_confidence": round(mean_confidence, 6),
                        "table_count": structure["tables"],
                        "row_count": structure["rows"],
                        "elapsed_seconds": evidence["elapsed_seconds"],
                    }
                )
        finally:
            document.close()

        if sha256_file(source) != source_digest:
            raise RuntimeError("PDF nguồn thay đổi trong lúc OCR")
        passed = sum(item["status"] == "PASS" for item in report["pages"])
        report["status"] = (
            "PASS"
            if passed == report["page_count"]
            else "PARTIAL"
            if passed
            else "FAILED"
        )
    except Exception as error:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json_atomic(output, report)

    if report["status"] not in {"PASS", "PARTIAL"}:
        print(
            (report.get("error") or {}).get("message", "Geometry OCR failed"),
            file=sys.stderr,
        )
        return 2
    print(
        f"EASYOCR_GEOMETRY={report['status']} pages={report['page_count']} "
        f"elapsed={report['elapsed_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
