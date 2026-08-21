from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit

from .config import Settings
from .math_exam import (
    MATH_REPAIR_VERSION,
    MATH_RESOLVED_ISSUE_CODES,
    analyze_math_pages,
    merge_page_markdown,
    normalize_math_markdown,
    validate_math_markdown,
)
from .models import ConversionMode, DocumentMetrics, QualityIssue, RepairProfile
from .normalizer import (
    ConversionDocument,
    HybridUnavailableError,
    HybridWorkerError,
    MathVlmWorkerError,
    PipelineUnavailableError,
    ScanRequiresOcrError,
    analyze_pdf_text_layer,
    clean_markdown,
    extract_basic,
    sha256_file,
)
from .structural_profiles import apply_structural_profile, count_gfm_tables

ProgressCallback = Callable[[str, int, str], None]


class PipelineBridge:
    """Adapter over rag_pipeline with an explicit, labelled basic fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._imports: dict[str, Any] | None = None
        self._import_error: str | None = None

    def probe(self) -> dict[str, Any]:
        modules = self._load_pipeline()
        rag_available = modules is not None
        odl_available = (
            rag_available
            and self.settings.odl_python.is_file()
            and self.settings.odl_worker.is_file()
        )
        hybrid_installed = (
            odl_available
            and self.settings.hybrid_worker.is_file()
            and self.settings.hybrid_executable.is_file()
        )
        hybrid_backend_reachable = self._hybrid_backend_reachable()
        math_vlm_installed = (
            self.settings.vlm_python.is_file() and self.settings.math_worker.is_file()
        )
        scan_python = self.settings.scan_python or self.settings.vlm_python
        scan_worker = self.settings.scan_worker
        geometry_ocr_installed = (
            scan_python.is_file() and scan_worker is not None and scan_worker.is_file()
        )
        return {
            "rag_pipeline_available": rag_available,
            "opendataloader_available": odl_available,
            "hybrid_installed": hybrid_installed,
            "hybrid_backend_reachable": hybrid_backend_reachable,
            "hybrid_available": (
                odl_available
                and self.settings.hybrid_worker.is_file()
                and hybrid_backend_reachable
            ),
            "math_vlm_installed": math_vlm_installed,
            "math_vlm_model": self.settings.math_model_id,
            "math_vlm_page_limit": self.settings.math_max_vlm_pages,
            "geometry_ocr_installed": geometry_ocr_installed,
            "basic_fallback_available": True,
            "structural_profiles_available": True,
            "pipeline_root": str(self.settings.pipeline_root),
            "import_error": self._import_error,
        }

    def convert(
        self,
        source: Path,
        work_directory: Path,
        mode: ConversionMode,
        repair_profile: RepairProfile,
        progress: ProgressCallback,
    ) -> ConversionDocument:
        pdf_analysis = (
            analyze_pdf_text_layer(source)
            if source.suffix.casefold() == ".pdf"
            else None
        )
        if (
            source.suffix.casefold() == ".pdf"
            and repair_profile == RepairProfile.MATH_EXAM
        ):
            if pdf_analysis is not None and (
                pdf_analysis.requires_ocr or mode == ConversionMode.FAST
            ):
                progress(
                    "EXTRACT",
                    24,
                    "Đề toán dạng scan: Qwen3-VL đang đọc trực tiếp theo từng trang",
                )
                result = self._convert_math_vlm_direct(
                    source,
                    work_directory,
                    progress,
                )
                return self._apply_profile(result, repair_profile, progress)
        if (
            pdf_analysis is not None
            and pdf_analysis.requires_ocr
            and repair_profile != RepairProfile.MATH_EXAM
            and self._scan_worker_ready()
        ):
            progress(
                "EXTRACT",
                24,
                (
                    f"Đã phát hiện {pdf_analysis.image_dominant_pages}/"
                    f"{pdf_analysis.page_count} trang quét; Geometry OCR đang "
                    "đọc theo tọa độ và dựng lại lưới bảng"
                ),
            )
            try:
                result = self._convert_pdf_scan_geometry(
                    source,
                    work_directory,
                    progress,
                )
                if repair_profile == RepairProfile.AUTO:
                    result = self._maybe_apply_math_vlm(
                        source,
                        work_directory,
                        result,
                        repair_profile,
                        progress,
                        allow_vlm=True,
                    )
                return self._apply_profile(result, repair_profile, progress)
            except HybridWorkerError:
                if mode == ConversionMode.FAST or not self._hybrid_ready():
                    raise
                progress(
                    "EXTRACT",
                    26,
                    "Geometry OCR chưa hoàn tất; đang chuyển sang Hybrid tùy chọn",
                )
        if mode == ConversionMode.FAST:
            if source.suffix.casefold() == ".pdf":
                if pdf_analysis is not None and pdf_analysis.requires_ocr:
                    raise ScanRequiresOcrError(
                        "PDF là tài liệu quét ảnh nhưng Geometry OCR chưa sẵn "
                        "sàng; hãy chạy lại installer để kiểm tra EasyOCR."
                    )
            progress("EXTRACT", 32, "Đang trích xuất nhanh bằng bộ đọc native")
            result = extract_basic(source)
            return self._apply_profile(result, repair_profile, progress)
        modules = self._load_pipeline()
        if modules is None:
            if source.suffix.casefold() == ".pdf":
                if pdf_analysis is not None and pdf_analysis.requires_ocr:
                    raise HybridUnavailableError(
                        "PDF cần OCR nhưng rag_pipeline chưa sẵn sàng. Hãy chạy "
                        "install_windows.ps1 rồi mở web bằng run_windows.ps1."
                    )
            if mode == ConversionMode.STRUCTURE:
                raise PipelineUnavailableError(
                    "rag_pipeline chưa sẵn sàng; hãy chạy installer hoặc chọn chế độ Nhanh."
                )
            progress("EXTRACT", 32, "Pipeline chưa sẵn sàng, chuyển sang bộ đọc cơ bản")
            result = extract_basic(source)
            result.issues.insert(
                0,
                QualityIssue(
                    severity="WARNING",
                    code="PIPELINE_UNAVAILABLE_BASIC_USED",
                    message="Không nạp được rag_pipeline; kết quả đã được gắn nhãn BASIC_FALLBACK.",
                ),
            )
            return self._apply_profile(result, repair_profile, progress)
        result: ConversionDocument
        try:
            if source.suffix.casefold() == ".pdf":
                if not (
                    self.settings.odl_python.is_file()
                    and self.settings.odl_worker.is_file()
                ):
                    raise PipelineUnavailableError(
                        "Không tìm thấy venv_opendataloader hoặc worker PDF."
                    )
                analysis = pdf_analysis or analyze_pdf_text_layer(source)
                if analysis.requires_ocr:
                    self._require_hybrid_ready()
                    progress(
                        "EXTRACT",
                        26,
                        (
                            "Đã phát hiện "
                            f"{analysis.image_dominant_pages}/{analysis.page_count} "
                            "trang quét; Hybrid OCR vi,en đang xử lý cục bộ"
                        ),
                    )
                    result = self._convert_pdf_hybrid(
                        source,
                        work_directory,
                        modules,
                        progress,
                        route_reason="SCAN_TEXT_LAYER_INSUFFICIENT",
                    )
                else:
                    progress("EXTRACT", 28, "OpenDataLoader đang phân tích PDF")
                    try:
                        result = self._convert_pdf(
                            source,
                            work_directory,
                            modules,
                            progress,
                        )
                    except Exception as local_error:
                        if not self._hybrid_ready():
                            raise
                        progress(
                            "EXTRACT",
                            34,
                            "OpenDataLoader Local chưa đạt; đang thử Hybrid cục bộ",
                        )
                        result = self._convert_pdf_hybrid(
                            source,
                            work_directory,
                            modules,
                            progress,
                            route_reason=(
                                "LOCAL_PIPELINE_FAILED_"
                                f"{type(local_error).__name__.upper()}"
                            ),
                        )
            elif source.suffix.casefold() == ".docx":
                progress("EXTRACT", 30, "Đang đọc cấu trúc DOCX native")
                result = self._convert_docx(source, modules, progress)
            else:
                raise PipelineUnavailableError("Chỉ hỗ trợ PDF và DOCX")
        except Exception as error:
            if isinstance(
                error,
                (HybridUnavailableError, HybridWorkerError, ScanRequiresOcrError),
            ):
                raise
            if mode == ConversionMode.STRUCTURE:
                if isinstance(error, PipelineUnavailableError):
                    raise
                raise PipelineUnavailableError(
                    f"Pipeline cấu trúc thất bại: {type(error).__name__}: {error}"
                ) from error
            progress("EXTRACT", 34, "Pipeline chính thất bại, đang dùng bộ đọc cơ bản")
            result = extract_basic(source)
            result.issues.insert(
                0,
                QualityIssue(
                    severity="WARNING",
                    code="PRIMARY_PIPELINE_FAILED_BASIC_USED",
                    message=(
                        "Pipeline chính không hoàn tất; đã dùng fallback cơ bản. "
                        f"Nguyên nhân: {type(error).__name__}."
                    ),
                ),
            )
        if source.suffix.casefold() == ".pdf":
            result = self._maybe_apply_math_vlm(
                source,
                work_directory,
                result,
                repair_profile,
                progress,
                allow_vlm=result.backend != "RAG_PIPELINE_HYBRID_OCR",
            )
        return self._apply_profile(result, repair_profile, progress)

    @staticmethod
    def _apply_profile(
        result: ConversionDocument,
        requested: RepairProfile,
        progress: ProgressCallback,
    ) -> ConversionDocument:
        progress("NORMALIZE", 82, "Đang hậu kiểm và sửa cấu trúc có kiểm soát")
        effective_profile = (
            RepairProfile.MATH_EXAM
            if result.repair_profile_applied == RepairProfile.MATH_EXAM
            else requested
        )
        repair = apply_structural_profile(result.markdown, effective_profile)
        result.markdown = repair.markdown
        result.repair_profile_requested = requested
        result.repair_profile_applied = repair.applied
        result.repair_version = repair.repair_version
        result.repair_fingerprint_score = repair.fingerprint_score
        result.resolved_issue_codes = list(repair.resolved_issue_codes)
        result.metrics = result.metrics.model_copy(
            update={
                "headings": len(re.findall(r"(?m)^#{1,6}\s+\S", repair.markdown)),
                "tables": count_gfm_tables(repair.markdown),
                "markdown_characters": len(repair.markdown),
                "broken_characters": repair.markdown.count("\ufffd"),
            }
        )
        if repair.applied == RepairProfile.PETROLIMEX_MATERIALS:
            result.issues.append(
                QualityIssue(
                    severity="INFO",
                    code="STRUCTURAL_PROFILE_APPLIED",
                    message=(
                        "Đã nhận diện chắc chắn và sửa cấu trúc Quy trình quản lý "
                        "vật tư; hậu kiểm chuyên biệt đã PASS."
                    ),
                )
            )
        return result

    def _math_worker_ready(self) -> bool:
        return (
            self.settings.vlm_python.is_file() and self.settings.math_worker.is_file()
        )

    def _scan_worker_ready(self) -> bool:
        scan_python = self.settings.scan_python or self.settings.vlm_python
        scan_worker = self.settings.scan_worker
        return (
            scan_python.is_file() and scan_worker is not None and scan_worker.is_file()
        )

    def _run_scan_worker(
        self,
        source: Path,
        run_directory: Path,
        progress: ProgressCallback,
    ) -> dict[str, Any]:
        if not self._scan_worker_ready():
            raise HybridUnavailableError(
                "Geometry OCR chưa sẵn sàng; cần EasyOCR trong venv Qwen."
            )
        scan_python = self.settings.scan_python or self.settings.vlm_python
        scan_worker = self.settings.scan_worker
        assert scan_worker is not None
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise HybridWorkerError(
                f"Geometry OCR run directory đã tồn tại: {run_directory}"
            ) from error
        result_path = run_directory / "worker_result.json"
        progress_path = run_directory / "worker_progress.json"
        stdout_path = run_directory / "worker_stdout.log"
        stderr_path = run_directory / "worker_stderr.log"
        source_digest = sha256_file(source)
        command = [
            str(scan_python),
            str(scan_worker),
            "--input-pdf",
            str(source),
            "--output-json",
            str(result_path),
            "--progress-json",
            str(progress_path),
            "--target-long-side",
            str(self.settings.scan_target_long_side),
            "--device",
            "auto",
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCHDYNAMO_DISABLE": "1",
                "TORCH_COMPILE_DISABLE": "1",
            }
        )
        arguments: dict[str, Any] = {
            "args": command,
            "cwd": str(self.settings.product_root),
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "shell": False,
        }
        if os.name == "nt":
            arguments["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            arguments["start_new_session"] = True
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_log,
            stderr_path.open("w", encoding="utf-8") as stderr_log,
        ):
            arguments["stdout"] = stdout_log
            arguments["stderr"] = stderr_log
            try:
                process = subprocess.Popen(**arguments)
            except OSError as error:
                raise HybridWorkerError(
                    f"Không khởi chạy được Geometry OCR: {error}"
                ) from error
            started = time.monotonic()
            last_completed = -1
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                if (
                    time.monotonic() - started
                    > self.settings.scan_worker_timeout_seconds
                ):
                    self._terminate_process_tree(process)
                    raise HybridWorkerError(
                        "Geometry OCR vượt quá thời gian cho phép "
                        f"({self.settings.scan_worker_timeout_seconds:.0f} giây)."
                    )
                if progress_path.is_file():
                    try:
                        state = json.loads(progress_path.read_text(encoding="utf-8"))
                        completed = int(state.get("completed_pages") or 0)
                        total = max(1, int(state.get("total_pages") or 1))
                    except Exception:
                        completed = last_completed
                        total = 1
                    if completed >= 0 and completed != last_completed:
                        last_completed = completed
                        percent = min(61, 24 + round(37 * completed / total))
                        progress(
                            "EXTRACT",
                            percent,
                            f"Geometry OCR đã đọc {completed}/{total} trang",
                        )
                time.sleep(1.0)
        if sha256_file(source) != source_digest:
            raise HybridWorkerError("PDF nguồn thay đổi trong lúc Geometry OCR")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as error:
            tail = self._read_log_tail(stderr_path)
            detail = f"Geometry OCR không tạo JSON hợp lệ: {error}"
            if tail:
                detail += f"\nWorker stderr tail:\n{tail}"
            raise HybridWorkerError(detail) from error
        if payload.get("input_sha256") != source_digest:
            raise HybridWorkerError("Geometry OCR manifest không khớp PDF nguồn")
        if return_code != 0 or payload.get("status") not in {"PASS", "PARTIAL"}:
            worker_error = payload.get("error") or {}
            detail = (
                f"{worker_error.get('type', 'GeometryOcrFailure')}: "
                f"{worker_error.get('message', 'Geometry OCR thất bại')}"
            )
            tail = self._read_log_tail(stderr_path)
            if tail:
                detail += f"\nWorker stderr tail:\n{tail}"
            raise HybridWorkerError(detail)
        return payload

    def _convert_pdf_scan_geometry(
        self,
        source: Path,
        work_directory: Path,
        progress: ProgressCallback,
    ) -> ConversionDocument:
        payload = self._run_scan_worker(
            source,
            work_directory / "easyocr_geometry",
            progress,
        )
        page_count = int(payload.get("page_count") or 0)
        if page_count < 1:
            raise HybridWorkerError("Geometry OCR báo page_count không hợp lệ")
        page_markdown: dict[int, str] = {}
        failed: list[int] = []
        low_confidence: list[int] = []
        element_count = 0
        table_count = 0
        for item in payload.get("pages", []):
            if not isinstance(item, dict):
                continue
            page_number = int(item.get("page_number") or 0)
            if page_number < 1 or page_number > page_count:
                continue
            if item.get("status") != "PASS":
                failed.append(page_number)
                continue
            markdown = clean_markdown(str(item.get("markdown") or ""))
            if not markdown:
                failed.append(page_number)
                continue
            page_markdown[page_number] = markdown
            element_count += int(item.get("detection_count") or 0)
            table_count += int(item.get("table_count") or 0)
            if float(item.get("mean_confidence") or 0.0) < 0.46:
                low_confidence.append(page_number)
        missing = sorted(set(range(1, page_count + 1)) - set(page_markdown))
        failed = sorted(set(failed) | set(missing))
        confirmed_pages = len(page_markdown)
        for page_number in failed:
            page_markdown[page_number] = (
                "> **CẦN KIỂM TRA:** Geometry OCR không xác nhận được nội dung "
                "trang này.\n"
            )
        markdown = merge_page_markdown(
            page_markdown,
            {},
            include_pages=range(1, page_count + 1),
        )
        issues = [
            QualityIssue(
                severity="INFO",
                code="GEOMETRY_OCR_USED",
                message=(
                    "PDF scan được đọc bằng EasyOCR vi,en theo tọa độ; thứ tự "
                    "dòng là trên-xuống/trái-phải và bảng có đường kẻ được dựng "
                    "lại thành GFM."
                ),
                page_numbers=sorted(page_markdown),
            )
        ]
        if low_confidence:
            issues.append(
                QualityIssue(
                    severity="WARNING",
                    code="GEOMETRY_OCR_LOW_CONFIDENCE",
                    message="Một số trang có độ tin cậy OCR trung bình thấp.",
                    page_numbers=sorted(set(low_confidence)),
                )
            )
        disposition = "PASS_WITH_WARNINGS"
        if failed:
            disposition = "NEEDS_REVIEW"
            issues.append(
                QualityIssue(
                    severity="ERROR",
                    code="GEOMETRY_OCR_PAGE_FAILED",
                    message="Một số trang không vượt kiểm định OCR và không được suy đoán.",
                    page_numbers=failed,
                )
            )
        return ConversionDocument(
            markdown=markdown,
            metrics=DocumentMetrics(
                pages=page_count,
                elements=element_count,
                headings=len(re.findall(r"(?m)^#{1,6}\s+\S", markdown)),
                paragraphs=sum(
                    bool(line.strip())
                    and not line.lstrip().startswith(("#", "<!--", "|", ">"))
                    for line in markdown.splitlines()
                ),
                tables=table_count,
                lists=len(re.findall(r"(?m)^\s*[-*+]\s+", markdown)),
                images=0,
                markdown_characters=len(markdown),
                broken_characters=markdown.count("\ufffd"),
                provenance_coverage=confirmed_pages / page_count,
                ocr_invocations=1,
            ),
            issues=issues,
            engine_chain=["EASYOCR_GEOMETRY"],
            backend="EASYOCR_GEOMETRY",
            quality_disposition=disposition,
            source_sha256=sha256_file(source),
            page_markdown=page_markdown,
        )

    def _run_math_worker(
        self,
        source: Path,
        run_directory: Path,
        pages: list[int],
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not self._math_worker_ready():
            raise MathVlmWorkerError(
                "Thiếu venv Qwen hoặc qwen_math_worker.py. Mặc định cần "
                "<project>\\venv\\Scripts\\python.exe."
            )
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise MathVlmWorkerError(
                f"Math VLM run directory đã tồn tại: {run_directory}"
            ) from error
        result_path = run_directory / "worker_result.json"
        progress_path = run_directory / "worker_progress.json"
        stdout_path = run_directory / "worker_stdout.log"
        stderr_path = run_directory / "worker_stderr.log"
        source_digest = sha256_file(source)
        command = [
            str(self.settings.vlm_python),
            str(self.settings.math_worker),
            "--input-pdf",
            str(source),
            "--output-json",
            str(result_path),
            "--pages",
            ",".join(str(page) for page in pages),
            "--model-id",
            self.settings.math_model_id,
            "--target-long-side",
            str(self.settings.math_target_long_side),
            "--max-new-tokens",
            str(self.settings.math_max_new_tokens),
            "--progress-json",
            str(progress_path),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TORCHDYNAMO_DISABLE": "1",
                "TORCH_COMPILE_DISABLE": "1",
            }
        )
        arguments: dict[str, Any] = {
            "args": command,
            "cwd": str(self.settings.product_root),
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "shell": False,
        }
        if os.name == "nt":
            arguments["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            arguments["start_new_session"] = True
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_log,
            stderr_path.open("w", encoding="utf-8") as stderr_log,
        ):
            arguments["stdout"] = stdout_log
            arguments["stderr"] = stderr_log
            try:
                process = subprocess.Popen(**arguments)
            except OSError as error:
                raise MathVlmWorkerError(
                    f"Không khởi chạy được Qwen math worker: {error}"
                ) from error
            started = time.monotonic()
            last_completed = -1
            while True:
                return_code = process.poll()
                if return_code is not None:
                    break
                if (
                    time.monotonic() - started
                    > self.settings.math_worker_timeout_seconds
                ):
                    self._terminate_process_tree(process)
                    raise MathVlmWorkerError(
                        "Qwen math worker vượt quá thời gian cho phép "
                        f"({self.settings.math_worker_timeout_seconds:.0f} giây)."
                    )
                if progress is not None and progress_path.is_file():
                    try:
                        state = json.loads(progress_path.read_text(encoding="utf-8"))
                        completed = int(state.get("completed_pages") or 0)
                        total = max(1, int(state.get("total_pages") or len(pages)))
                    except Exception:
                        completed = last_completed
                        total = max(1, len(pages))
                    if completed >= 0 and completed != last_completed:
                        last_completed = completed
                        percent = min(79, 64 + round(14 * completed / total))
                        progress(
                            "VERIFY",
                            percent,
                            f"Qwen đã kiểm định {completed}/{total} trang toán",
                        )
                time.sleep(1.0)
        if sha256_file(source) != source_digest:
            raise MathVlmWorkerError("PDF nguồn thay đổi trong lúc Qwen xử lý")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as error:
            tail = self._read_log_tail(stderr_path)
            detail = f"Qwen math worker không tạo JSON hợp lệ: {error}"
            if tail:
                detail += f"\nWorker stderr tail:\n{tail}"
            raise MathVlmWorkerError(detail) from error
        if payload.get("input_sha256") != source_digest:
            raise MathVlmWorkerError("Math worker manifest không khớp PDF nguồn")
        if payload.get("requested_pages") != pages:
            raise MathVlmWorkerError("Math worker manifest không khớp danh sách trang")
        if return_code != 0 or payload.get("status") not in {"PASS", "PARTIAL"}:
            worker_error = payload.get("error") or {}
            detail = (
                f"{worker_error.get('type', 'MathWorkerFailure')}: "
                f"{worker_error.get('message', 'Qwen math worker thất bại')}"
            )
            tail = self._read_log_tail(stderr_path)
            if tail:
                detail += f"\nWorker stderr tail:\n{tail}"
            raise MathVlmWorkerError(detail)
        return payload

    @staticmethod
    def _math_replacements(payload: dict[str, Any]) -> tuple[dict[int, str], list[int]]:
        replacements: dict[int, str] = {}
        failed: list[int] = []
        for item in payload.get("pages", []):
            if not isinstance(item, dict):
                continue
            page_number = int(item.get("page_number") or 0)
            markdown = normalize_math_markdown(str(item.get("markdown") or ""))
            errors = validate_math_markdown(markdown)
            if item.get("status") == "PASS" and page_number > 0 and not errors:
                replacements[page_number] = markdown
            elif page_number > 0:
                failed.append(page_number)
        return replacements, sorted(set(failed))

    def _convert_math_vlm_direct(
        self,
        source: Path,
        work_directory: Path,
        progress: ProgressCallback,
    ) -> ConversionDocument:
        import pymupdf

        document = pymupdf.open(source)
        try:
            page_count = document.page_count
        finally:
            document.close()
        if page_count < 1:
            raise MathVlmWorkerError("PDF không có trang")
        pages = list(range(1, page_count + 1))
        progress(
            "EXTRACT",
            34,
            f"Qwen3-VL đã nạp một lần; đang đọc {page_count} trang đề toán",
        )
        payload = self._run_math_worker(
            source,
            work_directory / "qwen_math_direct",
            pages,
            progress,
        )
        replacements, failed = self._math_replacements(payload)
        if not replacements:
            raise MathVlmWorkerError("Không trang nào vượt kiểm định Math VLM")
        page_markdown = {
            page: replacements.get(
                page,
                "> **CẦN KIỂM TRA:** Math VLM không xác nhận được nội dung trang này.\n",
            )
            for page in pages
        }
        markdown = merge_page_markdown(
            page_markdown,
            {},
            include_pages=pages,
        )
        issues = [
            QualityIssue(
                severity="INFO",
                code="QWEN_MATH_DIRECT_USED",
                message=(
                    "Đề toán dạng scan được đọc trực tiếp bằng Qwen3-VL 2B 4-bit; "
                    "model nạp một lần cho toàn bộ job."
                ),
                page_numbers=sorted(replacements),
            )
        ]
        disposition = "PASS_WITH_WARNINGS"
        if failed:
            disposition = "NEEDS_REVIEW"
            issues.append(
                QualityIssue(
                    severity="ERROR",
                    code="MATH_VLM_PAGE_VALIDATION_FAILED",
                    message="Một số trang không vượt kiểm định và không được tự suy đoán.",
                    page_numbers=failed,
                )
            )
        return ConversionDocument(
            markdown=markdown,
            metrics=DocumentMetrics(
                pages=page_count,
                elements=len(replacements),
                headings=len(re.findall(r"(?m)^#{1,6}\s+\S", markdown)),
                paragraphs=sum(
                    bool(line.strip()) and not line.lstrip().startswith(("#", "<!--"))
                    for line in markdown.splitlines()
                ),
                tables=count_gfm_tables(markdown),
                lists=len(re.findall(r"(?m)^\s*[-*+]\s+", markdown)),
                images=0,
                markdown_characters=len(markdown),
                broken_characters=markdown.count("\ufffd"),
                provenance_coverage=len(replacements) / page_count,
                ocr_invocations=1,
                vlm_invocations=1,
                vlm_pages=len(replacements),
            ),
            issues=issues,
            engine_chain=["QWEN3_VL_MATH"],
            backend="QWEN3_VL_MATH_DIRECT",
            quality_disposition=disposition,
            source_sha256=sha256_file(source),
            repair_profile_applied=RepairProfile.MATH_EXAM,
            repair_version=MATH_REPAIR_VERSION,
            repair_fingerprint_score=1.0,
            resolved_issue_codes=list(MATH_RESOLVED_ISSUE_CODES),
            page_markdown=page_markdown,
        )

    def _maybe_apply_math_vlm(
        self,
        source: Path,
        work_directory: Path,
        result: ConversionDocument,
        requested: RepairProfile,
        progress: ProgressCallback,
        *,
        allow_vlm: bool,
    ) -> ConversionDocument:
        if requested not in {RepairProfile.AUTO, RepairProfile.MATH_EXAM}:
            return result
        if not result.page_markdown:
            if requested == RepairProfile.MATH_EXAM:
                raise MathVlmWorkerError(
                    "Không có provenance theo trang để chạy fallback đề toán."
                )
            return result
        risk = analyze_math_pages(
            result.page_markdown,
            forced=requested == RepairProfile.MATH_EXAM,
        )
        if not risk.is_math_document:
            return result
        flagged = list(risk.flagged_pages)
        if not flagged:
            if requested == RepairProfile.MATH_EXAM:
                result.repair_profile_applied = RepairProfile.MATH_EXAM
                result.repair_version = MATH_REPAIR_VERSION
                result.repair_fingerprint_score = 1.0
                result.issues.append(
                    QualityIssue(
                        severity="INFO",
                        code="MATH_FAST_PATH_ACCEPTED",
                        message=(
                            "Không phát hiện trang toán rủi ro cao; giữ kết quả "
                            "deterministic để tiết kiệm thời gian."
                        ),
                    )
                )
            return result

        ranked = sorted(
            (item for item in risk.page_risks if item.page_number in flagged),
            key=lambda item: (-item.score, item.page_number),
        )
        selected = sorted(
            item.page_number for item in ranked[: self.settings.math_max_vlm_pages]
        )
        deferred = sorted(set(flagged) - set(selected))
        if not allow_vlm:
            result.quality_disposition = "NEEDS_REVIEW"
            result.issues.append(
                QualityIssue(
                    severity="ERROR",
                    code="MATH_VLM_SKIPPED_AFTER_HYBRID",
                    message=(
                        "Đã phát hiện trang toán rủi ro nhưng Hybrid OCR đang giữ "
                        "GPU. Hãy chọn profile Đề toán để định tuyến trực tiếp, nhanh hơn."
                    ),
                    page_numbers=flagged,
                )
            )
            return result
        if not self._math_worker_ready():
            if requested == RepairProfile.MATH_EXAM:
                raise MathVlmWorkerError("Môi trường Qwen math chưa sẵn sàng")
            result.quality_disposition = "NEEDS_REVIEW"
            result.issues.append(
                QualityIssue(
                    severity="ERROR",
                    code="MATH_VLM_UNAVAILABLE",
                    message="Phát hiện đề toán rủi ro nhưng môi trường Qwen chưa sẵn sàng.",
                    page_numbers=flagged,
                )
            )
            return result

        progress(
            "VERIFY",
            64,
            (
                f"Phát hiện {len(flagged)} trang toán rủi ro; "
                f"Qwen chỉ đọc lại {len(selected)} trang để tối ưu thời gian"
            ),
        )
        payload = self._run_math_worker(
            source,
            work_directory / "qwen_math_selective",
            selected,
            progress,
        )
        replacements, failed = self._math_replacements(payload)
        if replacements:
            result.page_markdown.update(replacements)
            result.markdown = merge_page_markdown(result.page_markdown, {})
            if "QWEN3_VL_MATH" not in result.engine_chain:
                result.engine_chain.append("QWEN3_VL_MATH")
            result.backend += "+SELECTIVE_MATH_VLM"
            result.metrics = result.metrics.model_copy(
                update={
                    "headings": len(re.findall(r"(?m)^#{1,6}\s+\S", result.markdown)),
                    "tables": count_gfm_tables(result.markdown),
                    "markdown_characters": len(result.markdown),
                    "broken_characters": result.markdown.count("\ufffd"),
                    "vlm_invocations": result.metrics.vlm_invocations + 1,
                    "vlm_pages": result.metrics.vlm_pages + len(replacements),
                }
            )
        unresolved = sorted(
            set(failed) | set(deferred) | (set(selected) - set(replacements))
        )
        result.repair_profile_applied = RepairProfile.MATH_EXAM
        result.repair_version = MATH_REPAIR_VERSION
        result.repair_fingerprint_score = 1.0
        result.resolved_issue_codes = list(MATH_RESOLVED_ISSUE_CODES)
        result.issues.append(
            QualityIssue(
                severity="INFO",
                code="SELECTIVE_MATH_VLM_USED",
                message=(
                    f"Qwen3-VL chỉ đọc lại {len(replacements)}/{len(result.page_markdown)} "
                    "trang; các trang sạch giữ đường xử lý nhanh."
                ),
                page_numbers=sorted(replacements),
            )
        )
        if unresolved:
            result.quality_disposition = "NEEDS_REVIEW"
            result.issues.append(
                QualityIssue(
                    severity="ERROR",
                    code="MATH_PAGES_REQUIRE_REVIEW",
                    message=(
                        "Một số trang rủi ro chưa được thay thế do kiểm định hoặc "
                        "giới hạn thời gian selective fallback."
                    ),
                    page_numbers=unresolved,
                )
            )
        elif result.quality_disposition in {
            "NEEDS_REVIEW",
            "FALLBACK_REQUIRED",
            "QUARANTINE_REQUIRED",
        }:
            result.quality_disposition = "PASS_WITH_WARNINGS"
        return result

    def _load_pipeline(self) -> dict[str, Any] | None:
        if self._imports is not None:
            return self._imports
        source_root = self.settings.pipeline_source_root
        if not source_root.is_dir():
            self._import_error = f"Không thấy pipeline source: {source_root}"
            return None
        source_value = str(source_root)
        if source_value not in sys.path:
            sys.path.insert(0, source_value)
        try:
            from rag_pipeline.contracts import (
                EngineName,
                FileFormat,
                RouteDecision,
                RouteType,
            )
            from rag_pipeline.extractors.base import ExtractionRequest
            from rag_pipeline.extractors.docx_native import DocxNativeExtractor
            from rag_pipeline.extractors.opendataloader_local import (
                OpenDataLoaderLocalConfig,
                OpenDataLoaderLocalExtractor,
                WorkerInputRecord,
                WorkerOutputRecord,
            )
            from rag_pipeline.extractors.opendataloader_mapping import (
                DocumentIdentity,
                OpenDataLoaderJsonMapper,
            )
            from rag_pipeline.normalization.contract_native import render_element
            from rag_pipeline.stages.extraction_quality import ExtractionQualityStage
        except Exception as error:
            self._import_error = f"{type(error).__name__}: {error}"
            return None
        self._imports = {
            "EngineName": EngineName,
            "FileFormat": FileFormat,
            "RouteDecision": RouteDecision,
            "RouteType": RouteType,
            "ExtractionRequest": ExtractionRequest,
            "DocxNativeExtractor": DocxNativeExtractor,
            "OpenDataLoaderLocalConfig": OpenDataLoaderLocalConfig,
            "OpenDataLoaderLocalExtractor": OpenDataLoaderLocalExtractor,
            "WorkerInputRecord": WorkerInputRecord,
            "WorkerOutputRecord": WorkerOutputRecord,
            "DocumentIdentity": DocumentIdentity,
            "OpenDataLoaderJsonMapper": OpenDataLoaderJsonMapper,
            "render_element": render_element,
            "ExtractionQualityStage": ExtractionQualityStage,
        }
        return self._imports

    def _hybrid_endpoint(self) -> tuple[str, int] | None:
        try:
            parsed = urlsplit(self.settings.hybrid_url)
            port = parsed.port
        except ValueError:
            return None
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or port is None
            or not 1 <= port <= 65535
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
        ):
            return None
        return parsed.hostname, port

    def _hybrid_backend_reachable(self) -> bool:
        endpoint = self._hybrid_endpoint()
        if endpoint is None:
            return False
        try:
            with socket.create_connection(endpoint, timeout=0.25):
                return True
        except OSError:
            return False

    def _hybrid_ready(self) -> bool:
        return (
            self.settings.odl_python.is_file()
            and self.settings.odl_worker.is_file()
            and self.settings.hybrid_worker.is_file()
            and self._hybrid_backend_reachable()
        )

    def _require_hybrid_ready(self) -> None:
        endpoint = self._hybrid_endpoint()
        if endpoint is None:
            raise HybridUnavailableError(
                "DOCNORM_ODL_HYBRID_URL không phải địa chỉ localhost hợp lệ."
            )
        missing: list[str] = []
        if not self.settings.odl_python.is_file():
            missing.append("venv_opendataloader\\Scripts\\python.exe")
        if not self.settings.odl_worker.is_file():
            missing.append("opendataloader_local_worker.py")
        if not self.settings.hybrid_worker.is_file():
            missing.append("opendataloader_hybrid_document_worker.py")
        if missing:
            raise HybridUnavailableError(
                "Thiếu thành phần Hybrid OCR: " + ", ".join(missing)
            )
        if not self._hybrid_backend_reachable():
            raise HybridUnavailableError(
                "Hybrid OCR chưa chạy tại localhost:5002. Hãy đóng web rồi mở "
                "lại với DOCNORM_START_HYBRID=1 nếu cần backend tùy chọn."
            )

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                shell=False,
            )
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    @staticmethod
    def _read_log_tail(path: Path, maximum_characters: int = 5000) -> str:
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[-maximum_characters:]

    def _convert_pdf(
        self,
        source: Path,
        work_directory: Path,
        modules: dict[str, Any],
        progress: ProgressCallback,
    ) -> ConversionDocument:
        raw_root = work_directory / "opendataloader"
        extractor = modules["OpenDataLoaderLocalExtractor"](
            modules["OpenDataLoaderLocalConfig"](
                python_executable=self.settings.odl_python,
                worker_script=self.settings.odl_worker,
                output_root=raw_root,
                timeout_seconds=self.settings.pipeline_timeout_seconds,
            )
        )
        run = extractor.extract(source, run_id="local")
        progress("VERIFY", 58, "Đang ánh xạ cấu trúc và kiểm tra provenance")
        identity = modules["DocumentIdentity"](
            catalog_id="docnorm-web",
            family_id=f"document-{sha256_file(source)[:12]}",
            version_id=f"version-{sha256_file(source)[:16]}",
        )
        mapping = modules["OpenDataLoaderJsonMapper"](known_asset_policies={}).map_run(
            run, identity
        )
        return self._normalize_pipeline_document(
            mapping.document,
            modules,
            inherited_issues=mapping.mapping_issues,
            preferred_ids=mapping.indexable_element_ids,
            backend="RAG_PIPELINE",
            assets=[Path(record.path) for record in run.manifest.output.images],
            progress=progress,
        )

    def _convert_pdf_hybrid(
        self,
        source: Path,
        work_directory: Path,
        modules: dict[str, Any],
        progress: ProgressCallback,
        *,
        route_reason: str,
    ) -> ConversionDocument:
        run = self._run_hybrid_document_worker(
            source,
            work_directory / "opendataloader_hybrid",
            modules,
        )
        progress("VERIFY", 58, "Đang ánh xạ kết quả OCR và kiểm tra provenance")
        digest = sha256_file(source)
        identity = modules["DocumentIdentity"](
            catalog_id="docnorm-web",
            family_id=f"document-{digest[:12]}",
            version_id=f"version-{digest[:16]}",
        )
        mapping = modules["OpenDataLoaderJsonMapper"](known_asset_policies={}).map_run(
            run, identity
        )
        hybrid_engine = modules["EngineName"].OPENDATALOADER_HYBRID
        hybrid_elements = [
            element.model_copy(update={"engine": hybrid_engine})
            for element in mapping.document.elements
        ]
        hybrid_document = mapping.document.model_copy(
            update={
                "engine_chain": [hybrid_engine],
                "elements": hybrid_elements,
            }
        )
        result = self._normalize_pipeline_document(
            hybrid_document,
            modules,
            inherited_issues=mapping.mapping_issues,
            preferred_ids=mapping.indexable_element_ids,
            backend="RAG_PIPELINE_HYBRID_OCR",
            assets=[Path(record.path) for record in run.manifest.output.images],
            progress=progress,
            ocr_invocations=1,
        )
        result.issues.insert(
            0,
            QualityIssue(
                severity="INFO",
                code="LOCAL_HYBRID_OCR_USED",
                message=(
                    "Đã chạy Hybrid OCR cục bộ (docling-fast, full, vi+en, "
                    f"fail-closed). Lý do định tuyến: {route_reason}."
                ),
            ),
        )
        return result

    def _run_hybrid_document_worker(
        self,
        source: Path,
        run_directory: Path,
        modules: dict[str, Any],
    ) -> Any:
        self._require_hybrid_ready()
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise HybridWorkerError(
                f"Hybrid run directory đã tồn tại: {run_directory}"
            ) from error
        raw_directory = run_directory / "raw"
        result_path = run_directory / "worker_result.json"
        stdout_path = run_directory / "worker_stdout.log"
        stderr_path = run_directory / "worker_stderr.log"
        source_sha256_before = sha256_file(source)
        command = [
            str(self.settings.odl_python),
            str(self.settings.hybrid_worker),
            "--input-pdf",
            str(source),
            "--artifact-dir",
            str(raw_directory),
            "--result-json",
            str(result_path),
            "--local-worker",
            str(self.settings.odl_worker),
            "--hybrid-url",
            self.settings.hybrid_url,
            "--hybrid-timeout-ms",
            str(self.settings.hybrid_timeout_ms),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "TORCHDYNAMO_DISABLE": "1",
                "TORCH_COMPILE_DISABLE": "1",
            }
        )
        popen_arguments: dict[str, Any] = {
            "args": command,
            "cwd": str(self.settings.hybrid_worker.parent),
            "env": environment,
            "stdin": subprocess.DEVNULL,
            "shell": False,
        }
        if os.name == "nt":
            popen_arguments["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_arguments["start_new_session"] = True

        with (
            stdout_path.open("w", encoding="utf-8") as stdout_log,
            stderr_path.open("w", encoding="utf-8") as stderr_log,
        ):
            popen_arguments["stdout"] = stdout_log
            popen_arguments["stderr"] = stderr_log
            try:
                process = subprocess.Popen(**popen_arguments)
            except OSError as error:
                raise HybridWorkerError(
                    f"Không khởi chạy được Hybrid worker: {error}"
                ) from error
            try:
                return_code = process.wait(
                    timeout=self.settings.hybrid_process_timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                self._terminate_process_tree(process)
                raise HybridWorkerError(
                    "Hybrid OCR vượt quá thời gian cho phép "
                    f"({self.settings.hybrid_process_timeout_seconds:.0f} giây)."
                ) from error

        if sha256_file(source) != source_sha256_before:
            raise HybridWorkerError("PDF nguồn đã thay đổi trong lúc chạy Hybrid OCR")
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception as error:
            tail = self._read_log_tail(stderr_path)
            detail = f"Hybrid worker không tạo manifest hợp lệ: {error}"
            if tail:
                detail += f"\nWorker stderr tail:\n{tail}"
            raise HybridWorkerError(detail) from error
        if return_code != 0 or payload.get("status") != "PASS":
            worker_error = payload.get("error")
            if isinstance(worker_error, dict):
                detail = (
                    f"{worker_error.get('type', 'WorkerFailure')}: "
                    f"{worker_error.get('message', 'Hybrid OCR thất bại')}"
                )
            else:
                detail = "Hybrid OCR worker thất bại"
            tail = self._read_log_tail(stderr_path)
            if tail:
                detail += f"\nWorker stderr tail:\n{tail}"
            raise HybridWorkerError(detail)
        if payload.get("engine") != "OPENDATALOADER_HYBRID":
            raise HybridWorkerError("Hybrid manifest khai báo engine không hợp lệ")

        try:
            input_record = modules["WorkerInputRecord"].model_validate(payload["input"])
            output_record = modules["WorkerOutputRecord"].model_validate(
                payload["output"]
            )
        except Exception as error:
            raise HybridWorkerError(
                f"Hybrid manifest không khớp contract: {type(error).__name__}"
            ) from error
        if input_record.sha256 != source_sha256_before:
            raise HybridWorkerError("Hybrid manifest không khớp SHA256 của PDF nguồn")
        self._verify_hybrid_artifacts(raw_directory, output_record)
        manifest = SimpleNamespace(input=input_record, output=output_record)
        return SimpleNamespace(
            run_id="hybrid-full",
            run_directory=run_directory,
            raw_artifact_directory=raw_directory,
            result_manifest_path=result_path,
            stdout_log_path=stdout_path,
            stderr_log_path=stderr_path,
            manifest=manifest,
        )

    @staticmethod
    def _verify_hybrid_artifacts(raw_directory: Path, output: Any) -> None:
        raw_root = raw_directory.resolve()
        if Path(output.artifact_directory).resolve() != raw_root:
            raise HybridWorkerError("Hybrid artifact directory không khớp manifest")
        records = [output.markdown, output.structured_json, *output.images]
        for record in records:
            path = Path(record.path).resolve()
            if raw_root not in path.parents or not path.is_file():
                raise HybridWorkerError(
                    f"Hybrid artifact nằm ngoài raw directory: {record.relative_path}"
                )
            if path.stat().st_size != record.size_bytes:
                raise HybridWorkerError(
                    f"Hybrid artifact sai kích thước: {record.relative_path}"
                )
            if sha256_file(path) != record.sha256:
                raise HybridWorkerError(
                    f"Hybrid artifact sai SHA256: {record.relative_path}"
                )

    def _convert_docx(
        self,
        source: Path,
        modules: dict[str, Any],
        progress: ProgressCallback,
    ) -> ConversionDocument:
        digest = sha256_file(source)
        route = modules["RouteDecision"](
            route_id=f"docnorm-docx-{digest[:16]}",
            catalog_id="docnorm-web",
            version_id=f"version-{digest[:16]}",
            file_format=modules["FileFormat"].DOCX,
            route_type=modules["RouteType"].DOCX_NATIVE,
            primary_engine=modules["EngineName"].DOCX_NATIVE,
            fallback_engines=[],
            reason_code="docnorm_web_docx_native",
        )
        request = modules["ExtractionRequest"](
            source_path=source,
            catalog_id="docnorm-web",
            family_id=f"document-{digest[:12]}",
            version_id=f"version-{digest[:16]}",
            route=route,
            run_id=f"docx-{digest[:16]}",
        )
        output = modules["DocxNativeExtractor"]().extract(request)
        progress("VERIFY", 58, "Đang kiểm tra cấu trúc DOCX")
        return self._normalize_pipeline_document(
            output.document,
            modules,
            inherited_issues=(),
            preferred_ids=None,
            backend="RAG_PIPELINE",
            assets=[],
            progress=progress,
        )

    def _normalize_pipeline_document(
        self,
        document: Any,
        modules: dict[str, Any],
        *,
        inherited_issues: Any,
        preferred_ids: Any,
        backend: str,
        assets: list[Path],
        progress: ProgressCallback,
        ocr_invocations: int = 0,
    ) -> ConversionDocument:
        outcome = modules["ExtractionQualityStage"]().evaluate(
            document,
            inherited_issues=inherited_issues,
        )
        selected = list(outcome.assessment.indexable_element_ids)
        if preferred_ids is not None:
            preferred = set(preferred_ids)
            selected = [
                element_id for element_id in selected if element_id in preferred
            ]
        selected_set = set(selected)
        if not selected:
            raise PipelineUnavailableError(
                "Quality gate không tìm thấy element đủ điều kiện chuẩn hóa."
            )
        progress("NORMALIZE", 76, "Đang chuẩn hóa element sang Markdown GFM")
        render_element = modules["render_element"]
        rendered: list[str] = []
        page_rendered: dict[int, list[str]] = {}
        counts: Counter[str] = Counter()
        pages: set[int] = set()
        for element in document.elements:
            if element.element_id not in selected_set:
                continue
            rendered_element = render_element(element)
            rendered.append(rendered_element)
            counts[element.element_type.value] += 1
            if element.source_anchor.page_number is not None:
                page_number = element.source_anchor.page_number
                pages.add(page_number)
                page_rendered.setdefault(page_number, []).append(rendered_element)
        markdown = clean_markdown("\n\n".join(rendered))
        page_markdown = {
            page_number: clean_markdown("\n\n".join(parts))
            for page_number, parts in page_rendered.items()
        }
        if document.page_count and page_markdown:
            markdown = merge_page_markdown(page_markdown, {})
        issues = [
            QualityIssue(
                severity=issue.severity.value,
                code=issue.reason_code,
                message=issue.message,
                page_numbers=list(issue.page_numbers),
            )
            for issue in outcome.report.issues
        ]
        disposition = outcome.report.disposition.value
        if disposition in {"FALLBACK_REQUIRED", "QUARANTINE_REQUIRED"}:
            disposition = "NEEDS_REVIEW"
        page_count = document.page_count
        coverage = (
            len(pages) / page_count
            if page_count and pages
            else 1.0
            if page_count is None
            else 0.0
        )
        metrics = DocumentMetrics(
            pages=page_count,
            elements=len(selected),
            headings=counts["HEADING"] + counts["TITLE"],
            paragraphs=counts["PARAGRAPH"] + counts["CAPTION"],
            tables=counts["TABLE"],
            lists=counts["LIST"],
            images=len(document.assets),
            markdown_characters=len(markdown),
            broken_characters=markdown.count("\ufffd"),
            provenance_coverage=coverage,
            ocr_invocations=ocr_invocations,
        )
        return ConversionDocument(
            markdown=markdown,
            metrics=metrics,
            issues=issues,
            engine_chain=[engine.value for engine in document.engine_chain],
            backend=backend,
            quality_disposition=disposition,
            source_sha256=document.source_sha256,
            assets=[path for path in assets if path.is_file()],
            page_markdown=page_markdown,
        )


def copy_assets(source_paths: list[Path], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    seen_hashes: set[str] = set()
    for source in source_paths:
        digest = sha256_file(source)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        target = destination / source.name
        counter = 1
        while target.exists():
            target = destination / f"{source.stem}-{counter}{source.suffix}"
            counter += 1
        shutil.copy2(source, target)
        copied.append(target)
    return copied
