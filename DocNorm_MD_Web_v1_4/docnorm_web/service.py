from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .math_exam import validate_math_markdown
from .models import (
    JobRecord,
    JobStage,
    JobStatus,
    QualitySummary,
    RepairProfile,
)
from .normalizer import ConversionError, OutputQualityError
from .pipeline_bridge import PipelineBridge, copy_assets
from .store import JobStore
from .structural_profiles import validate_petrolimex_materials_markdown


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConversionService:
    def __init__(self, settings: Settings, store: JobStore) -> None:
        self.settings = settings
        self.store = store
        self.bridge = PipelineBridge(settings)

    def run_job(self, job_id: str) -> None:
        started = time.perf_counter()
        now = datetime.now(UTC)
        self.store.update(
            job_id,
            status=JobStatus.RUNNING,
            stage=JobStage.EXTRACT,
            progress_percent=18,
            message="Đang chuẩn bị bộ trích xuất",
            started_at=now,
        )
        record = self.store.get(job_id)
        source = Path(record.source_path)
        run_directory = Path(record.run_directory)

        def progress(stage: str, percent: int, message: str) -> None:
            stage_map = {
                "EXTRACT": JobStage.EXTRACT,
                "VERIFY": JobStage.VERIFY,
                "NORMALIZE": JobStage.NORMALIZE,
            }
            self.store.update(
                job_id,
                stage=stage_map.get(stage, JobStage.EXTRACT),
                progress_percent=percent,
                message=message,
                elapsed_seconds=time.perf_counter() - started,
            )

        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(2.0):
                try:
                    current = self.store.get(job_id)
                    if current.status != JobStatus.RUNNING:
                        return
                    self.store.update(
                        job_id,
                        elapsed_seconds=time.perf_counter() - started,
                    )
                except Exception:
                    return

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"docnorm-heartbeat-{job_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()

        try:
            converted = self.bridge.convert(
                source,
                run_directory,
                record.mode,
                record.repair_profile,
                progress,
            )
            self.store.update(
                job_id,
                stage=JobStage.VERIFY,
                progress_percent=68,
                message="Đang tổng hợp báo cáo chất lượng",
            )
            if converted.metrics.broken_characters:
                raise ConversionError(
                    "Markdown còn chứa ký tự thay thế U+FFFD; kết quả không được công bố."
                )
            if converted.repair_profile_applied == RepairProfile.MATH_EXAM:
                math_errors = validate_math_markdown(converted.markdown)
                if math_errors:
                    raise OutputQualityError(
                        "Đầu ra toán bị chặn bởi hậu kiểm: "
                        + ", ".join(math_errors)
                        + ". Nội dung lỗi không được công bố để tránh đưa rác vào chunking."
                    )
            fail_closed_backend = converted.backend.startswith(
                (
                    "QWEN3_VL_MATH",
                    "EASYOCR_GEOMETRY",
                    "RAG_PIPELINE_HYBRID_OCR",
                )
            )
            if fail_closed_backend and converted.quality_disposition in {
                "NEEDS_REVIEW",
                "FALLBACK_REQUIRED",
                "QUARANTINE_REQUIRED",
            }:
                raise OutputQualityError(
                    "Một hoặc nhiều trang không vượt kiểm định; DocNorm đã dừng "
                    "trước khi xuất Markdown không đáng tin cậy."
                )
            if fail_closed_backend and converted.metrics.pages:
                marker_count = len(
                    {
                        int(value)
                        for value in re.findall(
                            r"(?m)^<!-- page: (\d+) -->\s*$",
                            converted.markdown,
                        )
                    }
                )
                if marker_count != converted.metrics.pages:
                    raise OutputQualityError(
                        "Provenance theo trang không đầy đủ; kết quả không được công bố."
                    )
            output_directory = run_directory / "output"
            output_directory.mkdir(parents=True, exist_ok=True)
            markdown_path = output_directory / "document.normalized.md"
            markdown_path.write_text(converted.markdown, encoding="utf-8")
            copied_assets = copy_assets(
                converted.assets,
                output_directory / "assets",
            )
            converted.metrics.images = max(
                converted.metrics.images,
                len(copied_assets),
            )
            self.store.update(
                job_id,
                stage=JobStage.NORMALIZE,
                progress_percent=86,
                message="Markdown đã chuẩn hóa, đang đóng gói kết quả",
                metrics=converted.metrics,
                quality=QualitySummary(
                    disposition=converted.quality_disposition,
                    issues=converted.issues,
                    engine_chain=converted.engine_chain,
                    repair_profile_requested=converted.repair_profile_requested,
                    repair_profile_applied=converted.repair_profile_applied,
                    repair_version=converted.repair_version,
                    repair_fingerprint_score=converted.repair_fingerprint_score,
                    resolved_issue_codes=converted.resolved_issue_codes,
                ),
            )
            report_path, bundle_path = self._package(
                job_id=job_id,
                converted=converted,
                output_directory=output_directory,
                markdown_path=markdown_path,
                assets=copied_assets,
                elapsed_seconds=time.perf_counter() - started,
            )
            finished = datetime.now(UTC)
            self.store.update(
                job_id,
                status=JobStatus.COMPLETED,
                stage=JobStage.DONE,
                progress_percent=100,
                message="Chuẩn hóa hoàn tất",
                finished_at=finished,
                elapsed_seconds=time.perf_counter() - started,
                markdown_path=str(markdown_path),
                report_path=str(report_path),
                bundle_path=str(bundle_path),
                metrics=converted.metrics,
                quality=QualitySummary(
                    disposition=converted.quality_disposition,
                    issues=converted.issues,
                    engine_chain=converted.engine_chain,
                    repair_profile_requested=converted.repair_profile_requested,
                    repair_profile_applied=converted.repair_profile_applied,
                    repair_version=converted.repair_version,
                    repair_fingerprint_score=converted.repair_fingerprint_score,
                    resolved_issue_codes=converted.resolved_issue_codes,
                ),
            )
        except Exception as error:
            code = getattr(error, "code", "CONVERSION_FAILED")
            self.store.update(
                job_id,
                status=JobStatus.FAILED,
                stage=JobStage.DONE,
                progress_percent=100,
                message=self._public_error(error),
                finished_at=datetime.now(UTC),
                elapsed_seconds=time.perf_counter() - started,
                error_code=code,
                error_detail=f"{type(error).__name__}: {error}",
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2.5)

    def update_markdown(self, job_id: str, content: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record.status != JobStatus.COMPLETED or not record.markdown_path:
            raise ConversionError("Job chưa có Markdown để chỉnh sửa")
        markdown_path = Path(record.markdown_path)
        normalized = content.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
        if "\ufffd" in normalized:
            raise ConversionError("Không thể lưu Markdown còn ký tự U+FFFD")
        if record.quality.repair_profile_applied == RepairProfile.PETROLIMEX_MATERIALS:
            validate_petrolimex_materials_markdown(normalized)
        if record.quality.repair_profile_applied == RepairProfile.MATH_EXAM:
            math_errors = validate_math_markdown(normalized)
            if math_errors:
                raise ConversionError(
                    "Markdown toán không vượt kiểm định: " + ", ".join(math_errors)
                )
        markdown_path.write_text(normalized, encoding="utf-8")
        metrics = record.metrics.model_copy(
            update={
                "markdown_characters": len(normalized),
                "broken_characters": 0,
            }
        )
        report_path = Path(record.report_path or "")
        if report_path.is_file():
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            payload["metrics"] = metrics.model_dump(mode="json")
            payload["edited_by_user"] = True
            payload["markdown_sha256"] = _sha256(markdown_path)
            report_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        self._rebuild_bundle(record, markdown_path, report_path)
        self.store.update(job_id, metrics=metrics)
        return {
            "saved_at": datetime.now(UTC).isoformat(),
            "sha256": _sha256(markdown_path),
        }

    def _package(
        self,
        *,
        job_id: str,
        converted: Any,
        output_directory: Path,
        markdown_path: Path,
        assets: list[Path],
        elapsed_seconds: float,
    ) -> tuple[Path, Path]:
        report_path = output_directory / "quality_report.json"
        report = {
            "schema_version": "1.0",
            "product": "DocNorm MD",
            "job_id": job_id,
            "backend": converted.backend,
            "source_sha256": converted.source_sha256,
            "quality_disposition": converted.quality_disposition,
            "metrics": converted.metrics.model_dump(mode="json"),
            "issues": [issue.model_dump(mode="json") for issue in converted.issues],
            "engine_chain": converted.engine_chain,
            "structural_profile": {
                "requested": converted.repair_profile_requested.value,
                "applied": converted.repair_profile_applied.value,
                "repair_version": converted.repair_version,
                "fingerprint_score": converted.repair_fingerprint_score,
                "resolved_issue_codes": converted.resolved_issue_codes,
            },
            "external_calls": 0,
            "elapsed_seconds": elapsed_seconds,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifacts = [markdown_path, report_path, *assets]
        manifest_path = output_directory / "manifest.json"
        manifest = {
            "schema_version": "1.0",
            "job_id": job_id,
            "source_sha256": converted.source_sha256,
            "artifacts": [
                {
                    "path": path.relative_to(output_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in artifacts
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        bundle_path = output_directory / "docnorm_result.zip"
        self._write_zip(bundle_path, output_directory, [*artifacts, manifest_path])
        return report_path, bundle_path

    def _rebuild_bundle(
        self,
        record: JobRecord,
        markdown_path: Path,
        report_path: Path,
    ) -> None:
        if not record.bundle_path:
            return
        output_directory = markdown_path.parent
        assets = [
            path for path in (output_directory / "assets").glob("*") if path.is_file()
        ]
        manifest_path = output_directory / "manifest.json"
        artifacts = [markdown_path, report_path, *assets]
        manifest = {
            "schema_version": "1.0",
            "job_id": record.job_id,
            "source_sha256": _sha256(Path(record.source_path)),
            "artifacts": [
                {
                    "path": path.relative_to(output_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in artifacts
                if path.is_file()
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._write_zip(
            Path(record.bundle_path),
            output_directory,
            [path for path in [*artifacts, manifest_path] if path.is_file()],
        )

    @staticmethod
    def _write_zip(bundle: Path, root: Path, files: list[Path]) -> None:
        temporary = bundle.with_suffix(".zip.tmp")
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in files:
                resolved = path.resolve()
                if root.resolve() not in resolved.parents:
                    raise ConversionError("Artifact nằm ngoài thư mục kết quả")
                archive.write(resolved, resolved.relative_to(root).as_posix())
        temporary.replace(bundle)

    @staticmethod
    def _public_error(error: Exception) -> str:
        if isinstance(error, ConversionError):
            return str(error)
        return "Chuẩn hóa thất bại. Mở mục chẩn đoán để kiểm tra môi trường."
