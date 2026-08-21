from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .config import get_settings
from .models import (
    ConversionMode,
    JobRecord,
    JobStatus,
    MarkdownUpdate,
    RepairProfile,
)
from .normalizer import ConversionError
from .security import safe_display_name, validate_document
from .service import ConversionService
from .store import JobNotFoundError, JobStore

settings = get_settings()
store = JobStore(settings.runs_root)
service = ConversionService(settings, store)
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docnorm")
templates = Jinja2Templates(directory=str(settings.templates_root))

app = FastAPI(
    title="DocNorm MD",
    version="1.4.0",
    docs_url="/api/docs",
    redoc_url=None,
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["127.0.0.1", "localhost", "testserver"],
)
app.mount("/static", StaticFiles(directory=str(settings.static_root)), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self'; script-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
    )
    return response


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"max_upload_mb": settings.max_upload_bytes // 1024 // 1024},
    )


@app.get("/api/health")
def health() -> dict:
    capabilities = service.bridge.probe()
    return {
        "status": (
            "ok"
            if capabilities["rag_pipeline_available"]
            or capabilities["geometry_ocr_installed"]
            else "degraded"
        ),
        "product": "DocNorm MD",
        "version": "1.4.0",
        "supported_formats": ["pdf", "docx"],
        "max_upload_bytes": settings.max_upload_bytes,
        "capabilities": {
            key: value
            for key, value in capabilities.items()
            if key not in {"pipeline_root", "import_error"}
        },
    }


@app.post("/api/jobs", status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    mode: ConversionMode = Form(ConversionMode.AUTO),
    repair_profile: RepairProfile = Form(RepairProfile.AUTO),
) -> dict:
    display_name = safe_display_name(file.filename)
    suffix = Path(display_name).suffix.casefold()
    if suffix not in {".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ PDF và DOCX")
    if repair_profile == RepairProfile.MATH_EXAM and suffix != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Profile Đề toán hiện chỉ áp dụng cho PDF.",
        )
    job_id = uuid.uuid4().hex
    run_directory = settings.runs_root / job_id
    input_directory = run_directory / "input"
    input_directory.mkdir(parents=True, exist_ok=False)
    source_path = input_directory / f"source{suffix}"
    size = 0
    try:
        with source_path.open("xb") as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Tệp vượt quá giới hạn")
                output.write(chunk)
        validate_document(source_path, settings.max_upload_bytes)
    except HTTPException:
        shutil.rmtree(run_directory, ignore_errors=True)
        raise
    except ConversionError as error:
        shutil.rmtree(run_directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        await file.close()
    record = JobRecord(
        job_id=job_id,
        filename=display_name,
        size_bytes=size,
        content_type=file.content_type or "application/octet-stream",
        mode=mode,
        repair_profile=repair_profile,
        source_path=str(source_path),
        run_directory=str(run_directory),
    )
    store.add(record)
    background_tasks.add_task(executor.submit, service.run_job, job_id)
    return store.get(job_id).public_dict()


@app.get("/api/jobs")
def list_jobs(limit: int = 8) -> dict:
    limit = min(50, max(1, limit))
    return {"items": [job.public_dict() for job in store.list(limit)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job(job_id).public_dict()


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> dict:
    record = _completed_job(job_id)
    markdown = Path(record.markdown_path or "").read_text(encoding="utf-8")
    return {
        "job_id": job_id,
        "markdown": markdown,
        "metrics": record.metrics.model_dump(mode="json"),
        "quality": record.quality.model_dump(mode="json"),
        "downloads": {
            "markdown": f"/api/jobs/{job_id}/download/markdown",
            "bundle": f"/api/jobs/{job_id}/download/bundle",
            "report": f"/api/jobs/{job_id}/download/report",
        },
    }


@app.patch("/api/jobs/{job_id}/markdown")
def update_markdown(job_id: str, update: MarkdownUpdate) -> dict:
    try:
        return service.update_markdown(job_id, update.content)
    except JobNotFoundError as error:
        raise HTTPException(status_code=404, detail="Không tìm thấy job") from error
    except ConversionError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/jobs/{job_id}/download/{artifact}")
def download(job_id: str, artifact: str):
    record = _completed_job(job_id)
    mapping = {
        "markdown": (record.markdown_path, "text/markdown", "document.normalized.md"),
        "bundle": (record.bundle_path, "application/zip", "docnorm_result.zip"),
        "report": (record.report_path, "application/json", "quality_report.json"),
    }
    if artifact not in mapping:
        raise HTTPException(status_code=404, detail="Artifact không tồn tại")
    raw_path, media_type, filename = mapping[artifact]
    path = Path(raw_path or "")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact không tồn tại")
    return FileResponse(path, media_type=media_type, filename=filename)


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    record = _job(job_id)
    if record.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Không thể xóa job đang chạy")
    run_directory = Path(record.run_directory).resolve()
    if settings.runs_root.resolve() not in run_directory.parents:
        raise HTTPException(status_code=409, detail="Run directory không hợp lệ")
    store.delete(job_id)
    shutil.rmtree(run_directory, ignore_errors=False)


def _job(job_id: str) -> JobRecord:
    try:
        valid_id = uuid.UUID(hex=job_id).hex == job_id
    except (ValueError, AttributeError):
        valid_id = False
    if not valid_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    try:
        return store.get(job_id)
    except (JobNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Không tìm thấy job") from error


def _completed_job(job_id: str) -> JobRecord:
    record = _job(job_id)
    if record.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail="Job chưa hoàn tất")
    return record
