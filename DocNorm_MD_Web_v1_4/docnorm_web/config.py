from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    product_root: Path
    project_root: Path
    runs_root: Path
    pipeline_root: Path
    odl_python: Path
    odl_worker: Path
    hybrid_worker: Path
    hybrid_executable: Path
    vlm_python: Path
    math_worker: Path
    scan_python: Path | None = None
    scan_worker: Path | None = None
    hybrid_url: str = "http://127.0.0.1:5002"
    hybrid_timeout_ms: int = 3_600_000
    hybrid_process_timeout_seconds: float = 7_200.0
    host: str = "127.0.0.1"
    port: int = 8010
    max_upload_bytes: int = 100 * 1024 * 1024
    pipeline_timeout_seconds: float = 900.0
    math_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    math_max_vlm_pages: int = 8
    math_target_long_side: int = 1900
    math_max_new_tokens: int = 1800
    math_worker_timeout_seconds: float = 3600.0
    scan_target_long_side: int = 2200
    scan_worker_timeout_seconds: float = 7200.0

    @property
    def templates_root(self) -> Path:
        return self.product_root / "docnorm_web" / "templates"

    @property
    def static_root(self) -> Path:
        return self.product_root / "docnorm_web" / "static"

    @property
    def pipeline_source_root(self) -> Path:
        return self.pipeline_root / "src"

    def prepare(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    product_root = Path(__file__).resolve().parents[1]
    default_project_root = product_root.parent
    project_root = _env_path("DOCNORM_PROJECT_ROOT", default_project_root)
    pipeline_root = _env_path(
        "DOCNORM_PIPELINE_ROOT",
        project_root / "pipeline_v2",
    )
    settings = Settings(
        product_root=product_root,
        project_root=project_root,
        runs_root=_env_path("DOCNORM_RUNS_ROOT", product_root / "runs"),
        pipeline_root=pipeline_root,
        odl_python=_env_path(
            "DOCNORM_ODL_PYTHON",
            project_root / "venv_opendataloader" / "Scripts" / "python.exe",
        ),
        odl_worker=_env_path(
            "DOCNORM_ODL_WORKER",
            pipeline_root
            / "src"
            / "rag_pipeline"
            / "workers"
            / "opendataloader_local_worker.py",
        ),
        hybrid_worker=_env_path(
            "DOCNORM_ODL_HYBRID_WORKER",
            product_root / "docnorm_web" / "opendataloader_hybrid_document_worker.py",
        ),
        hybrid_executable=_env_path(
            "DOCNORM_ODL_HYBRID_EXECUTABLE",
            project_root
            / "venv_opendataloader"
            / "Scripts"
            / "opendataloader-pdf-hybrid.exe",
        ),
        vlm_python=_env_path(
            "DOCNORM_VLM_PYTHON",
            project_root / "venv" / "Scripts" / "python.exe",
        ),
        math_worker=_env_path(
            "DOCNORM_MATH_WORKER",
            product_root / "docnorm_web" / "qwen_math_worker.py",
        ),
        scan_python=_env_path(
            "DOCNORM_SCAN_PYTHON",
            project_root / "venv" / "Scripts" / "python.exe",
        ),
        scan_worker=_env_path(
            "DOCNORM_SCAN_WORKER",
            product_root / "docnorm_web" / "easyocr_geometry_worker.py",
        ),
        hybrid_url=os.getenv(
            "DOCNORM_ODL_HYBRID_URL",
            "http://127.0.0.1:5002",
        ),
        hybrid_timeout_ms=int(os.getenv("DOCNORM_ODL_HYBRID_TIMEOUT_MS", "3600000")),
        hybrid_process_timeout_seconds=float(
            os.getenv("DOCNORM_ODL_HYBRID_PROCESS_TIMEOUT_SECONDS", "7200")
        ),
        host=os.getenv("DOCNORM_HOST", "127.0.0.1"),
        port=int(os.getenv("DOCNORM_PORT", "8010")),
        max_upload_bytes=int(
            os.getenv("DOCNORM_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024))
        ),
        pipeline_timeout_seconds=float(
            os.getenv("DOCNORM_PIPELINE_TIMEOUT_SECONDS", "900")
        ),
        math_model_id=os.getenv(
            "DOCNORM_MATH_MODEL_ID",
            "Qwen/Qwen3-VL-2B-Instruct",
        ),
        math_max_vlm_pages=max(
            1,
            int(os.getenv("DOCNORM_MATH_MAX_VLM_PAGES", "8")),
        ),
        math_target_long_side=max(
            1000,
            int(os.getenv("DOCNORM_MATH_TARGET_LONG_SIDE", "1900")),
        ),
        math_max_new_tokens=max(
            512,
            int(os.getenv("DOCNORM_MATH_MAX_NEW_TOKENS", "1800")),
        ),
        math_worker_timeout_seconds=float(
            os.getenv("DOCNORM_MATH_WORKER_TIMEOUT_SECONDS", "3600")
        ),
        scan_target_long_side=max(
            1400,
            int(os.getenv("DOCNORM_SCAN_TARGET_LONG_SIDE", "2200")),
        ),
        scan_worker_timeout_seconds=float(
            os.getenv("DOCNORM_SCAN_WORKER_TIMEOUT_SECONDS", "7200")
        ),
    )
    settings.prepare()
    return settings
