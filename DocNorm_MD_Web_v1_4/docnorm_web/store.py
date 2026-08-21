from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import JobRecord, JobStage, JobStatus


class JobNotFoundError(KeyError):
    pass


class JobStore:
    def __init__(self, runs_root: Path) -> None:
        self.runs_root = runs_root.resolve()
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for path in self.runs_root.glob("*/job.json"):
            try:
                record = JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if record.status in {JobStatus.QUEUED, JobStatus.RUNNING}:
                record = record.model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "stage": JobStage.DONE,
                        "progress_percent": 100,
                        "message": "Job bị gián đoạn do ứng dụng được khởi động lại.",
                        "error_code": "SERVER_RESTARTED",
                        "finished_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC),
                    }
                )
            self._jobs[record.job_id] = JobRecord.model_validate(record.model_dump())
            if (
                record.status == JobStatus.FAILED
                and record.error_code == "SERVER_RESTARTED"
            ):
                self._persist(self._jobs[record.job_id])

    def add(self, record: JobRecord) -> JobRecord:
        with self._lock:
            if record.job_id in self._jobs:
                raise ValueError(f"Job đã tồn tại: {record.job_id}")
            self._jobs[record.job_id] = record
            self._persist(record)
            return record.model_copy(deep=True)

    def get(self, job_id: str) -> JobRecord:
        with self._lock:
            try:
                return self._jobs[job_id].model_copy(deep=True)
            except KeyError as error:
                raise JobNotFoundError(job_id) from error

    def update(self, job_id: str, **changes: Any) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            changes["updated_at"] = datetime.now(UTC)
            record = self._jobs[job_id].model_copy(update=changes)
            record = JobRecord.model_validate(record.model_dump())
            self._jobs[job_id] = record
            self._persist(record)
            return record.model_copy(deep=True)

    def list(self, limit: int = 8) -> list[JobRecord]:
        with self._lock:
            ordered = sorted(
                self._jobs.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )
            return [item.model_copy(deep=True) for item in ordered[:limit]]

    def delete(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(job_id)
            return self._jobs.pop(job_id).model_copy(deep=True)

    def _persist(self, record: JobRecord) -> None:
        run_directory = Path(record.run_directory).resolve()
        if self.runs_root not in run_directory.parents:
            raise ValueError("Run directory nằm ngoài DOCNORM_RUNS_ROOT")
        run_directory.mkdir(parents=True, exist_ok=True)
        target = run_directory / "job.json"
        temporary = run_directory / "job.json.tmp"
        temporary.write_text(
            record.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
