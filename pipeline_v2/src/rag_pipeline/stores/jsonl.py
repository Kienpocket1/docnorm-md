from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from pydantic import BaseModel


ModelType = TypeVar(
    "ModelType",
    bound=BaseModel,
)


class JsonlStoreError(ValueError):
    """Raised when a JSONL artifact cannot be loaded."""


def write_models_jsonl(
    records: Iterable[BaseModel],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            for record in records:
                output_file.write(
                    record.model_dump_json()
                )
                output_file.write("\n")

        os.replace(
            temporary_path,
            destination,
        )
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def load_models_jsonl(
    input_path: str | Path,
    model_type: type[ModelType],
) -> list[ModelType]:
    path = Path(input_path)

    if not path.is_file():
        raise JsonlStoreError(
            f"JSONL artifact does not exist: {path}"
        )

    records: list[ModelType] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line_number, raw_line in enumerate(
            input_file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                records.append(
                    model_type.model_validate_json(line)
                )
            except Exception as error:
                raise JsonlStoreError(
                    "Invalid JSONL record at "
                    f"line {line_number}: {path.name}"
                ) from error

    return records