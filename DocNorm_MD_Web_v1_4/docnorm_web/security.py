from __future__ import annotations

import re
import zipfile
from pathlib import Path

from .normalizer import ConversionError, UnsupportedFileError


class InvalidDocumentError(ConversionError):
    code = "INVALID_DOCUMENT"


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_display_name(value: str | None) -> str:
    name = Path(value or "document").name.strip()
    name = _SAFE_NAME.sub("_", name)
    return name[:180] or "document"


def validate_document(path: Path, maximum_bytes: int) -> None:
    if not path.is_file() or path.stat().st_size < 1:
        raise InvalidDocumentError("Tệp tải lên rỗng")
    if path.stat().st_size > maximum_bytes:
        raise InvalidDocumentError("Tệp vượt quá giới hạn dung lượng")
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise InvalidDocumentError("Phần mở rộng PDF không khớp nội dung tệp")
        return
    if suffix == ".docx":
        if not zipfile.is_zipfile(path):
            raise InvalidDocumentError("DOCX không phải gói OpenXML hợp lệ")
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                names = {member.filename for member in members}
                if (
                    "[Content_Types].xml" not in names
                    or "word/document.xml" not in names
                ):
                    raise InvalidDocumentError("DOCX thiếu cấu trúc tài liệu bắt buộc")
                if len(members) > 10_000:
                    raise InvalidDocumentError("DOCX chứa quá nhiều thành phần")
                total_uncompressed = sum(member.file_size for member in members)
                if total_uncompressed > 500 * 1024 * 1024:
                    raise InvalidDocumentError("DOCX có kích thước giải nén bất thường")
                for member in members:
                    if member.file_size > 100 * 1024 * 1024:
                        raise InvalidDocumentError("DOCX chứa thành phần quá lớn")
                    if (
                        member.compress_size
                        and member.file_size / member.compress_size > 250
                    ):
                        raise InvalidDocumentError("DOCX có tỷ lệ nén bất thường")
        except zipfile.BadZipFile as error:
            raise InvalidDocumentError("DOCX bị hỏng") from error
        return
    raise UnsupportedFileError("Chỉ hỗ trợ tệp .pdf và .docx")
