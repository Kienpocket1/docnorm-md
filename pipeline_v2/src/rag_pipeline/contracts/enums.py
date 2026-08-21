from enum import StrEnum


class QuarantineType(StrEnum):
    """Three official quarantine queues."""

    RESOLVER = "RESOLVER"
    EXTRACTION = "EXTRACTION"
    INDEXING = "INDEXING"


class ResumeFrom(StrEnum):
    """Executable stages from which a quarantined run can resume."""

    CATALOG = "CATALOG"
    RESOLVE = "RESOLVE"
    VERSION = "VERSION"
    ROUTE = "ROUTE"
    EXTRACT = "EXTRACT"
    EXTRACTION_QUALITY = "EXTRACTION_QUALITY"
    NORMALIZE = "NORMALIZE"
    CHUNK = "CHUNK"
    INDEX_VALIDATE = "INDEX_VALIDATE"
    PUBLISH = "PUBLISH"


class ResolutionOutcome(StrEnum):
    SAME_VERSION = "SAME_VERSION"
    NEW_VERSION = "NEW_VERSION"
    UNRELATED = "UNRELATED"
    CONFLICT = "CONFLICT"
    REJECTED = "REJECTED"


class FileFormat(StrEnum):
    PDF = "PDF"
    DOCX = "DOCX"
    DOC = "DOC"
    IMAGE = "IMAGE"
    UNKNOWN = "UNKNOWN"


class StageStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PASS = "PASS"
    FAIL = "FAIL"
    QUARANTINED = "QUARANTINED"
    REJECTED = "REJECTED"
    SKIPPED = "SKIPPED"


class FailureDomain(StrEnum):
    """Diagnostic subtype, not a quarantine queue."""

    CATALOG = "CATALOG"
    RESOLUTION = "RESOLUTION"
    ROUTING = "ROUTING"
    TEXT = "TEXT"
    TABLE = "TABLE"
    VISUAL = "VISUAL"
    PROVENANCE = "PROVENANCE"
    CHUNK = "CHUNK"
    INDEX = "INDEX"
    PUBLISH = "PUBLISH"
    SECURITY = "SECURITY"


class PublicationState(StrEnum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    REJECTED = "REJECTED"


class QuarantineStatus(StrEnum):
    OPEN = "OPEN"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"


class RouteType(StrEnum):
    PDF_OPENDATALOADER = "PDF_OPENDATALOADER"
    DOCX_NATIVE = "DOCX_NATIVE"
    DOC_CONVERT = "DOC_CONVERT"
    IMAGE_VISUAL = "IMAGE_VISUAL"
    UNSUPPORTED = "UNSUPPORTED"

class EngineName(StrEnum):
    OPENDATALOADER_LOCAL = "OPENDATALOADER_LOCAL"
    OPENDATALOADER_HYBRID = "OPENDATALOADER_HYBRID"
    DOCX_NATIVE = "DOCX_NATIVE"
    LIBREOFFICE = "LIBREOFFICE"
    QWEN3_VL = "QWEN3_VL"
    MINERU = "MINERU"
    MANUAL = "MANUAL"


class AssetClassification(StrEnum):
    """Whether an extracted visual carries indexable document meaning."""

    INFORMATIVE = "INFORMATIVE"
    DECORATIVE = "DECORATIVE"
    UNKNOWN = "UNKNOWN"


class AssetRole(StrEnum):
    """Semantic role assigned to one canonical extracted asset."""

    BRAND_LOGO = "BRAND_LOGO"
    FLOWCHART = "FLOWCHART"
    TABLE_IMAGE = "TABLE_IMAGE"
    FIGURE = "FIGURE"
    SIGNATURE = "SIGNATURE"
    PAGE_DECORATION = "PAGE_DECORATION"
    UNKNOWN = "UNKNOWN"


class QualitySeverity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class QualityDisposition(StrEnum):
    """Decision emitted by the extraction-quality stage."""

    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FALLBACK_REQUIRED = "FALLBACK_REQUIRED"
    QUARANTINE_REQUIRED = "QUARANTINE_REQUIRED"


class ElementType(StrEnum):
    TITLE = "TITLE"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST = "LIST"
    TABLE = "TABLE"
    IMAGE = "IMAGE"
    FLOWCHART = "FLOWCHART"
    CAPTION = "CAPTION"
    FORMULA = "FORMULA"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    UNKNOWN = "UNKNOWN"

class VersionAction(StrEnum):
    CREATE_FAMILY_AND_VERSION = (
        "CREATE_FAMILY_AND_VERSION"
    )
    CREATE_VERSION = "CREATE_VERSION"
    REUSE_VERSION = "REUSE_VERSION"
