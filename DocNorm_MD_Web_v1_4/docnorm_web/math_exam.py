from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

MATH_REPAIR_VERSION = "2026-08-21-1.4.0"
MATH_RESOLVED_ISSUE_CODES = (
    "math_two_column_reading_order",
    "math_duplicate_option_labels",
    "math_matrix_label_alignment",
    "math_orphan_matrix_glyphs",
    "math_solution_heading_consistency",
    "math_runaway_generation",
    "math_source_coverage",
)

_QUESTION_RE = re.compile(r"(?mi)^\s*(?:#{1,6}\s*)?(?:\*{0,2})?Câu\s+(\d+)\b")
_OPTION_RE = re.compile(r"(?mi)^\s*(?:[-*+]\s*)?(?:\*{0,2})?([A-D])[.)]\*{0,2}\s+")
_DUPLICATE_OPTION_RE = re.compile(
    r"(?mi)^\s*(?:[-*+]\s*)?(?:\*{0,2})?([A-D])[.)]\*{0,2}\s+"
    r"(?:[◯○⃝]\s*)?(?:\*{0,2})?([A-D])[.)]"
)
_MATH_TERMS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bma trận\b",
        r"\bđịnh thức\b",
        r"\bdet\s*[A-Za-z]?\b",
        r"\bvector\b|\bvectơ\b",
        r"\bspan\s*\(",
        r"\bhệ phương trình\b",
        r"\bkhông gian vector\b|\bkhông gian vectơ\b",
        r"\b(?:F|v)\s*[_\[]?\d",
        r"\[[^\]\n]{1,8}\]_[A-Za-z0-9]",
        r"\\begin\{(?:bmatrix|pmatrix|vmatrix|matrix)\}",
        r"[αβγλμ]|\\(?:alpha|beta|gamma|lambda|mu)\b",
    )
)
_ORPHAN_GLYPH_LINE_RE = re.compile(
    r"(?mi)^\s*(?:ï|ò|ó|T|⌈|⌉|⌊|⌋|⎡|⎤|⎣|⎦)(?:\s+(?:ï|ò|ó|T))*\s*$"
)
_SOLUTION_HEADING_RE = re.compile(
    r"(?mi)^\s*(?:#{1,6}\s*)?\[?Hướng\s+dẫn\s+giải\]?\s*:?[ \t]*$"
)
_PAGE_MARKER_RE = re.compile(r"(?m)^<!-- page: (\d+) -->\s*$")
_IGNORABLE_REPEAT_RE = re.compile(r"^(?:\$\$|---+|<!--.*-->|\|[\s|:\-]+\|)$")


@dataclass(frozen=True, slots=True)
class MathPageRisk:
    page_number: int
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MathRiskReport:
    is_math_document: bool
    question_count: int
    page_risks: tuple[MathPageRisk, ...]
    flagged_pages: tuple[int, ...]


def _clean(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def _option_order_is_suspicious(markdown: str) -> bool:
    blocks = _QUESTION_RE.split(markdown)
    bodies = blocks[2::2] if len(blocks) >= 3 else [markdown]
    for body in bodies:
        labels = [match.group(1).upper() for match in _OPTION_RE.finditer(body)]
        if len(labels) < 3:
            continue
        numeric = [ord(label) - ord("A") for label in labels]
        if len(set(labels)) != len(labels) or numeric != sorted(numeric):
            return True
    return False


def _has_numeric_grid(markdown: str) -> bool:
    consecutive = 0
    for line in markdown.splitlines():
        tokens = re.findall(r"(?<!\w)[+−-]?(?:\d+(?:[.,]\d+)?|[αβγλμ])(?!\w)", line)
        if len(tokens) >= 3 and "|" not in line:
            consecutive += 1
            if consecutive >= 2:
                return True
        elif line.strip():
            consecutive = 0
    return False


def score_math_page(page_number: int, markdown: str) -> MathPageRisk:
    text = _clean(markdown)
    reasons: list[str] = []
    score = 0
    questions = len(_QUESTION_RE.findall(text))
    if questions:
        score += min(3, questions + 1)
        reasons.append("question_numbering")

    term_matches = sum(pattern.search(text) is not None for pattern in _MATH_TERMS)
    if term_matches:
        score += min(4, term_matches)
        reasons.append("math_or_matrix_terms")

    option_count = len(_OPTION_RE.findall(text))
    if option_count >= 4:
        score += 2
        reasons.append("multiple_choice_options")
    elif option_count >= 2:
        score += 1

    if _DUPLICATE_OPTION_RE.search(text):
        score += 5
        reasons.append("duplicate_option_prefix")
    if _option_order_is_suspicious(text):
        score += 4
        reasons.append("noncanonical_option_order")
    if _ORPHAN_GLYPH_LINE_RE.search(text):
        score += 4
        reasons.append("orphan_matrix_glyph")
    if _has_numeric_grid(text):
        score += 3
        reasons.append("flattened_numeric_grid")
    if "[Hướng dẫn giải]" in text and not re.search(
        r"(?mi)^\s*#{2,3}\s+\[Hướng dẫn giải\]\s*$", text
    ):
        score += 2
        reasons.append("inconsistent_solution_heading")

    return MathPageRisk(
        page_number=page_number,
        score=score,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def analyze_math_pages(
    page_markdown: Mapping[int, str],
    *,
    forced: bool = False,
) -> MathRiskReport:
    risks = tuple(
        score_math_page(page_number, markdown)
        for page_number, markdown in sorted(page_markdown.items())
    )
    question_count = sum(
        len(_QUESTION_RE.findall(_clean(markdown)))
        for markdown in page_markdown.values()
    )
    pages_with_math_terms = sum(
        any(pattern.search(markdown) for pattern in _MATH_TERMS)
        for markdown in page_markdown.values()
    )
    is_math_document = forced or (question_count >= 2 and pages_with_math_terms >= 1)
    threshold = 3 if forced else 6
    flagged = tuple(
        risk.page_number
        for risk in risks
        if is_math_document and risk.score >= threshold
    )
    return MathRiskReport(
        is_math_document=is_math_document,
        question_count=question_count,
        page_risks=risks,
        flagged_pages=flagged,
    )


def normalize_math_markdown(markdown: str) -> str:
    text = _clean(markdown).strip()
    text = re.sub(r"^```(?:markdown|md)?\s*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n```\s*$", "", text)
    text = _SOLUTION_HEADING_RE.sub("### [Hướng dẫn giải]", text)
    text = re.sub(
        r"(?mi)^\s*(?:#{1,6}\s*)?Câu\s+(\d+)\s*[:.]?\s*$",
        r"## Câu \1",
        text,
    )
    text = re.sub(
        r"(?mi)^\s*(?:[-*+]\s*)?([A-D])[.)]\s+(.+)$",
        lambda match: f"- **{match.group(1).upper()}.** {match.group(2).strip()}",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text + "\n" if text else ""


def _fold_lexical(value: str) -> list[str]:
    text = unicodedata.normalize("NFD", value.casefold())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    return re.findall(r"[a-z0-9]+", text)


def _content_lines(markdown: str) -> list[str]:
    output: list[str] = []
    for raw_line in _clean(markdown).splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line or _IGNORABLE_REPEAT_RE.fullmatch(line):
            continue
        output.append(line.casefold())
    return output


def _repetition_errors(markdown: str) -> list[str]:
    """Detect model degeneration without rejecting normal repeated syntax.

    Validation is page-local when provenance markers exist. This prevents a
    legitimate form header repeated on several pages from looking like a model
    loop while still catching a sentence or a multi-line solution emitted over
    and over on one page.
    """

    text = _clean(markdown)
    parts = _PAGE_MARKER_RE.split(text)
    bodies = parts[2::2] if len(parts) >= 3 else [text]
    errors: list[str] = []
    for body in bodies:
        lines = _content_lines(body)
        meaningful = [line for line in lines if len(line) >= 24]
        counts = Counter(meaningful)
        if counts and max(counts.values()) >= 4:
            errors.append("runaway_repeated_line")

        # A repeated solution block can change one trailing line and evade the
        # exact-line threshold. Three-line shingles catch that failure mode.
        shingles = Counter(
            tuple(lines[index : index + 3])
            for index in range(max(0, len(lines) - 2))
            if sum(len(item) for item in lines[index : index + 3]) >= 120
        )
        if shingles and max(shingles.values()) >= 3:
            errors.append("runaway_repeated_block")
    return errors


def _source_alignment_errors(markdown: str, source_text: str) -> list[str]:
    source = _clean(source_text).strip()
    if not source:
        return []
    output = _clean(markdown)
    errors: list[str] = []

    source_questions = set(_QUESTION_RE.findall(source))
    output_questions = set(_QUESTION_RE.findall(output))
    if source_questions and not source_questions.issubset(output_questions):
        errors.append("source_question_missing")
    source_options = {match.group(1).upper() for match in _OPTION_RE.finditer(source)}
    output_options = {match.group(1).upper() for match in _OPTION_RE.finditer(output)}
    if source_options == {"A", "B", "C", "D"} and not source_options.issubset(
        output_options
    ):
        errors.append("source_option_missing")

    source_tokens = _fold_lexical(source)
    output_tokens = _fold_lexical(output)
    if len(source_tokens) >= 30:
        source_counts = Counter(source_tokens)
        output_counts = Counter(output_tokens)
        matched = sum(
            min(count, output_counts.get(token, 0))
            for token, count in source_counts.items()
        )
        recall = matched / max(1, sum(source_counts.values()))
        if recall < 0.58:
            errors.append("source_lexical_coverage_low")
        if len(output_tokens) > max(260, int(len(source_tokens) * 2.15)):
            errors.append("output_bloated_vs_source")
    return errors


def validate_math_markdown(
    markdown: str,
    *,
    source_text: str = "",
) -> tuple[str, ...]:
    text = _clean(markdown)
    errors: list[str] = []
    if len(text.strip()) < 20:
        errors.append("output_too_short")
    if "\ufffd" in text:
        errors.append("replacement_character")
    if "```" in text:
        errors.append("markdown_code_fence")
    if _DUPLICATE_OPTION_RE.search(text):
        errors.append("duplicate_option_prefix")
    if _option_order_is_suspicious(text):
        errors.append("noncanonical_option_order")
    if _ORPHAN_GLYPH_LINE_RE.search(text):
        errors.append("orphan_matrix_glyph")
    errors.extend(_repetition_errors(text))
    errors.extend(_source_alignment_errors(text, source_text))
    question_numbers = [int(value) for value in _QUESTION_RE.findall(text)]
    if len(question_numbers) != len(set(question_numbers)):
        errors.append("duplicate_question_block")
    if question_numbers and question_numbers != sorted(question_numbers):
        errors.append("nonmonotonic_question_order")
    question_parts = _QUESTION_RE.split(text)
    option_sets = [
        {match.group(1).upper() for match in _OPTION_RE.finditer(body)}
        for body in question_parts[2::2]
    ]
    expected_four_options = any(
        labels == {"A", "B", "C", "D"} for labels in option_sets
    )
    if expected_four_options and any(
        labels and labels != {"A", "B", "C", "D"} for labels in option_sets
    ):
        errors.append("incomplete_option_set")
    for environment in ("bmatrix", "pmatrix", "vmatrix", "matrix", "aligned"):
        if text.count(f"\\begin{{{environment}}}") != text.count(
            f"\\end{{{environment}}}"
        ):
            errors.append(f"unbalanced_{environment}")
    if text.count("$$") % 2:
        errors.append("unbalanced_display_math")
    return tuple(dict.fromkeys(errors))


def merge_page_markdown(
    page_markdown: Mapping[int, str],
    replacements: Mapping[int, str],
    *,
    include_pages: Iterable[int] | None = None,
) -> str:
    pages = sorted(include_pages if include_pages is not None else page_markdown)
    sections: list[str] = []
    for page_number in pages:
        content = replacements.get(page_number, page_markdown.get(page_number, ""))
        content = content.strip()
        if not content:
            continue
        sections.append(f"<!-- page: {page_number} -->\n\n{content}")
    return "\n\n".join(sections).strip() + "\n" if sections else ""
