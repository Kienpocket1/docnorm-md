from __future__ import annotations

from pathlib import Path

import pymupdf

from docnorm_web.config import Settings
from docnorm_web.math_exam import (
    MATH_REPAIR_VERSION,
    analyze_math_pages,
    merge_page_markdown,
    normalize_math_markdown,
    score_math_page,
    validate_math_markdown,
)
from docnorm_web.models import DocumentMetrics, RepairProfile
from docnorm_web.normalizer import ConversionDocument
from docnorm_web.pipeline_bridge import PipelineBridge
from docnorm_web.structural_profiles import apply_structural_profile

BROKEN_MATRIX_PAGE = r"""
## Câu 5

Cho ma trận A và tính det A.

1 2 3
4 α 6
7 8 α

A =

A. ⃝ C. $\alpha=1$
B. ⃝ D. $\alpha=2$

[Hướng dẫn giải]
"""


def test_math_risk_flags_only_structurally_risky_pages() -> None:
    pages = {
        1: BROKEN_MATRIX_PAGE,
        2: "## Câu 11\n\nChứng minh tập hợp đã cho là không gian vector.\n",
        3: "## Câu 12\n\nTrình bày lập luận và kết luận.\n",
    }
    report = analyze_math_pages(pages, forced=True)
    assert report.is_math_document is True
    assert 1 in report.flagged_pages
    assert 3 not in report.flagged_pages
    risk = score_math_page(1, BROKEN_MATRIX_PAGE)
    assert "duplicate_option_prefix" in risk.reasons
    assert "flattened_numeric_grid" in risk.reasons


def test_auto_math_detection_requires_questions_and_math_content() -> None:
    report = analyze_math_pages(
        {
            1: "## Câu 1\n\nTìm định thức của ma trận A.\n",
            2: "## Câu 2\n\nCho vector v và cơ sở B.\n",
        }
    )
    assert report.is_math_document is True
    unrelated = analyze_math_pages({1: "# Quy trình\n\nNội dung hành chính.\n"})
    assert unrelated.is_math_document is False


def test_math_normalizer_canonicalizes_questions_options_and_solution_heading() -> None:
    normalized = normalize_math_markdown(
        "Câu 8\n\nA. Một\nB. Hai\nC. Ba\nD. Bốn\n\n[Hướng dẫn giải]\nNội dung."
    )
    assert "## Câu 8" in normalized
    assert "- **A.** Một" in normalized
    assert "- **B.** Hai" in normalized
    assert "### [Hướng dẫn giải]" in normalized
    assert validate_math_markdown(normalized) == ()


def test_math_validator_blocks_runaway_repetition_from_real_regression() -> None:
    repeated = "\n\n".join(
        "+> Ý D: không đúng vì không thỏa mãn điều kiện về vector." for _ in range(40)
    )
    markdown = (
        "<!-- page: 1 -->\n\n## Câu 2\n\n"
        "- **A.** Một\n- **B.** Hai\n- **C.** Ba\n- **D.** Bốn\n\n"
        "### [Hướng dẫn giải]\n\n" + repeated
    )
    errors = validate_math_markdown(markdown)
    assert "runaway_repeated_line" in errors
    assert "runaway_repeated_block" in errors


def test_math_validator_checks_source_coverage_and_bloat() -> None:
    source = (
        "Câu 5. Cho ma trận A, tính định thức và tìm alpha. "
        "A. 1 B. 2 C. 3 D. 4. Hướng dẫn giải dùng phép biến đổi hàng."
    )
    output = "## Câu 5\n\nNội dung bị mất gần hết.\n"
    errors = validate_math_markdown(output, source_text=source * 3)
    assert "source_lexical_coverage_low" in errors


def test_three_option_exam_is_not_assumed_to_have_four_choices() -> None:
    markdown = (
        "## Câu 1\n\n- **A.** Một\n- **B.** Hai\n- **C.** Ba\n\n"
        "### [Hướng dẫn giải]\n\nĐáp án A.\n"
    )
    assert "incomplete_option_set" not in validate_math_markdown(markdown)


def test_math_validator_rejects_duplicate_labels_and_unbalanced_matrix() -> None:
    invalid = r"""
## Câu 3

A. C. Phải tồn tại phần tử đơn vị.

$A = \begin{bmatrix}1 & 2 \\ 3 & 4$
"""
    errors = validate_math_markdown(invalid)
    assert "duplicate_option_prefix" in errors
    assert "unbalanced_bmatrix" in errors


def test_page_merge_preserves_provenance_and_replaces_only_selected_page() -> None:
    merged = merge_page_markdown(
        {1: "Trang gốc một", 2: "Trang gốc hai"},
        {2: "Trang hai đã sửa"},
    )
    assert "<!-- page: 1 -->" in merged
    assert "Trang gốc một" in merged
    assert "<!-- page: 2 -->" in merged
    assert "Trang hai đã sửa" in merged
    assert "Trang gốc hai" not in merged


def test_math_profile_is_recorded_without_petrolimex_fingerprint() -> None:
    result = apply_structural_profile(
        "## Câu 1\n\nNội dung câu hỏi.\n",
        RepairProfile.MATH_EXAM,
    )
    assert result.applied == RepairProfile.MATH_EXAM
    assert result.repair_version == MATH_REPAIR_VERSION


def test_selective_vlm_replaces_only_risky_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "math.pdf"
    document = pymupdf.open()
    document.new_page()
    document.new_page()
    document.save(source)
    document.close()
    executable = tmp_path / "python.exe"
    worker = tmp_path / "worker.py"
    executable.write_bytes(b"stub")
    worker.write_text("# stub\n", encoding="utf-8")
    settings = Settings(
        product_root=tmp_path,
        project_root=tmp_path,
        runs_root=tmp_path / "runs",
        pipeline_root=tmp_path / "pipeline",
        odl_python=executable,
        odl_worker=worker,
        hybrid_worker=worker,
        hybrid_executable=executable,
        vlm_python=executable,
        math_worker=worker,
        math_max_vlm_pages=1,
    )
    bridge = PipelineBridge(settings)
    page_two = "## Câu 11\n\nTrình bày lời giải văn xuôi đầy đủ.\n"
    base = ConversionDocument(
        markdown=BROKEN_MATRIX_PAGE + "\n" + page_two,
        metrics=DocumentMetrics(pages=2, elements=2, provenance_coverage=1.0),
        issues=[],
        engine_chain=["OPENDATALOADER_LOCAL"],
        backend="RAG_PIPELINE",
        quality_disposition="NEEDS_REVIEW",
        source_sha256="unused-in-mocked-worker",
        page_markdown={1: BROKEN_MATRIX_PAGE, 2: page_two},
    )
    repaired_page = r"""
## Câu 5

Cho ma trận $A = \begin{bmatrix}1 & 2 & 3 \\ 4 & \alpha & 6 \\ 7 & 8 & \alpha\end{bmatrix}$.

- **A.** $\alpha=1$
- **B.** $\alpha=2$
- **C.** $\alpha=3$
- **D.** $\alpha=4$

### [Hướng dẫn giải]

$\det A = \alpha^2 - 29\alpha + 132$.
"""

    def fake_worker(_source, _run_directory, pages, _progress=None):
        assert pages == [1]
        return {
            "status": "PASS",
            "pages": [
                {
                    "page_number": 1,
                    "status": "PASS",
                    "markdown": repaired_page,
                }
            ],
        }

    monkeypatch.setattr(bridge, "_run_math_worker", fake_worker)
    result = bridge._maybe_apply_math_vlm(
        source,
        tmp_path / "job",
        base,
        RepairProfile.MATH_EXAM,
        lambda *_args: None,
        allow_vlm=True,
    )
    assert "\\begin{bmatrix}" in result.markdown
    assert page_two.strip() in result.markdown
    assert "A. ⃝ C." not in result.markdown
    assert result.metrics.vlm_pages == 1
    assert result.repair_profile_applied == RepairProfile.MATH_EXAM
    assert result.quality_disposition == "PASS_WITH_WARNINGS"
