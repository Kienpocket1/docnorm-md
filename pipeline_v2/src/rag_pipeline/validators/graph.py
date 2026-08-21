from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.contracts import (
    ElementType,
    ExtractedDocument,
    ExtractionQualityIssue,
    FailureDomain,
    QualitySeverity,
)


GRAPH_VALIDATOR_VERSION = "2026-08-18-6.1A.1"


class NarrativeCoverageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    visual_element_id: str
    sufficient: bool
    score: float = Field(ge=0, le=1)
    evidence_element_ids: list[str]
    narrative_character_count: int = Field(ge=0)
    matched_keywords: list[str]
    explicit_override: bool = False


class GraphValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validator_version: str
    valid: bool
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)
    issues: list[ExtractionQualityIssue]


@dataclass(frozen=True, slots=True)
class NarrativeCoverageConfig:
    minimum_characters: int = 60
    minimum_score: float = 0.65
    neighbor_block_distance: int = 4


def _stable_issue_id(
    source_sha256: str,
    reason_code: str,
    *parts: object,
) -> str:
    payload = "|".join(
        [source_sha256, reason_code, *(str(part) for part in parts)]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return f"quality-issue-{digest}"


class NarrativeCoverageValidator:
    KEYWORDS = (
        "lưu đồ",
        "quy trình",
        "bước",
        "nếu",
        "đạt",
        "không đạt",
        "nhánh",
        "tiếp theo",
        "flowchart",
        "process",
    )

    def __init__(
        self,
        config: NarrativeCoverageConfig | None = None,
    ) -> None:
        self.config = config or NarrativeCoverageConfig()

        if self.config.minimum_characters < 1:
            raise ValueError("minimum_characters must be positive")

        if not 0 <= self.config.minimum_score <= 1:
            raise ValueError("minimum_score must be between zero and one")

        if self.config.neighbor_block_distance < 0:
            raise ValueError(
                "neighbor_block_distance must be non-negative"
            )

    def evaluate(
        self,
        document: ExtractedDocument,
        visual_element_id: str,
    ) -> NarrativeCoverageResult:
        by_id = {
            element.element_id: element
            for element in document.elements
        }
        visual = by_id.get(visual_element_id)

        if visual is None:
            raise KeyError(
                f"Visual element not found: {visual_element_id}"
            )

        if visual.element_type not in {
            ElementType.IMAGE,
            ElementType.FLOWCHART,
        }:
            raise ValueError(
                "Narrative coverage requires IMAGE or FLOWCHART"
            )

        override = visual.metadata.get("narrative_coverage")

        if isinstance(override, bool):
            return NarrativeCoverageResult(
                visual_element_id=visual.element_id,
                sufficient=override,
                score=1.0 if override else 0.0,
                evidence_element_ids=[],
                narrative_character_count=0,
                matched_keywords=[],
                explicit_override=True,
            )

        if isinstance(override, str):
            normalized = override.strip().casefold()

            if normalized in {
                "sufficient",
                "complete",
                "pass",
                "đủ",
            }:
                return NarrativeCoverageResult(
                    visual_element_id=visual.element_id,
                    sufficient=True,
                    score=1.0,
                    evidence_element_ids=[],
                    narrative_character_count=0,
                    matched_keywords=[],
                    explicit_override=True,
                )

            if normalized in {
                "insufficient",
                "missing",
                "fail",
                "thiếu",
            }:
                return NarrativeCoverageResult(
                    visual_element_id=visual.element_id,
                    sufficient=False,
                    score=0.0,
                    evidence_element_ids=[],
                    narrative_character_count=0,
                    matched_keywords=[],
                    explicit_override=True,
                )

        candidates = []
        visual_anchor = visual.source_anchor
        visual_section = visual.metadata.get("section_index")

        for element in document.elements:
            if element.element_type not in {
                ElementType.PARAGRAPH,
                ElementType.CAPTION,
                ElementType.LIST,
                ElementType.HEADING,
            }:
                continue

            same_page = (
                visual_anchor.page_number is not None
                and element.source_anchor.page_number
                == visual_anchor.page_number
            )
            same_section = (
                visual_anchor.page_number is None
                and element.metadata.get("section_index")
                == visual_section
            )
            close_block = abs(
                element.source_anchor.block_index
                - visual_anchor.block_index
            ) <= self.config.neighbor_block_distance

            if (same_page or same_section) and close_block:
                candidates.append(element)

        narrative = "\n".join(
            element.content.strip()
            for element in candidates
            if element.content.strip()
        )
        normalized_narrative = narrative.casefold()
        matched_keywords = sorted(
            {
                keyword
                for keyword in self.KEYWORDS
                if keyword in normalized_narrative
            }
        )
        character_score = min(
            len(narrative) / max(self.config.minimum_characters, 1),
            1.0,
        )
        keyword_score = min(len(matched_keywords) / 3, 1.0)
        score = 0.6 * character_score + 0.4 * keyword_score
        sufficient = (
            len(narrative) >= self.config.minimum_characters
            and score >= self.config.minimum_score
        )
        return NarrativeCoverageResult(
            visual_element_id=visual.element_id,
            sufficient=sufficient,
            score=round(score, 6),
            evidence_element_ids=[
                element.element_id
                for element in candidates
            ],
            narrative_character_count=len(narrative),
            matched_keywords=matched_keywords,
            explicit_override=False,
        )


class GraphTopologyValidator:
    def validate(
        self,
        payload: Mapping[str, Any],
        *,
        source_sha256: str,
        page_number: int | None,
        element_id: str,
    ) -> GraphValidationResult:
        raw_nodes = payload.get("nodes")
        raw_edges = payload.get("edges")
        reason_details: list[str] = []

        if not isinstance(raw_nodes, list) or not raw_nodes:
            raw_nodes = []
            reason_details.append("nodes_missing_or_empty")

        if not isinstance(raw_edges, list):
            raw_edges = []
            reason_details.append("edges_missing")

        node_ids: list[str] = []
        decision_ids: set[str] = set()

        for node in raw_nodes:
            if not isinstance(node, Mapping):
                reason_details.append("node_not_object")
                continue

            node_id = str(node.get("id") or "").strip()

            if not node_id:
                reason_details.append("node_id_missing")
                continue

            node_ids.append(node_id)

            if str(node.get("type") or "").casefold() == "decision":
                decision_ids.add(node_id)

        if len(node_ids) != len(set(node_ids)):
            reason_details.append("duplicate_node_id")

        known_nodes = set(node_ids)
        outgoing: dict[str, int] = {
            node_id: 0
            for node_id in known_nodes
        }

        for edge in raw_edges:
            if not isinstance(edge, Mapping):
                reason_details.append("edge_not_object")
                continue

            source = str(
                edge.get("source") or edge.get("from") or ""
            ).strip()
            target = str(
                edge.get("target") or edge.get("to") or ""
            ).strip()

            if source not in known_nodes or target not in known_nodes:
                reason_details.append("edge_endpoint_missing")
                continue

            outgoing[source] += 1

        for decision_id in decision_ids:
            if outgoing.get(decision_id, 0) < 2:
                reason_details.append("decision_branch_count_invalid")

        reason_details = sorted(set(reason_details))
        issues: list[ExtractionQualityIssue] = []

        if reason_details:
            issues.append(
                ExtractionQualityIssue(
                    issue_id=_stable_issue_id(
                        source_sha256,
                        "graph_topology_invalid",
                        page_number,
                        element_id,
                        ",".join(reason_details),
                    ),
                    reason_code="graph_topology_invalid",
                    severity=QualitySeverity.ERROR,
                    failure_domain=FailureDomain.VISUAL,
                    message="Flowchart graph topology is invalid",
                    page_numbers=(
                        [page_number]
                        if page_number is not None
                        else []
                    ),
                    element_ids=[element_id],
                    triggers_fallback=True,
                    details={"violations": reason_details},
                )
            )

        return GraphValidationResult(
            validator_version=GRAPH_VALIDATOR_VERSION,
            valid=not issues,
            node_count=len(node_ids),
            edge_count=len(raw_edges),
            issues=issues,
        )
