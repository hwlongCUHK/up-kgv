"""
ReviewItem: the core data structure passed between all tiers of KG-Verify-HITL.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ReviewItem:
    triple_id: str
    original_triple: tuple[str, str, str]       # (head, relation, tail)
    head_type: str
    tail_type: str

    # Machine tier outputs
    machine_classification: str                  # one of the states from config
    detector_signals: dict = field(default_factory=dict)   # {detector_name: score}
    composite_score: float = 0.0               # aggregated suspicion score
    primary_signal: str = "none"               # strongest detector signal

    # Agent tier outputs
    agent_verdicts: list[dict] = field(default_factory=list)     # one per invoked agent
    repair_candidates: list[dict] = field(default_factory=list)  # from Agent F

    # Evidence package (for human reviewer)
    evidence_package: dict = field(default_factory=dict)

    # Review routing
    risk_level: str = "LOW"                    # LOW | MEDIUM | HIGH | CRITICAL
    review_priority: float = 0.0              # 0.0–1.0
    required_reviewer_role: str = "ANNOTATOR"  # ANNOTATOR | BIOMEDICAL_EXPERT | SENIOR_EXPERT
    allowed_actions: list[str] = field(default_factory=list)

    # Human decision (filled in by reviewer or Agent G in simulation mode)
    human_decision: Optional[str] = None
    human_corrected_head: Optional[str] = None
    human_corrected_relation: Optional[str] = None
    human_corrected_tail: Optional[str] = None
    human_reason: Optional[str] = None
    reviewer_id: Optional[str] = None
    review_timestamp: Optional[str] = None
    adjudication_status: Optional[str] = None

    # Simulation flag (set when Agent G is used instead of real human)
    simulated_review: bool = False

    @classmethod
    def create(
        cls,
        h: str,
        r: str,
        t: str,
        h_type: str,
        t_type: str,
        classification: str,
        composite_score: float,
        primary_signal: str,
        detector_signals: dict,
        evidence_package: dict,
    ) -> "ReviewItem":
        """Factory method for creating a ReviewItem from pipeline output."""
        triple_id = str(uuid.uuid4())[:8] + f"_{h[:10]}_{r[:10]}_{t[:10]}"
        return cls(
            triple_id=triple_id,
            original_triple=(h, r, t),
            head_type=h_type,
            tail_type=t_type,
            machine_classification=classification,
            detector_signals=detector_signals,
            composite_score=composite_score,
            primary_signal=primary_signal,
            evidence_package=evidence_package,
        )

    def apply_agent_verdict(self, verdict: dict) -> None:
        """Append an agent verdict to this item."""
        self.agent_verdicts.append(verdict)
        if verdict.get("repair_candidates"):
            self.repair_candidates.extend(verdict["repair_candidates"])

    def apply_simulated_review(self, decision: dict) -> None:
        """Apply Agent G's simulated human decision."""
        self.human_decision = decision.get("human_decision")
        self.human_corrected_head = decision.get("human_corrected_head")
        self.human_corrected_relation = decision.get("human_corrected_relation")
        self.human_corrected_tail = decision.get("human_corrected_tail")
        self.human_reason = decision.get("human_reason")
        self.reviewer_id = decision.get("reviewer_id", "simulated_reviewer_g")
        self.review_timestamp = datetime.utcnow().isoformat()
        self.simulated_review = True

    def to_dict(self) -> dict:
        """Serialize to dict for TSV export or JSON logging."""
        return {
            "triple_id": self.triple_id,
            "original_triple": list(self.original_triple),
            "head": self.original_triple[0],
            "relation": self.original_triple[1],
            "tail": self.original_triple[2],
            "head_type": self.head_type,
            "tail_type": self.tail_type,
            "machine_classification": self.machine_classification,
            "composite_score": self.composite_score,
            "primary_signal": self.primary_signal,
            "detector_signals": json.dumps(self.detector_signals, ensure_ascii=False),
            "agent_verdicts": json.dumps(self.agent_verdicts, ensure_ascii=False),
            "repair_candidates": json.dumps(self.repair_candidates, ensure_ascii=False),
            "evidence_package": self.evidence_package,   # keep as dict for agents
            "risk_level": self.risk_level,
            "review_priority": self.review_priority,
            "required_reviewer_role": self.required_reviewer_role,
            "human_decision": self.human_decision,
            "human_corrected_head": self.human_corrected_head,
            "human_corrected_relation": self.human_corrected_relation,
            "human_corrected_tail": self.human_corrected_tail,
            "human_reason": self.human_reason,
            "reviewer_id": self.reviewer_id,
            "review_timestamp": self.review_timestamp,
            "simulated_review": self.simulated_review,
        }
