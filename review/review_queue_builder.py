"""
ReviewQueueBuilder: assigns priority scores and reviewer roles to ReviewItems,
then sorts the queue for efficient human annotation.
"""
import logging
from typing import Optional

from kg_verify.config import HIGH_RISK_RELATIONS, PRIORITY_WEIGHTS
from .review_item import ReviewItem

logger = logging.getLogger(__name__)


class ReviewQueueBuilder:
    """Builds a prioritized, role-assigned review queue from MACHINE_SUSPECTED items.

    Args:
        priority_weights: optional override for PRIORITY_WEIGHTS config dict.
            Useful for ablation configs (G0, G1) that test different weight distributions.
        uncertain_conf_boost: H1 innovation — composite_score-weighted priority boost for
            UNCERTAIN zone items (conflicted IMPLAUSIBLE+PLAUSIBLE ensemble results).
            Empirical basis: UNCERTAIN TRUE_ERRORs have avg composite_score=0.836 vs
            FALSE_ALARMs=0.497 (discrimination=0.339). With default weights, UNCERTAIN items
            get agent_conf=0 → they are deprioritized below PLAUSIBLE items despite high
            schema signal. This boost applies: priority += uncertain_conf_boost * composite_score
            for items where Agent A's final verdict is UNCERTAIN. Default 0.0 = disabled.
    """

    def __init__(
        self,
        priority_weights: Optional[dict] = None,
        uncertain_conf_boost: float = 0.0,
    ) -> None:
        self._priority_weights = priority_weights or PRIORITY_WEIGHTS
        self._uncertain_conf_boost = uncertain_conf_boost

    def build(
        self,
        items: list[ReviewItem],
        degree_map: Optional[dict[str, int]] = None,
    ) -> list[ReviewItem]:
        """
        Assign priority and reviewer role to all items, return sorted queue.

        Args:
            items: list of ReviewItem objects from agent layer
            degree_map: optional entity -> degree dict for impact scoring
        """
        for item in items:
            item.review_priority = self.compute_priority(item, degree_map)
            item.risk_level = self._compute_risk_level(item)
            item.required_reviewer_role = self.assign_reviewer_role(item)
            item.allowed_actions = self._compute_allowed_actions(item)

        # Sort: highest priority first
        sorted_items = sorted(items, key=lambda x: x.review_priority, reverse=True)
        logger.info(
            "ReviewQueueBuilder: %d items sorted, top priority=%.4f",
            len(sorted_items),
            sorted_items[0].review_priority if sorted_items else 0.0,
        )
        return sorted_items

    def compute_priority(
        self,
        item: ReviewItem,
        degree_map: Optional[dict[str, int]] = None,
    ) -> float:
        w = self._priority_weights
        score = (
            w["composite_score"] * item.composite_score
            + w["agent_confidence"] * self._agent_confidence_score(item)
            + w["impact_score"] * self._impact_score(item, degree_map)
            + w["degree_score"] * self._degree_score(item, degree_map)
            + w["medical_risk_score"] * self._medical_risk_score(item)
            + w.get("repair_quality", 0.0) * self._repair_quality_score(item)
        )
        # H1 uncertain zone boost: give composite_score-proportional priority to
        # UNCERTAIN items (conflicted ensemble: one agent IMPLAUSIBLE, one PLAUSIBLE).
        # Motivation: UNCERTAIN TRUE_ERRORs have avg composite_score=0.836 vs FA=0.497
        # (discrimination=0.339). Without this boost, all UNCERTAIN items score near-zero
        # (agent_conf=0), preventing composite_score from surfacing suspicious ambiguous cases.
        if self._uncertain_conf_boost > 0.0 and self._is_uncertain_item(item):
            score += self._uncertain_conf_boost * item.composite_score
        return min(1.0, score)

    def _is_uncertain_item(self, item: ReviewItem) -> bool:
        """Return True if Agent A's final ensemble verdict is UNCERTAIN."""
        for v in item.agent_verdicts:
            if v.get("agent", "").startswith("A_"):
                return v.get("verdict", "") == "UNCERTAIN"
        return False

    def _repair_quality_score(self, item: ReviewItem) -> float:
        """Agent F repair quality signal: HIGH-confidence repair candidates → higher priority.

        A6 fix: Agent F (RepairSynthesizer) outputs were previously architectural
        dead weight — repair_candidates were stored but not used in ranking. This
        method contributes Agent F's evidence to the priority formula.
        """
        if not item.repair_candidates:
            return 0.0
        high_conf = sum(
            1 for rc in item.repair_candidates if rc.get("confidence") == "HIGH"
        )
        # Also check recommended_action in agent verdicts for Agent F signal
        has_review_repair = any(
            v.get("recommended_action") in ("REVIEW_REPAIR", "REVIEW_DELETE")
            and v.get("agent", "").startswith("F_")
            for v in item.agent_verdicts
        )
        base = high_conf / max(len(item.repair_candidates), 1)
        return min(1.0, base + (0.2 if has_review_repair else 0.0))

    def _agent_confidence_score(self, item: ReviewItem) -> float:
        """Priority signal from Agent A's verdict ONLY.

        Round 4 fix: restrict to Agent A (SemanticPlausibilityAgent) only.

        Root cause identified in Round 3 ablation (3 seeds):
        - Specialist agents D (ContradictionArbiter), E (ContextualCoherenceVerifier),
          F (RepairSynthesizer) give HIGH-confidence negative verdicts for legitimate
          but unusual medical relations (DOMAIN_MISMATCH, PREFER_T1, etc.).
        - Round 2's MAX fix amplified this: any single specialist HIGH:DOMAIN_MISMATCH
          for a false positive → agent_confidence = 1.0, same as a TRUE error.
        - Result: false positives in high_risk_medical and schema_borderline routes
          crowded the top-K, dropping A0 AUPRC to 0.277 (seed=789) vs A1=0.794.
        - A1 (flat Agent A only) achieves 0.750 mean AUPRC precisely because Agent A
          is well-calibrated for semantic plausibility across all relation types.

        New formula: use ONLY Agent A's highest negative verdict confidence.
        Specialist agents contribute via repair_quality signal (Agent F) and
        medical_risk_score (static relation-type signal), not via agent_confidence.
        """
        best = 0.0
        negative_verdicts = ("IMPLAUSIBLE", "DOMAIN_MISMATCH", "PREFER_T1", "PREFER_T2")
        for v in item.agent_verdicts:
            # Only trust Agent A for the primary confidence signal
            if not v.get("agent", "").startswith("A_"):
                continue
            verdict = v.get("verdict", v.get("assessment", "UNCERTAIN"))
            if verdict not in negative_verdicts:
                continue
            conf = v.get("confidence", "LOW")
            if conf == "HIGH":
                best = max(best, 1.0)
            elif conf == "MEDIUM":
                best = max(best, 0.6)
            else:
                best = max(best, 0.3)
        return best

    def _impact_score(self, item: ReviewItem, degree_map: Optional[dict] = None) -> float:
        """Hub entities affect more downstream triples → higher impact."""
        if degree_map is None:
            return 0.0
        h_deg = degree_map.get(item.original_triple[0], 0)
        t_deg = degree_map.get(item.original_triple[2], 0)
        max_deg = max(degree_map.values(), default=1)
        return min(1.0, (h_deg + t_deg) / (2 * max_deg))

    def _degree_score(self, item: ReviewItem, degree_map: Optional[dict] = None) -> float:
        """Hub entities score higher — uses config HUB_DEGREE_THRESHOLD.

        Round 2 fix: previous code hardcoded threshold=50, which in the full
        4.58M-triple KG (max_degree=33,617) evaluates True for nearly every
        entity, making this component constant noise. Now uses the config
        value (5000) so only genuine hubs receive the degree bonus.
        """
        if degree_map is None:
            return 0.0
        from kg_verify.config import HUB_DEGREE_THRESHOLD
        h_deg = degree_map.get(item.original_triple[0], 0)
        t_deg = degree_map.get(item.original_triple[2], 0)
        return 1.0 if (h_deg >= HUB_DEGREE_THRESHOLD or t_deg >= HUB_DEGREE_THRESHOLD) else 0.0

    def _medical_risk_score(self, item: ReviewItem) -> float:
        """Drug safety relations get maximum medical risk score."""
        return 1.0 if item.original_triple[1] in HIGH_RISK_RELATIONS else 0.0

    def _compute_risk_level(self, item: ReviewItem) -> str:
        rel = item.original_triple[1]
        if rel in HIGH_RISK_RELATIONS:
            return "CRITICAL"
        if any(v.get("confidence") == "HIGH" and v.get("verdict") == "IMPLAUSIBLE"
               for v in item.agent_verdicts):
            return "HIGH"
        if item.composite_score >= 0.7:
            return "HIGH"
        if item.composite_score >= 0.4:
            return "MEDIUM"
        return "LOW"

    def assign_reviewer_role(self, item: ReviewItem) -> str:
        rel = item.original_triple[1]
        if rel in HIGH_RISK_RELATIONS:
            return "SENIOR_EXPERT"
        if item.repair_candidates:
            for rc in item.repair_candidates:
                if rc.get("repair_type") in ("HEAD_REPLACE", "TAIL_REPLACE", "ENTITY_MERGE"):
                    return "BIOMEDICAL_EXPERT"
        if item.primary_signal == "type_conflict":
            return "BIOMEDICAL_EXPERT"
        return "ANNOTATOR"

    def _compute_allowed_actions(self, item: ReviewItem) -> list[str]:
        """Compute the set of human decisions that make sense for this item."""
        base = ["APPROVE_AS_IS", "REJECT_MACHINE_FLAG", "DEFER_TO_EXPERT"]
        if item.primary_signal in ("self_loop", "exact_duplicate"):
            base.append("APPROVE_DELETE")
        if any(rc.get("repair_type") == "RELATION_REPLACE" for rc in item.repair_candidates):
            base.append("APPROVE_RELATION_REPLACE")
        if any(rc.get("repair_type") in ("HEAD_REPLACE", "TAIL_REPLACE")
               for rc in item.repair_candidates):
            base.append("APPROVE_ENTITY_REPLACE")
        if any(rc.get("repair_type") == "ADD_REVERSE" for rc in item.repair_candidates):
            base.append("APPROVE_ADD_REVERSE")
        if any(rc.get("repair_type") == "ENTITY_MERGE" for rc in item.repair_candidates):
            base.append("APPROVE_ENTITY_MERGE")
        # Always allow delete as fallback
        if "APPROVE_DELETE" not in base:
            base.append("APPROVE_DELETE")
        return base
