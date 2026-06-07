"""
ScoringModule: aggregates all detector signals into a classification state.
Never outputs CONFIRMED_ERROR — only AUTO_ACTION_SAFE or MACHINE_SUSPECTED.
"""
import logging
from typing import Optional

from kg_verify.config import (
    AUTO_ACTION_SAFE,
    AUTO_CLEAN,
    AUTO_CLEAN_SCORE,
    MACHINE_SUSPECTED,
    POTENTIALLY_REFLEXIVE_RELATIONS,
    SCORING_WEIGHTS,
)

logger = logging.getLogger(__name__)


class ScoringModule:
    """
    Combines schema, cardinality, and structural scores into a
    composite suspicion score, then assigns a classification state.
    """

    AUTO_SAFE_SIGNALS = {"self_loop", "exact_duplicate", "parse_error"}
    NON_REFLEXIVE_THRESHOLD = 0.0  # self-loop on non-reflexive relation → always safe

    def classify(
        self,
        h: str,
        r: str,
        t: str,
        schema_score: float,
        cardinality_score: float,
        structural_score: float,
        explicit_signals: Optional[list[str]] = None,
    ) -> tuple[str, float, str]:
        """
        Args:
            h, r, t: triple
            schema_score: from SchemaPatternAnalyzer.score()
            cardinality_score: from CardinalityAnalyzer.score()
            structural_score: from TopologyScorer.score()
            explicit_signals: list of signal names already flagged (e.g. "self_loop")

        Returns:
            classification: one of AUTO_CLEAN, AUTO_ACTION_SAFE, MACHINE_SUSPECTED
            composite_score: float in [0, 1]
            primary_signal: name of strongest contributing signal
        """
        explicit_signals = explicit_signals or []

        # ── AUTO_ACTION_SAFE: self-loop on non-reflexive relation ──────────────
        if "self_loop" in explicit_signals and r not in POTENTIALLY_REFLEXIVE_RELATIONS:
            return AUTO_ACTION_SAFE, 1.0, "self_loop"

        # ── AUTO_ACTION_SAFE: exact duplicate ─────────────────────────────────
        if "exact_duplicate" in explicit_signals:
            return AUTO_ACTION_SAFE, 1.0, "exact_duplicate"

        # ── Composite weighted score ──────────────────────────────────────────
        w = SCORING_WEIGHTS
        composite = (
            w["schema_violation"] * schema_score
            + w["cardinality_anomaly"] * cardinality_score
            + w["structural_score"] * structural_score
        )

        # ── AUTO_CLEAN ────────────────────────────────────────────────────────
        if composite < AUTO_CLEAN_SCORE and not explicit_signals:
            return AUTO_CLEAN, composite, "none"

        # ── Determine primary signal ──────────────────────────────────────────
        signal_scores = {
            "schema_violation": schema_score * w["schema_violation"],
            "cardinality_anomaly": cardinality_score * w["cardinality_anomaly"],
            "structural_score": structural_score * w["structural_score"],
        }
        primary = max(signal_scores, key=signal_scores.__getitem__)
        if explicit_signals:
            primary = explicit_signals[0]

        # ── MACHINE_SUSPECTED ─────────────────────────────────────────────────
        return MACHINE_SUSPECTED, composite, primary

    def classify_batch(
        self,
        triples: list[tuple[str, str, str]],
        schema_scores: dict[tuple, float],
        cardinality_scores: dict[tuple, float],
        structural_scores: dict[tuple, float],
        self_loops: set[tuple],
        duplicates: set[tuple],
        type_conflicts: set[tuple],
    ) -> list[dict]:
        """
        Classify all triples and return a list of result dicts.
        """
        results = []
        seen: set[tuple] = set()

        for h, r, t in triples:
            triple_key = (h, r, t)
            explicit: list[str] = []

            if triple_key in self_loops:
                explicit.append("self_loop")
            if triple_key in seen:
                explicit.append("exact_duplicate")
            else:
                seen.add(triple_key)
            if triple_key in type_conflicts:
                explicit.append("type_conflict")

            schema_s = schema_scores.get(triple_key, 0.0)
            card_s = cardinality_scores.get((h, r), 0.0)
            struct_s = structural_scores.get(triple_key, 0.0)

            classification, score, primary = self.classify(
                h, r, t, schema_s, card_s, struct_s, explicit
            )
            results.append({
                "triple": triple_key,
                "classification": classification,
                "composite_score": score,
                "primary_signal": primary,
                "explicit_signals": explicit,
                "schema_score": schema_s,
                "cardinality_score": card_s,
                "structural_score": struct_s,
            })

        stats = {s: sum(1 for r in results if r["classification"] == s)
                 for s in [AUTO_CLEAN, AUTO_ACTION_SAFE, MACHINE_SUSPECTED]}
        logger.info("ScoringModule: %s", stats)
        return results
