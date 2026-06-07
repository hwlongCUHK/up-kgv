"""
CardinalityAnalyzer: flags heads with far more tail entities than typical
for a given relation AND entity type.

Domain-adjusted cardinality: compares a head's tail count against the median
among other heads of the SAME entity type for the same relation. This prevents
flagging legitimate high-frequency procedures (e.g., TACE treating many cancers)
as anomalies simply because they have many tails for "治疗".
"""
import logging
from collections import defaultdict
from statistics import median
from typing import Optional

from kg_verify.config import CARDINALITY_MULTIPLIER

logger = logging.getLogger(__name__)


class CardinalityAnalyzer:
    """Detects (head, relation) pairs with anomalously high tail counts.

    When entity_types are provided at fit() time, uses per-(relation, head_type)
    medians (domain-adjusted). Falls back to per-relation median otherwise.
    """

    def __init__(self, multiplier: float = CARDINALITY_MULTIPLIER) -> None:
        self.multiplier = multiplier
        self._relation_tails: dict[str, dict[str, set[str]]] = {}
        self._median_cardinality: dict[str, float] = {}
        # typed: (relation, head_type) → median tail count
        self._typed_median: dict[tuple[str, str], float] = {}
        self._entity_types: dict[str, str] = {}
        self._fitted = False

    def fit(
        self,
        triples: list[tuple[str, str, str]],
        entity_types: Optional[dict[str, str]] = None,
    ) -> None:
        """Compute tail cardinality per (head, relation) pair.

        Args:
            triples: list of (head, relation, tail) tuples
            entity_types: optional entity → type mapping for domain-adjusted scoring
        """
        rel_head_tails: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for h, r, t in triples:
            rel_head_tails[r][h].add(t)

        self._relation_tails = {r: dict(heads) for r, heads in rel_head_tails.items()}

        for r, heads in self._relation_tails.items():
            counts = [len(tails) for tails in heads.values()]
            self._median_cardinality[r] = median(counts) if counts else 1.0

        # Domain-adjusted: per-(relation, head_type) median
        if entity_types:
            self._entity_types = entity_types
            typed_rel_counts: dict[tuple[str, str], list[int]] = defaultdict(list)
            for r, heads in self._relation_tails.items():
                for h, tails in heads.items():
                    h_type = entity_types.get(h, "Unknown")
                    typed_rel_counts[(r, h_type)].append(len(tails))
            self._typed_median = {
                key: median(counts) if counts else 1.0
                for key, counts in typed_rel_counts.items()
            }
            logger.info(
                "CardinalityAnalyzer fitted: %d relations, %d (rel, type) strata",
                len(self._relation_tails),
                len(self._typed_median),
            )
        else:
            self._typed_median = {}
            logger.info(
                "CardinalityAnalyzer fitted: %d relations (no entity types)", len(self._relation_tails)
            )

        self._fitted = True

    def score(self, h: str, r: str, h_type: Optional[str] = None) -> float:
        """
        Returns cardinality anomaly score in [0, 1].
        1.0 = extremely high cardinality relative to median.

        Uses domain-adjusted (per entity type) median when h_type is provided
        and typed statistics are available, else falls back to per-relation median.
        """
        assert self._fitted, "Call fit() first"
        heads = self._relation_tails.get(r, {})
        tail_count = len(heads.get(h, set()))

        # Prefer typed median for domain-adjusted comparison
        typed_key = (r, h_type) if h_type else None
        if typed_key and typed_key in self._typed_median:
            median_card = self._typed_median[typed_key]
        else:
            # Fall back: use entity_types stored at fit time
            stored_type = self._entity_types.get(h)
            fallback_key = (r, stored_type) if stored_type else None
            if fallback_key and fallback_key in self._typed_median:
                median_card = self._typed_median[fallback_key]
            else:
                median_card = self._median_cardinality.get(r, 1.0)

        if median_card <= 0:
            return 0.0
        ratio = tail_count / median_card
        if ratio >= self.multiplier:
            # Normalize: ratio=10x → 0.5, 100x → 1.0
            return min(1.0, (ratio - self.multiplier) / (100.0 - self.multiplier) * 0.5 + 0.5)
        return 0.0

    def find_anomalies(self, threshold: float = 0.5) -> list[dict]:
        """Return all (head, relation) pairs with cardinality score >= threshold."""
        assert self._fitted, "Call fit() first"
        results = []
        for r, heads in self._relation_tails.items():
            for h, tails in heads.items():
                tail_count = len(tails)
                h_type = self._entity_types.get(h)
                score = self.score(h, r, h_type)
                if score >= threshold:
                    typed_key = (r, h_type) if h_type else None
                    median_used = (
                        self._typed_median.get(typed_key, self._median_cardinality.get(r, 1.0))
                        if typed_key else self._median_cardinality.get(r, 1.0)
                    )
                    results.append({
                        "head": h,
                        "relation": r,
                        "tail_count": tail_count,
                        "median_for_stratum": median_used,
                        "head_type": h_type or "Unknown",
                        "ratio": tail_count / max(median_used, 1.0),
                        "cardinality_score": score,
                        "signal": "cardinality_anomaly",
                    })
        logger.info(
            "CardinalityAnalyzer: %d anomalous (head, relation) pairs", len(results)
        )
        return results
