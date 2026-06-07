"""
SchemaPatternAnalyzer: learns (type_h, relation, type_t) co-occurrence frequencies.
Triples with rare type patterns are flagged as potentially anomalous.
"""
import logging
from collections import Counter, defaultdict
from typing import Optional

from kg_verify.config import SCHEMA_THRESHOLD_BORDERLINE, SCHEMA_THRESHOLD_EXTREME

logger = logging.getLogger(__name__)


class SchemaPatternAnalyzer:
    """
    Fits schema pattern frequencies from the full KG.
    Scores each triple by rarity of its (type_h, relation, type_t) pattern.
    """

    def __init__(
        self,
        threshold_extreme: float = SCHEMA_THRESHOLD_EXTREME,
        threshold_borderline: float = SCHEMA_THRESHOLD_BORDERLINE,
    ) -> None:
        self.threshold_extreme = threshold_extreme
        self.threshold_borderline = threshold_borderline
        self._pattern_count: Counter = Counter()
        self._relation_count: Counter = Counter()
        self._relation_patterns: dict[str, list[tuple]] = defaultdict(list)
        self._fitted = False

    def fit(
        self,
        triples: list[tuple[str, str, str]],
        entity_types: dict[str, str],
    ) -> None:
        """Count (type_h, relation, type_t) co-occurrences over the full KG."""
        self._pattern_count.clear()
        self._relation_count.clear()
        self._relation_patterns.clear()

        for h, r, t in triples:
            h_type = entity_types.get(h, "Unknown")
            t_type = entity_types.get(t, "Unknown")
            pattern = (h_type, r, t_type)
            self._pattern_count[pattern] += 1
            self._relation_count[r] += 1

        for (h_type, r, t_type), cnt in self._pattern_count.items():
            self._relation_patterns[r].append((h_type, t_type, cnt))

        self._fitted = True
        logger.info(
            "SchemaPatternAnalyzer fitted: %d unique (type,rel,type) patterns",
            len(self._pattern_count),
        )

    def score(
        self,
        h: str,
        r: str,
        t: str,
        entity_types: Optional[dict[str, str]] = None,
        h_type: Optional[str] = None,
        t_type: Optional[str] = None,
    ) -> float:
        """
        Returns suspicion score in [0, 1].
        1.0 = pattern never seen before; 0.0 = dominant pattern.
        """
        assert self._fitted, "Call fit() before score()"
        if h_type is None:
            h_type = entity_types.get(h, "Unknown") if entity_types else "Unknown"
        if t_type is None:
            t_type = entity_types.get(t, "Unknown") if entity_types else "Unknown"

        pattern = (h_type, r, t_type)
        total_for_rel = self._relation_count.get(r, 0)
        pattern_cnt = self._pattern_count.get(pattern, 0)

        if total_for_rel == 0:
            return 1.0

        freq = pattern_cnt / total_for_rel
        return 1.0 - freq

    def is_extreme(self, score: float) -> bool:
        return score >= (1.0 - self.threshold_extreme)

    def is_borderline(self, score: float) -> bool:
        return (1.0 - self.threshold_borderline) <= score < (1.0 - self.threshold_extreme)

    def top_patterns(self, relation: str, n: int = 3) -> list[dict]:
        """Return top-n (type_h, type_t) patterns for a relation by frequency."""
        patterns = self._relation_patterns.get(relation, [])
        total = self._relation_count.get(relation, 1)
        sorted_patterns = sorted(patterns, key=lambda x: x[2], reverse=True)[:n]
        return [
            {"head_type": ht, "tail_type": tt, "count": c, "frequency": c / total}
            for ht, tt, c in sorted_patterns
        ]
