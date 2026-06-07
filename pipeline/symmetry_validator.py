"""
SymmetryValidator: identifies symmetric relations and finds missing reverse triples.
Outputs ADD_REVERSE candidates (not confirmed errors).
"""
import logging
from collections import Counter

from kg_verify.config import POTENTIALLY_REFLEXIVE_RELATIONS, SYMMETRIC_COVERAGE_THRESHOLD

logger = logging.getLogger(__name__)


class SymmetryValidator:
    """
    Detects symmetric relations by checking what fraction of (A,r,B) triples
    also have (B,r,A). If > threshold, r is treated as symmetric.
    Then finds all (A,r,B) triples missing the reverse.
    """

    def __init__(self, coverage_threshold: float = SYMMETRIC_COVERAGE_THRESHOLD) -> None:
        self.coverage_threshold = coverage_threshold
        self._symmetric_relations: set[str] = set()
        self._triple_set: set[tuple[str, str, str]] = set()
        self._fitted = False

    def fit(self, triples: list[tuple[str, str, str]]) -> None:
        """Determine which relations are symmetric from the data."""
        self._triple_set = set(triples)

        relation_pairs: dict[str, list[tuple[str, str]]] = {}
        for h, r, t in triples:
            if r not in relation_pairs:
                relation_pairs[r] = []
            relation_pairs[r].append((h, t))

        self._symmetric_relations = set(POTENTIALLY_REFLEXIVE_RELATIONS)

        for r, pairs in relation_pairs.items():
            if len(pairs) < 10:
                continue
            sym_count = sum(1 for h, t in pairs if (t, r, h) in self._triple_set)
            coverage = sym_count / len(pairs)
            if coverage >= self.coverage_threshold:
                self._symmetric_relations.add(r)

        logger.info(
            "SymmetryValidator: %d symmetric relations detected: %s",
            len(self._symmetric_relations),
            self._symmetric_relations,
        )
        self._fitted = True

    def find_missing_reverses(self) -> list[dict]:
        """
        Find all (h, r, t) where r is symmetric and (t, r, h) does not exist.
        Returns ADD_REVERSE candidates.
        """
        assert self._fitted, "Call fit() first"
        candidates = []
        for h, r, t in self._triple_set:
            if r in self._symmetric_relations and (t, r, h) not in self._triple_set:
                candidates.append({
                    "triple": (h, r, t),
                    "signal": "missing_reverse",
                    "candidate_action": "ADD_REVERSE",
                    "missing_triple": (t, r, h),
                    "auto_safe": True,
                })
        logger.info(
            "SymmetryValidator: %d missing reverse triples found", len(candidates)
        )
        return candidates

    @property
    def symmetric_relations(self) -> set[str]:
        return self._symmetric_relations
