"""
TopologyScorer: computes structural suspicion scores from degree and connectivity.
Uses precomputed topology info or computes it inline.
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TopologyScorer:
    """
    Combines degree information and sink/source status into a scalar
    structural suspicion score.

    Key insight from CPubMed-KGv2_0 topology:
    - 63.5% of nodes are degree-1 (leaf nodes)
    - 64.4% are sink-only nodes (in-degree > 0, out-degree == 0)
    - These are not necessarily errors but have high ambiguity
    """

    def __init__(self) -> None:
        self._out_degree: dict[str, int] = defaultdict(int)
        self._in_degree: dict[str, int] = defaultdict(int)
        self._total_degree: dict[str, int] = {}
        self._max_degree: int = 1
        self._fitted = False

    def fit(self, triples: list[tuple[str, str, str]]) -> None:
        """Compute degree stats from the full triple list."""
        self._out_degree.clear()
        self._in_degree.clear()

        for h, r, t in triples:
            self._out_degree[h] += 1
            self._in_degree[t] += 1

        all_entities: set[str] = set(self._out_degree.keys()) | set(self._in_degree.keys())
        self._total_degree = {
            e: self._out_degree.get(e, 0) + self._in_degree.get(e, 0)
            for e in all_entities
        }
        self._max_degree = max(self._total_degree.values(), default=1)
        self._fitted = True
        logger.info(
            "TopologyScorer: %d entities, max degree %d",
            len(all_entities), self._max_degree,
        )

    def score(self, h: str, t: str) -> float:
        """
        Structural suspicion score in [0, 1].
        Higher = structurally more suspicious.

        Logic:
        - Sink-only tail (no out-edges): slight flag
        - Source-only head (no in-edges): slight flag
        - Both entities are degree-1 leaves: higher score
        """
        assert self._fitted, "Call fit() first"
        h_out = self._out_degree.get(h, 0)
        h_in = self._in_degree.get(h, 0)
        t_out = self._out_degree.get(t, 0)
        t_in = self._in_degree.get(t, 0)
        h_deg = h_out + h_in
        t_deg = t_out + t_in

        score = 0.0

        # Both leaf nodes
        if h_deg == 1 and t_deg == 1:
            score += 0.5

        # Head is sink-only (isolated in the reverse direction)
        if h_in > 0 and h_out == 0:
            score += 0.2

        # Tail is source-only (no in-edges, which is structurally unusual for a tail)
        if t_out > 0 and t_in == 0:
            score += 0.2

        # Unknown entities (not seen in fit set)
        if h not in self._total_degree or t not in self._total_degree:
            score += 0.3

        return min(1.0, score)

    def get_degree(self, entity: str) -> int:
        return self._total_degree.get(entity, 0)

    def is_hub(self, entity: str) -> bool:
        from kg_verify.config import HUB_DEGREE_THRESHOLD
        return self.get_degree(entity) >= HUB_DEGREE_THRESHOLD
