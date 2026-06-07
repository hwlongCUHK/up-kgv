"""
EvidencePackageBuilder: assembles a complete evidence package for each
MACHINE_SUSPECTED triple before sending to the agent layer.
"""
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)


class EvidencePackageBuilder:
    """
    Bundles all relevant context for a triple into an evidence_package dict
    that LLM agents and human reviewers can inspect.
    """

    def __init__(
        self,
        triples: list[tuple[str, str, str]],
        entity_types: dict[str, str],
        schema_analyzer,      # SchemaPatternAnalyzer
        topology_scorer,      # TopologyScorer
        type_conflicts: dict[str, set[str]],
    ) -> None:
        self._entity_types = entity_types
        self._schema_analyzer = schema_analyzer
        self._topology_scorer = topology_scorer
        self._type_conflicts = type_conflicts

        # Build adjacency lists (undirected)
        self._adj_out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._adj_in: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._triple_set: set[tuple[str, str, str]] = set(triples)

        for h, r, t in triples:
            self._adj_out[h].append((r, t))
            self._adj_in[t].append((r, h))

        logger.info("EvidencePackageBuilder ready: %d triples indexed", len(triples))

    def build(
        self,
        h: str,
        r: str,
        t: str,
        h_type: Optional[str] = None,
        t_type: Optional[str] = None,
    ) -> dict:
        """Build evidence package for the triple (h, r, t)."""
        h_type = h_type or self._entity_types.get(h, "Unknown")
        t_type = t_type or self._entity_types.get(t, "Unknown")

        return {
            "triple": {"head": h, "relation": r, "tail": t},
            "head_type": h_type,
            "tail_type": t_type,
            "schema_stats": {
                "pattern_frequency": 1.0 - self._schema_analyzer.score(
                    h, r, t, h_type=h_type, t_type=t_type
                ),
                "dominant_patterns": self._schema_analyzer.top_patterns(r, n=3),
            },
            "neighbor_summary_head": self._summarize_neighbors_out(h, n=10),
            "neighbor_summary_tail": self._summarize_neighbors_in(t, n=10),
            "contradiction_pairs": self._find_contradictions(h, t),
            "symmetric_gap": self._check_symmetric_gap(h, r, t),
            "entity_type_history": {
                "head": list(self._type_conflicts.get(h, {h_type})),
                "tail": list(self._type_conflicts.get(t, {t_type})),
            },
            "topology": {
                "head_degree": self._topology_scorer.get_degree(h),
                "tail_degree": self._topology_scorer.get_degree(t),
                "head_is_hub": self._topology_scorer.is_hub(h),
                "tail_is_hub": self._topology_scorer.is_hub(t),
            },
        }

    def _summarize_neighbors_out(self, entity: str, n: int = 10) -> list[dict]:
        """Return top-n out-neighbors (relation, tail) for an entity."""
        neighbors = self._adj_out.get(entity, [])[:n]
        return [
            {"relation": r, "neighbor": nbr, "neighbor_type": self._entity_types.get(nbr, "?")}
            for r, nbr in neighbors
        ]

    def _summarize_neighbors_in(self, entity: str, n: int = 10) -> list[dict]:
        """Return top-n in-neighbors (relation, head) for an entity."""
        neighbors = self._adj_in.get(entity, [])[:n]
        return [
            {"relation": r, "neighbor": nbr, "neighbor_type": self._entity_types.get(nbr, "?")}
            for r, nbr in neighbors
        ]

    def _find_contradictions(self, h: str, t: str) -> list[dict]:
        """Find triples (h, r2, t) where r2 might contradict known relations."""
        from kg_verify.config import CONTRADICTORY_RELATION_PAIRS
        contradictions = []
        for r_out, neighbor in self._adj_out.get(h, []):
            if neighbor == t:
                # found (h, r_out, t) — check contradictory pairs
                for (r1, r2) in CONTRADICTORY_RELATION_PAIRS:
                    if r_out == r1:
                        contradictions.append({
                            "head": h, "relation": r_out, "tail": t,
                            "contradicts_with": r2,
                        })
        return contradictions

    def _check_symmetric_gap(self, h: str, r: str, t: str) -> dict:
        """Check if the reverse triple (t, r, h) exists."""
        reverse_exists = (t, r, h) in self._triple_set
        return {
            "reverse_triple": (t, r, h),
            "reverse_exists": reverse_exists,
        }
