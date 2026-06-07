"""
TypeConflictChecker: finds entities with multiple @@type labels in the KG.
Does NOT output CONFIRMED_ERROR — type conflicts need Agent C + human review.
"""
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TypeConflictChecker:
    """
    Same entity string can appear with different @@type labels in the KG.
    This creates ambiguity that must be resolved by Agent C (EntityLinkerTyper).
    """

    def run(
        self,
        entity_types: dict[str, str],
        triples: list[tuple[str, str, str]],
    ) -> tuple[dict[str, set[str]], list[dict]]:
        """
        Detects entities with multiple type labels.

        Args:
            entity_types: dict entity_name -> type (only last-seen type; use raw_types)
            triples: full triple list to extract all type occurrences

        Returns:
            multi_type_entities: entity -> set of all seen types
            signals: list of signal dicts for affected triples
        """
        # Rebuild full type map from raw triples (entity_types only has last-seen)
        raw_types: dict[str, set[str]] = defaultdict(set)
        for h_raw, rel, t_raw in triples:
            if "@@" in h_raw:
                h, h_type = h_raw.rsplit("@@", 1) if "@@" in h_raw else (h_raw, "Unknown")
                raw_types[h].add(h_type)
            if "@@" in t_raw:
                t, t_type = t_raw.rsplit("@@", 1) if "@@" in t_raw else (t_raw, "Unknown")
                raw_types[t].add(t_type)

        multi_type: dict[str, set[str]] = {
            e: types for e, types in raw_types.items() if len(types) > 1
        }

        signals = []
        for h, r, t in triples:
            if h in multi_type or t in multi_type:
                signals.append({
                    "triple": (h, r, t),
                    "signal": "type_conflict",
                    "conflict_head": list(multi_type.get(h, set())),
                    "conflict_tail": list(multi_type.get(t, set())),
                    "auto_safe": False,
                    "note": "requires_agent_c",
                })

        logger.info(
            "TypeConflictChecker: %d entities with multiple types; %d triples flagged",
            len(multi_type), len(signals),
        )
        return multi_type, signals

    def run_from_raw_file(self, kg_path: str) -> dict[str, set[str]]:
        """
        Quick scan of KG file without loading all triples into memory.
        Returns dict of entities with conflicting types.
        """
        raw_types: dict[str, set[str]] = defaultdict(set)
        with open(kg_path, encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                for raw in (parts[0], parts[2]):
                    if "@@" in raw:
                        e, t = raw.rsplit("@@", 1)
                        raw_types[e].add(t)

        multi = {e: ts for e, ts in raw_types.items() if len(ts) > 1}
        logger.info(
            "TypeConflictChecker (file scan): %d entities with multiple types", len(multi)
        )
        return multi
