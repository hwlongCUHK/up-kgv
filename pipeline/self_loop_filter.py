"""
SelfLoopFilter: detects triples where head entity == tail entity.
Note: some reflexive relations (同义词, 相同概念) may legitimately self-loop.
Outputs AUTO_ACTION_SAFE signals, not AUTO_CONFIRMED_ERROR.
"""
import logging
from typing import Iterable

from kg_verify.config import AUTO_ACTION_SAFE, POTENTIALLY_REFLEXIVE_RELATIONS

logger = logging.getLogger(__name__)


class SelfLoopFilter:
    """Detects head == tail self-loops and classifies them."""

    def run(
        self,
        triples: Iterable[tuple[str, str, str]],
    ) -> list[dict]:
        """
        Args:
            triples: iterable of (head, relation, tail)

        Returns:
            List of signal dicts for detected self-loops.
            Each dict: {triple, signal, candidate_action, auto_safe}
        """
        signals = []
        for h, r, t in triples:
            if h == t:
                is_reflexive = r in POTENTIALLY_REFLEXIVE_RELATIONS
                signals.append({
                    "triple": (h, r, t),
                    "signal": "self_loop",
                    "candidate_action": "NO_OP" if is_reflexive else "DELETE",
                    "auto_safe": not is_reflexive,
                    "classification": AUTO_ACTION_SAFE if not is_reflexive else None,
                    "note": "reflexive_relation" if is_reflexive else None,
                })

        logger.info("SelfLoopFilter: %d self-loops detected", len(signals))
        return signals
