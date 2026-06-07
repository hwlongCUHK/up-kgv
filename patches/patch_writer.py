"""
PatchWriter: accepts ONLY human-approved patches and writes them to disk.
PatchValidator must pass before any file write occurs.
Invariant: only HUMAN_APPROVED_* decisions trigger a write.
"""
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

from kg_verify.config import WRITE_DECISIONS
from .patch_schema import DECISION_TO_OPERATION, KGPatch
from .patch_validator import PatchValidator
from ..review.review_item import ReviewItem

logger = logging.getLogger(__name__)


class PatchWriter:
    """Writes approved patches to JSON files in the approved_patches/ directory."""

    def __init__(
        self,
        output_dir: str | Path,
        triple_set: set[tuple[str, str, str]],
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._validator = PatchValidator(triple_set)
        self._triple_set = triple_set

    def write(self, review_item: ReviewItem) -> KGPatch:
        """
        Write a patch for an approved review item.

        Args:
            review_item: must have human_decision in WRITE_DECISIONS

        Returns:
            The written KGPatch

        Raises:
            AssertionError: if human_decision is not in WRITE_DECISIONS
        """
        assert review_item.human_decision in WRITE_DECISIONS, (
            f"PatchWriter: decision '{review_item.human_decision}' is not in WRITE_DECISIONS. "
            "Only HUMAN_APPROVED_* decisions may trigger a write."
        )

        patch = self._build_patch(review_item)
        self._validator.validate(patch)
        self._write_to_disk(patch)

        # Update the in-memory triple set to prevent downstream duplicates
        if patch.operation == "DELETE_TRIPLE":
            before_triple = (
                patch.before["head"], patch.before["relation"], patch.before["tail"]
            )
            self._triple_set.discard(before_triple)
        elif patch.after is not None:
            after_triple = (
                patch.after.get("head", ""),
                patch.after.get("relation", ""),
                patch.after.get("tail", ""),
            )
            if all(after_triple):
                self._triple_set.add(after_triple)

        logger.info(
            "PatchWriter: wrote patch %s (op=%s, by=%s)",
            patch.patch_id, patch.operation, patch.approved_by,
        )
        return patch

    def _build_patch(self, item: ReviewItem) -> KGPatch:
        h, r, t = item.original_triple
        operation = DECISION_TO_OPERATION.get(item.human_decision, "NO_OP")

        # Determine before/after
        before = {"head": h, "relation": r, "tail": t}
        after: dict | None = None

        if item.human_decision == "APPROVE_DELETE":
            after = None
        elif item.human_decision == "APPROVE_RELATION_REPLACE":
            after = {
                "head": item.human_corrected_head or h,
                "relation": item.human_corrected_relation or r,
                "tail": item.human_corrected_tail or t,
            }
        elif item.human_decision == "APPROVE_ENTITY_REPLACE":
            # Determine operation: HEAD or TAIL replace
            new_h = item.human_corrected_head or h
            new_t = item.human_corrected_tail or t
            operation = "HEAD_REPLACE" if new_h != h else "TAIL_REPLACE"
            after = {"head": new_h, "relation": r, "tail": new_t}
        elif item.human_decision == "APPROVE_ADD_REVERSE":
            operation = "ADD_REVERSE"
            after = {"head": t, "relation": r, "tail": h}
        elif item.human_decision == "APPROVE_ENTITY_MERGE":
            operation = "ENTITY_MERGE"
            after = {
                "head": item.human_corrected_head or h,
                "relation": r,
                "tail": item.human_corrected_tail or t,
            }

        # Derive confidence from agent verdicts
        high_conf_count = sum(
            1 for v in item.agent_verdicts if v.get("confidence") == "HIGH"
        )
        confidence = "HIGH" if high_conf_count > 0 else "MEDIUM"

        return KGPatch(
            patch_id=f"patch_{uuid.uuid4().hex[:8]}",
            triple_id=item.triple_id,
            operation=operation,
            before=before,
            after=after,
            approved_by=item.reviewer_id or "unknown",
            approval_timestamp=item.review_timestamp or datetime.utcnow().isoformat(),
            evidence_refs=[v.get("agent", "?") for v in item.agent_verdicts],
            confidence=confidence,
            risk_level=item.risk_level,
            simulated=item.simulated_review,
            audit_note=item.human_reason,
        )

    def _write_to_disk(self, patch: KGPatch) -> None:
        filename = self._output_dir / f"{patch.patch_id}.json"
        data = {
            "patch_id": patch.patch_id,
            "triple_id": patch.triple_id,
            "operation": patch.operation,
            "before": patch.before,
            "after": patch.after,
            "approved_by": patch.approved_by,
            "approval_timestamp": patch.approval_timestamp,
            "evidence_refs": patch.evidence_refs,
            "confidence": patch.confidence,
            "risk_level": patch.risk_level,
            "simulated": patch.simulated,
            "audit_note": patch.audit_note,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
