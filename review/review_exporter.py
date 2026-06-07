"""
ReviewExporter: exports review queue to TSV for human annotation,
and imports human decisions back from TSV.
"""
import csv
import json
import logging
from pathlib import Path
from typing import Optional

from .review_item import ReviewItem

logger = logging.getLogger(__name__)

TSV_FIELDNAMES = [
    "triple_id", "head", "relation", "tail", "head_type", "tail_type",
    "machine_classification", "composite_score", "primary_signal",
    "detector_signals", "agent_verdicts", "repair_candidates",
    "risk_level", "review_priority", "required_reviewer_role",
    "human_decision", "human_corrected_head", "human_corrected_relation",
    "human_corrected_tail", "human_reason", "reviewer_id",
    "review_timestamp", "simulated_review",
]


class ReviewExporter:
    """Handles import/export of review items to/from TSV."""

    def export(self, items: list[ReviewItem], output_path: str | Path) -> None:
        """Export review items to TSV for human annotation."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TSV_FIELDNAMES, delimiter="\t")
            writer.writeheader()
            for item in items:
                full_row = item.to_dict()
                # TSV_FIELDNAMES excludes evidence_package (too large / not serializable)
                row = {k: full_row.get(k, "") for k in TSV_FIELDNAMES}
                writer.writerow(row)

        logger.info("Exported %d review items to %s", len(items), output_path)

    def import_decisions(
        self,
        items: list[ReviewItem],
        decisions_path: str | Path,
    ) -> list[ReviewItem]:
        """
        Import human decisions from TSV and apply to matching ReviewItems.
        Only items with non-empty human_decision are updated.
        """
        decisions_path = Path(decisions_path)
        if not decisions_path.exists():
            logger.warning("Decisions file not found: %s", decisions_path)
            return items

        decisions: dict[str, dict] = {}
        with open(decisions_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("human_decision"):
                    decisions[row["triple_id"]] = row

        updated = 0
        for item in items:
            if item.triple_id in decisions:
                d = decisions[item.triple_id]
                item.human_decision = d.get("human_decision") or None
                item.human_corrected_head = d.get("human_corrected_head") or None
                item.human_corrected_relation = d.get("human_corrected_relation") or None
                item.human_corrected_tail = d.get("human_corrected_tail") or None
                item.human_reason = d.get("human_reason") or None
                item.reviewer_id = d.get("reviewer_id") or None
                item.review_timestamp = d.get("review_timestamp") or None
                updated += 1

        logger.info(
            "Imported decisions for %d / %d items from %s",
            updated, len(items), decisions_path,
        )
        return items


def export_auto_safe(
    signals: list[dict],
    output_path: str | Path,
) -> None:
    """Export AUTO_ACTION_SAFE signals to TSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["head", "relation", "tail", "signal", "candidate_action", "auto_safe", "note"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for sig in signals:
            row = {
                "head": sig["triple"][0],
                "relation": sig["triple"][1],
                "tail": sig["triple"][2],
                **{k: v for k, v in sig.items() if k != "triple"},
            }
            writer.writerow(row)

    logger.info("Exported %d AUTO_ACTION_SAFE signals to %s", len(signals), output_path)
