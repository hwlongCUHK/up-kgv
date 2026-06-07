"""
AuditLog: append-only log of all machine, agent, and human decisions.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

from ..patches.patch_schema import KGPatch

logger = logging.getLogger(__name__)


class AuditLog:
    """Append-only JSONL audit log for full decision traceability."""

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record_auto_action(self, signal: dict) -> None:
        """Log an AUTO_ACTION_SAFE event."""
        self._append({
            "event_type": "auto_action",
            "timestamp": datetime.utcnow().isoformat(),
            "triple": list(signal["triple"]),
            "signal": signal.get("signal"),
            "candidate_action": signal.get("candidate_action"),
            "auto_safe": signal.get("auto_safe"),
        })

    def record_agent_verdict(self, triple_id: str, verdict: dict) -> None:
        """Log an agent verdict."""
        self._append({
            "event_type": "agent_verdict",
            "timestamp": datetime.utcnow().isoformat(),
            "triple_id": triple_id,
            "agent": verdict.get("agent"),
            "verdict": verdict,
        })

    def record_patch(self, patch: KGPatch) -> None:
        """Log a written patch (always human-approved or simulated)."""
        self._append({
            "event_type": "patch_written",
            "timestamp": datetime.utcnow().isoformat(),
            "patch_id": patch.patch_id,
            "triple_id": patch.triple_id,
            "operation": patch.operation,
            "before": patch.before,
            "after": patch.after,
            "approved_by": patch.approved_by,
            "risk_level": patch.risk_level,
            "simulated": patch.simulated,
            "audit_note": patch.audit_note,
        })

    def _append(self, record: dict) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
