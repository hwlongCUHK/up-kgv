"""
KGPatch data model and allowed operations.
"""
from dataclasses import dataclass
from typing import Optional

ALLOWED_OPERATIONS = {
    "DELETE_TRIPLE",
    "ADD_TRIPLE",
    "RELATION_REPLACE",
    "HEAD_REPLACE",
    "TAIL_REPLACE",
    "ENTITY_TYPE_CORRECT",
    "ENTITY_MERGE",
    "ADD_REVERSE",
    "NO_OP",
}

# Map human decisions → patch operations
DECISION_TO_OPERATION = {
    "APPROVE_DELETE": "DELETE_TRIPLE",
    "APPROVE_RELATION_REPLACE": "RELATION_REPLACE",
    "APPROVE_ENTITY_REPLACE": "HEAD_REPLACE",  # writer checks which entity changed
    "APPROVE_ADD_REVERSE": "ADD_REVERSE",
    "APPROVE_ENTITY_MERGE": "ENTITY_MERGE",
}


@dataclass
class KGPatch:
    patch_id: str
    triple_id: str
    operation: str                  # one of ALLOWED_OPERATIONS
    before: dict                    # {head, relation, tail}
    after: Optional[dict]           # None for DELETE_TRIPLE
    approved_by: str                # reviewer_id
    approval_timestamp: str
    evidence_refs: list[str]        # which agents/detectors informed this
    confidence: str                 # HIGH | MEDIUM | LOW
    risk_level: str
    simulated: bool = False         # True if Agent G (not real human)
    audit_note: Optional[str] = None
