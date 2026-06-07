"""
PatchValidator: validates a KGPatch before writing.
Enforces schema constraints, deduplication, and no-self-loop invariants.
"""
import logging
from .patch_schema import ALLOWED_OPERATIONS, KGPatch

logger = logging.getLogger(__name__)


class PatchValidator:
    """Validates patches before write. Raises AssertionError on any violation."""

    def __init__(
        self,
        triple_set: set[tuple[str, str, str]],
    ) -> None:
        """
        Args:
            triple_set: set of existing (h, r, t) triples for dedup check
        """
        self._triple_set = triple_set

    def validate(self, patch: KGPatch) -> None:
        """
        Validate a patch. Raises AssertionError if any constraint is violated.
        """
        assert patch.approved_by is not None, f"Patch {patch.patch_id}: no approver"
        assert patch.operation in ALLOWED_OPERATIONS, (
            f"Patch {patch.patch_id}: unknown operation '{patch.operation}'"
        )

        if patch.operation != "NO_OP" and patch.operation != "DELETE_TRIPLE":
            assert patch.after is not None, (
                f"Patch {patch.patch_id}: 'after' required for {patch.operation}"
            )
            assert patch.before != patch.after, (
                f"Patch {patch.patch_id}: 'before' == 'after' — no-op"
            )

        if patch.after is not None:
            # No self-loop in new triple
            after_h = patch.after.get("head")
            after_t = patch.after.get("tail")
            if after_h and after_t:
                assert after_h != after_t, (
                    f"Patch {patch.patch_id}: would create self-loop ({after_h})"
                )

            # No duplicate
            if patch.operation != "DELETE_TRIPLE":
                new_triple = (
                    patch.after.get("head", ""),
                    patch.after.get("relation", ""),
                    patch.after.get("tail", ""),
                )
                assert new_triple not in self._triple_set, (
                    f"Patch {patch.patch_id}: would create duplicate triple {new_triple}"
                )

        logger.debug("PatchValidator: patch %s passed all checks", patch.patch_id)
