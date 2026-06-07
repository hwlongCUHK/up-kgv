"""
KG loader: reads CPubMed-KGv2_0.txt and returns triples + entity type map.
Format per line: head@@type TAB relation TAB tail@@type
"""
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_kg(
    path: str | Path,
    max_triples: Optional[int] = None,
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """
    Load the KG file into triples and entity type map.

    Args:
        path: Path to CPubMed-KGv2_0.txt (tab-separated with @@type suffix)
        max_triples: If set, only load this many triples (for debugging)

    Returns:
        triples: list of (head, relation, tail)
        entity_types: dict of entity_name -> type_string
    """
    path = Path(path)
    triples: list[tuple[str, str, str]] = []
    entity_types: dict[str, str] = {}
    skipped = 0

    with open(path, encoding="utf-8") as f:
        next(f)  # skip header line
        for i, line in enumerate(f):
            if max_triples and i >= max_triples:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                skipped += 1
                continue

            h_raw, rel, t_raw = parts[0], parts[1], parts[2]

            if "@@" in h_raw:
                h, h_type = h_raw.rsplit("@@", 1)
            else:
                h, h_type = h_raw, "Unknown"

            if "@@" in t_raw:
                t, t_type = t_raw.rsplit("@@", 1)
            else:
                t, t_type = t_raw, "Unknown"

            triples.append((h, rel, t))
            entity_types[h] = h_type
            entity_types[t] = t_type

    logger.info(
        "Loaded %d triples, %d entities, skipped %d malformed lines",
        len(triples), len(entity_types), skipped,
    )
    return triples, entity_types
