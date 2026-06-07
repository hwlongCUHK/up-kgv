"""
KGLookupTool: Qwen-Agent BaseTool wrapper for querying entity neighborhood.
Called by LLM agents via function calling.
"""
import json
import logging
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# Global KG index (populated once at startup by the orchestrator)
_KG_INDEX: dict = {}


def init_kg_index(
    triples: list[tuple[str, str, str]],
    entity_types: dict[str, str],
) -> None:
    """Initialize the global KG index used by KGLookupTool."""
    global _KG_INDEX
    from collections import defaultdict

    out_edges: dict[str, list] = defaultdict(list)
    in_edges: dict[str, list] = defaultdict(list)
    for h, r, t in triples:
        out_edges[h].append({"relation": r, "entity": t, "type": entity_types.get(t, "Unknown")})
        in_edges[t].append({"relation": r, "entity": h, "type": entity_types.get(h, "Unknown")})

    _KG_INDEX = {
        "out_edges": dict(out_edges),
        "in_edges": dict(in_edges),
        "entity_types": entity_types,
    }
    logger.info("KG index initialized: %d entities", len(entity_types))


@register_tool("kg_lookup")
class KGLookupTool(BaseTool):
    """Returns neighborhood context for an entity from the knowledge graph."""

    description = (
        "Query the Chinese biomedical knowledge graph for an entity's connections. "
        "Returns out-edges (what this entity points to) and in-edges (what points to it), "
        "along with entity types."
    )
    parameters = [
        {
            "name": "entity",
            "type": "string",
            "description": "The entity name to look up (without @@type suffix)",
            "required": True,
        },
        {
            "name": "max_neighbors",
            "type": "integer",
            "description": "Maximum number of neighbors to return per direction (default: 15)",
            "required": False,
        },
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params)

        entity: str = params["entity"]
        max_n: int = int(params.get("max_neighbors", 15))

        if not _KG_INDEX:
            return json.dumps({"error": "KG index not initialized"})

        out = _KG_INDEX["out_edges"].get(entity, [])[:max_n]
        inc = _KG_INDEX["in_edges"].get(entity, [])[:max_n]
        etype = _KG_INDEX["entity_types"].get(entity, "Unknown")

        result = {
            "entity": entity,
            "type": etype,
            "out_degree": len(_KG_INDEX["out_edges"].get(entity, [])),
            "in_degree": len(_KG_INDEX["in_edges"].get(entity, [])),
            "out_edges": out,
            "in_edges": inc,
        }
        return json.dumps(result, ensure_ascii=False)
