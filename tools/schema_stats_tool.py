"""
SchemaStatsTool: Qwen-Agent BaseTool wrapper for schema pattern statistics.
Called by LLM agents to understand how common a (type_h, relation, type_t) pattern is.
"""
import json
import logging
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

# Global schema analyzer (populated once at startup)
_SCHEMA_ANALYZER = None


def init_schema_analyzer(schema_analyzer) -> None:
    """Register the fitted SchemaPatternAnalyzer for use by SchemaStatsTool."""
    global _SCHEMA_ANALYZER
    _SCHEMA_ANALYZER = schema_analyzer
    logger.info("Schema analyzer registered in SchemaStatsTool")


@register_tool("schema_stats")
class SchemaStatsTool(BaseTool):
    """Returns schema frequency statistics for a relation's entity-type patterns."""

    description = (
        "Query schema statistics for a (head_type, relation, tail_type) pattern. "
        "Returns how frequently this entity-type combination appears for the relation, "
        "and what the dominant patterns are. Use this to assess whether a triple's "
        "entity types are typical or anomalous for the given relation."
    )
    parameters = [
        {
            "name": "head_type",
            "type": "string",
            "description": "The entity type of the head entity",
            "required": True,
        },
        {
            "name": "relation",
            "type": "string",
            "description": "The relation name",
            "required": True,
        },
        {
            "name": "tail_type",
            "type": "string",
            "description": "The entity type of the tail entity",
            "required": True,
        },
    ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params)

        if _SCHEMA_ANALYZER is None:
            return json.dumps({"error": "Schema analyzer not initialized"})

        head_type: str = params["head_type"]
        relation: str = params["relation"]
        tail_type: str = params["tail_type"]

        score = _SCHEMA_ANALYZER.score(
            h="", r=relation, t="",
            h_type=head_type, t_type=tail_type,
        )
        frequency = 1.0 - score
        top_patterns = _SCHEMA_ANALYZER.top_patterns(relation, n=5)

        result = {
            "head_type": head_type,
            "relation": relation,
            "tail_type": tail_type,
            "pattern_frequency": round(frequency, 4),
            "suspicion_score": round(score, 4),
            "assessment": (
                "EXTREME_OUTLIER" if _SCHEMA_ANALYZER.is_extreme(score)
                else "BORDERLINE" if _SCHEMA_ANALYZER.is_borderline(score)
                else "NORMAL"
            ),
            "dominant_patterns": top_patterns,
        }
        return json.dumps(result, ensure_ascii=False)
