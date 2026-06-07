"""
Shared base utilities for all KG-Verify-HITL LLM agents.
Provides: LLM config, JSON output parsing, message building utilities.
"""
import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_llm_cfg() -> dict:
    """Return the shared LLM config for all agents."""
    from kg_verify.config import LLM_CFG
    return LLM_CFG


def parse_agent_json(response_text: str, fallback: Optional[dict] = None) -> dict:
    """
    Extract and parse JSON from an LLM response.
    Handles markdown code blocks and bare JSON objects.

    Args:
        response_text: raw LLM response string
        fallback: returned if parsing fails

    Returns:
        Parsed dict, or fallback if parsing fails.
    """
    if fallback is None:
        fallback = {"verdict": "UNCERTAIN", "confidence": "LOW", "reason": "parse_failed"}

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", response_text).strip("` \n")

    # Try to find JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    logger.warning("JSON parse failed for response: %s...", response_text[:200])
    return fallback


def extract_final_text(messages: list) -> str:
    """
    Extract the last assistant message text from a Qwen-Agent response.
    Handles both dict messages and Qwen-Agent Message objects.
    Note: Qwen-Agent run() yields List[Message] at each step; pass the last yielded list here.
    """
    for msg in reversed(messages):
        # Normalise: Qwen-Agent Message object or plain dict
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
        elif hasattr(msg, "role"):
            role = msg.role
            content = msg.content
        else:
            continue

        if role == "assistant":
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block["text"]
                    if hasattr(block, "text"):
                        return block.text
                    if isinstance(block, str):
                        return block
            return str(content) if content else ""
    return ""


def build_triple_message(evidence_package: dict) -> str:
    """Format evidence package as a structured message for Agent A.

    V2 (Loop 2): if route_hint is present, show it prominently and OMIT
    neighbor_summary (which adds noise without precision — see 4-round ablation).
    If no route_hint, fall back to minimal text-only format (triple + types).

    Legacy full-context format is no longer used by Agent A in V2.
    """
    pkg = evidence_package
    triple = pkg["triple"]
    lines = [
        f"【三元组】头实体：{triple['head']}（类型：{pkg['head_type']}）",
        f"         关系：{triple['relation']}",
        f"         尾实体：{triple['tail']}（类型：{pkg['tail_type']}）",
        "",
    ]

    # V2 core: route hint section (machine-generated suspicion brief)
    route_hint = pkg.get("route_hint", {})
    if route_hint:
        lines.extend([
            "【预筛查信号】（系统自动生成，供参考）",
            f"  路由类型：{route_hint.get('route_type', '未知')}",
            f"  判断依据：{route_hint.get('reason', '综合评分超过阈值')}",
            f"  该类型组合频率：{route_hint.get('pattern_frequency', 'N/A')}",
            "",
        ])

    # Schema statistics (available for all routes)
    schema_stats = pkg.get("schema_stats", {})
    if schema_stats and schema_stats.get("dominant_patterns"):
        lines.extend([
            f"【主要类型模式】{schema_stats['dominant_patterns']}",
            "",
        ])

    # Contradiction detection (when available)
    if pkg.get("contradiction_pairs"):
        lines.append(f"【矛盾检测】发现潜在矛盾：{pkg['contradiction_pairs']}")

    return "\n".join(lines)
