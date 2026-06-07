"""
Agent C — EntityLinkerTyper
Resolves entity type conflicts and types Unknown entities.
Detects alias/merge candidates.
Always runs BEFORE Agent A when type_conflict is the primary signal.
"""
import logging
from qwen_agent.agents import FnCallAgent

from .base_agent import build_triple_message, extract_final_text, get_llm_cfg, parse_agent_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是中文生物医学实体标注专家，你的判断将被用于辅助人工审核，不会自动写回知识图谱。
请判断：
1. 出现多种类型标注时，根据实体名称和关系上下文确定正确类型。
2. 类型为Unknown时，根据名称和关系上下文推断正确类型。
3. 该实体是否可能是已有实体的同义词或别名（需合并）。

请基于知识图谱邻居信息进行推理，使用kg_lookup工具查询相关实体。

返回JSON（不含注释，不含markdown代码块）：
{"canonical_type": "...", "aliases_detected": [], "merge_candidate": null, "confidence": "HIGH|MEDIUM|LOW", "reason": "一句话"}
"""


class EntityLinkerTyper(FnCallAgent):
    """Agent C: resolves entity types and detects alias/merge candidates."""

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT,
            name="EntityLinkerTyper",
            description="Resolves entity type conflicts and detects aliases in Chinese biomedical KG",
        )

    def run_on_triple(self, evidence_package: dict) -> dict:
        """
        Run Agent C on a single triple's evidence package.

        Args:
            evidence_package: built by EvidencePackageBuilder

        Returns:
            dict with keys: canonical_type, aliases_detected, merge_candidate,
                            confidence, reason
        """
        triple_msg = build_triple_message(evidence_package)
        messages = [{"role": "user", "content": triple_msg}]

        last_response: list = []
        for response in self.run(messages=messages):
            last_response = response  # Qwen-Agent yields List[Message] each step

        response_text = extract_final_text(last_response)
        result = parse_agent_json(
            response_text,
            fallback={
                "canonical_type": "Unknown",
                "aliases_detected": [],
                "merge_candidate": None,
                "confidence": "LOW",
                "reason": "parse_failed",
            },
        )
        result["agent"] = "C_EntityLinkerTyper"
        logger.debug("Agent C result for %s: %s", evidence_package["triple"], result)
        return result
