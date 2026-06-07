"""
Agent E — ContextualCoherenceVerifier
For hub entities with rich neighborhood context, assesses whether a triple
fits the entity's established medical specialty domain.
Invoked when Agent A returns UNCERTAIN and deg(h) >= 50.
"""
import logging
from qwen_agent.agents import FnCallAgent

from .base_agent import build_triple_message, extract_final_text, get_llm_cfg, parse_agent_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是中文生物医学知识图谱一致性分析专家，你的判断将被用于辅助人工审核。
该三元组的头实体或尾实体是知识图谱中的高连接度（枢纽）实体，拥有丰富的领域上下文。
请使用kg_lookup工具查询两个实体的邻居信息，然后评估：
1. 头实体的医学领域背景（通过其邻居推断）与该关系是否相符？
2. 尾实体的医学领域背景与该关系是否相符？
3. 这条三元组与两个实体各自建立的医学领域背景是否整体一致？

注意：枢纽实体通常具有多学科背景，请避免因领域跨越而误判为不一致。

返回JSON（不含注释，不含markdown代码块）：
{"assessment": "COHERENT|DOMAIN_MISMATCH|UNCERTAIN", "confidence": "HIGH|MEDIUM|LOW", "reason": "一句话"}
"""


class ContextualCoherenceVerifier(FnCallAgent):
    """Agent E: verifies domain coherence for triples involving hub entities."""

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT,
            name="ContextualCoherenceVerifier",
            description="Verifies contextual domain coherence for hub entity triples",
        )

    def run_on_triple(self, evidence_package: dict) -> dict:
        triple_msg = build_triple_message(evidence_package)
        messages = [{"role": "user", "content": triple_msg}]

        last_response: list = []
        for response in self.run(messages=messages):
            last_response = response  # Qwen-Agent yields List[Message] each step

        response_text = extract_final_text(last_response)
        result = parse_agent_json(
            response_text,
            fallback={
                "assessment": "UNCERTAIN",
                "confidence": "LOW",
                "reason": "parse_failed",
            },
        )
        result["agent"] = "E_ContextualCoherenceVerifier"
        return result
