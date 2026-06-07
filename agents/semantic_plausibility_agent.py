"""
Agent A — SemanticPlausibilityAgent
Determines if a triple makes a medically true claim using the LLM as a
frozen Chinese biomedical knowledge base.
"""
import logging
from qwen_agent.agents import FnCallAgent

from .base_agent import build_triple_message, extract_final_text, get_llm_cfg, parse_agent_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是一位中文生物医学知识专家，你的判断将被用于辅助人工审核，不会自动写回知识图谱。

本系统已通过统计预筛查标记了可疑三元组，并在【预筛查信号】中提供了统计判断依据。

请按以下步骤推理：
0. 阅读【预筛查信号】——了解该三元组被统计系统标记的具体原因
1. 该关系对这两类实体是否语义合理？（基于类型层面）
2. 头尾实体之间的具体医学事实是否成立？（基于实体层面）
3. 综合预筛查信号与你的医学知识，给出最终判断

请使用kg_lookup工具查询头尾实体的医学上下文，使用schema_stats工具查询该类型组合的统计分布。

注意：
- 预筛查信号提供了统计依据，但罕见模式不一定代表错误（可能是合理的罕见医疗关系）
- 你的判断是信号，不是最终决定
- 对于不确定的情况，请返回UNCERTAIN而非强行给出判断
- 中文医学专有名词可能有多种表达方式，请综合考虑

返回JSON（不含注释，不含markdown代码块）：
{"verdict": "PLAUSIBLE|IMPLAUSIBLE|UNCERTAIN", "confidence": "HIGH|MEDIUM|LOW", "reason": "一句话"}
"""


class SemanticPlausibilityAgent(FnCallAgent):
    """Agent A: assesses whether a triple's claim is medically plausible."""

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup", "schema_stats"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT,
            name="SemanticPlausibilityAgent",
            description="Assesses medical plausibility of triples in Chinese biomedical KG",
        )

    def run_on_triple(self, evidence_package: dict) -> dict:
        """
        Run Agent A on a single triple's evidence package.

        Returns:
            dict with keys: verdict, confidence, reason
        """
        triple_msg = build_triple_message(evidence_package)
        messages = [{"role": "user", "content": triple_msg}]

        last_response: list = []
        for response in self.run(messages=messages):
            last_response = response  # Qwen-Agent yields List[Message] each step

        response_text = extract_final_text(last_response)
        result = parse_agent_json(
            response_text,
            fallback={"verdict": "UNCERTAIN", "confidence": "LOW", "reason": "parse_failed"},
        )
        result["agent"] = "A_SemanticPlausibilityAgent"
        logger.debug("Agent A result for %s: %s", evidence_package["triple"], result)
        return result
