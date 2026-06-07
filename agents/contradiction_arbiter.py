"""
Agent D — ContradictionArbiter
Given two triples making opposing claims about the same entity pair,
assesses which is more likely correct — for human reviewer's consideration.
"""
import logging
from qwen_agent.agents import FnCallAgent

from .base_agent import build_triple_message, extract_final_text, get_llm_cfg, parse_agent_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是中文生物医学知识冲突分析专家，你的判断将被用于辅助人工审核，不会自动写回知识图谱。
给定两条关于同一实体对的三元组，它们使用了语义相反的关系（如"治疗"和"禁忌"），请分析：
1. 这两条三元组是否真的相互矛盾？还是关注的临床场景不同？
2. 在医学上哪条更有可能正确？请给出医学证据支持。
3. 是否存在两者在不同临床语境（如不同剂量、不同适应症）下均成立的可能？

请使用kg_lookup工具分别查询两个实体的上下文背景。

注意：请不要把"治疗"和"禁忌"之间的冲突简单归结为错误——某些药物可能对某病有治疗作用，
但在特定并发症情况下又有禁忌，两条记录可能都是正确的。

返回JSON（不含注释，不含markdown代码块）：
{"assessment": "PREFER_T1|PREFER_T2|BOTH_SUSPECT|CONTEXT_DEPENDENT", "confidence": "HIGH|MEDIUM|LOW", "reason": "一句话"}
"""


class ContradictionArbiter(FnCallAgent):
    """Agent D: assesses which of two contradictory triples is more likely correct."""

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT,
            name="ContradictionArbiter",
            description="Arbitrates between contradictory triples in Chinese biomedical KG",
        )

    def run_on_contradiction_pair(
        self,
        evidence_package_t1: dict,
        evidence_package_t2: dict,
    ) -> dict:
        """
        Assess two contradictory triples.

        Args:
            evidence_package_t1: evidence for triple 1
            evidence_package_t2: evidence for triple 2

        Returns:
            dict with keys: assessment, confidence, reason
        """
        t1 = evidence_package_t1["triple"]
        t2 = evidence_package_t2["triple"]
        msg = (
            f"【三元组T1】头：{t1['head']}（{evidence_package_t1['head_type']}）"
            f" → 关系：{t1['relation']} → 尾：{t1['tail']}（{evidence_package_t1['tail_type']}）\n"
            f"【三元组T2】头：{t2['head']}（{evidence_package_t2['head_type']}）"
            f" → 关系：{t2['relation']} → 尾：{t2['tail']}（{evidence_package_t2['tail_type']}）\n"
            f"\n这两条三元组的头实体相同，尾实体相同，但关系语义相反，请分析哪条更可能正确。\n"
            f"\n{build_triple_message(evidence_package_t1)}\n"
        )
        messages = [{"role": "user", "content": msg}]

        last_response: list = []
        for response in self.run(messages=messages):
            last_response = response  # Qwen-Agent yields List[Message] each step

        response_text = extract_final_text(last_response)
        result = parse_agent_json(
            response_text,
            fallback={
                "assessment": "CONTEXT_DEPENDENT",
                "confidence": "LOW",
                "reason": "parse_failed",
            },
        )
        result["agent"] = "D_ContradictionArbiter"
        return result
