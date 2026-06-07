"""
Agent B — RelationLabelAuditor
Determines if the relation label is the most precise among the 45-relation vocabulary.
Invoked on cardinality anomaly or borderline schema violations.
"""
import logging
from qwen_agent.agents import FnCallAgent

from .base_agent import build_triple_message, extract_final_text, get_llm_cfg, parse_agent_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是中文生物医学知识图谱关系标注专家，你的判断将被用于辅助人工审核。

本知识图谱共有45种关系类型，你的任务是判断给定三元组的当前关系标注是否最精确。
常见关系包括：治疗、禁忌、适应症、不良反应、并发症、属于、包含、临床表现、病因、
诊断、相关、同义词、药物治疗、手术治疗等。

请使用schema_stats工具查询该类型组合下各关系的统计分布。
请判断：
1. 当前关系是否语义准确描述了头尾实体之间的关系？
2. 是否存在45种关系中更精确的替代标注？
3. 基数异常（该头实体的同类关系数量远超中位数）是否提示错误？

返回JSON（不含注释，不含markdown代码块）：
{"verdict": "CORRECT|RELABEL", "suggested_relation": null, "confidence": "HIGH|MEDIUM|LOW", "reason": "一句话"}
"""


class RelationLabelAuditor(FnCallAgent):
    """Agent B: audits relation label precision within the 45-relation vocabulary."""

    def __init__(self) -> None:
        super().__init__(
            function_list=["schema_stats"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT,
            name="RelationLabelAuditor",
            description="Audits relation label precision in Chinese biomedical KG",
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
                "verdict": "CORRECT",
                "suggested_relation": None,
                "confidence": "LOW",
                "reason": "parse_failed",
            },
        )
        result["agent"] = "B_RelationLabelAuditor"
        return result
