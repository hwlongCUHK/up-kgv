"""
Agent F — RepairSynthesizer
Generates ranked repair candidates for human reviewer selection.
NEVER executes repairs — only proposes options with evidence and risk assessment.
requires_human_approval is ALWAYS true.
"""
import logging
from qwen_agent.agents import FnCallAgent

from .base_agent import build_triple_message, extract_final_text, get_llm_cfg, parse_agent_json

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是中文生物医学知识图谱修复方案设计专家，你的候选方案将提交给人工审核员，不会自动执行。
给定一条疑似错误的三元组及其异常信号，请生成修复候选方案。

原则：
- 优先考虑关系替换（风险最低，只需改变关系类型）
- 其次考虑实体替换（需要更强证据，影响更广）
- 若无法确定修复方向，返回NEED_EXPERT，不要强行给出不确定的方案
- 每个候选方案必须有具体的证据支持和风险评估

请使用kg_lookup工具查询邻居上下文，使用schema_stats工具查询统计分布支持。

返回JSON（不含注释，不含markdown代码块，repair_candidates为数组）：
{
  "repair_candidates": [
    {
      "repair_type": "RELATION_REPLACE|HEAD_REPLACE|TAIL_REPLACE|DELETE|ADD_REVERSE",
      "corrected_triple": {"head": "...", "relation": "...", "tail": "..."},
      "confidence": "HIGH|MEDIUM|LOW",
      "evidence": ["证据1", "证据2"],
      "risk": "LOW|MEDIUM|HIGH"
    }
  ],
  "recommended_action": "REVIEW_REPAIR|REVIEW_DELETE|NO_SAFE_REPAIR|NEED_EXPERT",
  "requires_human_approval": true,
  "reason": "一句话"
}
"""


class RepairSynthesizer(FnCallAgent):
    """Agent F: generates repair candidates for human reviewer selection."""

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup", "schema_stats"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT,
            name="RepairSynthesizer",
            description="Generates ranked repair candidates for human approval in Chinese biomedical KG",
        )

    def run_on_triple(self, evidence_package: dict) -> dict:
        """
        Generate repair candidates for the given triple.

        Returns:
            dict with repair_candidates list, recommended_action, requires_human_approval, reason
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
                "repair_candidates": [],
                "recommended_action": "NEED_EXPERT",
                "requires_human_approval": True,
                "reason": "parse_failed",
            },
        )

        # Enforce the invariant: requires_human_approval is ALWAYS true
        result["requires_human_approval"] = True
        result["agent"] = "F_RepairSynthesizer"
        return result
