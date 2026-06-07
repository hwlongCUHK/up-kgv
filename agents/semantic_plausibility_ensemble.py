"""
V3.0 Two-Round Agent A Ensemble + V3.1 Cascade Arbitration.

V3.0 Motivation (Loop 3 empirical finding):
  - agent_confidence alone → AUPRC=0.9576 (fixed eval, 504 items, n_true=81)
  - mixed priority formula → 0.7022 (-0.236)
  - Root cause: schema-rare-but-medically-valid FALSE_ALARMs rank equally with
    TRUE_ERRORs when both get IMPLAUSIBLE/HIGH from a single Agent A call.

V3.0 Innovation: run Agent A twice with different evidence contexts.
  Round 1 (schema-aware): sees schema_stats.dominant_patterns — may anchor on
    statistical rarity (causing false IMPLAUSIBLE verdicts for schema-rare-but-valid items)
  Round 2 (semantic-only): no schema stats, only kg_lookup — pure medical knowledge

V3.0 Ensemble calibration:
  Both IMPLAUSIBLE     → IMPLAUSIBLE/HIGH   (1.0) cross-context error agreement
  One IMPLAUSIBLE      → IMPLAUSIBLE/MEDIUM (0.6) single-context flag
  Conflicted           → UNCERTAIN          (0.0) schema-rare but semantically valid
  Both non-IMPLAUSIBLE → PLAUSIBLE          (0.0) confirmed false alarm

V3.1 Motivation (Loop 4 empirical finding):
  - 16.6% of pipeline items land in UNCERTAIN/HIGH (conflicted zone, score=0.0)
  - These 83 items per seed are not all false alarms — some are hidden TRUE_ERRORs
    where the semantic agent was fooled by superficially plausible context
  - V3.0 discards ALL conflicted items equally (score=0.0), losing these true errors

V3.1 Innovation: Cascade Arbitration for conflicted cases only.
  Stage 3 (ArbitrationAgent): runs only on UNCERTAIN/HIGH items (~17% of pipeline)
    Synthesizes both conflicting verdicts to give a final IMPLAUSIBLE/PLAUSIBLE/UNCERTAIN call
    Calibrated to LOW confidence to avoid over-promoting → score=0.3 if IMPLAUSIBLE
  Cost: +17% API calls, targeted at exactly the ambiguous zone
"""
import logging
from qwen_agent.agents import FnCallAgent

from .base_agent import build_triple_message, extract_final_text, get_llm_cfg, parse_agent_json
from .semantic_plausibility_agent import SemanticPlausibilityAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_SEMANTIC_ONLY = """\
你是一位中文生物医学知识专家，你的判断将被用于辅助人工审核，不会自动写回知识图谱。

请基于你的生物医学领域知识，判断以下知识图谱三元组在医学上是否成立。

推理步骤：
1. 该关系对这两类实体是否在语义上合理？（类型层面）
2. 头尾实体之间的具体医学事实是否成立？（使用kg_lookup查询医学上下文）
3. 综合以上分析，给出最终判断

注意：
- 请仅基于医学事实和语义推理进行判断，不依赖统计频率或模式信息
- 对于不确定的情况，请返回UNCERTAIN而非强行给出判断
- 中文医学专有名词可能有多种表达方式，请综合考虑

返回JSON（不含注释，不含markdown代码块）：
{"verdict": "PLAUSIBLE|IMPLAUSIBLE|UNCERTAIN", "confidence": "HIGH|MEDIUM|LOW", "reason": "一句话"}
"""


def _build_semantic_only_message(evidence_package: dict) -> str:
    """Format evidence package for semantic-only Agent A.

    Strips schema_stats and route_hint — the agent sees only triple + types
    (+ contradiction detection when present). This ensures the semantic round
    is genuinely independent of the statistical pre-screening signal.
    """
    pkg = evidence_package
    triple = pkg["triple"]
    lines = [
        f"【三元组】头实体：{triple['head']}（类型：{pkg['head_type']}）",
        f"         关系：{triple['relation']}",
        f"         尾实体：{triple['tail']}（类型：{pkg['tail_type']}）",
        "",
    ]
    if pkg.get("contradiction_pairs"):
        lines.append(f"【矛盾检测】发现潜在矛盾：{pkg['contradiction_pairs']}")
    return "\n".join(lines)


class SemanticOnlyAgentA(FnCallAgent):
    """Agent A variant: pure semantic reasoning without schema statistics.

    Uses kg_lookup only (no schema_stats tool). Receives a stripped-down
    evidence package with triple + types, no statistical pre-screening signals.
    This creates the independent semantic signal for the ensemble.
    """

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT_SEMANTIC_ONLY,
            name="SemanticOnlyAgentA",
            description="Pure semantic plausibility assessment without schema statistics",
        )

    def run_on_triple(self, evidence_package: dict) -> dict:
        triple_msg = _build_semantic_only_message(evidence_package)
        messages = [{"role": "user", "content": triple_msg}]

        last_response: list = []
        for response in self.run(messages=messages):
            last_response = response

        response_text = extract_final_text(last_response)
        result = parse_agent_json(
            response_text,
            fallback={"verdict": "UNCERTAIN", "confidence": "LOW", "reason": "parse_failed"},
        )
        result["agent"] = "A_SemanticOnlyAgentA"
        logger.debug(
            "Semantic-only result for %s: %s",
            evidence_package.get("triple", {}),
            result,
        )
        return result


def _ensemble_verdict(r1: dict, r2: dict) -> dict:
    """Combine schema-aware (r1) and semantic-only (r2) verdicts.

    Ensemble calibration table:
      R1=IMPLAUSIBLE + R2=IMPLAUSIBLE → IMPLAUSIBLE/HIGH   strong cross-context agreement
      R1=IMPLAUSIBLE + R2=UNCERTAIN   → IMPLAUSIBLE/MEDIUM single-context flag
      R1=UNCERTAIN   + R2=IMPLAUSIBLE → IMPLAUSIBLE/MEDIUM single-context flag (other side)
      R1=IMPLAUSIBLE + R2=PLAUSIBLE   → UNCERTAIN          conflicted: schema-rare but valid
      R1=PLAUSIBLE   + R2=IMPLAUSIBLE → UNCERTAIN          conflicted: schema-rare but valid
      R1=UNCERTAIN   + R2=UNCERTAIN   → UNCERTAIN/LOW      weak signal
      any PLAUSIBLE  (no IMPLAUSIBLE) → PLAUSIBLE/LOW      confirmed false alarm

    The conflicted cases map to UNCERTAIN (not IMPLAUSIBLE) so that
    _agent_confidence_score() returns 0.0 for them — they will NOT be
    promoted in the review queue even if composite_score is high.
    This is the key architectural fix: schema-rare-but-medically-valid items
    stop crowding the top-K list.
    """
    v1 = r1.get("verdict", "UNCERTAIN")
    v2 = r2.get("verdict", "UNCERTAIN")
    r1_reason = r1.get("reason", "")
    r2_reason = r2.get("reason", "")

    if v1 == "IMPLAUSIBLE" and v2 == "IMPLAUSIBLE":
        return {
            "verdict": "IMPLAUSIBLE",
            "confidence": "HIGH",
            "reason": f"双重确认错误: schema=[{r1_reason}] semantic=[{r2_reason}]",
        }
    elif v1 == "IMPLAUSIBLE" and v2 == "UNCERTAIN":
        return {
            "verdict": "IMPLAUSIBLE",
            "confidence": "MEDIUM",
            "reason": f"统计标记疑似错误，语义不确定: [{r1_reason}]",
        }
    elif v1 == "UNCERTAIN" and v2 == "IMPLAUSIBLE":
        return {
            "verdict": "IMPLAUSIBLE",
            "confidence": "MEDIUM",
            "reason": f"语义标记疑似错误，统计不确定: [{r2_reason}]",
        }
    elif v1 == "IMPLAUSIBLE" and v2 == "PLAUSIBLE":
        return {
            "verdict": "UNCERTAIN",
            "confidence": "HIGH",
            "reason": f"冲突判断(统计异常但语义合理): schema=[IMPLAUSIBLE: {r1_reason}] semantic=[PLAUSIBLE: {r2_reason}]",
        }
    elif v1 == "PLAUSIBLE" and v2 == "IMPLAUSIBLE":
        return {
            "verdict": "UNCERTAIN",
            "confidence": "HIGH",
            "reason": f"冲突判断(语义异常但统计合理): schema=[PLAUSIBLE: {r1_reason}] semantic=[IMPLAUSIBLE: {r2_reason}]",
        }
    elif v1 == "UNCERTAIN" and v2 == "UNCERTAIN":
        return {
            "verdict": "UNCERTAIN",
            "confidence": "LOW",
            "reason": f"双方均不确定: schema=[{r1_reason}] semantic=[{r2_reason}]",
        }
    else:
        # PLAUSIBLE+UNCERTAIN or both PLAUSIBLE
        return {
            "verdict": "PLAUSIBLE",
            "confidence": "LOW",
            "reason": f"schema=[{v1}: {r1_reason}] semantic=[{v2}: {r2_reason}]",
        }


def _max_verdict(r1: dict, r2: dict) -> dict:
    """E-MAX aggregation: take the higher-confidence IMPLAUSIBLE verdict.

    Used by D3 control (SemanticPlausibilityEnsembleMaxAgent) where both
    calls use the same schema-aware evidence. By taking MAX we give repeated
    sampling the best possible advantage — any improvement of D0 over D3
    must come from context-diversity, not from aggregation alone.

    Confidence rank: HIGH > MEDIUM > LOW. Ties go to r1.
    """
    _conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

    def _key(r: dict) -> tuple[int, int]:
        v = r.get("verdict", "UNCERTAIN")
        c = r.get("confidence", "LOW")
        return (1 if v == "IMPLAUSIBLE" else 0, _conf_rank.get(c, 0))

    winner = r1 if _key(r1) >= _key(r2) else r2
    result = dict(winner)
    result["reason"] = (
        f"E-MAX: r1=[{r1.get('verdict')}/{r1.get('confidence')}] "
        f"r2=[{r2.get('verdict')}/{r2.get('confidence')}] → {winner.get('verdict')}/{winner.get('confidence')}"
    )
    return result


class SemanticPlausibilityEnsembleAgent:
    """V3.0 Two-Round Agent A Ensemble.

    Runs Agent A twice with different evidence contexts to create calibrated
    confidence scores within the MACHINE_SUSPECTED bucket:

      - Round 1: schema-aware (current Agent A, sees schema_stats)
      - Round 2: semantic-only (no schema_stats, only kg_lookup)

    Cross-context agreement → HIGH confidence (TRUE_ERROR signal)
    Single-context flag     → MEDIUM confidence (uncertain, needs review)
    Conflicted              → UNCERTAIN (schema-rare-but-medically-valid FALSE_ALARM)
    Both non-IMPLAUSIBLE    → PLAUSIBLE (false alarm)

    The result uses agent name "A_SemanticPlausibilityEnsemble" so that
    _agent_confidence_score() picks it up correctly (checks for prefix "A_").

    Additional metadata fields returned:
      - schema_aware_verdict: R1 verdict string
      - semantic_verdict: R2 verdict string
    """

    def __init__(self) -> None:
        self._schema_agent = SemanticPlausibilityAgent()
        self._semantic_agent = SemanticOnlyAgentA()

    def run_on_triple(self, evidence_package: dict) -> dict:
        """Run both rounds and combine via ensemble vote."""
        r1 = self._schema_agent.run_on_triple(evidence_package)
        r2 = self._semantic_agent.run_on_triple(evidence_package)

        ensemble = _ensemble_verdict(r1, r2)
        ensemble["agent"] = "A_SemanticPlausibilityEnsemble"
        ensemble["schema_aware_verdict"] = r1.get("verdict", "UNCERTAIN")
        ensemble["semantic_verdict"] = r2.get("verdict", "UNCERTAIN")

        logger.debug(
            "Ensemble result for %s: schema=%s semantic=%s → %s/%s",
            evidence_package.get("triple", {}),
            r1.get("verdict"),
            r2.get("verdict"),
            ensemble["verdict"],
            ensemble["confidence"],
        )
        return ensemble


class SemanticPlausibilityEnsembleMaxAgent:
    """D3 (E-MAX) control: calls Agent A twice with IDENTICAL schema-aware evidence.

    Purpose: ablation control for D0. Establishes whether D0's gain over C0 comes
    from context-diversity (schema-aware vs semantic-only) or simply from calling
    Agent A twice (repeated sampling). Uses MAX aggregation to give repeated
    sampling the best possible advantage.

    Interpretation:
      D0 >> D3 → context-diversity is the key innovation (supports main claim)
      D0 ≈ D3 → gain is from repeated sampling, not context diversity (undermines claim)

    The result uses agent name "A_SemanticPlausibilityEnsembleMax" so that
    _agent_confidence_score() picks it up correctly (checks for prefix "A_").
    """

    def __init__(self) -> None:
        self._agent1 = SemanticPlausibilityAgent()
        self._agent2 = SemanticPlausibilityAgent()

    def run_on_triple(self, evidence_package: dict) -> dict:
        """Run Agent A twice on IDENTICAL schema-aware evidence, take MAX confidence."""
        r1 = self._agent1.run_on_triple(evidence_package)
        r2 = self._agent2.run_on_triple(evidence_package)

        result = _max_verdict(r1, r2)
        result["agent"] = "A_SemanticPlausibilityEnsembleMax"
        result["r1_verdict"] = r1.get("verdict", "UNCERTAIN")
        result["r2_verdict"] = r2.get("verdict", "UNCERTAIN")

        logger.debug(
            "E-MAX result for %s: r1=%s/%s r2=%s/%s → %s/%s",
            evidence_package.get("triple", {}),
            r1.get("verdict"), r1.get("confidence"),
            r2.get("verdict"), r2.get("confidence"),
            result["verdict"], result["confidence"],
        )
        return result


class SemanticPlausibilityEnsemblePrimeAgent:
    """D4 (D3-prime) control: calls Agent A twice with IDENTICAL schema-aware evidence,
    applies D0's 7-case ensemble logic (not MAX).

    Purpose: cleanly isolates context-diversity from aggregation-logic contribution.
    Comparison chain:
      D3  (same context, MAX aggregation)
      D4  (same context, D0's ensemble logic)  ← this agent
      D0  (different context, D0's ensemble logic)

    If D0 >> D4: context-diversity drives the gain.
    If D4 >> D3: D0's ensemble logic (not MAX) is responsible for gains.
    If D4 ≈ D3: aggregation logic doesn't matter, it's context-diversity.

    Note: with identical context, R1 and R2 will mostly agree (same verdicts due to
    same input, with slight temperature variance). Conflicted cases are rare but
    possible when LLM sampling produces different outputs on the same prompt.

    The result uses agent name "A_SemanticPlausibilityEnsemblePrime" so that
    _agent_confidence_score() picks it up correctly (checks for prefix "A_").
    """

    def __init__(self) -> None:
        self._agent1 = SemanticPlausibilityAgent()
        self._agent2 = SemanticPlausibilityAgent()

    def run_on_triple(self, evidence_package: dict) -> dict:
        """Run Agent A twice on IDENTICAL schema-aware evidence, apply D0's ensemble logic."""
        r1 = self._agent1.run_on_triple(evidence_package)
        r2 = self._agent2.run_on_triple(evidence_package)

        result = _ensemble_verdict(r1, r2)
        result["agent"] = "A_SemanticPlausibilityEnsemblePrime"
        result["r1_verdict"] = r1.get("verdict", "UNCERTAIN")
        result["r2_verdict"] = r2.get("verdict", "UNCERTAIN")

        logger.debug(
            "D3-prime result for %s: r1=%s/%s r2=%s/%s → %s/%s",
            evidence_package.get("triple", {}),
            r1.get("verdict"), r1.get("confidence"),
            r2.get("verdict"), r2.get("confidence"),
            result["verdict"], result["confidence"],
        )
        return result


# ─── V3.1: Cascade Arbitration ────────────────────────────────────────────────

SYSTEM_PROMPT_ARBITRATION = """\
你是一位专业的中文生物医学知识图谱质量审核员。

你将收到一个知识图谱三元组，以及对它的两种相互矛盾的自动评估。在给出裁定之前，你**必须**先使用 kg_lookup 工具主动检索新证据。

**强制步骤（必须按顺序执行）：**
1. 首先用 kg_lookup 查询头实体，检索它在知识图谱中的实际关系（重点关注与本三元组相同的关系类型）
2. 然后用 kg_lookup 查询尾实体，确认其实际的实体类型和医学角色
3. 基于上述**新检索的证据**（而非仅凭两种评估的理由）做出最终裁定

裁定原则：
- 只有在新证据**明确支持**该三元组存在医学错误时，才判断为IMPLAUSIBLE
- 若新证据发现该关系在同类实体间确实存在，或头实体确实具备该关系，则判断为PLAUSIBLE
- 若新证据无法提供明确支持，则判断为UNCERTAIN（不要强行推断）

重要：不要仅凭两种评估的理由进行推理，必须以kg_lookup检索到的实际知识图谱数据为依据。

返回JSON（不含注释，不含markdown代码块）：
{"verdict": "IMPLAUSIBLE|PLAUSIBLE|UNCERTAIN", "confidence": "HIGH|MEDIUM|LOW", "reason": "基于新检索证据的一句话裁定理由", "evidence_found": true|false}
"""


def _build_arbitration_message(evidence_package: dict, r1: dict, r2: dict) -> str:
    """Build message for evidence-gated arbitration agent (V3.2).

    Forces the agent to retrieve new KG evidence before deciding.
    The conflicting verdicts are provided as context but not the decision basis.
    """
    pkg = evidence_package
    triple = pkg.get("triple", {})
    head = triple.get("head", "?")
    relation = triple.get("relation", "?")
    tail = triple.get("tail", "?")
    lines = [
        f"【待裁定三元组】",
        f"  头实体：{head}（类型：{pkg.get('head_type', '?')}）",
        f"  关系：{relation}",
        f"  尾实体：{tail}（类型：{pkg.get('tail_type', '?')}）",
        "",
        f"【之前的两种矛盾评估（仅供参考，不作为裁定依据）】",
        f"  评估一（统计视角）：{r1.get('verdict', '?')} — {r1.get('reason', '无')}",
        f"  评估二（语义视角）：{r2.get('verdict', '?')} — {r2.get('reason', '无')}",
        "",
        f"**现在请按以下步骤操作：**",
        f"步骤1：用kg_lookup查询【{head}】，找到它在图谱中有哪些【{relation}】类型的关系",
        f"步骤2：用kg_lookup查询【{tail}】，确认其实际医学角色和类型",
        f"步骤3：根据查询结果，给出最终裁定（只有查到明确证据才能判IMPLAUSIBLE）",
    ]
    return "\n".join(lines)


class ArbitrationAgent(FnCallAgent):
    """V3.1 Stage 3: Resolve conflicted ensemble verdicts via targeted arbitration.

    Runs ONLY when the two-round ensemble gives UNCERTAIN (conflicted or double-UNCERTAIN).
    Input: both conflicting verdicts + their reasons + original triple.
    Output: final IMPLAUSIBLE/PLAUSIBLE/UNCERTAIN with LOW confidence.

    Calibrated to LOW confidence to avoid over-promoting ambiguous items.
    Score mapping: IMPLAUSIBLE/LOW → 0.3 (below MEDIUM=0.6, above UNCERTAIN=0.0).
    """

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT_ARBITRATION,
            name="ArbitrationAgent",
            description="Arbitrates conflicted ensemble verdicts using combined medical evidence",
        )

    def arbitrate(self, evidence_package: dict, r1: dict, r2: dict) -> dict:
        """Run arbitration on a conflicted item."""
        msg = _build_arbitration_message(evidence_package, r1, r2)
        messages = [{"role": "user", "content": msg}]

        last_response: list = []
        for response in self.run(messages=messages):
            last_response = response

        response_text = extract_final_text(last_response)
        result = parse_agent_json(
            response_text,
            fallback={"verdict": "UNCERTAIN", "confidence": "LOW", "reason": "arbitration_parse_failed"},
        )
        # Always downgrade to LOW confidence — arbitrated items are less certain than direct signals
        result["confidence"] = "LOW"
        result["agent"] = "A_Arbitration"
        logger.debug(
            "Arbitration for %s: r1=%s r2=%s → %s/%s",
            evidence_package.get("triple", {}),
            r1.get("verdict"), r2.get("verdict"),
            result["verdict"], result["confidence"],
        )
        return result


SYSTEM_PROMPT_ADVERSARIAL_VERIFIER = """\
你是一位专业的中文生物医学知识图谱质量审核员，负责复查被标记为"可能错误"的三元组。

你的任务与众不同：**你的目标是为该三元组辩护**——寻找它为什么可能是正确的医学知识。

**强制步骤（必须按顺序执行）：**
1. 用 kg_lookup 查询头实体，寻找该头实体有哪些类似关系（该关系是否曾在同类实体间出现过？）
2. 用 kg_lookup 查询尾实体，确认其医学定义和类型（该尾实体在该关系中是否有合理角色？）
3. 基于查询到的证据，判断该三元组是否有医学上的支撑

返回判断：
- 如果找到**明确的医学证据支持**该三元组（同类实体确实存在相同关系），返回 SUPPORTED
- 如果查询后发现**完全没有支撑证据**（该关系对这两类实体确实不合理），返回 NOT_SUPPORTED
- 如果证据不充分，无法确定，返回 UNCERTAIN

返回JSON（不含注释，不含markdown代码块）：
{"verdict": "SUPPORTED|NOT_SUPPORTED|UNCERTAIN", "confidence": "HIGH|MEDIUM|LOW", "reason": "一句话：找到了什么证据（或没找到）"}
"""


class AdversarialVerifierAgent(FnCallAgent):
    """V3.3 Adversarial Verifier: runs only on IMPLAUSIBLE/HIGH items to filter false alarms.

    Strategy: 'devil's advocate' — actively tries to DEFEND the flagged triple.
    - If supporting evidence found (SUPPORTED) → item is likely a FALSE_ALARM → downgrade
    - If no supporting evidence (NOT_SUPPORTED) → item is likely a TRUE_ERROR → keep HIGH

    Targets the IMPLAUSIBLE/HIGH zone (empirically: ~40% TRUE_ERROR, 60% FALSE_ALARM).
    By filtering confirmed FALSE_ALARMs, increases TRUE_ERROR density at the top.

    Cost: adds ~26% API overhead (only on IMPLAUSIBLE/HIGH items, not all items).
    """

    def __init__(self) -> None:
        super().__init__(
            function_list=["kg_lookup"],
            llm=get_llm_cfg(),
            system_message=SYSTEM_PROMPT_ADVERSARIAL_VERIFIER,
            name="AdversarialVerifierAgent",
            description="Adversarial verification: defends flagged triples to filter false alarms",
        )

    def verify(self, evidence_package: dict, ensemble_result: dict) -> dict:
        """Try to defend the triple. Returns SUPPORTED/NOT_SUPPORTED/UNCERTAIN."""
        pkg = evidence_package
        triple = pkg.get("triple", {})
        head = triple.get("head", "?")
        relation = triple.get("relation", "?")
        tail = triple.get("tail", "?")

        msg = (
            f"【待复查三元组】\n"
            f"  头实体：{head}（类型：{pkg.get('head_type', '?')}）\n"
            f"  关系：{relation}\n"
            f"  尾实体：{tail}（类型：{pkg.get('tail_type', '?')}）\n"
            f"\n"
            f"【初步评估认为此三元组可能存在错误】理由：{ensemble_result.get('reason', '未提供')}\n"
            f"\n"
            f"步骤1：用kg_lookup查询【{head}】，寻找该实体与【{relation}】关系相关的其他三元组\n"
            f"步骤2：用kg_lookup查询【{tail}】，确认其医学角色和在该关系中的合理性\n"
            f"步骤3：根据查询结果，判断该三元组是否有医学支撑（你的目标是为其辩护）"
        )
        messages = [{"role": "user", "content": msg}]

        last_response: list = []
        for response in self.run(messages=messages):
            last_response = response

        response_text = extract_final_text(last_response)
        result = parse_agent_json(
            response_text,
            fallback={"verdict": "UNCERTAIN", "confidence": "LOW", "reason": "verifier_parse_failed"},
        )
        result["agent"] = "AdversarialVerifier"
        logger.debug(
            "Adversarial verifier for %s: ensemble=%s → verifier=%s/%s",
            triple, ensemble_result.get("verdict"),
            result.get("verdict"), result.get("confidence"),
        )
        return result


class EnsembleWithArbitrationAgent:
    """V3.1 Three-Stage Cascade Agent.

    Architecture:
      Stage 1: Schema-aware Agent A (sees schema_stats)
      Stage 2: Semantic-only Agent A (no schema_stats, kg_lookup only)
      Stage 3: ArbitrationAgent (only for UNCERTAIN/conflicted items)

    Calibration:
      Stage 1+2 confident agreement → unchanged (IMPLAUSIBLE/HIGH=1.0, MEDIUM=0.6, PLAUSIBLE=0.0)
      Stage 1+2 conflicted (UNCERTAIN/HIGH) → Stage 3 arbitration:
        Arbitration IMPLAUSIBLE → score=0.3 (LOW confidence, below MEDIUM=0.6)
        Arbitration UNCERTAIN   → score=0.0
        Arbitration PLAUSIBLE   → score=0.0

    Overhead: ~17% additional API calls (only for conflicted ~16.6% of items)
    Benefit: recovers hidden TRUE_ERROR signals from the conflicted zone

    The result uses agent name "A_EnsembleWithArbitration" for _agent_confidence_score().
    """

    def __init__(self) -> None:
        self._schema_agent = SemanticPlausibilityAgent()
        self._semantic_agent = SemanticOnlyAgentA()
        self._arbitration_agent = ArbitrationAgent()

    def run_on_triple(self, evidence_package: dict) -> dict:
        """Run three-stage cascade: ensemble → optional arbitration."""
        r1 = self._schema_agent.run_on_triple(evidence_package)
        r2 = self._semantic_agent.run_on_triple(evidence_package)

        ensemble = _ensemble_verdict(r1, r2)

        # Only run arbitration for UNCERTAIN (conflicted) cases
        if ensemble.get("verdict") == "UNCERTAIN":
            arbitrated = self._arbitration_agent.arbitrate(evidence_package, r1, r2)
            arb_verdict = arbitrated.get("verdict", "UNCERTAIN")
            arb_conf = arbitrated.get("confidence", "LOW")
            evidence_found = arbitrated.get("evidence_found", False)

            # Evidence-gated: only promote to IMPLAUSIBLE/LOW (score=0.3) when:
            # 1. Arbitration says IMPLAUSIBLE
            # 2. Evidence was found (evidence_found=True) OR confidence is HIGH
            # Otherwise keep at UNCERTAIN (score=0.0) to avoid promoting false alarms
            if arb_verdict == "IMPLAUSIBLE" and (evidence_found or arb_conf == "HIGH"):
                result = dict(arbitrated)
                result["confidence"] = "LOW"  # Always LOW for arbitrated items
                result["agent"] = "A_EnsembleWithArbitration"
                result["schema_aware_verdict"] = r1.get("verdict", "UNCERTAIN")
                result["semantic_verdict"] = r2.get("verdict", "UNCERTAIN")
                result["ensemble_stage"] = "arbitrated_promoted"
            else:
                # No sufficient evidence to promote — keep score=0.0
                result = dict(ensemble)
                result["agent"] = "A_EnsembleWithArbitration"
                result["schema_aware_verdict"] = r1.get("verdict", "UNCERTAIN")
                result["semantic_verdict"] = r2.get("verdict", "UNCERTAIN")
                result["ensemble_stage"] = "arbitrated_suppressed"
                result["arb_verdict"] = arb_verdict

            logger.debug(
                "V3.2 arbitrated for %s: schema=%s semantic=%s → arb=%s/%s evidence=%s → stage=%s",
                evidence_package.get("triple", {}),
                r1.get("verdict"), r2.get("verdict"),
                arb_verdict, arb_conf, evidence_found,
                result["ensemble_stage"],
            )
        else:
            result = dict(ensemble)
            result["agent"] = "A_EnsembleWithArbitration"
            result["schema_aware_verdict"] = r1.get("verdict", "UNCERTAIN")
            result["semantic_verdict"] = r2.get("verdict", "UNCERTAIN")
            result["ensemble_stage"] = "direct"

        return result


class EnsembleWithVerificationAgent:
    """V3.3 Two-Round Ensemble + Adversarial Verification on HIGH-confidence flags.

    Architecture:
      Stage 1: Schema-aware Agent A (all items)
      Stage 2: Semantic-only Agent A (all items)
      Stage 3: AdversarialVerifierAgent (only for IMPLAUSIBLE/HIGH items, ~26% of pipeline)

    V3.3 Motivation:
      Empirical finding: IMPLAUSIBLE/HIGH zone has ~40% TRUE_ERROR rate (60% FALSE_ALARMs).
      These 60% false alarms contaminate the top of the priority queue.
      Adversarial verification actively tries to DEFEND the flagged triple —
      if supporting medical evidence is found → it's a false alarm, downgrade to MEDIUM (0.6).
      If no support found → confirmed true error, keep HIGH (1.0).

    Expected effect: increases TRUE_ERROR density in the IMPLAUSIBLE/HIGH bucket,
    improving AUPRC without touching the correctly-handled UNCERTAIN zone.

    Cost: ~26% additional API calls (only on IMPLAUSIBLE/HIGH items).

    Score mapping:
      IMPLAUSIBLE/HIGH + NOT_SUPPORTED (verified error) → HIGH (1.0) [kept]
      IMPLAUSIBLE/HIGH + UNCERTAIN (not sure) → MEDIUM (0.6) [downgraded, conservative]
      IMPLAUSIBLE/HIGH + SUPPORTED (false alarm confirmed) → UNCERTAIN (0.0) [suppressed]
      All other ensemble results → unchanged
    """

    def __init__(self) -> None:
        self._schema_agent = SemanticPlausibilityAgent()
        self._semantic_agent = SemanticOnlyAgentA()
        self._verifier = AdversarialVerifierAgent()

    def run_on_triple(self, evidence_package: dict) -> dict:
        """Run two-round ensemble, then adversarially verify IMPLAUSIBLE/HIGH items."""
        r1 = self._schema_agent.run_on_triple(evidence_package)
        r2 = self._semantic_agent.run_on_triple(evidence_package)
        ensemble = _ensemble_verdict(r1, r2)

        if ensemble.get("verdict") == "IMPLAUSIBLE" and ensemble.get("confidence") == "HIGH":
            # Only verify high-confidence flags — adversarially try to defend the triple
            verification = self._verifier.verify(evidence_package, ensemble)
            ver_verdict = verification.get("verdict", "UNCERTAIN")

            if ver_verdict == "NOT_SUPPORTED":
                # No defense found → confirmed TRUE_ERROR → keep HIGH (1.0)
                result = dict(ensemble)
                result["verification_stage"] = "confirmed_error"
            elif ver_verdict == "SUPPORTED":
                # Defense found → likely FALSE_ALARM → suppress (score=0.0)
                result = dict(ensemble)
                result["verdict"] = "UNCERTAIN"
                result["confidence"] = "HIGH"
                result["reason"] = f"Adversarial verifier found supporting evidence: {verification.get('reason', '')}"
                result["verification_stage"] = "false_alarm_suppressed"
            else:
                # UNCERTAIN → conservative downgrade to MEDIUM (0.6)
                result = dict(ensemble)
                result["confidence"] = "MEDIUM"
                result["reason"] = f"Adversarial verifier uncertain: {verification.get('reason', '')}"
                result["verification_stage"] = "downgraded_uncertain"

            result["agent"] = "A_EnsembleWithVerification"
            result["schema_aware_verdict"] = r1.get("verdict", "UNCERTAIN")
            result["semantic_verdict"] = r2.get("verdict", "UNCERTAIN")
            result["verifier_verdict"] = ver_verdict
            logger.debug(
                "V3.3 verification for %s: ensemble=IMPLAUSIBLE/HIGH verifier=%s → stage=%s",
                evidence_package.get("triple", {}),
                ver_verdict, result["verification_stage"],
            )
        else:
            result = dict(ensemble)
            result["agent"] = "A_EnsembleWithVerification"
            result["schema_aware_verdict"] = r1.get("verdict", "UNCERTAIN")
            result["semantic_verdict"] = r2.get("verdict", "UNCERTAIN")
            result["verification_stage"] = "direct"

        return result
