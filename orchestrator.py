"""
KGVerifyHITLOrchestrator: 6-stage HITL pipeline orchestrator.

Stage 1: Deterministic detection (SelfLoopFilter, SchemaPatternAnalyzer, etc.)
Stage 2: Machine triage (ScoringModule classifies each triple)
Stage 3: Evidence package building (EvidencePackageBuilder per MACHINE_SUSPECTED)
Stage 4: Agent reasoning (LLM agents invoked per ROUTING_TABLE)
Stage 5: Review queue construction and export
Stage 6: Patch writing (only after human/simulated decisions)
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import config
from .pipeline import (
    EvidencePackageBuilder,
    CardinalityAnalyzer,
    SchemaPatternAnalyzer,
    ScoringModule,
    SelfLoopFilter,
    SymmetryValidator,
    TopologyScorer,
    TypeConflictChecker,
    load_kg,
)
from .tools import init_kg_index, init_schema_analyzer
from .agents import (
    ContradictionArbiter,
    ContextualCoherenceVerifier,
    EntityLinkerTyper,
    RelationLabelAuditor,
    RepairSynthesizer,
    SemanticPlausibilityAgent,
    SemanticPlausibilityEnsembleAgent,
    SimulatedHumanReviewerAgent,
)
from .review import ReviewItem, ReviewQueueBuilder, ReviewExporter, export_auto_safe
from .patches import PatchWriter
from .audit import AuditLog

logger = logging.getLogger(__name__)

ROUTING_TABLE = {
    # V2.2 fix (Loop 2, Round 2): Remove Agent F (RepairSynthesizer) from all routes.
    # C-series ablation (seed=202): C4 (no repair) = 0.9442 >> C0 (with repair) = 0.7065.
    # Agent F hurts ranking by -0.238 AUPRC. Hypothesis: repair suggestions inflate
    # priority scores for false positives (triples routed to schema_extreme or
    # high_risk_medical are disproportionately valid triples flagged by statistical
    # rarity, not genuine errors). Removing Agent F gives cleaner Agent A signal.
    # R4 fix preserved: Agent A in every active route for _agent_confidence_score().
    "auto_safe":          [],
    "type_conflict":      ["agent_c", "agent_a"],
    "unknown_type":       ["agent_c", "agent_a"],
    "schema_borderline":  ["agent_a", "agent_b"],
    "schema_extreme":     ["agent_a"],
    "cardinality":        ["agent_a", "agent_b"],
    "contradiction":      ["agent_a", "agent_d"],
    "uncertain_leaf":     ["agent_a"],
    "uncertain_hub":      ["agent_a"],
    "high_risk_medical":  ["agent_a"],
}

# V2: Machine-generated suspicion brief per route type.
# Replaces raw neighbor context (which hurt Agent A calibration in 4-round ablation)
# with a structured explanation of WHY the triple was statistically flagged.
# Route hints guide Agent A's reasoning step 0 ("why is this suspicious?")
# before it applies domain knowledge.
ROUTE_HINT_REASONS: dict[str, str] = {
    "schema_extreme":     "统计预筛查：该(类型,关系,类型)组合在知识图谱中极为罕见（频率<0.5%），属于极端模式违反，需重点核实",
    "schema_borderline":  "统计预筛查：该(类型,关系,类型)组合的频率介于0.5%-5%之间，属于边缘异常，请判断是否合理的罕见关系",
    "cardinality":        "统计预筛查：该实体在此关系上的连接数量显著超出同类实体中位数，请检查是否存在错误录入或实体混淆",
    "contradiction":      "结构检测：该三元组与图谱中的现有三元组存在语义矛盾（如同时存在'治疗'与'禁忌'），请核实哪一条正确",
    "high_risk_medical":  "领域规则：该三元组涉及高风险医疗关系（禁忌症/副作用/毒性/药物治疗等），错误可能造成医疗安全风险，请严格核实",
    "uncertain_hub":      "统计预筛查：涉及高连接度枢纽实体，综合可疑评分超过检测阈值，请判断该关系在医学上是否成立",
    "uncertain_leaf":     "统计预筛查：该三元组的综合可疑评分超过检测阈值，schema模式分析认为此类型组合不常见，请语义核实",
    "type_conflict":      "结构检测：该实体在知识图谱中被赋予多种不一致的类型标注，可能存在实体混淆或标注错误",
}


def _build_route_hint(route: str, res: dict, schema_analyzer=None) -> dict:
    """Build a structured route hint for Agent A's evidence package.

    V2 innovation: instead of raw neighbor context, Agent A receives a
    machine-generated 'suspicion brief' explaining the statistical reason
    for flagging. This gives targeted reasoning guidance without the noise
    of neighborhood lists.
    """
    schema_score = res.get("schema_score", 0.0)
    pattern_freq = round(1.0 - schema_score, 4)  # higher score → rarer pattern
    return {
        "route_type": route,
        "reason": ROUTE_HINT_REASONS.get(route, "统计预筛查：综合评分超过可疑阈值"),
        "pattern_frequency": pattern_freq,
        "schema_violation_score": round(schema_score, 4),
    }


def _agent_a_pkg(pkg: dict) -> dict:
    """V2: return the route-conditioned minimal evidence package for Agent A.

    Strips noisy neighbor context; keeps triple, types, schema stats, contradiction
    pairs, and route_hint (if USE_ROUTE_HINTS=True and route was injected).
    Importable by run_ablation.py for consistent V2 evidence format.
    """
    return {
        "triple": pkg.get("triple", {}),
        "head_type": pkg.get("head_type", "Unknown"),
        "tail_type": pkg.get("tail_type", "Unknown"),
        "schema_stats": pkg.get("schema_stats", {"pattern_frequency": 0.0, "dominant_patterns": []}),
        "contradiction_pairs": pkg.get("contradiction_pairs", []),
        "route_hint": pkg.get("route_hint", {}),
    }


def _determine_route(result: dict, topology_scorer: TopologyScorer) -> str:
    """Map a scored triple to a routing key."""
    h, r, t = result["triple"]
    signals = result["explicit_signals"]
    schema_s = result["schema_score"]
    card_s = result["cardinality_score"]

    if r in config.HIGH_RISK_RELATIONS:
        return "high_risk_medical"
    if "type_conflict" in signals:
        return "type_conflict"
    if "self_loop" in signals or "exact_duplicate" in signals:
        return "auto_safe"
    if schema_s >= (1.0 - config.SCHEMA_THRESHOLD_EXTREME):
        return "schema_extreme"
    if card_s >= 0.5:
        return "cardinality"
    if schema_s >= (1.0 - config.SCHEMA_THRESHOLD_BORDERLINE):
        return "schema_borderline"
    if topology_scorer.is_hub(h) or topology_scorer.is_hub(t):
        return "uncertain_hub"
    return "uncertain_leaf"


class KGVerifyHITLOrchestrator:
    """Full 6-stage KG verification pipeline."""

    def __init__(self) -> None:
        self._agents: dict = {}
        self._triple_set: set[tuple[str, str, str]] = set()
        self._output_dir = config.OUTPUT_DIR
        self._flat_llm_mode: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(
        self,
        kg_path: Optional[str] = None,
        max_triples: Optional[int] = None,
        max_agent_items: Optional[int] = None,
        flat_llm_mode: bool = False,
    ) -> dict:
        """
        Execute the full pipeline end-to-end.

        Args:
            kg_path: path to KG file (defaults to config.KG_FILE)
            max_triples: limit triples loaded (for debugging)
            max_agent_items: limit MACHINE_SUSPECTED items sent to agents

        Returns:
            Summary dict with counts per stage
        """
        t0 = time.time()
        kg_path = str(kg_path or config.KG_FILE)
        self._flat_llm_mode = flat_llm_mode or config.FLAT_LLM_MODE

        # ── Stage 1: Load KG ──────────────────────────────────────────────────
        logger.info("=== Stage 1: Loading KG ===")
        triples, entity_types = load_kg(kg_path, max_triples=max_triples)
        self._triple_set = set(triples)

        # ── Stage 1b: Fit detectors ───────────────────────────────────────────
        logger.info("=== Stage 1b: Fitting detectors ===")
        schema_analyzer = SchemaPatternAnalyzer()
        schema_analyzer.fit(triples, entity_types)

        cardinality_analyzer = CardinalityAnalyzer()
        cardinality_analyzer.fit(triples, entity_types=entity_types)

        topology_scorer = TopologyScorer()
        topology_scorer.fit(triples)

        symmetry_validator = SymmetryValidator()
        symmetry_validator.fit(triples)

        type_checker = TypeConflictChecker()
        multi_type, type_conflict_signals = type_checker.run(entity_types, triples)

        type_conflict_triples: set[tuple] = {
            sig["triple"] for sig in type_conflict_signals
        }

        self_loop_filter = SelfLoopFilter()
        self_loop_signals = self_loop_filter.run(triples)
        self_loop_triples = {sig["triple"] for sig in self_loop_signals}

        # ── Stage 2: Scoring ──────────────────────────────────────────────────
        logger.info("=== Stage 2: Scoring triples ===")
        scoring_module = ScoringModule()

        schema_scores = {
            (h, r, t): schema_analyzer.score(h, r, t, entity_types=entity_types)
            for h, r, t in triples
        }
        cardinality_scores = {
            (h, r): cardinality_analyzer.score(h, r, entity_types.get(h))
            for h, r, t in triples
        }
        structural_scores = {
            (h, r, t): topology_scorer.score(h, t)
            for h, r, t in triples
        }

        scored = scoring_module.classify_batch(
            triples=triples,
            schema_scores=schema_scores,
            cardinality_scores={(h, r, t): cardinality_scores.get((h, r), 0.0)
                                 for h, r, t in triples},
            structural_scores=structural_scores,
            self_loops=self_loop_triples,
            duplicates=set(),   # duplicates detected inside classify_batch
            type_conflicts=type_conflict_triples,
        )

        auto_safe = [r for r in scored if r["classification"] == config.AUTO_ACTION_SAFE]
        machine_suspected = [r for r in scored if r["classification"] == config.MACHINE_SUSPECTED]

        logger.info(
            "Stage 2: %d auto_safe, %d machine_suspected, %d auto_clean",
            len(auto_safe),
            len(machine_suspected),
            sum(1 for r in scored if r["classification"] == config.AUTO_CLEAN),
        )

        # Export auto-safe signals
        all_auto_signals = self_loop_signals + symmetry_validator.find_missing_reverses()
        export_auto_safe(all_auto_signals, self._output_dir / "auto_action_safe.tsv")

        # ── Stage 3: Evidence package building ────────────────────────────────
        logger.info("=== Stage 3: Building evidence packages ===")
        init_kg_index(triples, entity_types)
        init_schema_analyzer(schema_analyzer)

        evidence_builder = EvidencePackageBuilder(
            triples=triples,
            entity_types=entity_types,
            schema_analyzer=schema_analyzer,
            topology_scorer=topology_scorer,
            type_conflicts=multi_type,
        )

        # Limit for development
        suspect_subset = machine_suspected[:max_agent_items] if max_agent_items else machine_suspected

        # Build ReviewItems for all MACHINE_SUSPECTED
        review_items: list[ReviewItem] = []
        for res in suspect_subset:
            h, r, t = res["triple"]
            pkg = evidence_builder.build(
                h, r, t,
                h_type=entity_types.get(h, "Unknown"),
                t_type=entity_types.get(t, "Unknown"),
            )
            item = ReviewItem.create(
                h=h, r=r, t=t,
                h_type=entity_types.get(h, "Unknown"),
                t_type=entity_types.get(t, "Unknown"),
                classification=res["classification"],
                composite_score=res["composite_score"],
                primary_signal=res["primary_signal"],
                detector_signals={
                    "schema_score": res["schema_score"],
                    "cardinality_score": res["cardinality_score"],
                    "structural_score": res["structural_score"],
                },
                evidence_package=pkg,
            )
            review_items.append(item)

        # ── Stage 4: Agent reasoning ──────────────────────────────────────────
        logger.info("=== Stage 4: Running LLM agents on %d items ===", len(review_items))
        self._init_agents()

        degree_map = topology_scorer._total_degree
        self._run_agents(review_items, scored_map={
            (h, r, t): res for res in scored for h, r, t in [res["triple"]]
        }, topology_scorer=topology_scorer)

        # ── Stage 5: Review queue ─────────────────────────────────────────────
        logger.info("=== Stage 5: Building review queue ===")
        queue_builder = ReviewQueueBuilder()
        sorted_items = queue_builder.build(review_items, degree_map=degree_map)

        exporter = ReviewExporter()
        exporter.export(sorted_items, self._output_dir / "review_items.tsv")
        logger.info("Exported review queue: %s", self._output_dir / "review_items.tsv")

        # ── Stage 5b: Simulation mode ─────────────────────────────────────────
        if config.PIPELINE_MODE == "simulation":
            logger.info("=== Stage 5b: Simulating human review (Agent G) ===")
            sorted_items = self._simulate_human_review(sorted_items)
            # Re-export TSV with simulated decisions filled in
            exporter.export(sorted_items, self._output_dir / "review_items_simulated.tsv")
            logger.info("Exported simulated review results: review_items_simulated.tsv")

        # ── Stage 6: Patch writing ────────────────────────────────────────────
        logger.info("=== Stage 6: Writing approved patches ===")
        audit_log = AuditLog(self._output_dir / "audit_log.jsonl")
        patch_writer = PatchWriter(
            output_dir=self._output_dir / "approved_patches",
            triple_set=self._triple_set,
        )

        patches_written = 0
        for item in sorted_items:
            if item.human_decision in config.WRITE_DECISIONS:
                try:
                    patch = patch_writer.write(item)
                    audit_log.record_patch(patch)
                    patches_written += 1
                except AssertionError as e:
                    logger.warning("Patch validation failed for %s: %s", item.triple_id, e)

        total_time = time.time() - t0
        summary = {
            "total_triples": len(triples),
            "total_entities": len(entity_types),
            "auto_safe": len(auto_safe),
            "machine_suspected": len(machine_suspected),
            "review_items": len(sorted_items),
            "patches_written": patches_written,
            "pipeline_mode": config.PIPELINE_MODE,
            "total_time_sec": round(total_time, 1),
        }
        logger.info("=== Pipeline complete: %s ===", summary)
        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # Agent routing and dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _init_agents(self) -> None:
        """Lazy-initialize all LLM agents (shared across items).

        V3.0: when USE_ENSEMBLE_AGENT_A=True, uses SemanticPlausibilityEnsembleAgent
        (two-round schema-aware + semantic-only) instead of single-call Agent A.
        """
        if not self._agents:
            agent_a = (
                SemanticPlausibilityEnsembleAgent()
                if config.USE_ENSEMBLE_AGENT_A
                else SemanticPlausibilityAgent()
            )
            self._agents = {
                "agent_a": agent_a,
                "agent_b": RelationLabelAuditor(),
                "agent_c": EntityLinkerTyper(),
                "agent_d": ContradictionArbiter(),
                "agent_e": ContextualCoherenceVerifier(),
                "agent_f": RepairSynthesizer(),
                "agent_g": SimulatedHumanReviewerAgent(),
            }
        logger.info("Agents initialized: %s", list(self._agents.keys()))

    def _run_agents(
        self,
        items: list[ReviewItem],
        scored_map: dict,
        topology_scorer: TopologyScorer,
    ) -> None:
        """Route each item through the appropriate agents (sequential for safety)."""
        for item in items:
            h, r, t = item.original_triple
            res = scored_map.get((h, r, t), {})
            route = _determine_route(res or {
                "triple": (h, r, t),
                "explicit_signals": [],
                "schema_score": item.detector_signals.get("schema_score", 0.0),
                "cardinality_score": item.detector_signals.get("cardinality_score", 0.0),
            }, topology_scorer)

            if self._flat_llm_mode:
                agent_names = ["agent_a"]
            else:
                agent_names = ROUTING_TABLE.get(route, ["agent_a"])
            pkg = item.evidence_package

            # V2: inject route hint into evidence package before agent invocation.
            # Route hint replaces raw neighbor context (empirically harmful in R3/R4 ablations).
            if config.USE_ROUTE_HINTS:
                pkg["route_hint"] = _build_route_hint(route, res or {
                    "schema_score": item.detector_signals.get("schema_score", 0.0),
                })

            for agent_name in agent_names:
                try:
                    verdict = self._invoke_agent(agent_name, item, pkg, topology_scorer)
                    if verdict:
                        item.apply_agent_verdict(verdict)
                        # Agent C → Agent A handoff: enrich evidence with corrected type
                        if agent_name == "agent_c" and "canonical_type" in verdict:
                            if item.head_type == "Unknown":
                                pkg["head_type"] = verdict["canonical_type"]
                                item.head_type = verdict["canonical_type"]
                except Exception as e:
                    logger.error("Agent %s failed on %s: %s", agent_name, item.triple_id, e)

    @staticmethod
    def _minimal_pkg(pkg: dict) -> dict:
        """Return a minimal evidence package for specialist agents.

        Round 4 fix: specialist agents (B, C, D, E, F) receive only the core
        triple + type information. Ablation A3 (no evidence context → +0.245
        AUPRC vs A0) showed that neighbor_summary and topology context injected
        into specialist prompts causes over-flagging of legitimate relations that
        happen to sit in unusual KG neighborhoods.
        """
        return {
            "triple": pkg.get("triple", {}),
            "head_type": pkg.get("head_type", "Unknown"),
            "tail_type": pkg.get("tail_type", "Unknown"),
            "contradictory_relations": pkg.get("contradictory_relations", []),
            "head_is_hub": pkg.get("topology", {}).get("head_is_hub", False),
        }

    def _invoke_agent(
        self,
        agent_name: str,
        item: ReviewItem,
        pkg: dict,
        topology_scorer: TopologyScorer,
    ) -> dict | None:
        """Invoke a specific agent on a review item.

        V2: Agent A gets route-conditioned minimal context (_agent_a_pkg).
        Specialist agents (B, C, D, E, F) get triple+type only (_minimal_pkg).
        """
        agent = self._agents.get(agent_name)
        if agent is None:
            return None

        if agent_name == "agent_a":
            # V2: route-conditioned minimal pkg (route_hint + schema_stats only)
            return agent.run_on_triple(_agent_a_pkg(pkg))
        elif agent_name == "agent_b":
            return agent.run_on_triple(self._minimal_pkg(pkg))
        elif agent_name == "agent_c":
            return agent.run_on_triple(self._minimal_pkg(pkg))
        elif agent_name == "agent_d":
            # Agent D contradiction arbiter: include contradictions in minimal pkg
            return agent.run_on_contradiction_pair(
                self._minimal_pkg(pkg), self._minimal_pkg(pkg)
            )
        elif agent_name == "agent_e":
            h = item.original_triple[0]
            if topology_scorer.is_hub(h):
                return agent.run_on_triple(self._minimal_pkg(pkg))
            return None
        elif agent_name == "agent_f":
            return agent.run_on_triple(self._minimal_pkg(pkg))
        return None

    def _simulate_human_review(self, items: list[ReviewItem]) -> list[ReviewItem]:
        """Use Agent G to simulate human review decisions for all items."""
        agent_g: SimulatedHumanReviewerAgent = self._agents["agent_g"]
        for item in items:
            try:
                decision = agent_g.run_on_review_item(item.to_dict())
                item.apply_simulated_review(decision)
            except Exception as e:
                logger.error("Agent G failed on %s: %s", item.triple_id, e)
                item.human_decision = "DEFER_TO_EXPERT"
                item.reviewer_id = "simulated_reviewer_g_error"
        return items
