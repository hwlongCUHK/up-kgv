# UP-KGV: Uncertainty-Prioritized Knowledge Graph Verification

A human-in-the-loop pipeline for prioritizing error review in large, automatically extracted biomedical knowledge graphs.

## Overview

UP-KGV addresses the problem: given a KG with millions of automatically extracted triples and a limited review budget *K*, which triples should human curators inspect first?

The key insight is that **schema–semantic disagreement is informative**. When a triple looks structurally anomalous but is semantically plausible (or vice versa), it is disproportionately likely to be a true error. UP-KGV exploits this signal to build a ranked review queue without requiring labeled training data from the target KG.

Evaluated on CPubMed-KGv2\_0 (4.58M triples), UP-KGV achieves **AUPRC = 0.782** and finds **2.7× more errors per review** than unsorted inspection at *K* = 100.

## Pipeline

```
KG (4.58M triples)
    │
    ▼ Stage 1 — Deterministic detectors
    │   SelfLoopFilter, SchemaPatternAnalyzer, CardinalityAnalyzer,
    │   TopologyScorer, TypeConflictChecker
    │   → ~12K Machine_Suspected triples
    │
    ▼ Stage 2 — Structural scoring
    │   Composite anomaly score C(x) per triple
    │   Low-scoring triples → AUTO_CLEAN (bypass LLM)
    │
    ▼ Stage 3 — Evidence package construction
    │   Per-triple: entity types, schema stats, neighbour summaries,
    │   contradiction pairs, symmetry gaps, topology features
    │
    ▼ Stage 4 — Specialist LLM routing
    │   SemanticPlausibilityEnsemble  (two-round: schema-aware + semantic-only)
    │   RelationLabelAuditor          (schema-extreme triples)
    │   EntityLinkerTyper             (borderline schema + type conflict)
    │   ContradictionArbiter          (contradiction pairs)
    │   ContextualCoherenceVerifier   (uncertain hub entities)
    │
    ▼ Stage 5 — Priority scoring
    │   S(x) = 0.55·A(x) + 0.35·C(x) + 0.10·R(x) + λ·1[V(x)=UNCERTAIN]·C(x)
    │   Top-K queue exported for human review
    │
    ▼ Stage 6 — Patch writing (after human decisions)
        Approved repairs written back to KG
```

## Installation

```bash
pip install -e .
# or with uv:
uv sync
```

Requires a `.env` file with your API key:

```env
DEEPSEEK_API_KEY=your_key_here
# Optional: use a different model family for gold labels
OPENAI_API_KEY=your_key_here
```

## Usage

```python
from kg_verify.orchestrator import KGVerifyHITLOrchestrator

pipeline = KGVerifyHITLOrchestrator()
pipeline.run()
# Exports review_items.tsv to outputs/ for human review
```

**Simulation mode** (LLM plays the human reviewer, for benchmarking):

```bash
PIPELINE_MODE=simulation python -m kg_verify.orchestrator
```

**Enable the two-round ensemble** (recommended for production):

```bash
ENSEMBLE_AGENT_A=1 python -m kg_verify.orchestrator
```

## Configuration

All thresholds, weights, and model settings are in `config.py`. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `AUTO_CLEAN_SCORE` | 0.15 | C(x) below this → skip LLM review |
| `SCHEMA_THRESHOLD_EXTREME` | 0.005 | Pattern frequency below this → RelationLabelAuditor |
| `HUB_DEGREE_THRESHOLD` | 5000 | Degree above this → ContextualCoherenceVerifier |
| `USE_ENSEMBLE_AGENT_A` | False | Enable two-round schema/semantic ensemble |
| `PRIORITY_WEIGHTS` | see config | Weights for S(x) scoring formula |

## Repository Structure

```
kg_verify/
├── agents/          # LLM specialist agents (5 specialists)
├── audit/           # Audit logging
├── patches/         # KG patch writing (post-review repairs)
├── pipeline/        # Statistical detectors and evidence builder
├── review/          # Review queue construction and export
├── tools/           # KG index and schema stats utilities
├── config.py        # All configuration in one place
└── orchestrator.py  # 6-stage pipeline entry point
```

## Citation

If you use this code, please cite:

```bibtex
@article{upkgv2025,
  title   = {UP-KGV: Uncertainty-Prioritized Knowledge Graph Verification},
  author  = {[Authors]},
  year    = {2025}
}
```

## License

MIT
