"""Canonical improvement loop phase order for this RAG project.

Both the CRAG LangGraph pipeline and offline eval runner follow this sequence:

    analysis → verification → search → test → evaluation → verification → feedback

Each phase maps to concrete nodes (runtime) or scripts (offline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PhaseName = Literal[
    "analysis",
    "verification",
    "search",
    "test",
    "evaluation",
    "verification_post",
    "feedback",
]


@dataclass(frozen=True)
class PipelinePhase:
    """One step in the analysis-to-feedback improvement loop."""

    name: PhaseName
    label: str
    graph_nodes: tuple[str, ...]
    eval_target: str | None = None


# Ordered loop — do not reorder without updating graph.py and run_evals.py together.
IMPROVEMENT_LOOP: tuple[PipelinePhase, ...] = (
    PipelinePhase(
        name="analysis",
        label="Query analysis (intent, slots, ambiguity)",
        graph_nodes=("analyze_query",),
        eval_target="evals/offline/run_clarification_eval.py",
    ),
    PipelinePhase(
        name="verification",
        label="Input verification (clarification / rewrite decision)",
        graph_nodes=("decide_rewrite", "rewrite_query"),
        eval_target=None,
    ),
    PipelinePhase(
        name="search",
        label="Hybrid retrieval (vector + BM25)",
        graph_nodes=("hybrid_retrieve",),
        eval_target="evals/offline/run_retrieval_eval.py",
    ),
    PipelinePhase(
        name="test",
        label="Retrieval quality test (expand, rerank, relevance filter)",
        graph_nodes=("test_retrieval",),
        eval_target="tests/unit",
    ),
    PipelinePhase(
        name="evaluation",
        label="Answer generation + LLM-as-judge",
        graph_nodes=("generate_answer", "run_judge"),
        eval_target="evals/offline/run_answer_eval.py",
    ),
    PipelinePhase(
        name="verification_post",
        label="Grounding verification",
        graph_nodes=("evaluate_grounding",),
        eval_target="evals/offline/run_judge_eval.py",
    ),
    PipelinePhase(
        name="feedback",
        label="Feedback loop (retry retrieval / regeneration / reject)",
        graph_nodes=(
            "retry_retrieval",
            "retry_generation",
            "retry_with_policy",
            "finalize_ok",
            "finalize_rejected",
        ),
        eval_target="evals/offline/run_feedback_eval.py",
    ),
)


def graph_node_order() -> list[str]:
    """Flat list of CRAG graph nodes in pipeline phase order."""
    nodes: list[str] = []
    for phase in IMPROVEMENT_LOOP:
        nodes.extend(phase.graph_nodes)
    return nodes
