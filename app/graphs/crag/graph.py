"""CRAG (Corrective RAG) LangGraph graph construction and compilation.

Improvement loop order (matches ``app.core.improvement_loop``):

    analyze_query          → analysis
    decide_rewrite         → verification
    rewrite_query          → verification (conditional)
    hybrid_retrieve        → search
    test_retrieval         → test
    generate_answer        → evaluation
    run_judge              → evaluation
    evaluate_grounding     → verification (post)
    [feedback routers]     → retry_retrieval | retry_generation | retry_with_policy
                             | finalize_ok | finalize_rejected
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graphs.crag.nodes import (
    analyze_query,
    ask_clarification,
    decide_rewrite,
    evaluate_grounding,
    finalize_ok,
    finalize_rejected,
    generate_answer,
    hybrid_retrieve,
    retry_generation,
    retry_retrieval,
    retry_with_policy,
    rewrite_query,
    run_judge,
    test_retrieval,
)
from app.graphs.crag.routes import (
    route_after_analyze,
    route_after_feedback_eval,
    route_after_rewrite_decision,
)
from app.graphs.crag.state import CRAGState


def build_crag_graph() -> StateGraph:
    """Construct and compile the CRAG StateGraph.

    Returns:
        A compiled LangGraph ``StateGraph`` ready for ``.invoke()`` or
        ``.stream()`` calls.
    """
    graph = StateGraph(CRAGState)

    # ── Register nodes (improvement-loop order) ───────────────────────────────
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("decide_rewrite", decide_rewrite)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("test_retrieval", test_retrieval)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("run_judge", run_judge)
    graph.add_node("evaluate_grounding", evaluate_grounding)
    graph.add_node("retry_retrieval", retry_retrieval)
    graph.add_node("retry_generation", retry_generation)
    graph.add_node("retry_with_policy", retry_with_policy)
    graph.add_node("finalize_ok", finalize_ok)
    graph.add_node("finalize_rejected", finalize_rejected)

    # ── Phase 1–2: analysis → verification ───────────────────────────────────
    graph.set_entry_point("analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        {"ask_clarification": "ask_clarification", "decide_rewrite": "decide_rewrite"},
    )
    graph.add_edge("ask_clarification", END)

    graph.add_conditional_edges(
        "decide_rewrite",
        route_after_rewrite_decision,
        {"rewrite_query": "rewrite_query", "hybrid_retrieve": "hybrid_retrieve"},
    )
    graph.add_edge("rewrite_query", "hybrid_retrieve")

    # ── Phase 3–4: search → test ──────────────────────────────────────────────
    graph.add_edge("hybrid_retrieve", "test_retrieval")

    # ── Phase 5–6: evaluation → verification ───────────────────────────────────
    graph.add_edge("test_retrieval", "generate_answer")
    graph.add_edge("generate_answer", "run_judge")
    graph.add_edge("run_judge", "evaluate_grounding")

    # ── Phase 7: feedback loop ────────────────────────────────────────────────
    graph.add_conditional_edges(
        "evaluate_grounding",
        route_after_feedback_eval,
        {
            "end": "finalize_ok",
            "reject": "finalize_rejected",
            "retry_retrieval": "retry_retrieval",
            "retry_generation": "retry_generation",
            "retry_with_policy": "retry_with_policy",
        },
    )
    graph.add_edge("finalize_ok", END)
    graph.add_edge("finalize_rejected", END)
    graph.add_edge("retry_retrieval", "decide_rewrite")
    graph.add_edge("retry_generation", "generate_answer")
    graph.add_edge("retry_with_policy", "decide_rewrite")

    return graph.compile()


crag_chain = build_crag_graph()
