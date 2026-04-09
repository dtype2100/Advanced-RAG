"""CRAG (Corrective RAG) LangGraph graph construction and compilation.

Graph flow:
    analyze_query
      ↓ (needs_clarification?)
    [ask_clarification] → END (clarification_needed)
      ↓
    decide_rewrite
      ↓ (needs_rewrite?)
    [rewrite_query]
      ↓
    hybrid_retrieve
      ↓ (should_expand?)
    [expand_context]
      ↓
    rerank_context
      ↓
    generate_answer
      ↓
    evaluate_grounding
      ↓ (grounding_score < threshold AND retries < max?)
    [retry_with_policy] → decide_rewrite (loop, max 3×)
      ↓
    END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graphs.crag.nodes import (
    analyze_query,
    ask_clarification,
    decide_rewrite,
    evaluate_grounding,
    expand_context,
    generate_answer,
    hybrid_retrieve,
    rerank_context,
    retry_with_policy,
    rewrite_query,
)
from app.graphs.crag.routes import (
    route_after_analyze,
    route_after_grounding,
    route_after_retrieve,
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

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("decide_rewrite", decide_rewrite)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("expand_context", expand_context)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_grounding", evaluate_grounding)
    graph.add_node("retry_with_policy", retry_with_policy)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.set_entry_point("analyze_query")

    # ── Conditional: clarification ────────────────────────────────────────────
    graph.add_conditional_edges(
        "analyze_query",
        route_after_analyze,
        {"ask_clarification": "ask_clarification", "decide_rewrite": "decide_rewrite"},
    )
    graph.add_edge("ask_clarification", END)

    # ── Conditional: rewrite ──────────────────────────────────────────────────
    graph.add_conditional_edges(
        "decide_rewrite",
        route_after_rewrite_decision,
        {"rewrite_query": "rewrite_query", "hybrid_retrieve": "hybrid_retrieve"},
    )
    graph.add_edge("rewrite_query", "hybrid_retrieve")

    # ── Conditional: context expansion ───────────────────────────────────────
    graph.add_conditional_edges(
        "hybrid_retrieve",
        route_after_retrieve,
        {"expand_context": "expand_context", "rerank_context": "rerank_context"},
    )
    graph.add_edge("expand_context", "rerank_context")
    graph.add_edge("rerank_context", "generate_answer")
    graph.add_edge("generate_answer", "evaluate_grounding")

    # ── Conditional: hallucination feedback loop (max 3×) ────────────────────
    graph.add_conditional_edges(
        "evaluate_grounding",
        route_after_grounding,
        {"retry_with_policy": "retry_with_policy", "end": END},
    )
    graph.add_edge("retry_with_policy", "decide_rewrite")

    return graph.compile()


crag_chain = build_crag_graph()
