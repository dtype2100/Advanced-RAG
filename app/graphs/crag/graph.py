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
      ↓ (grounding ok?) → END
      ↓ (low grounding, retries remain)
    run_judge
      ↓ (accept?) → END
      ↓ (retry?) → retry_with_policy → decide_rewrite (loop, max 3×)
      ↓ (reject?) → mark_rejected → END
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
    mark_rejected,
    rerank_context,
    retry_with_policy,
    rewrite_query,
    run_judge,
)
from app.graphs.crag.routes import (
    route_after_analyze,
    route_after_grounding,
    route_after_judge_eval,
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

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("decide_rewrite", decide_rewrite)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("hybrid_retrieve", hybrid_retrieve)
    graph.add_node("expand_context", expand_context)
    graph.add_node("rerank_context", rerank_context)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("evaluate_grounding", evaluate_grounding)
    graph.add_node("run_judge", run_judge)
    graph.add_node("retry_with_policy", retry_with_policy)
    graph.add_node("mark_rejected", mark_rejected)

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

    graph.add_conditional_edges(
        "hybrid_retrieve",
        route_after_retrieve,
        {"expand_context": "expand_context", "rerank_context": "rerank_context"},
    )
    graph.add_edge("expand_context", "rerank_context")
    graph.add_edge("rerank_context", "generate_answer")
    graph.add_edge("generate_answer", "evaluate_grounding")

    graph.add_conditional_edges(
        "evaluate_grounding",
        route_after_grounding,
        {"run_judge": "run_judge", "end": END},
    )

    graph.add_conditional_edges(
        "run_judge",
        route_after_judge_eval,
        {
            "retry_with_policy": "retry_with_policy",
            "mark_rejected": "mark_rejected",
            "end": END,
        },
    )
    graph.add_edge("retry_with_policy", "decide_rewrite")
    graph.add_edge("mark_rejected", END)

    return graph.compile()


crag_chain = build_crag_graph()
