"""LangGraph self-corrective RAG graph construction.

Pipeline (with optional reranking):

    retrieve → [rerank_documents] → grade_documents →[relevant?]→ generate → END
                                                      →[irrelevant]→ rewrite_query → retrieve (loop)

The ``rerank_documents`` node is always present in the graph.  When
``RERANKER_PROVIDER=none`` (the default) it acts as a transparent pass-through
so no performance overhead is incurred.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.rag.nodes import (
    generate,
    grade_documents,
    rerank_documents,
    retrieve,
    rewrite_query,
    should_retry,
)
from app.rag.state import RAGState


def build_rag_graph() -> StateGraph:
    """Build and compile the self-corrective RAG graph.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank_documents", rerank_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "rerank_documents")
    graph.add_edge("rerank_documents", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        should_retry,
        {
            "generate": "generate",
            "rewrite": "rewrite_query",
        },
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


rag_chain = build_rag_graph()
