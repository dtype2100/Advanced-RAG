"""Node implementations for the CRAG graph.

Each node receives the full ``CRAGState``, performs one responsibility,
and returns a partial state dict with only the keys it has modified.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.graphs.crag.state import CRAGState
from app.providers.llm_provider import get_llm

logger = logging.getLogger(__name__)


def _active_query(state: CRAGState) -> str:
    """Return the most refined query available in state."""
    return (
        state.get("rewritten_query") or state.get("clarified_query") or state.get("user_query", "")
    )


# ── analyze_query ─────────────────────────────────────────────────────────────


def analyze_query(state: CRAGState) -> dict:
    """Detect ambiguity, missing slots, and intent from the raw user query."""
    from app.rag.policies.clarification_policy import needs_clarification
    from app.rag.query.clarification import generate_clarification_question
    from app.rag.query.query_analyzer import analyze

    query = state.get("clarified_query") or state["user_query"]
    analysis = analyze(query)
    clarify = needs_clarification(analysis)

    result: dict = {"needs_clarification": clarify}
    if clarify:
        result["clarification_question"] = generate_clarification_question(analysis)
    return result


# ── ask_clarification ─────────────────────────────────────────────────────────


def ask_clarification(state: CRAGState) -> dict:
    """Mark the graph as pending user input and surface the clarification question."""
    logger.info("Clarification needed: %s", state.get("clarification_question"))
    return {"final_status": "clarification_needed"}


# ── decide_rewrite ────────────────────────────────────────────────────────────


def decide_rewrite(state: CRAGState) -> dict:
    """Apply the rewrite policy to determine if the query should be reformulated."""
    from app.rag.policies.rewrite_policy import needs_rewrite

    query = state.get("clarified_query") or state["user_query"]
    return {"needs_rewrite": needs_rewrite(query, state)}


# ── rewrite_query ─────────────────────────────────────────────────────────────


def rewrite_query(state: CRAGState) -> dict:
    """Rewrite the current query for improved retrieval."""
    from app.rag.query.query_rewriter import rewrite

    query = state.get("clarified_query") or state["user_query"]
    rewritten = rewrite(query)
    logger.info("Query rewritten: %s → %s", query, rewritten)
    return {"rewritten_query": rewritten}


# ── hybrid_retrieve ───────────────────────────────────────────────────────────


def hybrid_retrieve(state: CRAGState) -> dict:
    """Retrieve child chunks via hybrid search with optional multi-query RRF fusion."""
    from app.rag.retrievers.corpus_registry import get_corpus_docs
    from app.rag.retrievers.hybrid_retriever import hybrid_retrieve as run_hybrid
    from app.rag.retrievers.hybrid_retriever import reciprocal_rank_fusion

    query = _active_query(state)
    attempt = state.get("retrieval_attempt", 0) + 1
    logger.info("Retrieval attempt %d for: %s", attempt, query)

    top_k = state.get("top_k") or settings.max_retrieval_docs
    queries = [query]
    if settings.multi_query:
        from app.rag.query.multi_query_generator import generate_multi_query

        queries = generate_multi_query(query, n=3)
        logger.info("Multi-query: %d variants", len(queries))

    corpus_docs = get_corpus_docs()
    retrieval_k = max(top_k, top_k * 4)
    ranked_lists = [
        run_hybrid(q, corpus_docs=corpus_docs or None, top_k=retrieval_k) for q in queries
    ]
    children = (
        reciprocal_rank_fusion(ranked_lists)[:retrieval_k]
        if len(ranked_lists) > 1
        else ranked_lists[0][:retrieval_k]
    )
    return {"retrieved_children": children, "retrieval_attempt": attempt}


# ── expand_context ────────────────────────────────────────────────────────────


def expand_context(state: CRAGState) -> dict:
    """Expand child hits to parent chunks when parent metadata is available."""
    from app.rag.retrievers.corpus_registry import get_docstore
    from app.rag.retrievers.parent_child_retriever import fetch_parents
    from app.rag.types import chunk_text, normalize_chunks

    children = normalize_chunks(state.get("retrieved_children", []))
    logger.info("Expanding %d child chunks to parent context", len(children))

    docstore = get_docstore()
    if docstore:
        expanded = fetch_parents(children, docstore)
        return {"expanded_contexts": expanded}

    expanded: list[dict] = []
    seen_texts: set[str] = set()
    for child in children:
        meta = child.get("metadata") or {}
        parent_text = meta.get("parent_text")
        text = parent_text or chunk_text(child)
        if text and text not in seen_texts:
            seen_texts.add(text)
            expanded.append(
                {
                    "text": text,
                    "score": child.get("score", 0.0),
                    "metadata": {**meta, "expanded": "true"},
                }
            )
    return {"expanded_contexts": expanded or children}


# ── rerank_context ────────────────────────────────────────────────────────────


def rerank_context(state: CRAGState) -> dict:
    """Rerank retrieved/expanded chunks and keep the top candidates."""
    contexts = state.get("expanded_contexts") or state.get("retrieved_children", [])
    query = _active_query(state)
    logger.info("Reranking %d context chunks", len(contexts))

    from app.providers.reranker_provider import get_reranker

    reranker = get_reranker()
    if reranker is not None:
        top_k = state.get("top_k") or settings.rerank_top_k
        contexts = reranker.rerank(query, contexts, top_k=top_k)

    return {"expanded_contexts": contexts}


# ── generate_answer ───────────────────────────────────────────────────────────


def generate_answer(state: CRAGState) -> dict:
    """Generate a grounded answer from the ranked context chunks."""
    query = _active_query(state)
    contexts = state.get("expanded_contexts") or state.get("retrieved_children", [])
    if contexts:
        context_str = "\n\n---\n\n".join(c["text"] if isinstance(c, dict) else c for c in contexts)
    else:
        context_str = "(No relevant documents found)"
    logger.info("Generating answer from %d context chunks", len(contexts))

    history = state.get("chat_history") or []
    history_block = ""
    if history:
        lines = [f"{m.get('role', 'user').title()}: {m.get('content', '')}" for m in history[-6:]]
        history_block = "Conversation history:\n" + "\n".join(lines) + "\n\n"

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant. Answer the user's question based ONLY on the "
                    "provided context. If the context does not contain enough information, say so. "
                    "Use conversation history only for disambiguation, not as a factual source."
                )
            ),
            HumanMessage(
                content=(f"{history_block}Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:")
            ),
        ]
    )
    return {"answer": response.content.strip()}


# ── run_judge ─────────────────────────────────────────────────────────────────


def run_judge(state: CRAGState) -> dict:
    """Run the LLM-as-judge evaluator and store the structured verdict."""
    from app.rag.evaluators.llm_judge_evaluator import judge

    query = _active_query(state)
    answer = state.get("answer", "")
    contexts = state.get("expanded_contexts") or state.get("retrieved_children", [])

    verdict = judge(question=query, answer=answer, contexts=contexts)
    logger.info(
        "Judge overall=%.2f passed=%s",
        verdict.overall_score,
        verdict.passed,
    )
    return {"judge_verdict": verdict}


# ── evaluate_grounding ────────────────────────────────────────────────────────


def evaluate_grounding(state: CRAGState) -> dict:
    """Score how well the generated answer is grounded in the retrieved context."""
    from app.rag.evaluators.grounding_evaluator import evaluate

    score = evaluate(
        answer=state.get("answer", ""),
        contexts=state.get("expanded_contexts") or state.get("retrieved_children", []),
        question=_active_query(state),
    )
    logger.info("Grounding score: %.2f", score)
    return {"grounding_score": score}


# ── retry_with_policy ─────────────────────────────────────────────────────────


def mark_rejected(state: CRAGState) -> dict:
    """Mark the pipeline as rejected after judge evaluation at max retries."""
    logger.info("Judge rejected answer after max retries")
    return {"final_status": "rejected"}


def retry_with_policy(state: CRAGState) -> dict:
    """Increment the hallucination attempt counter and reset query for retry."""
    attempt = state.get("hallucination_attempt", 0) + 1
    logger.info("Hallucination retry %d/%d", attempt, settings.max_retries)
    return {
        "hallucination_attempt": attempt,
        "needs_rewrite": True,
    }
