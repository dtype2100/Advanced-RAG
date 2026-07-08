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
from app.providers.vectorstore_provider import get_vectorstore

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
    """Retrieve relevant child chunks using hybrid (vector + BM25) search.

    When ``multi_query`` mode is enabled via env var ``MULTI_QUERY=1``,
    generates additional query variants and fuses the retrieval results.
    """
    import os

    query = _active_query(state)
    attempt = state.get("retrieval_attempt", 0) + 1
    logger.info("Retrieval attempt %d for: %s", attempt, query)

    queries = [query]
    if os.getenv("MULTI_QUERY", "0") == "1":
        from app.rag.query.multi_query_generator import generate_multi_query

        queries = generate_multi_query(query, n=3)
        logger.info("Multi-query: %d variants", len(queries))

    store = get_vectorstore()
    seen: dict[str, dict] = {}
    for q in queries:
        for r in store.search(q, top_k=settings.max_retrieval_docs):
            seen.setdefault(r["text"], r)

    children = list(seen.values())
    return {"retrieved_children": children, "retrieval_attempt": attempt}


# ── test_retrieval ────────────────────────────────────────────────────────────


def test_retrieval(state: CRAGState) -> dict:
    """Test retrieved context quality: expand, rerank, and relevance filter.

    Phase 4 of the improvement loop (search → **test** → evaluation).
    """
    from app.providers.reranker_provider import get_reranker
    from app.rag.guards.relevance_guard import filter_relevant
    from app.rag.policies.expansion_policy import should_expand

    children = state.get("retrieved_children", [])
    query = _active_query(state)

    if should_expand(state):
        contexts = expand_context(state).get("expanded_contexts", children)
    else:
        contexts = children
    logger.info("Test phase: evaluating %d candidate chunks", len(contexts))

    reranker = get_reranker()
    if reranker is not None:
        contexts = reranker.rerank(query, contexts)

    filtered = filter_relevant(contexts)
    logger.info(
        "Test phase: %d → %d contexts passed relevance filter",
        len(contexts),
        len(filtered),
    )
    return {"expanded_contexts": filtered}


# ── expand_context ────────────────────────────────────────────────────────────


def expand_context(state: CRAGState) -> dict:
    """Expand child hits to parent / larger chunks (small-to-big strategy)."""
    children = state.get("retrieved_children", [])
    logger.info("Expanding %d child chunks to parent context", len(children))
    return {"expanded_contexts": children}


# ── rerank_context ────────────────────────────────────────────────────────────


def rerank_context(state: CRAGState) -> dict:
    """Rerank retrieved/expanded chunks and keep the top candidates."""
    contexts = state.get("expanded_contexts") or state.get("retrieved_children", [])
    query = _active_query(state)
    logger.info("Reranking %d context chunks", len(contexts))

    from app.providers.reranker_provider import get_reranker

    reranker = get_reranker()
    if reranker is not None:
        contexts = reranker.rerank(query, contexts)

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

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful AI assistant. Answer the user's question based ONLY on the "
                    "provided context. If the context does not contain enough information, say so."
                )
            ),
            HumanMessage(content=f"Context:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"),
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
    )
    logger.info("Grounding score: %.2f", score)
    return {"grounding_score": score}


# ── retry_with_policy ─────────────────────────────────────────────────────────


def retry_with_policy(state: CRAGState) -> dict:
    """Increment the hallucination attempt counter and reset query for retry."""
    attempt = state.get("hallucination_attempt", 0) + 1
    logger.info("Hallucination retry %d/%d", attempt, settings.max_retries)
    return {
        "hallucination_attempt": attempt,
        "needs_rewrite": True,
    }


def retry_retrieval(state: CRAGState) -> dict:
    """Feedback: low faithfulness — loop back to search with a rewritten query."""
    attempt = state.get("hallucination_attempt", 0) + 1
    retrieval = state.get("retrieval_attempt", 0) + 1
    logger.info(
        "Feedback retry_retrieval: hallucination=%d retrieval=%d",
        attempt,
        retrieval,
    )
    return {
        "hallucination_attempt": attempt,
        "retrieval_attempt": retrieval,
        "needs_rewrite": True,
    }


def retry_generation(state: CRAGState) -> dict:
    """Feedback: low overall score — regenerate answer without re-retrieval."""
    attempt = state.get("hallucination_attempt", 0) + 1
    logger.info("Feedback retry_generation: attempt %d/%d", attempt, settings.max_retries)
    return {"hallucination_attempt": attempt}


def finalize_ok(state: CRAGState) -> dict:
    """Mark the pipeline as successfully completed."""
    return {"final_status": "ok"}


def finalize_rejected(state: CRAGState) -> dict:
    """Mark the pipeline as rejected after exhausting feedback retries."""
    answer = state.get("answer", "")
    suffix = " [Answer did not pass quality verification]"
    if suffix not in answer:
        answer = f"{answer}{suffix}".strip()
    return {"final_status": "rejected", "answer": answer}
