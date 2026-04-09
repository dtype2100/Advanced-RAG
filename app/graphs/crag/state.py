"""LangGraph state definition for the CRAG pipeline.

Each field maps to a named step in the graph; conditional edges read these
flags to decide which branch to take.  All fields are optional (``total=False``)
so that nodes only need to return the keys they actually update.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    pass


class CRAGState(TypedDict, total=False):
    """Shared state flowing through the CRAG graph.

    Attributes:
        user_query:             Original, unmodified question from the user.
        clarified_query:        Query after the user has answered a clarification
                                question (if clarification was triggered).
        rewritten_query:        Query after ``query_rewriter`` has reformulated it
                                for better retrieval.
        needs_clarification:    True when ``query_analyzer`` detects missing slots
                                or ambiguous intent and the clarification policy
                                decides to prompt the user.
        needs_rewrite:          True when the rewrite policy decides that the
                                current query should be reformulated before retrieval.
        clarification_question: The concrete question to surface to the user when
                                ``needs_clarification`` is True.
        retrieved_children:     Child-level (small) chunk texts from the retriever.
        expanded_contexts:      Parent / big chunks after context-expansion (parent-
                                child or small-to-big).
        retrieval_attempt:      Number of retrieval + rewrite cycles executed so far.
        hallucination_attempt:  Number of hallucination-check + re-generation cycles.
        grounding_score:        Float in [0, 1] from the grounding evaluator.
        judge_verdict:          Structured ``JudgeVerdict`` from the LLM-as-judge
                                evaluator; ``None`` if judge was not invoked.
        answer:                 The LLM-generated answer string.
        final_status:           Terminal status tag: ``"ok"`` | ``"max_retries"``
                                | ``"clarification_needed"`` | ``"rejected"``.
    """

    user_query: str
    clarified_query: str
    rewritten_query: str
    needs_clarification: bool
    needs_rewrite: bool
    clarification_question: str
    retrieved_children: list[str]
    expanded_contexts: list[str]
    retrieval_attempt: int
    hallucination_attempt: int
    grounding_score: float
    judge_verdict: Any  # JudgeVerdict | None
    answer: str
    final_status: str
