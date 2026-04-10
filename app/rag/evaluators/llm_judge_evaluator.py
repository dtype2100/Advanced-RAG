"""LLM-as-judge evaluator — structured rubric-based answer quality assessment.

Uses a dedicated judge LLM (via ``judge_llm_provider``) to score answers on
multiple dimensions (correctness, faithfulness, completeness, conciseness).
Returns a structured ``JudgeVerdict`` dataclass for downstream policy decisions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_io_log import log_llm_io
from app.providers.judge_llm_provider import get_judge_llm

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert evaluator assessing the quality of a RAG-generated answer.
Score the answer on the following dimensions (each 0.0–1.0):

- correctness:   Is the answer factually accurate based on the context?
- faithfulness:  Does the answer stay within what the context supports (no hallucination)?
- completeness:  Does the answer address all parts of the question?
- conciseness:   Is the answer appropriately concise without omitting key info?

Return a JSON object with ONLY these keys:
{"correctness": <float>, "faithfulness": <float>, "completeness": <float>,
 "conciseness": <float>, "reasoning": "<one sentence>"}
"""

_HUMAN = """\
Question: {question}

Context:
{context}

Answer:
{answer}
"""


@dataclass
class JudgeVerdict:
    """Structured result from the LLM judge evaluator."""

    correctness: float = 0.0
    faithfulness: float = 0.0
    completeness: float = 0.0
    conciseness: float = 0.0
    reasoning: str = ""
    raw_response: str = ""
    error: str = ""

    @property
    def overall_score(self) -> float:
        """Weighted average of all dimension scores.

        Faithfulness is weighted double to penalise hallucination more heavily.
        """
        return (
            self.correctness * 1.0
            + self.faithfulness * 2.0
            + self.completeness * 1.0
            + self.conciseness * 0.5
        ) / 4.5

    @property
    def passed(self) -> bool:
        """True when the overall score meets the minimum threshold (0.6)."""
        return self.overall_score >= 0.6


def judge(
    question: str,
    answer: str,
    contexts: list[str],
) -> JudgeVerdict:
    """Run the LLM judge and return a structured ``JudgeVerdict``.

    Args:
        question: Original user question.
        answer:   Generated answer to evaluate.
        contexts: Retrieved context chunks used to generate the answer.

    Returns:
        ``JudgeVerdict`` with per-dimension scores and an overall score.
    """
    if not answer or not contexts:
        return JudgeVerdict(error="Missing answer or contexts")

    context_str = "\n\n---\n\n".join(contexts[:5])

    user_text = _HUMAN.format(
        question=question,
        context=context_str,
        answer=answer,
    )
    log_llm_io(
        "llm_judge",
        system=_SYSTEM,
        user=user_text,
        user_query=question,
        context=context_str,
        prior_answer=answer,
    )
    try:
        llm = get_judge_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=user_text),
            ]
        )
        raw = response.content.strip()
        log_llm_io("llm_judge", assistant=raw)
        data = json.loads(raw)

        verdict = JudgeVerdict(
            correctness=float(data.get("correctness", 0.0)),
            faithfulness=float(data.get("faithfulness", 0.0)),
            completeness=float(data.get("completeness", 0.0)),
            conciseness=float(data.get("conciseness", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            raw_response=raw,
        )
        logger.info(
            "Judge verdict: overall=%.2f (corr=%.2f, faith=%.2f, comp=%.2f, conc=%.2f)",
            verdict.overall_score,
            verdict.correctness,
            verdict.faithfulness,
            verdict.completeness,
            verdict.conciseness,
        )
        return verdict

    except json.JSONDecodeError as exc:
        logger.warning("Judge evaluator: failed to parse JSON response — %s", exc)
        raw_resp = getattr(response, "content", "") or ""
        return JudgeVerdict(error=f"JSON parse error: {exc}", raw_response=raw_resp)
    except Exception as exc:
        logger.warning("Judge evaluator failed: %s", exc, exc_info=True)
        return JudgeVerdict(error=str(exc))
