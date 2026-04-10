"""Chat / RAG query endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.chat_service import run_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=ChatResponse, tags=["rag"])
async def rag_query(req: ChatRequest) -> ChatResponse:
    """Run the full CRAG pipeline (analyse → retrieve → rerank → generate → evaluate)."""
    if not settings.using_vllm and not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail=("LLM backend not configured. Set LLM_BACKEND=vllm or provide OPENAI_API_KEY."),
        )

    try:
        result = run_chat(question=req.question, session_id=req.session_id)
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    raw_src = result.get("retrieved_children") or result.get("expanded_contexts", [])
    sources = [x["text"] if isinstance(x, dict) else str(x) for x in raw_src]

    return ChatResponse(
        question=req.question,
        answer=result.get("answer", ""),
        sources=sources,
        retries=result.get("hallucination_attempt", 0),
        clarification_needed=result.get("final_status") == "clarification_needed",
        clarification_question=result.get("clarification_question"),
    )
