"""Chat / RAG query endpoints."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.rag.citations.source_citations import build_source_citations
from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse
from app.services.chat_service import run_chat, stream_chat

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
        result = run_chat(
            question=req.question,
            session_id=req.session_id,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    raw_sources = result.get("expanded_contexts") or result.get("retrieved_children") or []
    return ChatResponse(
        question=req.question,
        answer=result.get("answer", ""),
        sources=build_source_citations(raw_sources),
        retries=result.get("hallucination_attempt", 0),
        clarification_needed=result.get("final_status") == "clarification_needed",
        clarification_question=result.get("clarification_question"),
    )


@router.post("/query/stream", tags=["rag"])
async def rag_query_stream(req: ChatRequest) -> StreamingResponse:
    """Stream CRAG graph node updates as Server-Sent Events."""
    if not settings.using_vllm and not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail=("LLM backend not configured. Set LLM_BACKEND=vllm or provide OPENAI_API_KEY."),
        )

    def event_generator():
        try:
            for event in stream_chat(
                question=req.question,
                session_id=req.session_id,
                top_k=req.top_k,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as exc:
            logger.exception("RAG stream failed")
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
