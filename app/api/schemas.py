"""Pydantic request / response schemas for the API."""

from pydantic import BaseModel, Field

# ── Documents ────────────────────────────────────────────────────────────────


class DocumentInput(BaseModel):
    text: str = Field(..., min_length=1, description="Document text to ingest")
    metadata: dict[str, str] = Field(default_factory=dict, description="Optional metadata")


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    message: str
    count: int


# ── Query ────────────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    retries: int


# ── Search ───────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    text: str
    score: float
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    results: list[SearchResult]


# ── Health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    collection: str
