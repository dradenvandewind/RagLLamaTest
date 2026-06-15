"""Pydantic schemas for the RAG API."""

from typing import Any
from pydantic import BaseModel, HttpUrl, Field


class IngestRequest(BaseModel):
    url: str = Field(..., description="YouTube URL to ingest")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (title, author, etc.)",
    )


class IngestResponse(BaseModel):
    message: str
    url: str
    status: str


class ChatRequest(BaseModel):
    question: str = Field(..., description="Question to ask the RAG system")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    model: str


class HealthResponse(BaseModel):
    status: str
    index_ready: bool


class IndexStatsResponse(BaseModel):
    total_chunks: int
