"""
RAG LlamaIndex - async FastAPI API
YouTube video ingestion + LLM querying
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import chromadb
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import BaseModel

from .ingestion import VideoIngestionPipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    HealthResponse,
    IndexStatsResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Globals (initialized at startup)
# ──────────────────────────────────────────────
_index: VectorStoreIndex | None = None
_ingestion_pipeline: VideoIngestionPipeline | None = None


def _build_index() -> VectorStoreIndex:
    """Builds or loads the index from ChromaDB."""
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/chroma_db"))
    chroma_collection = chroma_client.get_or_create_collection("video_rag")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and cleanup."""
    global _index, _ingestion_pipeline

    logger.info("🚀 Starting the RAG LlamaIndex application...")

    # Global LlamaIndex configuration
    Settings.llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.embed_model = OpenAIEmbedding(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
    )

    # Load the index
    _index = await asyncio.to_thread(_build_index)
    _ingestion_pipeline = VideoIngestionPipeline(index=_index)
    logger.info("✅ ChromaDB index loaded.")

    yield

    logger.info("🛑 Shutting down the application.")


# ──────────────────────────────────────────────
# Application FastAPI
# ──────────────────────────────────────────────
app = FastAPI(
    title="RAG LlamaIndex - Video",
    description="Async RAG pipeline for querying video transcripts",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """API health check."""
    return HealthResponse(status="ok", index_ready=_index is not None)


@app.get("/stats", response_model=IndexStatsResponse)
async def stats():
    """Vector index statistics."""
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/chroma_db"))
    collection = chroma_client.get_or_create_collection("video_rag")
    count = await asyncio.to_thread(collection.count)
    return IndexStatsResponse(total_chunks=count)


@app.post("/ingest/url", response_model=IngestResponse)
async def ingest_url(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest a YouTube video via its URL.
    Ingestion is started as a background task.
    """
    if _ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    background_tasks.add_task(_ingestion_pipeline.ingest_youtube_url, req.url, req.metadata)
    return IngestResponse(
        message=f"Ingestion started for: {req.url}",
        url=req.url,
        status="pending",
    )


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """
    Ingest a transcript file (.txt / .srt / .vtt).
    """
    if _ingestion_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    background_tasks.add_task(
        _ingestion_pipeline.ingest_text,
        text,
        {"source": file.filename, "type": "file_upload"},
    )
    return IngestResponse(
        message=f"File '{file.filename}' is being ingested.",
        url=file.filename,
        status="pending",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Query the RAG system (full non-streamed response).
    """
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    query_engine = _index.as_query_engine(
        similarity_top_k=req.top_k,
        streaming=False,
    )
    response = await asyncio.to_thread(query_engine.query, req.question)

    sources = [
        {
            "text": node.get_content()[:300],
            "score": round(node.score or 0.0, 4),
            "metadata": node.metadata,
        }
        for node in (response.source_nodes or [])
    ]

    return ChatResponse(
        answer=str(response),
        sources=sources,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Query the RAG system with a streaming response (Server-Sent Events).
    """
    if _index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")

    async def token_generator() -> AsyncGenerator[str, None]:
        query_engine = _index.as_query_engine(
            similarity_top_k=req.top_k,
            streaming=True,
        )
        streaming_response = await asyncio.to_thread(query_engine.query, req.question)
        for token in streaming_response.response_gen:
            yield f"data: {token}\n\n"
            await asyncio.sleep(0)  # Yield control to the event loop
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.delete("/index")
async def reset_index():
    """Reset the index (delete all chunks)."""
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/chroma_db"))
    await asyncio.to_thread(chroma_client.delete_collection, "video_rag")
    await asyncio.to_thread(chroma_client.get_or_create_collection, "video_rag")
    global _index
    _index = await asyncio.to_thread(_build_index)
    return {"message": "Index reset successfully."}
