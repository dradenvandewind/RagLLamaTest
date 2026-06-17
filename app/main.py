"""
RAG LlamaIndex - async FastAPI API
YouTube video ingestion + LLM querying
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import uuid

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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.llms.openai import OpenAI
#from llama_index.llms.anthropic import Anthropic
from llama_index.llms.ollama import Ollama
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
_job_status: dict[str, dict] = {}

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
    """
    Settings.llm = OpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    """
    """
    Settings.llm = Anthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    """
    
    Settings.llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        request_timeout=300.0,
    )
    
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
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
    job_id = str(uuid.uuid4())
    _job_status[job_id] = {"status": "pending", "url": req.url}
    background_tasks.add_task(_run_ingest, job_id, req.url, req.metadata)
    return IngestResponse(
    message=f"Ingestion started for: {req.url}",
    url=req.url,
    status="pending",
)

async def _run_ingest(job_id: str, url: str, metadata: dict):
    try:
        n = await _ingestion_pipeline.ingest_youtube_url(url, metadata)
        _job_status[job_id] = {"status": "done", "chunks": n}
    except Exception as exc:
        _job_status[job_id] = {"status": "error", "detail": str(exc)}

@app.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str):
    return _job_status.get(job_id, {"status": "not_found"})

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
    global _index, _ingestion_pipeline   # ← ajouter _ingestion_pipeline
    chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_PATH", "/app/chroma_db"))
    await asyncio.to_thread(chroma_client.delete_collection, "video_rag")
    await asyncio.to_thread(chroma_client.get_or_create_collection, "video_rag")
    _index = await asyncio.to_thread(_build_index)
    _ingestion_pipeline = VideoIngestionPipeline(index=_index)  # ← reconstruire
    return {"message": "Index reset successfully."}