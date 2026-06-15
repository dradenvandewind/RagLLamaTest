"""
Async video ingestion pipeline.
Supports:
  - YouTube transcription via youtube-transcript-api
  - Plain text / .srt / .vtt files
"""

import asyncio
import logging
import re
from typing import Any

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

import xml.etree.ElementTree as ET
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    CouldNotRetrieveTranscript,
)

logger = logging.getLogger(__name__)

_YT_REGEX = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str | None:
    match = _YT_REGEX.search(url)
    return match.group(1) if match else None


def _fetch_transcript(video_id: str, languages: list[str] | None = None) -> str:
    langs = languages or ["fr", "en", "auto"]
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript) as exc:
        raise RuntimeError(f"Transcript unavailable for {video_id}") from exc
    except ET.ParseError as exc:
        # YouTube returned an empty/malformed XML response (rate-limit ou bloc réseau)
        raise RuntimeError(
            f"YouTube returned an invalid transcript response for {video_id} "
            "(possible rate-limit or geo-block)"
        ) from exc
    return " ".join(entry["text"] for entry in transcript_list)


def _parse_srt_vtt(raw: str) -> str:
    """Clean SRT/VTT files to keep only the text."""
    # Remove timestamps and HTML tags
    text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}", "", raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"WEBVTT.*?\n", "", text)
    return " ".join(text.split())


class VideoIngestionPipeline:
    """Async ingestion pipeline for video documents."""

    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self._pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=64),
            ]
        )

    async def ingest_youtube_url(
            self,
            url: str,
            extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        video_id = _extract_video_id(url)
        if not video_id:
            logger.error("Invalid YouTube URL: %s", url)
            return 0  # Ne pas lever ici non plus si appelé en background

        logger.info("📥 Fetching transcript for %s…", video_id)
        try:
            transcript = await asyncio.to_thread(_fetch_transcript, video_id)
        except Exception as exc:          # ← catch-all pour le background task
            logger.error("Transcript unavailable for %s: %s", video_id, exc)
            return 0                      # ← on retourne proprement, sans re-raise

        metadata = {
                "source": "youtube",
                "video_id": video_id,
                "url": url,
                **(extra_metadata or {}),
            }
        return await self._insert_text(transcript, metadata)

    async def ingest_text(
        self,
        raw_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Ingest plain text (or cleaned SRT/VTT content).
        Returns the number of inserted chunks.
        """
        # Clean SRT/VTT if applicable
        if "-->" in raw_text:
            raw_text = _parse_srt_vtt(raw_text)

        return await self._insert_text(raw_text, metadata or {})

    async def _insert_text(self, text: str, metadata: dict[str, Any]) -> int:
        """Convert text into nodes and insert them into the index."""
        doc = Document(text=text, metadata=metadata)

        nodes = await asyncio.to_thread(self._pipeline.run, documents=[doc])
        logger.info("📦 %d chunks generated for %s", len(nodes), metadata.get("source", "?"))

        await asyncio.to_thread(self.index.insert_nodes, nodes)
        logger.info("✅ %d chunks inserted into the index.", len(nodes))
        return len(nodes)
