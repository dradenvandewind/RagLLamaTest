"""
Async video ingestion pipeline.
"""

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests
import yt_dlp
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from youtube_transcript_api._errors import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
)

logger = logging.getLogger(__name__)

COOKIES_PATH = os.getenv("YT_COOKIES_PATH", "/app/cookies.txt")

_YT_REGEX = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([A-Za-z0-9_-]{11})"
)


def _extract_video_id(url: str) -> str | None:
    match = _YT_REGEX.search(url)
    return match.group(1) if match else None


def _fetch_transcript(video_id: str, languages: list[str] | None = None) -> str:
    langs = languages or ["fr", "en"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    cookies = COOKIES_PATH if os.path.exists(COOKIES_PATH) else None

    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": langs,
        "subtitlesformat": "json3",
        "quiet": True,
        "cookiesfrombrowser": ("firefox", "/root/.mozilla/firefox"),
        "extractor_args": {
            "youtube": {
                "player_client": ["web"],
            }
        },
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    subtitles = info.get("subtitles") or {}
    auto_captions = info.get("automatic_captions") or {}

    for lang in langs:
        for source in (subtitles, auto_captions):
            if lang in source:
                for fmt in source[lang]:
                    if fmt.get("ext") == "json3":
                        resp = requests.get(fmt["url"])
                        data = resp.json()
                        text = " ".join(
                            event["segs"][0]["utf8"]
                            for event in data.get("events", [])
                            if event.get("segs")
                        )
                        return text.strip()

    raise RuntimeError(f"No transcript found for {video_id}")


def _parse_srt_vtt(raw: str) -> str:
    text = re.sub(r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}", "", raw)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"WEBVTT.*?\n", "", text)
    return " ".join(text.split())


class VideoIngestionPipeline:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self._pipeline = IngestionPipeline(
            transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=64)]
        )

    async def ingest_youtube_url(
        self,
        url: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> int:
        video_id = _extract_video_id(url)
        if not video_id:
            logger.error("Invalid YouTube URL: %s", url)
            return 0

        logger.info("📥 Fetching transcript for %s…", video_id)
        try:
            transcript = await asyncio.to_thread(_fetch_transcript, video_id)
        except Exception as exc:
            logger.error("Transcript unavailable for %s: %s", video_id, exc)
            return 0

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
        if "-->" in raw_text:
            raw_text = _parse_srt_vtt(raw_text)
        return await self._insert_text(raw_text, metadata or {})

    async def _insert_text(self, text: str, metadata: dict[str, Any]) -> int:
        doc = Document(text=text, metadata=metadata)
        nodes = await asyncio.to_thread(self._pipeline.run, documents=[doc])
        logger.info("📦 %d chunks generated for %s", len(nodes), metadata.get("source", "?"))
        await asyncio.to_thread(self.index.insert_nodes, nodes)
        logger.info("✅ %d chunks inserted into the index.", len(nodes))
        return len(nodes)