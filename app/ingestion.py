"""
Async video ingestion pipeline.
"""

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Any
import json

import requests
import google.auth.transport.requests
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

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


def _get_youtube_client():
    token_path = os.getenv("YOUTUBE_TOKEN_PATH", "/app/youtube_token.json")
    with open(token_path) as f:
        data = json.load(f)

    creds = Credentials(
        token=data["token"],
        refresh_token=data["refresh_token"],
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
    )
    if not creds.valid:
        creds.refresh(google.auth.transport.requests.Request())

    return build("youtube", "v3", credentials=creds)


def _fetch_transcript(video_id: str, languages: list[str] | None = None) -> str:
    langs = languages or ["fr", "en"]
    youtube = _get_youtube_client()

    # 1. List captions
    response = youtube.captions().list(
        part="snippet",
        videoId=video_id,
    ).execute()

    items = response.get("items", [])
    if not items:
        raise RuntimeError(f"No captions found for {video_id}")

    # 2. Choose the language
    caption_id = None
    for lang in langs:
        for item in items:
            if item["snippet"]["language"].startswith(lang):
                caption_id = item["id"]
                break
        if caption_id:
            break
    if not caption_id:
        caption_id = items[0]["id"]

    # 3. Download as SRT
    srt = youtube.captions().download(
        id=caption_id,
        tfmt="srt",
    ).execute()

    return _parse_srt_vtt(srt.decode("utf-8"))


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