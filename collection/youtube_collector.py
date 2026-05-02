"""
collection/youtube_collector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Fetch channel info, video list, and all comments via YouTube Data API v3.
"""

import logging
import time
from dataclasses import dataclass
from typing import Iterator
import datetime

import googleapiclient.discovery
import googleapiclient.errors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    video_id: str
    title: str
    published_at: str
    view_count: int
    like_count: int
    comment_count: int


@dataclass
class Comment:
    video_id: str
    video_title: str
    comment_id: str
    parent_id: str          # "" if top-level
    author: str
    text: str
    like_count: int
    published_at: str
    updated_at: str
    is_reply: bool


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class YouTubeCollector:
    """Wrapper around the YouTube Data API v3."""

    _API_SERVICE = "youtube"
    _API_VERSION = "v3"
    _QUOTA_RETRY_SLEEP = 60  # seconds to wait on quota error

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY is not set.")
        self._youtube = googleapiclient.discovery.build(
            self._API_SERVICE,
            self._API_VERSION,
            developerKey=api_key,
            cache_discovery=False,
        )

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------

    def resolve_channel_id(self, channel_id: str = "", handle: str = "") -> str:
        """Return the canonical channel ID given either a raw ID or a handle."""
        if channel_id:
            return channel_id

        if not handle:
            raise ValueError("Provide either channel_id or channel_handle.")

        handle_clean = handle.lstrip("@")
        response = (
            self._youtube.channels()
            .list(part="id", forHandle=handle_clean)
            .execute()
        )
        items = response.get("items", [])
        if not items:
            raise ValueError(f"Channel not found for handle: {handle}")
        return items[0]["id"]

    def get_channel_info(self, channel_id: str) -> dict:
        response = (
            self._youtube.channels()
            .list(
                part="snippet,contentDetails,statistics",
                id=channel_id,
            )
            .execute()
        )
        items = response.get("items", [])
        if not items:
            raise ValueError(f"No channel found with id={channel_id}")
        ch = items[0]
        return {
            "channel_id": channel_id,
            "title": ch["snippet"]["title"],
            "description": ch["snippet"]["description"],
            "subscriber_count": int(ch["statistics"].get("subscriberCount", 0)),
            "video_count": int(ch["statistics"].get("videoCount", 0)),
            "uploads_playlist_id": ch["contentDetails"]["relatedPlaylists"]["uploads"],
        }

    # ------------------------------------------------------------------
    # Video helpers
    # ------------------------------------------------------------------

    def iter_videos(
        self, uploads_playlist_id: str, max_videos: int = 0
    ) -> Iterator[VideoMeta]:
        """Yield VideoMeta objects for every video in the uploads playlist."""
        collected = 0
        page_token: str | None = None

        while True:
            playlist_resp = (
                self._youtube.playlistItems()
                .list(
                    part="contentDetails",
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=page_token,
                )
                .execute()
            )

            video_ids = [
                item["contentDetails"]["videoId"]
                for item in playlist_resp.get("items", [])
            ]

            if video_ids:
                stats_resp = (
                    self._youtube.videos()
                    .list(
                        part="snippet,statistics",
                        id=",".join(video_ids),
                    )
                    .execute()
                )

                for item in stats_resp.get("items", []):
                    stats = item.get("statistics", {})
                    snippet = item.get("snippet", {})
                    published_str = snippet.get("publishedAt", "")
                    if published_str:
                        try:
                            published_dt = datetime.datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                            cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=5*365)
                            if published_dt < cutoff:
                                return  # since videos are ordered newest first, stop here
                        except ValueError:
                            pass  # if parsing fails, still yield
                    yield VideoMeta(
                        video_id=item["id"],
                        title=snippet.get("title", ""),
                        published_at=published_str,
                        view_count=int(stats.get("viewCount", 0)),
                        like_count=int(stats.get("likeCount", 0)),
                        comment_count=int(stats.get("commentCount", 0)),
                    )
                    collected += 1
                    if max_videos and collected >= max_videos:
                        return

            page_token = playlist_resp.get("nextPageToken")
            if not page_token:
                break

    # ------------------------------------------------------------------
    # Comment helpers
    # ------------------------------------------------------------------

    def iter_comments(
        self, video: VideoMeta, max_comments: int = 0
    ) -> Iterator[Comment]:
        """Yield every top-level comment and reply for *video*."""
        collected = 0
        page_token: str | None = None

        while True:
            try:
                thread_resp = (
                    self._youtube.commentThreads()
                    .list(
                        part="snippet,replies",
                        videoId=video.video_id,
                        maxResults=100,
                        pageToken=page_token,
                        textFormat="plainText",
                    )
                    .execute()
                )
            except googleapiclient.errors.HttpError as exc:
                status = exc.resp.status
                if status in (400, 403):
                    # 403 = comments disabled by uploader
                    # 400 processingFailure = live stream / unsupported video type
                    logger.warning(
                        "Cannot retrieve comments for video '%s' (%s) — HTTP %d. Skipping.",
                        video.title,
                        video.video_id,
                        status,
                    )
                    return
                if status == 429:
                    logger.warning("Quota hit — sleeping %ds.", self._QUOTA_RETRY_SLEEP)
                    time.sleep(self._QUOTA_RETRY_SLEEP)
                    continue
                raise

            for thread in thread_resp.get("items", []):
                top = thread["snippet"]["topLevelComment"]
                top_snip = top["snippet"]

                yield Comment(
                    video_id=video.video_id,
                    video_title=video.title,
                    comment_id=top["id"],
                    parent_id="",
                    author=top_snip.get("authorDisplayName", ""),
                    text=top_snip.get("textDisplay", ""),
                    like_count=int(top_snip.get("likeCount", 0)),
                    published_at=top_snip.get("publishedAt", ""),
                    updated_at=top_snip.get("updatedAt", ""),
                    is_reply=False,
                )
                collected += 1
                if max_comments and collected >= max_comments:
                    return

                for reply in thread.get("replies", {}).get("comments", []):
                    r_snip = reply["snippet"]
                    yield Comment(
                        video_id=video.video_id,
                        video_title=video.title,
                        comment_id=reply["id"],
                        parent_id=top["id"],
                        author=r_snip.get("authorDisplayName", ""),
                        text=r_snip.get("textDisplay", ""),
                        like_count=int(r_snip.get("likeCount", 0)),
                        published_at=r_snip.get("publishedAt", ""),
                        updated_at=r_snip.get("updatedAt", ""),
                        is_reply=True,
                    )
                    collected += 1
                    if max_comments and collected >= max_comments:
                        return

            page_token = thread_resp.get("nextPageToken")
            if not page_token:
                break
