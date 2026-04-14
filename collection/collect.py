"""
collection/collect.py
~~~~~~~~~~~~~~~~~~~~~
Standalone entry point — Step 1: Data Collection.

Reads channels from channels.xlsx, fetches all videos and comments via the
YouTube Data API v3, and saves the raw data to CSV files in the output directory.

Usage:
    python -m collection.collect
    python -m collection.collect --channels-file my_channels.xlsx --max-videos 10
"""

import argparse
import logging
import os
import sys

import pandas as pd

# Allow running as `python collection/collect.py` directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from collection.channel_loader import load_channels_from_excel
from collection.youtube_collector import YouTubeCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1 — Collect YouTube comments from multiple channels."
    )
    parser.add_argument(
        "--channels-file", default=config.CHANNELS_FILE,
        help="Excel file with channel list (default: channels.xlsx)."
    )
    parser.add_argument(
        "--api-key", default=config.YOUTUBE_API_KEY,
        help="YouTube Data API v3 key."
    )
    parser.add_argument(
        "--max-videos", type=int, default=config.MAX_VIDEOS,
        help="Max videos per channel (0 = unlimited)."
    )
    parser.add_argument(
        "--max-comments", type=int, default=config.MAX_COMMENTS_PER_VIDEO,
        help="Max comments per video (0 = unlimited)."
    )
    parser.add_argument(
        "--output-dir", default=config.OUTPUT_DIR,
        help="Directory to save raw comment CSVs."
    )
    return parser.parse_args()


def collect_channel(
    collector: YouTubeCollector,
    channel_id: str,
    channel_handle: str,
    max_videos: int,
    max_comments: int,
) -> tuple[dict, list[dict]]:
    """Resolve channel, collect all videos + comments. Returns (channel_info, comments)."""
    resolved_id = collector.resolve_channel_id(
        channel_id=channel_id,
        handle=channel_handle,
    )
    channel_info = collector.get_channel_info(resolved_id)
    logger.info(
        "  Channel: %s | Videos: %s | Subscribers: %s",
        channel_info["title"],
        channel_info["video_count"],
        channel_info["subscriber_count"],
    )

    videos = list(
        collector.iter_videos(
            uploads_playlist_id=channel_info["uploads_playlist_id"],
            max_videos=max_videos,
        )
    )
    logger.info("  Found %d videos.", len(videos))

    comments: list[dict] = []
    for idx, video in enumerate(videos, 1):
        logger.info(
            "  [%d/%d] Collecting: %s (%d comments reported)",
            idx, len(videos), video.title, video.comment_count,
        )
        for c in collector.iter_comments(video, max_comments=max_comments):
            comments.append(
                {
                    "channel_id": channel_info["channel_id"],
                    "channel_name": channel_info["title"],
                    "video_id": c.video_id,
                    "video_title": c.video_title,
                    "comment_id": c.comment_id,
                    "parent_id": c.parent_id,
                    "author": c.author,
                    "text": c.text,
                    "like_count": c.like_count,
                    "published_at": c.published_at,
                    "updated_at": c.updated_at,
                    "is_reply": c.is_reply,
                }
            )

    logger.info("  Collected %d comments.", len(comments))
    return channel_info, comments


def run(args: argparse.Namespace) -> list[dict]:
    """
    Execute the collection pipeline and return all collected comment dicts.
    Can be called programmatically by main.py.
    """
    if not args.api_key:
        logger.error("No API key. Set YOUTUBE_API_KEY in .env or pass --api-key.")
        sys.exit(1)

    try:
        channels = load_channels_from_excel(args.channels_file)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Excel format error: %s", exc)
        sys.exit(1)

    if not channels:
        logger.error("No valid channels found in %s.", args.channels_file)
        sys.exit(1)

    logger.info("=== DATA COLLECTION ===")
    logger.info("Channels to process: %d  |  Source: %s", len(channels), args.channels_file)

    os.makedirs(args.output_dir, exist_ok=True)
    collector = YouTubeCollector(api_key=args.api_key)

    all_comments: list[dict] = []

    for ch_idx, entry in enumerate(channels, 1):
        label = entry["name"] or entry["channel_handle"] or entry["channel_id"]
        logger.info("--- Channel %d/%d: %s ---", ch_idx, len(channels), label)
        try:
            ch_info, comments = collect_channel(
                collector=collector,
                channel_id=entry["channel_id"],
                channel_handle=entry["channel_handle"],
                max_videos=args.max_videos,
                max_comments=args.max_comments,
            )
        except Exception as exc:
            logger.error("Failed to collect '%s': %s", label, exc)
            continue

        if comments:
            safe_name = ch_info["title"].replace("/", "_").replace(" ", "_")
            per_path = os.path.join(args.output_dir, f"raw_{safe_name}.csv")
            pd.DataFrame(comments).to_csv(per_path, index=False, encoding="utf-8-sig")
            logger.info("  Saved → %s", per_path)
            all_comments.extend(comments)

    if not all_comments:
        logger.warning("No comments collected from any channel.")
        return []

    # Save combined raw CSV
    combined_path = os.path.join(args.output_dir, "raw_all_channels.csv")
    pd.DataFrame(all_comments).to_csv(combined_path, index=False, encoding="utf-8-sig")
    logger.info(
        "Collection complete. Total comments: %d  |  Saved → %s",
        len(all_comments),
        combined_path,
    )
    return all_comments


if __name__ == "__main__":
    run(parse_args())
