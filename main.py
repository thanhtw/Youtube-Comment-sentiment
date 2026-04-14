"""
main.py
~~~~~~~
Full pipeline — runs both steps in sequence:

  Step 1 │ collection/collect.py   — fetch comments from YouTube
  Step 2 │ analysis/analyze.py     — sentiment & emotion analysis

You can also run each step independently:
    python -m collection.collect   [--help]
    python -m analysis.analyze     [--help]
"""

import argparse
import logging
import os
import sys

import pandas as pd

import config
from collection.collect import collect_channel, parse_args as collect_args, run as collect_run
from analysis.analyze import parse_args as analyze_args, run as analyze_run
from collection.channel_loader import load_channels_from_excel
from collection.youtube_collector import YouTubeCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full pipeline: collect YouTube comments then analyse sentiment & emotion."
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
        "--skip-emotion", action="store_true",
        help="Skip the HuggingFace emotion model (faster)."
    )
    parser.add_argument(
        "--skip-sentiment-model", action="store_true",
        help="Skip the HuggingFace sentiment model (VADER still runs)."
    )
    parser.add_argument(
        "--device", type=int, default=-1,
        help="Torch device: -1=CPU, 0=first GPU, etc."
    )
    parser.add_argument(
        "--output-dir", default=config.OUTPUT_DIR,
        help="Directory for all output files."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-channel collection (delegates to collection package)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.api_key:
        logger.error("No API key. Set YOUTUBE_API_KEY in .env or pass --api-key.")
        sys.exit(1)

    # ── Step 1: Data Collection ──────────────────────────────────────────────
    all_comments = collect_run(args)

    if not all_comments:
        logger.error("Collection produced no comments. Aborting analysis.")
        sys.exit(1)

    # ── Step 2: Sentiment Analysis ───────────────────────────────────────────
    df = pd.DataFrame(all_comments)
    analyze_run(df, args)


if __name__ == "__main__":
    main()
