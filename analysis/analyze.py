"""
analysis/analyze.py
~~~~~~~~~~~~~~~~~~~~
Standalone entry point — Step 2: Sentiment Analysis.

Reads the raw comments CSV produced by the collection step, runs VADER and
HuggingFace models, saves analyzed CSVs and an HTML report.

Usage:
    python -m analysis.analyze
    python -m analysis.analyze --input output/raw_all_channels.csv
    python -m analysis.analyze --skip-emotion          # VADER only, fast
"""

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from analysis.analyzer import (
    add_hf_emotion,
    add_hf_sentiment,
    add_language_column,
    add_vader_sentiment,
    compute_stats,
    generate_html_report,
)
from analysis.visualize import generate_figures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    default_input = os.path.join(config.OUTPUT_DIR, "raw_all_channels.csv")
    parser = argparse.ArgumentParser(
        description="Step 2 — Sentiment & emotion analysis on collected comments."
    )
    parser.add_argument(
        "--input", default=default_input,
        help=f"Raw comments CSV to analyse (default: {default_input})."
    )
    parser.add_argument(
        "--output-dir", default=config.OUTPUT_DIR,
        help="Directory to save analyzed CSVs and report."
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
        "--figures-dir", default=None,
        help="Directory to save publication figures (default: <output-dir>/figures/)."
    )
    parser.add_argument(
        "--figures-fmt", default="png", choices=["png", "pdf"],
        help="Figure format: png (300 dpi) or pdf (vector). Default: png."
    )
    return parser.parse_args()


def run(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Execute the analysis pipeline on *df* and save results.
    Can be called programmatically by main.py.
    Returns the analyzed DataFrame.
    """
    logger.info("=== SENTIMENT ANALYSIS ===")
    logger.info("Comments to analyse: %d", len(df))

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 0 — detect language of each comment (zh / en)
    logger.info("Detecting comment language …")
    df = add_language_column(df, text_col="text")

    # VADER — English only (lightweight, no model download)
    logger.info("Running VADER sentiment (English comments) …")
    df = add_vader_sentiment(df, text_col="text")

    # HuggingFace sentiment — separate model per language
    if not args.skip_sentiment_model:
        logger.info(
            "Running HuggingFace sentiment — EN: %s | ZH: %s …",
            config.ENGLISH_SENTIMENT_MODEL,
            config.CHINESE_SENTIMENT_MODEL,
        )
        df = add_hf_sentiment(
            df,
            en_model=config.ENGLISH_SENTIMENT_MODEL,
            zh_model=config.CHINESE_SENTIMENT_MODEL,
            text_col="text",
            batch_size=config.ANALYSIS_BATCH_SIZE,
            device=args.device,
        )

    # HuggingFace emotion — separate model per language
    if not args.skip_emotion:
        logger.info(
            "Running HuggingFace emotion — EN: %s | ZH: %s …",
            config.ENGLISH_EMOTION_MODEL,
            config.CHINESE_EMOTION_MODEL,
        )
        df = add_hf_emotion(
            df,
            en_model=config.ENGLISH_EMOTION_MODEL,
            zh_model=config.CHINESE_EMOTION_MODEL,
            text_col="text",
            batch_size=config.ANALYSIS_BATCH_SIZE,
            device=args.device,
        )

    # Save combined analyzed CSV
    analyzed_path = os.path.join(args.output_dir, "analyzed_all_channels.csv")
    df.to_csv(analyzed_path, index=False, encoding="utf-8-sig")
    logger.info("Saved analyzed data → %s", analyzed_path)

    # Save per-channel analyzed CSVs (if channel_name column exists)
    if "channel_name" in df.columns:
        for ch_name, ch_df in df.groupby("channel_name"):
            safe_name = str(ch_name).replace("/", "_").replace(" ", "_")
            per_path = os.path.join(args.output_dir, f"analyzed_{safe_name}.csv")
            ch_df.to_csv(per_path, index=False, encoding="utf-8-sig")
            logger.info("  Per-channel → %s", per_path)

    # HTML report
    stats = compute_stats(df)
    logger.info("Stats: %s", stats)

    channel_info = {
        "title": (
            ", ".join(df["channel_name"].unique())
            if "channel_name" in df.columns
            else "Unknown"
        ),
        "total_comments": len(df),
        "unique_videos": df["video_id"].nunique() if "video_id" in df.columns else "N/A",
    }

    report_path = os.path.join(args.output_dir, "report.html")
    generate_html_report(channel_info, stats, df, output_path=report_path)

    # Publication-quality figures
    figures_dir = getattr(args, "figures_dir", None) or os.path.join(args.output_dir, "figures")
    figures_fmt = getattr(args, "figures_fmt", "png")
    logger.info("Generating publication figures (%s) → %s …", figures_fmt, figures_dir)
    generate_figures(df, out_dir=figures_dir, fmt=figures_fmt)

    logger.info("Analysis complete. Open %s to view the report.", report_path)
    return df


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.input):
        logger.error(
            "Input file not found: %s\n"
            "Run 'python -m collection.collect' first to collect comments.",
            args.input,
        )
        sys.exit(1)

    logger.info("Loading comments from %s …", args.input)
    df = pd.read_csv(args.input, dtype=str)
    df["like_count"] = pd.to_numeric(df.get("like_count", 0), errors="coerce").fillna(0).astype(int)

    run(df, args)
