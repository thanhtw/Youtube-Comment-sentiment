"""
analysis/visualize.py
~~~~~~~~~~~~~~~~~~~~~
Publication-quality figure generation for the bilingual sentiment/emotion
analysis results.  All figures are saved as 300 dpi PNG (and optionally PDF)
so they can be embedded directly in a research paper.

Figures produced
----------------
  fig01_language_distribution.png
      Pie chart — proportion of English vs Chinese comments.

  fig02_sentiment_comparison.png
      Grouped bar chart — sentiment label distribution side-by-side for
      English (RoBERTa) and Chinese (BERT).

  fig03_emotion_distribution_en.png
      Horizontal bar chart — emotion label frequencies for English comments.

  fig04_emotion_distribution_zh.png
      Horizontal bar chart — emotion label frequencies for Chinese comments.

  fig05_vader_compound_distribution.png
      KDE + histogram — VADER compound score distribution (English only).

  fig06_sentiment_score_boxplot.png
      Box plot — model confidence score by sentiment label and language.

  fig07_emotion_score_heatmap.png
      Heatmap — mean emotion confidence score per language × emotion label.

  fig08_comments_per_channel.png
      Horizontal bar chart — comment count per channel (if multi-channel).

  fig09_top_emotions_by_sentiment.png
      Stacked bar chart — emotion composition broken down by sentiment label.

Usage (standalone)
------------------
    python -m analysis.visualize --input output/analyzed_all_channels.csv
    python -m analysis.visualize --input output/analyzed_all_channels.csv \\
        --output-dir output/figures --fmt pdf

Called programmatically from analysis/analyze.py after the pipeline finishes.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — matplotlib / seaborn
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; safe on servers
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    import seaborn as sns
    _viz_available = True
except ImportError:
    _viz_available = False
    logger.warning(
        "matplotlib or seaborn not installed — visualizations will be skipped. "
        "Run: pip install matplotlib seaborn scipy"
    )

# ---------------------------------------------------------------------------
# Global style  (Nature / IEEE-compatible academic theme)
# ---------------------------------------------------------------------------

# Muted, colour-blind-friendly palettes
_C_EN   = "#3A6EA5"   # steel blue  — English
_C_ZH   = "#B03A2E"   # muted red   — Chinese
_C_NEU  = "#7F8C8D"   # slate grey  — neutral sentiment
_C_POS  = "#1A7A4A"   # dark green  — positive
_C_NEG  = "#B03A2E"   # muted red   — negative

_SENTIMENT_PALETTE = {
    "positive": _C_POS, "Positive": _C_POS, "POSITIVE": _C_POS,
    "neutral":  _C_NEU, "Neutral":  _C_NEU, "NEUTRAL":  _C_NEU,
    "negative": _C_NEG, "Negative": _C_NEG, "NEGATIVE": _C_NEG,
}

# Tol bright palette — perceptually distinct, colour-blind safe
_EMOTION_PALETTE = {
    "Anger":    "#CC3311",
    "Disgust":  "#AA3377",
    "Fear":     "#EE7733",
    "Joy":      "#009988",
    "Neutral":  "#BBBBBB",
    "Sadness":  "#0077BB",
    "Surprise": "#33BBEE",
    "Love":     "#EE3377",
}

_LANG_PALETTE = {
    "en": _C_EN, "zh": _C_ZH,
    "English": _C_EN, "Chinese": _C_ZH,
}

# rcParams tuned for ACL / IEEE double-column paper embedding
_RC = {
    "font.family":               "DejaVu Sans",
    "font.size":                 10,
    "axes.titlesize":            11,
    "axes.titleweight":          "bold",
    "axes.labelsize":            10,
    "axes.labelweight":          "bold",
    "axes.linewidth":            0.8,
    "axes.edgecolor":            "#333333",
    "xtick.labelsize":           9,
    "ytick.labelsize":           9,
    "xtick.major.width":         0.8,
    "ytick.major.width":         0.8,
    "legend.fontsize":           9,
    "legend.title_fontsize":     9,
    "legend.frameon":            True,
    "legend.framealpha":         0.92,
    "legend.edgecolor":          "#CCCCCC",
    "figure.dpi":                100,
    "savefig.dpi":               300,
    "savefig.facecolor":         "white",
    "axes.facecolor":            "#FAFAFA",
    "axes.grid":                 True,
    "grid.color":                "#E0E0E0",
    "grid.linewidth":            0.6,
    "axes.spines.top":           False,
    "axes.spines.right":         False,
    "figure.constrained_layout.use": True,
}


def _apply_style() -> None:
    plt.rcParams.update(_RC)
    sns.set_theme(style="ticks", rc=_RC)


def _save(fig: "plt.Figure", path: str, fmt: str = "png") -> None:
    """Save *fig* at 300 dpi with a tight bounding box, then close."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", format=fmt)
    plt.close(fig)
    logger.info("  Saved → %s", path)


def _add_footnote(ax: "plt.Axes", n: int) -> None:
    """Add a sample-size footnote to the bottom-left of the axes."""
    ax.annotate(
        f"n = {n:,}",
        xy=(0, -0.08),
        xycoords="axes fraction",
        fontsize=8,
        color="#666666",
        ha="left",
    )


# ---------------------------------------------------------------------------
# Individual figure functions
# ---------------------------------------------------------------------------

def fig_language_distribution(df: pd.DataFrame, out_dir: str, fmt: str = "png") -> None:
    """Fig 01 — donut chart of comment language split."""
    if "language" not in df.columns:
        return
    counts = df["language"].map({"en": "English", "zh": "Chinese"}).value_counts()
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 4.5))
    colors = [_LANG_PALETTE.get(l, "#aaaaaa") for l in counts.index]
    wedges, _, autotexts = ax.pie(
        counts,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        pctdistance=0.75,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2.0, "width": 0.55},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")

    # Central count label
    total = counts.sum()
    ax.text(0, 0, f"{total:,}\ncomments", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#333333")

    legend_patches = [
        mpatches.Patch(color=colors[i], label=f"{counts.index[i]}  ({counts.iloc[i]:,})")
        for i in range(len(counts))
    ]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    ax.set_title("Comment Language Distribution", pad=10)
    _save(fig, os.path.join(out_dir, f"fig01_language_distribution.{fmt}"), fmt)


def fig_sentiment_comparison(df: pd.DataFrame, out_dir: str, fmt: str = "png") -> None:
    """Fig 02 — grouped bar chart comparing sentiment by language."""
    if "sentiment_label" not in df.columns or "language" not in df.columns:
        return

    rows = []
    lang_ns = {}
    for lang_code, lang_label in [("en", "English"), ("zh", "Chinese")]:
        sub = df[df["language"] == lang_code]["sentiment_label"].dropna()
        if sub.empty:
            continue
        lang_ns[lang_label] = len(sub)
        vc = sub.value_counts(normalize=True).mul(100).rename_axis("label").reset_index(name="pct")
        vc["language"] = lang_label
        rows.append(vc)
    if not rows:
        return

    data = pd.concat(rows, ignore_index=True)
    data["label_norm"] = data["label"].str.capitalize()
    label_order = ["Negative", "Neutral", "Positive"]
    present = [l for l in label_order if l in data["label_norm"].values]
    sent_colors = [_SENTIMENT_PALETTE.get(l, "#888") for l in present]

    lang_labels = data["language"].unique().tolist()
    x = np.arange(len(present))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, lang in enumerate(lang_labels):
        sub = data[data["language"] == lang].set_index("label_norm")["pct"]
        heights = [sub.get(l, 0.0) for l in present]
        offset = (i - (len(lang_labels) - 1) / 2) * width
        bars = ax.bar(
            x + offset, heights, width,
            label=f"{lang} (n={lang_ns.get(lang, 0):,})",
            color=_LANG_PALETTE.get(lang, "#888"),
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        for bar, h in zip(bars, heights):
            if h >= 1.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                        f"{h:.1f}%", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(present, fontsize=10)
    ax.set_ylabel("Percentage of Comments (%)")
    ax.set_title("Sentiment Distribution by Language")
    ax.legend(title="Language", loc="upper right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, data["pct"].max() * 1.18)
    ax.axhline(0, color="#333333", linewidth=0.8)
    _save(fig, os.path.join(out_dir, f"fig02_sentiment_comparison.{fmt}"), fmt)


def fig_emotion_bar(
    df: pd.DataFrame,
    out_dir: str,
    lang_code: str,
    lang_label: str,
    fig_num: int,
    fmt: str = "png",
) -> None:
    """Fig 03/04 — horizontal bar chart of emotion distribution for one language."""
    if "emotion_label" not in df.columns:
        return
    sub = (df[df["language"] == lang_code]["emotion_label"].dropna()
           if "language" in df.columns else df["emotion_label"].dropna())
    if sub.empty:
        return

    vc = sub.value_counts(normalize=True).mul(100).sort_values(ascending=True)
    bar_colors = [_EMOTION_PALETTE.get(l.capitalize(), "#AAAAAA") for l in vc.index]
    labels_cap = [l.capitalize() for l in vc.index]

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, len(vc) * 0.6 + 0.8)))
    bars = ax.barh(labels_cap, vc.values, color=bar_colors,
                   edgecolor="white", linewidth=0.8, height=0.6, zorder=3)

    x_max = vc.max()
    for bar, val in zip(bars, vc.values):
        ax.text(min(val + x_max * 0.015, x_max * 0.98),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                fontsize=9, fontweight="bold", color="#333333")

    ax.set_xlabel("Percentage of Comments (%)")
    ax.set_title(f"Emotion Distribution — {lang_label} Comments")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_xlim(0, x_max * 1.18)
    ax.invert_yaxis()
    _add_footnote(ax, len(sub))
    _save(fig, os.path.join(out_dir, f"fig0{fig_num}_emotion_distribution_{lang_code}.{fmt}"), fmt)


def fig_vader_compound(df: pd.DataFrame, out_dir: str, fmt: str = "png") -> None:
    """Fig 05 — KDE + histogram of VADER compound scores (English)."""
    if "vader_compound" not in df.columns:
        return
    scores = df["vader_compound"].dropna()
    if scores.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    # Shade sentiment zones
    ax.axvspan(-1.0,  -0.05, alpha=0.06, color=_C_NEG, zorder=0)
    ax.axvspan(-0.05,  0.05, alpha=0.06, color=_C_NEU, zorder=0)
    ax.axvspan( 0.05,  1.0,  alpha=0.06, color=_C_POS, zorder=0)

    ax.hist(scores, bins=40, density=True, alpha=0.30,
            color=_C_EN, edgecolor="white", linewidth=0.4, label="Histogram", zorder=2)
    sns.kdeplot(scores, ax=ax, color=_C_EN, linewidth=2.0, label="KDE", zorder=3)
    ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--", label="Neutral (0.0)")
    ax.axvline(scores.mean(), color=_C_NEG, linewidth=1.5,
               linestyle="-.", label=f"Mean ({scores.mean():.3f})", zorder=4)

    # Zone labels at top
    for xc, txt in [(-0.55, "Negative"), (0, "Neutral"), (0.55, "Positive")]:
        ax.text(xc, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.1,
                txt, ha="center", va="top", fontsize=8, color="#555555",
                style="italic")

    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Density")
    ax.set_title("VADER Compound Score Distribution (English Comments)")
    ax.legend(loc="upper left", fontsize=8)
    _add_footnote(ax, len(scores))
    _save(fig, os.path.join(out_dir, f"fig05_vader_compound_distribution.{fmt}"), fmt)


def fig_sentiment_score_boxplot(df: pd.DataFrame, out_dir: str, fmt: str = "png") -> None:
    """Fig 06 — violin + box plot of model confidence by sentiment label × language."""
    if not {"sentiment_label", "sentiment_score", "language"}.issubset(df.columns):
        return
    plot_df = df[["language", "sentiment_label", "sentiment_score"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df["Language"] = plot_df["language"].map({"en": "English", "zh": "Chinese"})
    plot_df["Sentiment"] = plot_df["sentiment_label"].str.capitalize()
    sent_order = [s for s in ["Negative", "Neutral", "Positive"]
                  if s in plot_df["Sentiment"].values]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    sns.violinplot(
        data=plot_df, x="Sentiment", y="sentiment_score", hue="Language",
        order=sent_order, palette=_LANG_PALETTE,
        inner=None, linewidth=0.6, alpha=0.25, cut=0,
        dodge=True, ax=ax,
    )
    sns.boxplot(
        data=plot_df, x="Sentiment", y="sentiment_score", hue="Language",
        order=sent_order, palette=_LANG_PALETTE,
        width=0.22, linewidth=1.0,
        flierprops={"marker": "o", "markersize": 2, "alpha": 0.3,
                    "markeredgewidth": 0.0},
        dodge=True, ax=ax,
    )

    # De-duplicate legend entries created by two overlapping plots
    handles, labels = ax.get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    ax.legend(h2, l2, title="Language", loc="lower right")

    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Model Confidence Score")
    ax.set_title("Sentiment Model Confidence by Label and Language")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _save(fig, os.path.join(out_dir, f"fig06_sentiment_score_boxplot.{fmt}"), fmt)


def fig_emotion_heatmap(df: pd.DataFrame, out_dir: str, fmt: str = "png") -> None:
    """Fig 07 — heatmap of mean emotion confidence per language × emotion."""
    if not {"emotion_label", "emotion_score", "language"}.issubset(df.columns):
        return
    plot_df = df[["language", "emotion_label", "emotion_score"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df["language"] = plot_df["language"].map({"en": "English", "zh": "Chinese"})
    pivot = (
        plot_df.groupby(["language", "emotion_label"])["emotion_score"]
        .mean()
        .unstack("emotion_label")
        .fillna(0)
    )
    # Sort columns by mean descending
    pivot = pivot[pivot.mean().sort_values(ascending=False).index]
    # Capitalise
    pivot.columns = [c.capitalize() for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(6.5, pivot.shape[1] * 1.1), 3.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=pivot.values.max(),
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Mean Confidence", "shrink": 0.8},
        annot_kws={"size": 9, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Mean Emotion Confidence Score by Language and Emotion")
    ax.set_xlabel("Emotion Label")
    ax.set_ylabel("Language")
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, os.path.join(out_dir, f"fig07_emotion_score_heatmap.{fmt}"), fmt)


def fig_comments_per_channel(df: pd.DataFrame, out_dir: str, fmt: str = "png") -> None:
    """Fig 08 — horizontal bar chart of comment count per channel."""
    if "channel_name" not in df.columns or df["channel_name"].nunique() < 2:
        return
    counts = df["channel_name"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(counts) * 0.55 + 0.6)))
    bars = ax.barh(counts.index, counts.values, color=_C_EN,
                   edgecolor="white", linewidth=0.8, height=0.6, zorder=3)
    for bar, val in zip(bars, counts.values):
        ax.text(val + counts.max() * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("Number of Comments")
    ax.set_title("Comment Count per Channel")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, counts.max() * 1.18)
    _save(fig, os.path.join(out_dir, f"fig08_comments_per_channel.{fmt}"), fmt)


def fig_top_emotions_by_sentiment(df: pd.DataFrame, out_dir: str, fmt: str = "png") -> None:
    """Fig 09 — 100 % stacked bar: emotion composition within each sentiment label."""
    if not {"sentiment_label", "emotion_label"}.issubset(df.columns):
        return
    plot_df = df[["sentiment_label", "emotion_label"]].dropna().copy()
    if plot_df.empty:
        return

    plot_df["sentiment_label"] = plot_df["sentiment_label"].str.capitalize()
    sent_order = [s for s in ["Negative", "Neutral", "Positive"]
                  if s in plot_df["sentiment_label"].values]
    ct = (
        plot_df.groupby(["sentiment_label", "emotion_label"])
        .size()
        .unstack("emotion_label")
        .reindex(sent_order)
        .fillna(0)
    )
    ct_pct = ct.div(ct.sum(axis=1), axis=0).mul(100)
    ct_pct.columns = [c.capitalize() for c in ct_pct.columns]

    emotion_order = [e for e in ["Joy", "Surprise", "Neutral", "Fear",
                                  "Sadness", "Disgust", "Anger", "Love"]
                     if e in ct_pct.columns]
    emotion_order += [c for c in ct_pct.columns if c not in emotion_order]
    ct_pct = ct_pct[emotion_order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bottom = np.zeros(len(ct_pct))
    bar_width = 0.55
    for col in emotion_order:
        vals = ct_pct[col].values
        color = _EMOTION_PALETTE.get(col.capitalize(), "#AAAAAA")
        ax.bar(ct_pct.index, vals, bottom=bottom, width=bar_width,
               color=color, label=col.capitalize(),
               edgecolor="white", linewidth=0.5, zorder=3)
        # Label segments ≥ 5 %
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 5:
                ax.text(i, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center", fontsize=8,
                        fontweight="bold", color="white")
        bottom += vals

    ax.legend(title="Emotion", bbox_to_anchor=(1.01, 1),
              loc="upper left", borderaxespad=0)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("Sentiment Label")
    ax.set_title("Emotion Composition within Each Sentiment Category")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylim(0, 105)
    _save(fig, os.path.join(out_dir, f"fig09_top_emotions_by_sentiment.{fmt}"), fmt)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_figures(
    df: pd.DataFrame,
    out_dir: str,
    fmt: str = "png",
) -> None:
    """
    Generate all publication-quality figures from an analysed DataFrame.

    Parameters
    ----------
    df      : Analysed comments DataFrame (output of analysis pipeline).
    out_dir : Directory to save figures.  Created if it does not exist.
    fmt     : Output format — 'png' (300 dpi) or 'pdf' (vector).
    """
    if not _viz_available:
        logger.warning("Skipping figure generation — matplotlib/seaborn not installed.")
        return

    os.makedirs(out_dir, exist_ok=True)
    _apply_style()

    logger.info("Generating publication figures in %s …", out_dir)

    fig_language_distribution(df, out_dir, fmt)
    fig_sentiment_comparison(df, out_dir, fmt)
    fig_emotion_bar(df, out_dir, "en", "English", fig_num=3, fmt=fmt)
    fig_emotion_bar(df, out_dir, "zh", "Chinese", fig_num=4, fmt=fmt)
    fig_vader_compound(df, out_dir, fmt)
    fig_sentiment_score_boxplot(df, out_dir, fmt)
    fig_emotion_heatmap(df, out_dir, fmt)
    fig_comments_per_channel(df, out_dir, fmt)
    fig_top_emotions_by_sentiment(df, out_dir, fmt)

    logger.info("Figure generation complete — %d figures saved to %s", 9, out_dir)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality figures from an analysed comments CSV."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to the analyzed comments CSV (e.g. output/analyzed_all_channels.csv)."
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save figures (default: <input_dir>/figures/)."
    )
    parser.add_argument(
        "--fmt", default="png", choices=["png", "pdf"],
        help="Output image format: png (300 dpi raster) or pdf (vector).  Default: png."
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = _parse_args()

    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input, dtype=str)
    for col in ("vader_compound", "sentiment_score", "emotion_score", "like_count"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    out_dir = args.output_dir or os.path.join(os.path.dirname(args.input), "figures")
    generate_figures(df, out_dir, fmt=args.fmt)
