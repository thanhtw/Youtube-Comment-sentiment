"""
analysis/analyzer.py
~~~~~~~~~~~~~~~~~~~~~
Bilingual sentiment and emotion analysis (English + Chinese).

Language detection
------------------
Each comment is classified as 'zh' (Chinese) or 'en' (English/other) by
measuring the ratio of CJK characters in the text.  No extra library needed.

English pipeline
----------------
  1. VADER — lexicon-based, no model download, instant.
     Compound score → positive / neutral / negative label.
  2. HuggingFace sentiment — deep-learning classifier (SST-2 fine-tune).
     Default: distilbert-base-uncased-finetuned-sst-2-english
  3. HuggingFace emotion — 7-class distilroberta model.
     Default: j-hartmann/emotion-english-distilroberta-base
     Labels: anger · disgust · fear · joy · neutral · sadness · surprise

Chinese pipeline
----------------
  1. HuggingFace sentiment — multilingual model that supports Simplified and
     Traditional Chinese.
     Default: lxyuan/distilbert-base-multilingual-cased-sentiments-student
     Labels: positive / neutral / negative
  2. HuggingFace emotion — multilingual emotion classifier.
     Default: michellejieli/emotion_text_classifier
     Labels: anger · fear · joy · love · sadness · surprise

No model training is required — all defaults are pre-trained, freely available
models downloaded on first run from HuggingFace Hub.  Any compatible
text-classification model can be swapped via .env settings.

Output columns added
--------------------
  language          'zh' or 'en'
  vader_compound    VADER score −1 to +1  (English only, NaN for Chinese)
  vader_label       positive / neutral / negative  (English only)
  sentiment_label   Model prediction for the comment's language
  sentiment_score   Confidence (0–1)
  emotion_label     Model prediction for the comment's language
  emotion_score     Confidence (0–1)
"""

import logging
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# VADER — pure Python, no torch needed (English only)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader_available = True
except ImportError:
    _vader_available = False
    logger.warning("vaderSentiment not installed — VADER scores will be skipped.")

# Transformers — optional but strongly recommended
try:
    from transformers import pipeline as hf_pipeline
    _transformers_available = True
except ImportError:
    _transformers_available = False
    logger.warning("transformers not installed — HuggingFace models will be skipped.")


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
                      r"\u3040-\u309f\u30a0-\u30ff]")  # CJK + kana
_ALPHA_RE = re.compile(r"[a-zA-Z\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
                        r"\u3040-\u309f\u30a0-\u30ff]")


def _detect_lang(text: str) -> str:
    """Return 'zh' when ≥30 % of alphabetic characters are CJK, else 'en'."""
    s = str(text)
    cjk = len(_CJK_RE.findall(s))
    alpha = len(_ALPHA_RE.findall(s))
    if alpha == 0:
        return "en"
    return "zh" if cjk / alpha >= 0.30 else "en"


def add_language_column(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Detect language of each comment and add a 'language' column ('zh'/'en')."""
    df["language"] = [_detect_lang(t) for t in df[text_col]]
    zh_count = (df["language"] == "zh").sum()
    en_count = (df["language"] == "en").sum()
    logger.info(
        "Language detection complete — English: %d  |  Chinese: %d", en_count, zh_count
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int = 512) -> str:
    return str(text)[:max_chars]


def _run_hf_pipe(pipe, texts: list[str], batch_size: int, desc: str) -> list[dict]:
    """Run a HuggingFace pipeline in batches and return flat list of result dicts."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_out = pipe(texts[i : i + batch_size])
        # top_k=1 returns list[list[dict]]; standard returns list[dict]
        for item in batch_out:
            results.append(item[0] if isinstance(item, list) else item)
    return results


# ---------------------------------------------------------------------------
# VADER sentiment  (English only)
# ---------------------------------------------------------------------------

def add_vader_sentiment(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add VADER scores for English comments.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
    lexicon approach — no model download needed.  It works well for social
    media text (short, informal, emojis, slang).

    Added columns (English rows only; NaN for Chinese):
      vader_compound  float  −1 (very negative) to +1 (very positive)
      vader_pos       float  proportion of positive sentiment tokens
      vader_neu       float  proportion of neutral sentiment tokens
      vader_neg       float  proportion of negative sentiment tokens
      vader_label     str    positive (≥0.05) / negative (≤−0.05) / neutral
    """
    if not _vader_available:
        logger.warning("Skipping VADER — vaderSentiment not installed.")
        return df

    analyzer = SentimentIntensityAnalyzer()

    # Initialise columns with NaN so Chinese rows stay empty
    for col in ("vader_compound", "vader_pos", "vader_neu", "vader_neg", "vader_label"):
        df[col] = np.nan if col != "vader_label" else None

    en_mask = df.get("language", pd.Series("en", index=df.index)) == "en"
    en_idx = df.index[en_mask]

    compounds, positives, neutrals, negatives = [], [], [], []
    for text in tqdm(df.loc[en_idx, text_col], desc="VADER (English)"):
        scores = analyzer.polarity_scores(str(text))
        compounds.append(scores["compound"])
        positives.append(scores["pos"])
        neutrals.append(scores["neu"])
        negatives.append(scores["neg"])

    df.loc[en_idx, "vader_compound"] = compounds
    df.loc[en_idx, "vader_pos"] = positives
    df.loc[en_idx, "vader_neu"] = neutrals
    df.loc[en_idx, "vader_neg"] = negatives
    df.loc[en_idx, "vader_label"] = [
        "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        for c in compounds
    ]
    return df


# ---------------------------------------------------------------------------
# HuggingFace sentiment  (bilingual — separate model per language)
# ---------------------------------------------------------------------------

def add_hf_sentiment(
    df: pd.DataFrame,
    en_model: str,
    zh_model: str,
    text_col: str = "text",
    batch_size: int = 32,
    device: int = -1,
) -> pd.DataFrame:
    """
    Add HuggingFace sentiment predictions using language-specific models.

    English model  (default: distilbert-base-uncased-finetuned-sst-2-english)
    ─────────────
    Fine-tuned on the Stanford Sentiment Treebank v2 (SST-2).
    Pre-trained model, no additional training required.
    Labels: POSITIVE / NEGATIVE

    Chinese model  (default: lxyuan/distilbert-base-multilingual-cased-sentiments-student)
    ─────────────
    Knowledge-distilled multilingual model trained on 7 languages including
    Simplified and Traditional Chinese.
    Pre-trained model, no additional training required.
    Labels: positive / neutral / negative

    Added columns:
      sentiment_label   str    model prediction label
      sentiment_score   float  confidence score (0–1)
    """
    if not _transformers_available:
        logger.warning("Skipping HF sentiment — transformers not installed.")
        return df

    df["sentiment_label"] = None
    df["sentiment_score"] = np.nan

    lang_col = df.get("language", pd.Series("en", index=df.index))

    for lang, model_name, desc in [
        ("en", en_model, f"HF sentiment English ({en_model.split('/')[-1]})"),
        ("zh", zh_model, f"HF sentiment Chinese ({zh_model.split('/')[-1]})"),
    ]:
        mask = lang_col == lang
        if not mask.any():
            continue

        idx = df.index[mask]
        texts = [_truncate(str(t)) for t in df.loc[idx, text_col]]
        logger.info("  [%s] Running %s on %d comments …", lang.upper(), model_name, len(texts))

        pipe = hf_pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512,
        )
        results = _run_hf_pipe(pipe, texts, batch_size, desc)

        df.loc[idx, "sentiment_label"] = [r["label"] for r in results]
        df.loc[idx, "sentiment_score"] = [round(r["score"], 4) for r in results]

    return df


# ---------------------------------------------------------------------------
# HuggingFace emotion  (bilingual — separate model per language)
# ---------------------------------------------------------------------------

def add_hf_emotion(
    df: pd.DataFrame,
    en_model: str,
    zh_model: str,
    text_col: str = "text",
    batch_size: int = 32,
    device: int = -1,
) -> pd.DataFrame:
    """
    Add HuggingFace emotion predictions using language-specific models.

    English model  (default: j-hartmann/emotion-english-distilroberta-base)
    ─────────────
    Fine-tuned DistilRoBERTa on 6 English emotion datasets covering social
    media, news, and dialogue.
    Pre-trained model, no additional training required.
    Labels: anger · disgust · fear · joy · neutral · sadness · surprise

    Chinese model  (default: michellejieli/emotion_text_classifier)
    ─────────────
    Multilingual RoBERTa-base fine-tuned for emotion classification.
    Handles Chinese, English, and mixed-language text.
    Pre-trained model, no additional training required.
    Labels: anger · fear · joy · love · sadness · surprise

    Added columns:
      emotion_label   str    predicted dominant emotion
      emotion_score   float  confidence score (0–1)
    """
    if not _transformers_available:
        logger.warning("Skipping HF emotion — transformers not installed.")
        return df

    df["emotion_label"] = None
    df["emotion_score"] = np.nan

    lang_col = df.get("language", pd.Series("en", index=df.index))

    for lang, model_name, desc in [
        ("en", en_model, f"HF emotion English ({en_model.split('/')[-1]})"),
        ("zh", zh_model, f"HF emotion Chinese ({zh_model.split('/')[-1]})"),
    ]:
        mask = lang_col == lang
        if not mask.any():
            continue

        idx = df.index[mask]
        texts = [_truncate(str(t)) for t in df.loc[idx, text_col]]
        logger.info("  [%s] Running %s on %d comments …", lang.upper(), model_name, len(texts))

        pipe = hf_pipeline(
            "text-classification",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512,
            top_k=1,
        )
        results = _run_hf_pipe(pipe, texts, batch_size, desc)

        df.loc[idx, "emotion_label"] = [r["label"] for r in results]
        df.loc[idx, "emotion_score"] = [round(r["score"], 4) for r in results]

    return df


# ---------------------------------------------------------------------------
# Aggregate statistics  (with per-language breakdown)
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    """Return aggregate stats including per-language breakdowns."""
    stats: dict = {"total_comments": len(df)}

    if "language" in df.columns:
        stats["language_breakdown"] = df["language"].value_counts().to_dict()

    if "video_id" in df.columns:
        stats["unique_videos"] = df["video_id"].nunique()

    if "channel_name" in df.columns:
        stats["comments_per_channel"] = (
            df.groupby("channel_name")["comment_id"].count().to_dict()
        )

    # Per-language stats
    lang_groups = {"en": "English", "zh": "Chinese"}
    for lang_code, lang_label in lang_groups.items():
        sub = df[df.get("language", pd.Series("en", index=df.index)) == lang_code]
        if sub.empty:
            continue

        prefix = lang_label

        if "vader_label" in sub.columns and sub["vader_label"].notna().any():
            vc = sub["vader_label"].value_counts(normalize=True).mul(100).round(2).to_dict()
            stats[f"{prefix} VADER sentiment %"] = vc
            stats[f"{prefix} mean VADER compound"] = round(sub["vader_compound"].mean(), 4)

        if "sentiment_label" in sub.columns and sub["sentiment_label"].notna().any():
            vc = sub["sentiment_label"].value_counts(normalize=True).mul(100).round(2).to_dict()
            stats[f"{prefix} HF sentiment %"] = vc

        if "emotion_label" in sub.columns and sub["emotion_label"].notna().any():
            vc = sub["emotion_label"].value_counts(normalize=True).mul(100).round(2).to_dict()
            stats[f"{prefix} emotion %"] = vc

    return stats


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html_report(
    channel_info: dict,
    stats: dict,
    df: pd.DataFrame,
    output_path: str,
    top_n: int = 20,
) -> None:
    """Write a self-contained bilingual HTML analysis report."""

    def _dict_to_table(d: dict) -> str:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items())
        return f"<table border='1' cellpadding='6'><tbody>{rows}</tbody></table>"

    def _df_to_html(dataframe: pd.DataFrame) -> str:
        return dataframe.to_html(index=False, border=1, escape=True)

    def _vc_table(series: pd.Series, label_col: str) -> str:
        tmp = series.value_counts().reset_index()
        tmp.columns = [label_col, "count"]
        tmp["percent"] = (tmp["count"] / tmp["count"].sum() * 100).round(2).astype(str) + " %"
        return _df_to_html(tmp)

    # Language split
    en_df = df[df.get("language", pd.Series("en", index=df.index)) == "en"] if "language" in df.columns else df
    zh_df = df[df.get("language", pd.Series("en", index=df.index)) == "zh"] if "language" in df.columns else pd.DataFrame()

    def _section(sub: pd.DataFrame, lang_label: str) -> str:
        if sub.empty:
            return f"<p>No {lang_label} comments found.</p>"

        parts = [f"<p><strong>Total comments:</strong> {len(sub)}</p>"]

        if "vader_label" in sub.columns and sub["vader_label"].notna().any():
            parts.append("<h3>VADER Sentiment (lexicon-based)</h3>")
            parts.append(_vc_table(sub["vader_label"].dropna(), "vader_label"))
            mean_c = round(sub["vader_compound"].mean(), 4)
            parts.append(f"<p>Mean compound score: <strong>{mean_c}</strong></p>")

        if "sentiment_label" in sub.columns and sub["sentiment_label"].notna().any():
            parts.append("<h3>Sentiment (HuggingFace model)</h3>")
            parts.append(_vc_table(sub["sentiment_label"].dropna(), "sentiment_label"))

        if "emotion_label" in sub.columns and sub["emotion_label"].notna().any():
            parts.append("<h3>Emotion Distribution (HuggingFace model)</h3>")
            parts.append(_vc_table(sub["emotion_label"].dropna(), "emotion_label"))

        return "".join(parts)

    # Top liked comments
    top_html = ""
    if "like_count" in df.columns:
        top_df = (
            df[["video_title", "author", "text", "like_count",
                "language", "sentiment_label", "emotion_label"]]
            .dropna(subset=["like_count"])
            .sort_values("like_count", ascending=False)
            .head(top_n)
        )
        top_html = _df_to_html(top_df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>YouTube Comment Analysis — {channel_info.get('title', '')}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; max-width: 1200px; }}
    h1 {{ color: #cc0000; }}
    h2 {{ color: #333; border-bottom: 2px solid #cc0000; padding-bottom: 4px; }}
    h3 {{ color: #555; margin-top: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    td, th {{ padding: 6px 10px; text-align: left; border: 1px solid #ddd; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    th {{ background: #4472C4; color: white; }}
    .lang-box {{ border: 1px solid #ddd; border-radius: 6px;
                 padding: 16px; margin-bottom: 24px; }}
    .lang-en {{ border-left: 5px solid #4472C4; }}
    .lang-zh {{ border-left: 5px solid #cc0000; }}
  </style>
</head>
<body>
  <h1>YouTube Comment Analysis</h1>

  <h2>Channel Info</h2>
  {_dict_to_table(channel_info)}

  <h2>Overall Summary</h2>
  {_dict_to_table(stats)}

  <h2>English Comments Analysis</h2>
  <div class="lang-box lang-en">
    {_section(en_df, "English")}
  </div>

  <h2>Chinese Comments Analysis (中文評論分析)</h2>
  <div class="lang-box lang-zh">
    {_section(zh_df, "Chinese")}
  </div>

  <h2>Top {top_n} Most-Liked Comments</h2>
  {top_html or "<p>N/A</p>"}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    logger.info("HTML report saved to %s", output_path)


