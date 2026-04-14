# YouTube Channel Comment Analyser

Collect every comment from multiple YouTube channels and analyse **sentiment** (VADER + HuggingFace) and **emotion** (HuggingFace `j-hartmann/emotion-english-distilroberta-base`).

The project is split into two independent sections that can be run together or separately:

| Section | Purpose |
|---|---|
| **1. Data Collection** (`collection/`) | Fetch all videos + comments from YouTube via the Data API v3 |
| **2. Sentiment Analysis** (`analysis/`) | Run VADER and HuggingFace models; generate CSV reports and HTML report |

---

## Project structure

```
emotion_project/
│
├── collection/                       ← Section 1: Data Collection
│   ├── __init__.py
│   ├── youtube_collector.py          YouTube API v3 wrapper
│   ├── channel_loader.py             Read channels from Excel
│   └── collect.py                    Standalone entry point
│
├── analysis/                         ← Section 2: Sentiment Analysis
│   ├── __init__.py
│   ├── analyzer.py                   VADER + HuggingFace models + HTML report
│   └── analyze.py                    Standalone entry point
│
├── main.py                           Full pipeline (runs both sections)
├── config.py                         Centralised settings (reads from .env)
├── channels.xlsx                     List of channels to collect
├── create_channels_template.py       Generate a blank channels.xlsx
├── requirements.txt
├── .env.example                      Copy to .env and fill in your values
└── output/                           Created automatically
    ├── raw_<ChannelName>.csv         Per-channel raw comments
    ├── raw_all_channels.csv          Combined raw comments
    ├── analyzed_<ChannelName>.csv    Per-channel analyzed comments
    ├── analyzed_all_channels.csv     Combined analyzed comments
    └── report.html                   Self-contained HTML report
```

---

## Quick start

### 1. Get a YouTube Data API v3 key

1. Go to <https://console.cloud.google.com/>
2. Create a project → Enable **YouTube Data API v3**
3. Create an **API key** (Credentials → Create Credentials → API key)

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set YOUTUBE_API_KEY and CHANNELS_FILE
```

### 3. Add your channels

```bash
# Generate a formatted channels.xlsx template
python create_channels_template.py

# Then open channels.xlsx and fill in your channels:
# | channel_id           | channel_handle | name       |
# | UCVHFbw7woebKtFFqtypAKfw |            | Kurzgesagt |
# |                      | @veritasium    | Veritasium |
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> **Tip:** Install PyTorch with CUDA support if you have a GPU:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## Run options

### Full pipeline (collect + analyse)

```bash
# Run both sections end-to-end
python main.py

# Limit scope for quick testing (5 videos, 100 comments per video)
python main.py --max-videos 5 --max-comments 100

# Skip heavy DL models — VADER only (very fast, no model download)
python main.py --skip-emotion --skip-sentiment-model

# Use a different channels file
python main.py --channels-file my_channels.xlsx

# Run analysis models on GPU 0
python main.py --device 0

# Save output to a custom directory
python main.py --output-dir results/
```

---

### Section 1 only — Data Collection

Fetches comments and saves raw CSVs. Does **not** run any analysis models.

```bash
# Collect all channels in channels.xlsx
python -m collection.collect

# Limit to 10 videos and 200 comments per video
python -m collection.collect --max-videos 10 --max-comments 200

# Use a different channels file
python -m collection.collect --channels-file my_channels.xlsx

# See all options
python -m collection.collect --help
```

**Output:** `output/raw_<ChannelName>.csv` and `output/raw_all_channels.csv`

---

### Section 2 only — Sentiment Analysis

Reads an existing raw CSV and runs analysis models. No API key needed.

```bash
# Analyse the combined raw CSV (produced by Section 1)
python -m analysis.analyze

# Analyse a specific file
python -m analysis.analyze --input output/raw_all_channels.csv

# VADER only — skip all HuggingFace models (fast)
python -m analysis.analyze --skip-emotion --skip-sentiment-model

# Skip only the emotion model (runs HF sentiment but not emotion)
python -m analysis.analyze --skip-emotion

# Run on GPU 0
python -m analysis.analyze --device 0

# See all options
python -m analysis.analyze --help
```

**Output:** `output/analyzed_<ChannelName>.csv`, `output/analyzed_all_channels.csv`, `output/report.html`

---

### All CLI flags

| Flag | Section | Default | Description |
|---|---|---|---|
| `--channels-file` | collect / main | `channels.xlsx` | Excel file listing channels |
| `--api-key` | collect / main | from `.env` | YouTube Data API v3 key |
| `--max-videos` | collect / main | `0` (unlimited) | Max videos per channel |
| `--max-comments` | collect / main | `0` (unlimited) | Max comments per video |
| `--input` | analyze | `output/raw_all_channels.csv` | Raw CSV to analyse |
| `--skip-emotion` | analyze / main | off | Skip HuggingFace emotion model |
| `--skip-sentiment-model` | analyze / main | off | Skip HuggingFace sentiment model |
| `--device` | analyze / main | `-1` (CPU) | Torch device (`0` = first GPU) |
| `--output-dir` | all | `output/` | Directory for all output files |

---

## Output files

| File | Description |
|---|---|
| `raw_<ChannelName>.csv` | Raw comments for one channel |
| `raw_all_channels.csv` | Combined raw comments from all channels |
| `analyzed_<ChannelName>.csv` | Analyzed comments for one channel |
| `analyzed_all_channels.csv` | Combined analyzed comments |
| `report.html` | Self-contained HTML report with stats & top comments |

### CSV columns

| Column | Description |
|---|---|
| `channel_id` | YouTube channel ID |
| `channel_name` | Channel display name |
| `video_id` | YouTube video ID |
| `video_title` | Video title |
| `comment_id` | Comment ID |
| `parent_id` | Parent comment ID (empty for top-level) |
| `author` | Comment author display name |
| `text` | Comment text |
| `like_count` | Number of likes on the comment |
| `published_at` | ISO 8601 publish timestamp |
| `updated_at` | ISO 8601 last-edit timestamp |
| `is_reply` | `True` if this is a reply |
| `vader_compound` | VADER compound score (−1 to +1) |
| `vader_label` | `positive` / `neutral` / `negative` |
| `sentiment_label` | HF model label (e.g. POSITIVE / NEGATIVE) |
| `sentiment_score` | HF model confidence score |
| `emotion_label` | Predicted emotion (anger / joy / sadness / …) |
| `emotion_score` | Emotion model confidence score |

---

## Emotion labels

The default model (`j-hartmann/emotion-english-distilroberta-base`) outputs:

`anger` · `disgust` · `fear` · `joy` · `neutral` · `sadness` · `surprise`

---

## API quota notes

YouTube Data API v3 has a **daily quota of 10 000 units** (free tier).

| Operation | Cost |
|---|---|
| `channels.list` | 1 unit |
| `playlistItems.list` (50 videos) | 1 unit |
| `videos.list` (50 videos) | 1 unit |
| `commentThreads.list` (100 threads) | 1 unit |

A channel with 500 videos and ~200 comments each will use roughly **1 500–2 000 units**.

---

## Customising models

Edit `.env` to swap HuggingFace models (any `text-classification` model works):

```env
EMOTION_MODEL=bhadresh-savani/distilbert-base-uncased-emotion
SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
```

