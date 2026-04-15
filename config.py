import os
from dotenv import load_dotenv

load_dotenv()

# YouTube API
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Channels Excel file (contains multiple channels)
CHANNELS_FILE = os.getenv("CHANNELS_FILE", "channels.xlsx")

# Collection settings
MAX_VIDEOS = int(os.getenv("MAX_VIDEOS", 0))       # 0 = collect all videos
MAX_COMMENTS_PER_VIDEO = int(os.getenv("MAX_COMMENTS_PER_VIDEO", 0))  # 0 = collect all

# Output settings
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
RAW_COMMENTS_FILE = os.path.join(OUTPUT_DIR, "comments_raw.csv")
ANALYZED_FILE = os.path.join(OUTPUT_DIR, "comments_analyzed.csv")
REPORT_FILE = os.path.join(OUTPUT_DIR, "report.html")

# Analysis settings
ANALYSIS_BATCH_SIZE = int(os.getenv("ANALYSIS_BATCH_SIZE", 32))

# ── English analysis models ──────────────────────────────────────────────────
# Sentiment: RoBERTa-base fine-tuned on ~124 M tweets (TweetEval benchmark)
# Labels: Negative / Neutral / Positive
ENGLISH_SENTIMENT_MODEL = os.getenv(
    "ENGLISH_SENTIMENT_MODEL",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
)
# Emotion: RoBERTa-large fine-tuned on 6 diverse English emotion datasets
# Labels: anger / disgust / fear / joy / neutral / sadness / surprise
ENGLISH_EMOTION_MODEL = os.getenv(
    "ENGLISH_EMOTION_MODEL",
    "j-hartmann/emotion-english-roberta-large",
)

# ── Chinese analysis models ───────────────────────────────────────────────────
# Sentiment: bert-base-chinese fine-tuned on Chinese financial news sentiment
# Labels: Negative / Neutral / Positive
CHINESE_SENTIMENT_MODEL = os.getenv(
    "CHINESE_SENTIMENT_MODEL",
    "hw2942/bert-base-chinese-finetuning-financial-news-sentiment-v2",
)
# Emotion: RoBERTa-large fine-tuned on 6 diverse emotion datasets
# Labels: anger / disgust / fear / joy / neutral / sadness / surprise
CHINESE_EMOTION_MODEL = os.getenv(
    "CHINESE_EMOTION_MODEL",
    "j-hartmann/emotion-english-roberta-large",
)
