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
# Sentiment: binary POSITIVE/NEGATIVE classifier fine-tuned on SST-2
ENGLISH_SENTIMENT_MODEL = os.getenv(
    "ENGLISH_SENTIMENT_MODEL",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
# Emotion: 7-class (anger/disgust/fear/joy/neutral/sadness/surprise)
ENGLISH_EMOTION_MODEL = os.getenv(
    "ENGLISH_EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base",
)

# ── Chinese analysis models ───────────────────────────────────────────────────
# Sentiment: multilingual model that handles Simplified & Traditional Chinese
# Labels: positive / neutral / negative
CHINESE_SENTIMENT_MODEL = os.getenv(
    "CHINESE_SENTIMENT_MODEL",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
)
# Emotion: multilingual emotion classifier (anger/fear/joy/love/sadness/surprise)
CHINESE_EMOTION_MODEL = os.getenv(
    "CHINESE_EMOTION_MODEL",
    "michellejieli/emotion_text_classifier",
)
