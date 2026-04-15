"""analysis package — bilingual sentiment and emotion analysis utilities."""
from .analyzer import (
    add_language_column,
    add_vader_sentiment,
    add_hf_sentiment,
    add_hf_emotion,
    compute_stats,
    generate_html_report,
)
from .visualize import generate_figures

__all__ = [
    "add_language_column",
    "add_vader_sentiment",
    "add_hf_sentiment",
    "add_hf_emotion",
    "compute_stats",
    "generate_html_report",
    "generate_figures",
]
