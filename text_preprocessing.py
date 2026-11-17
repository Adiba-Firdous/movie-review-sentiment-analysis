"""Text cleaning and token preparation for movie reviews."""

import re
from textblob import TextBlob


def clean_text(text: str) -> str:
    """Normalize a review while preserving useful words and punctuation."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)             # remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s']", " ", text)           # keep letters and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    """Tokenize cleaned text with TextBlob."""
    return [word for word in TextBlob(clean_text(text)).words if word.strip()]
