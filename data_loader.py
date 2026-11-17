"""Download and load the labeled NLTK Movie Reviews corpus."""

import nltk
from nltk.corpus import movie_reviews


def ensure_nltk_resources() -> None:
    """Download the corpus and tokenizer resources if they are missing."""
    resources = {
        "corpora/movie_reviews": "movie_reviews",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
    }
    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name, quiet=False)


def load_reviews() -> list[tuple[str, str]]:
    """Return (review_text, sentiment_label) pairs."""
    ensure_nltk_resources()
    records = []
    for label in movie_reviews.categories():
        for file_id in movie_reviews.fileids(label):
            records.append((movie_reviews.raw(file_id), label))
    return records
