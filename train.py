"""Train and evaluate a TextBlob Naive Bayes sentiment classifier."""

from pathlib import Path
import joblib
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data_loader import load_reviews
from src.text_preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentiment_model.pkl"


def main() -> None:
    print("Loading labeled movie reviews...")
    records = load_reviews()

    cleaned_records = [
        (clean_text(review), label)
        for review, label in records
        if clean_text(review)
    ]

    texts = [review for review, _ in cleaned_records]
    labels = [label for _, label in cleaned_records]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )

    train_data = list(zip(X_train, y_train))
    test_data = list(zip(X_test, y_test))

    print(f"Total reviews: {len(cleaned_records)}")
    print(f"Training reviews: {len(train_data)}")
    print(f"Testing reviews: {len(test_data)}")
    print("Training Naive Bayes classifier...")

    classifier = NaiveBayesClassifier(train_data)

    predictions = [classifier.classify(review) for review in X_test]

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, labels=["neg", "pos"],
        target_names=["Negative", "Positive"], zero_division=0
    ))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=["neg", "pos"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
