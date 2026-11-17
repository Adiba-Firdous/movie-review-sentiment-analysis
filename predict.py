"""Interactive command-line sentiment prediction."""

from pathlib import Path
import joblib
from src.text_preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentiment_model.pkl"


def main() -> None:
    if not MODEL_PATH.exists():
        print("Trained model not found. Run this first: python -m src.train")
        return

    classifier = joblib.load(MODEL_PATH)
    print("Movie Review Sentiment Predictor")
    print("Enter a review, or type 'exit' to quit.\n")

    while True:
        review = input("Enter a movie review: ").strip()
        if review.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not review:
            print("Please enter a review.")
            continue

        prediction = classifier.classify(clean_text(review))
        label = "POSITIVE" if prediction == "pos" else "NEGATIVE"
        print(f"Predicted sentiment: {label}\n")


if __name__ == "__main__":
    main()
