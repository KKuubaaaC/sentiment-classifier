"""
Inference: przewidywanie oceny (1-5) dla pojedynczego tekstu opinii.
Użycie:
  python -m src.predict "Tekst opinii..."
  echo "Tekst opinii..." | python -m src.predict
"""
from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

from src.preprocessing import PolishTextPreprocessor

# Ścieżki względem katalogu projektu (nad src/)
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def load_pipeline():
    """Ładuje preprocessor, model embeddingów i klasyfikator."""
    preprocessor = PolishTextPreprocessor(remove_emoji=True, remove_stopwords=False)
    encoder = SentenceTransformer(MODEL_NAME)
    classifier = joblib.load(MODELS_DIR / "best_classifier.pkl")
    return preprocessor, encoder, classifier


def predict(text: str, preprocessor=None, encoder=None, classifier=None) -> int:
    """
    Zwraca przewidywaną ocenę 1-5 dla tekstu opinii.
    Jeśli nie podasz preprocessor/encoder/classifier, zostaną załadowane (wolniejsze przy wielu wywołaniach).
    """
    own_pipeline = preprocessor is None and encoder is None and classifier is None
    if own_pipeline:
        preprocessor, encoder, classifier = load_pipeline()

    if not text or not str(text).strip():
        return 0  # lub rzuć wyjątek

    clean = preprocessor.preprocess_pipeline(str(text))
    emb = encoder.encode([clean])
    pred_01 = classifier.predict(emb)[0]
    return int(pred_01) + 1  # 0-4 -> 1-5


def main():
    import sys

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("Podaj tekst opinii jako argument lub na stdin.", file=sys.stderr)
        sys.exit(1)

    rating = predict(text)
    print(rating)


if __name__ == "__main__":
    main()
