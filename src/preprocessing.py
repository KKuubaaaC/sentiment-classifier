# src/preprocessing.py
import re
import unicodedata
from typing import Optional

# Opcjonalnie: pip install emoji
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

# Minimalna lista stopwords PL (możesz rozszerzyć lub załadować z pliku)
DEFAULT_POLISH_STOPWORDS = {
    "i", "w", "na", "z", "do", "nie", "że", "to", "się", "o", "ale", "jak",
    "jest", "po", "za", "od", "dla", "czy", "już", "tak", "bardzo", "ale",
    "tylko", "przez", "przed", "po", "nad", "pod", "bez", "aż", "gdy", "gdzie",
}


class PolishTextPreprocessor:
    def __init__(
        self,
        lowercase: bool = False,
        remove_emoji: bool = True,
        remove_stopwords: bool = False,
        stopwords: Optional[set] = None,
    ):
        self.lowercase = lowercase
        self.remove_emoji = remove_emoji
        self.remove_stopwords = remove_stopwords
        self.stopwords = stopwords or DEFAULT_POLISH_STOPWORDS

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        # URL
        text = re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)
        # HTML
        text = re.sub(r"<[^>]+>", "", text)
        # Unicode
        text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()
        # Emoji
        if self.remove_emoji and HAS_EMOJI:
            text = emoji.replace_emoji(text, replace="")
        elif self.remove_emoji and not HAS_EMOJI:
            text = re.sub(r"[\U0001F300-\U0001F9FF]", "", text)  # prosty zakres emoji
        # Wielokrotne spacje i białe znaki
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        return text.split()

    def preprocess_pipeline(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        if self.remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in self.stopwords]
        return " ".join(tokens)

    def preprocess_series(self, series):
        """Dla pandas: df['text_clean'] = preprocessor.preprocess_series(df['text'])"""
        import pandas as pd
        return series.astype(str).map(self.preprocess_pipeline)