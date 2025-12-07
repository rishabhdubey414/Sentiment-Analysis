# sentiment/preprocessing.py

import re
import nltk
import spacy
from nltk.corpus import stopwords

# NOTE:
# Run these once in your venv:
#   python -m spacy download en_core_web_sm
#   python -c "import nltk; nltk.download('stopwords')"

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Basic cleaning: lowercase, remove URLs, non-letters, extra spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-zA-Z']", " ", text)        # keep only letters and '
    text = re.sub(r"\s+", " ", text).strip()       # collapse spaces
    return text


def preprocess_text(text: str) -> str:
    """
    Full preprocessing using SpaCy + NLTK:
    - clean
    - tokenize via SpaCy
    - remove stopwords, punctuation, spaces
    - lemmatize
    """
    text = clean_text(text)
    doc = nlp(text)

    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space:
            continue
        lemma = token.lemma_.strip()
        if lemma and lemma not in stop_words:
            tokens.append(lemma)

    return " ".join(tokens)
