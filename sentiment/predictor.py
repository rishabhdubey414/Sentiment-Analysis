# sentiment/predictor.py

import os
import pickle
from django.conf import settings

from .preprocessing import preprocess_text

# Build absolute path to /sentiment/ml/
BASE_DIR = settings.BASE_DIR
ML_DIR = os.path.join(BASE_DIR, "sentiment", "ml")

MODEL_PATH = os.path.join(ML_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(ML_DIR, "vectorizer.pkl")

# Load artifacts once at startup
try:
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    MODEL_LOADED = True
except Exception as e:
    # You can log this
    print(f"[predictor] Error loading model/vectorizer: {e}")
    vectorizer = None
    model = None
    MODEL_LOADED = False


def predict_sentiment(text: str) -> str:
    """
    Takes raw user text, applies preprocessing, vectorization,
    and returns the model's predicted label as string.
    """
    if not MODEL_LOADED:
        return "Model not available"

    processed = preprocess_text(text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]

    # If your labels are 0/1, map them to strings:
    label_map = {
        0: "Negative",
        1: "Positive"
        # add more if multi-class
    }

    return label_map.get(prediction, str(prediction))
