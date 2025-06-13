import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
emotion_csv_path = os.path.join(BASE_DIR, 'dataset', 'combined_emotion.csv')
sentiment_csv_path = os.path.join(BASE_DIR, 'dataset', 'combined_sentiment_data.csv')

# Load datasets
emotion_df = pd.read_csv("C:\Sentiment Analysis\sentiment\dataset\combined_emotion.csv")
sentiment_df = pd.read_csv("C:\Sentiment Analysis\sentiment\dataset\combined_sentiment_data.csv")

# Extract features and labels
X_emotion = emotion_df['sentence']
y_emotion = emotion_df['emotion']

X_sentiment = sentiment_df['sentence']
y_sentiment = sentiment_df['sentiment']

# Train emotion model
emotion_model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
emotion_model.fit(X_emotion, y_emotion)

# Train sentiment model
sentiment_model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
sentiment_model.fit(X_sentiment, y_sentiment)



# Prediction function
def predict_sentiment(text):
    emotion = emotion_model.predict([text])[0]
    sentiment = sentiment_model.predict([text])[0]
    return {
        'emotion': emotion,
        'sentiment': sentiment
    }
