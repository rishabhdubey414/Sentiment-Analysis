📌 Sentiment Analysis — Django + ML + Streamlit + NLP Pipeline

A complete Sentiment Analysis system built using Python, Scikit-learn, TensorFlow (optional), Django, Streamlit, NLTK, SpaCy, Pandas, and Matplotlib.
It supports real-time predictions, REST API, and modern NLP preprocessing (tokenization, lemmatization, stopword removal).

🚀 Features
🔹 1. Machine Learning Model (Scikit-learn)

Trained using TF-IDF vectorization

Logistic Regression for sentiment classification

Achieves strong performance on combined_sentiment.csv dataset

Saved as:

/sentiment/ml/model.pkl

/sentiment/ml/vectorizer.pkl

🔹 2. Deep Learning Baseline (TensorFlow – optional)

LSTM model for sequence learning

Tokenizer + padded sequences

Exportable as .h5 for production inference

(Optional but supported in project upgrade.)

🔹 3. Advanced NLP Preprocessing

Implemented in sentiment/preprocessing.py:

✔ Lowercasing
✔ Removing noise & URLs
✔ Tokenization (SpaCy)
✔ Lemmatization
✔ Stopword removal (NLTK + SpaCy)
✔ Normalization

This ensures cleaner, more accurate features for the model.

🔹 4. Django Backend (Real-Time Predictions)

UI form input → ML model → response

REST endpoint:

GET /api/predict/?text=I love this
POST /api/predict/ { "text": "I dislike this product" }


JSON output:

{
  "text": "I love this",
  "sentiment": "Positive"
}

🔹 5. Streamlit Frontend

A modern alternative UI located in /streamlit_frontend/app.py:

Clean interface

Live sentiment predictions

Works independently from Django backend

🔹 6. Dataset

Located in:

/sentiment/dataset/
    |-- combined_sentiment.csv
    |-- combined_emotion.csv


Supports binary and multi-class emotion classification.

🔹 7. Training Notebook

(You will include after adding it)

model_training/sentiment_training.ipynb contains:

Data exploration

Preprocessing pipeline

TF-IDF + Logistic Regression

LSTM model

Accuracy, confusion matrix, visualizations

Saving .pkl + .h5 model files

📁 Project Structure
Sentiment Analysis/
│
├── mysite/                     # Django project
│
├── sentiment/                  # Django ML app
│   ├── dataset/
│   ├── ml/
│   │   ├── model.pkl
│   │   ├── vectorizer.pkl
│   ├── preprocessing.py
│   ├── predictor.py
│   ├── views.py
│   ├── urls.py
│   └── templates/
│
├── streamlit_frontend/
│   └── app.py
│
├── model_training/
│   └── sentiment_training.ipynb
│
└── requirements.txt

⚙️ Installation & Setup
1. Create virtual environment
python -m venv venv
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"

3. Run Django server
python manage.py runserver

4. Run Streamlit app
streamlit run streamlit_frontend/app.py

🧪 API Usage
Example (GET):
/api/predict/?text=This movie was amazing

Response:
{
  "text": "This movie was amazing",
  "sentiment": "Positive"
}

📊 Model Performance

(You will add after training)

Accuracy: XX%

Loss curves

Confusion matrix

Class distribution charts

📜 Tech Stack
Languages

Python 3.x

Libraries

Django

Scikit-learn

TensorFlow

Pandas

Matplotlib

NLTK

SpaCy

Streamlit

ML Techniques

TF-IDF vectorization

Logistic Regression

LSTM (optional)