from django.shortcuts import render
from .predictor import predict_sentiment  # ✅ Import the prediction function

def result(request):
    if request.method == 'POST':
        text = request.POST['text']  # Get text input from the form
        prediction = predict_sentiment(text)  # Predict sentiment
        return render(request, 'sentiment/result.html', {
            'text': text,
            'prediction': prediction
        })
    return render(request, 'sentiment/home.html')


def index(request):
    result = None
    text = ''
    if request.method == 'POST':
        text = request.POST.get('text')
        if text:
            result = predict_sentiment(text)
    return render(request, 'index.html', {'result': result, 'text': text})