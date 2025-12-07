# sentiment/views.py

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .predictor import predict_sentiment
import json


def index(request):
    """
    Renders an HTML page with a textarea.
    On POST, shows the predicted sentiment.
    """
    result = None
    text = ""

    if request.method == "POST":
        text = request.POST.get("text", "")
        if text:
            result = predict_sentiment(text)

    context = {
        "text": text,
        "result": result,
    }
    return render(request, "index.html", context)
    # change template name if you're using a different one


@csrf_exempt
def api_predict(request):
    """
    REST-style endpoint:
    - POST JSON: { "text": "your review" }
    - or GET: /api/predict/?text=your+review

    Returns JSON:
    {
        "text": "...",
        "sentiment": "Positive"
    }
    """
    if request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            text = body.get("text", "")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
    else:
        text = request.GET.get("text", "")

    if not text:
        return JsonResponse({"error": "No text provided"}, status=400)

    sentiment = predict_sentiment(text)

    return JsonResponse(
        {
            "text": text,
            "sentiment": sentiment,
        },
        status=200,
    )
