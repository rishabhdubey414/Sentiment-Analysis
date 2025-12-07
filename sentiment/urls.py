# sentiment/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),                # HTML UI
    path("api/predict/", views.api_predict, name="api_predict"),  # REST API
]
