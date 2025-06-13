from django.urls import path
from . import views

urlpatterns = [
    path('', views.result, name='home'),
    path('', views.index, name='index'),
    
]
