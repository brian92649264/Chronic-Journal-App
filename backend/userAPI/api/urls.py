from rest_framework import routers
from .views import LogListCreateView, LogRetrieveUpdateDestroyView, SpeechToTextAPIView
from django.urls import path, include

urlpatterns = [
    path('logs/', LogListCreateView.as_view(), name='log-list-create'),
    path('logs/<int:pk>/', LogRetrieveUpdateDestroyView.as_view(), name='log-detail'),
    path('speech-to-text/', SpeechToTextAPIView.as_view(), name='speech-to-text'),
]