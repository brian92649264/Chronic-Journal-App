from rest_framework import routers
from .views import LogListCreateView, LogRetrieveUpdateDestroyView, TranscribeAudioView, PainAnalysisView, ChartsView
from django.urls import path, include

urlpatterns = [
    path('logs/', LogListCreateView.as_view(), name='log-list-create'),
    path('logs/<int:pk>/', LogRetrieveUpdateDestroyView.as_view(), name='log-detail'),
    path('speech-to-text/', TranscribeAudioView.as_view(), name='transcribe'),
    path('patterns/', PainAnalysisView.as_view(), name='pain-analysis'),
    path('charts/',ChartsView.as_view(), name='chart-view'),
]