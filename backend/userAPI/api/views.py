from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ..models import Log
from .serializers import LogSerializer


class LogListCreateView(generics.ListCreateAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer


class LogRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer


class SpeechToTextAPIView(APIView):
    def get(self, request):
        return Response({"message": "Please use POST with an audio file for speech-to-text conversion."}, status=status.HTTP_200_OK)

    def post(self, request):
        # Placeholder for speech-to-text logic
        dummy_transcription = "This is a dummy transcription."
        return Response({"transcription": dummy_transcription}, status=status.HTTP_200_OK)

