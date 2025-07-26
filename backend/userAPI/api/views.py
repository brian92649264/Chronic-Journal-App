from rest_framework import generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models.functions import TruncMonth
from ..models import Log
from .serializers import LogSerializer
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from google.cloud import speech_v1 as speech
import os
import base64
import logging
from django.db.models import Q
from datetime import datetime, timedelta
from rest_framework.permissions import AllowAny
from django.db.models import Count, Avg
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

# Download NLTK data 
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(os.getcwd(), 'speech_to_text.json')

class LogListCreateView(generics.ListCreateAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer
    parser_classes = (MultiPartParser, FormParser, JSONParser)  

    def get_queryset(self):
        queryset = Log.objects.all()
        date_range = self.request.query_params.get('date_range', None)
        keyword = self.request.query_params.get('keyword', None)
        pain_level = self.request.query_params.get('pain_level', None)

        if date_range == 'last_month':
            last_month = datetime.now().date() - timedelta(days=30)
            queryset = queryset.filter(date__gte=last_month)
        elif date_range == 'last_week':
            last_week = datetime.now().date() - timedelta(days=7)
            queryset = queryset.filter(date__gte=last_week)
        elif keyword:
            queryset = queryset.filter(
                Q(location__icontains=keyword) | Q(details__icontains=keyword)
            )
        elif pain_level:
            queryset = queryset.filter(level__gte=int(pain_level))

        return queryset.order_by('-date')

class LogRetrieveUpdateDestroyView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Log.objects.all()
    serializer_class = LogSerializer
    parser_classes = (MultiPartParser, FormParser)

class TranscribeAudioView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def get(self, request, *args, **kwargs):
        logger.debug("GET request received for TranscribeAudioView")
        return Response(
            {
                "message": "Use POST with an 'audio' file (in .wav/LINEAR16 or webm/WEBM_OPUS format) or 'audio' base64 string."
            },
            status=status.HTTP_200_OK
        )

    def post(self, request, *args, **kwargs):
        logger.debug("POST request received: %s", request.POST)
        logger.debug("FILES: %s", request.FILES)

        file_obj = request.FILES.get('audio')
        base64_audio = request.POST.get('audio')

        if not file_obj and not base64_audio:
            logger.error("No audio file or base64 string provided")
            return Response({'error': 'No audio file or base64 string provided'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if file_obj:
                logger.debug("Processing file upload")
                content = file_obj.read()
            else:
                logger.debug("Processing base64 audio, length: %d", len(base64_audio))
                if base64_audio.startswith('data:'):
                    base64_audio = base64_audio.split('base64,')[1]
                content = base64.b64decode(base64_audio)
        except Exception as e:
            logger.error("Failed to decode audio: %s", str(e))
            return Response({'error': f'Invalid audio data: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        encoding = request.POST.get('encoding', 'LINEAR16')
        if encoding not in ['LINEAR16', 'WEBM_OPUS']:
            logger.error("Unsupported encoding: %s", encoding)
            return Response({'error': f'Unsupported encoding: {encoding}'}, status=status.HTTP_400_BAD_REQUEST)

        logger.debug("Audio config: encoding=%s, sampleRateHertz=16000", encoding)
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=16000,
            language_code="en-US",
        )

        audio = speech.RecognitionAudio(content=content)

        try:
            client = speech.SpeechClient()
            logger.debug("Sending audio to Google Speech-to-Text")
            response = client.recognize(config=config, audio=audio)
            logger.debug("Google Speech-to-Text response: %s", response)
        except Exception as e:
            logger.error("Google Speech-to-Text error: %s", str(e))
            return Response({'error': f'Speech recognition failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        logger.debug("Transcription result: %s", transcript.strip())
        return Response({"transcript": transcript.strip()}, status=status.HTTP_200_OK)
    
class ChartsView(APIView):
    """View to return data for charts."""
    def get(self, request):
        try:
            logger.debug("Starting ChartsView GET request")
            logs = Log.objects.all()
            logger.debug(f"Retrieved {logs.count()} logs")
            fields = ["id", "date", "location", "level", "details"]
            logs_values = logs.values(*fields)
            logger.debug("Fetched log values")

            # Total number of logs
            total_logs = logs.count()
            logger.debug(f"Total logs: {total_logs}")

            # Average pain level (overall)
            avg_pain = logs.aggregate(avg_level=Avg('level'))['avg_level'] or 0
            avg_pain = round(float(avg_pain), 1)
            logger.debug(f"Average pain: {avg_pain}")

            # Location counts and average pain levels, grouped by location and month
            location_stats = logs.annotate(
                month=TruncMonth('date')
            ).values('month', 'location').annotate(
                count=Count('id'),
                avg_level=Avg('level')
            ).order_by('month', 'location')
            logger.debug(f"Location stats: {list(location_stats)}")
            location_data = [
                {
                    'date': stat['month'].strftime('%Y-%m-%d'),  
                    'location': stat['location'].lower(),
                    'count': stat['count'],
                    'avg_level': round(float(stat['avg_level']), 1)
                }
                for stat in location_stats
            ]
            logger.debug(f"Location data: {location_data}")

            response_data = {
                'total_logs': total_logs,
                'average_pain': avg_pain,
                'location_data': location_data
            }
            logger.debug("Returning response data")
            return Response(response_data, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in ChartsView: {str(e)}", exc_info=True)
            return Response(
                {"error": "Failed to retrieve chart data"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
class PainAnalysisView(APIView):
    queryset = Log.objects.all()
    permission_classes = [AllowAny]

    def __init__(self):
        super().__init__()
        load_dotenv()
        self.api_token = os.environ.get("HF_API_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", token=self.api_token, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            token=self.api_token,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def get(self, request):
        try:
            logger.debug("Starting PainAnalysisView GET request")
            logs = self.queryset.values('id', 'date', 'location', 'level', 'details')
            logs = list(logs)
            
            if not logs:
                logger.debug("No logs found")
                return Response({"logs": [], "pattern": []}, status=status.HTTP_200_OK)
            
            details = [log['details'] or '' for log in logs]
            locations = [log['location'] or 'Unknown' for log in logs]
            levels = [log['level'] or 0 for log in logs]
            valid_indices = [i for i, d in enumerate(details) if d.strip()]
            valid_details = [details[i] for i in valid_indices]
            valid_locations = [locations[i] for i in valid_indices]
            valid_levels = [levels[i] for i in valid_indices]
            
            result_logs = [
                {
                    'id': log['id'],
                    'date': str(log['date']),
                    'location': log['location'],
                    'level': log['level'],
                    'details': log['details'],
                    'predicted_location': 'Unknown',
                    'probability': 0.0
                } 
                for log in logs
            ]
            
            # Count frequent locations
            location_counts = defaultdict(int)
            for log in logs:
                if log['location'] and log['location'] != 'Unknown':
                    location_counts[log['location']] += 1
            
            frequent_locations = [loc for loc, count in location_counts.items() if count >= 5]
            logger.debug(f" Frequent locations (3+ occurrences): {frequent_locations}")
            if not frequent_locations:
                logger.debug("No frequent locations found")
                return Response({"logs": result_logs, "pattern": ["No locations reported three or more times."]}, status=status.HTTP_200_OK)
            
            # Aggregate details for frequent locations
            location_details = defaultdict(list)
            for log in logs:
                if log['location'] in frequent_locations and log['details']:
                    location_details[log['location']].append(log['details'])
            
            # Extract triggers using NLTK
            stop_words = set(stopwords.words('english'))
            activity_keywords = [
                'sitting', 'screen time', 'workout', 'exercise', 'lifting', 'running',
                'walking', 'standing', 'sleeping', 'dehydration', 'medication', 'weather',
                'rainy weather', 'painting', 'gardening', 'typing', 'carrying', 'hunching',
                'stretching', 'resting', 'hydration', 'stress', 'awkward position',
                'physical exertion', 'warm-up', 'cooldown', 'uneven terrain', 'stairs',
                'pms', 'strength training', 'sun exposure', 'pushups', 'hiking', 'falling',
                'flashing lights', 'poor sleep posture', 'lunges', 'furniture assembly',
                'cold exposure', 'phone use', 'squatting', 'travel', 'poor footwear',
                'arthritis', 'angina', 'video calls', 'diet', 'injury'
            ]
            patterns = []
            format_index = 0 
            formats = [
                "Pain in the {location} is more frequent after days with {trigger}.",
                "{location} pain tends to appear after prolonged {trigger}, particularly on weekdays.",
                "{location} pain spikes correlate with {trigger} sessions.",
                "{location} pain often increases following {trigger} activities.",
                "Pain in the {location} is commonly reported after exposure to {trigger}.",
                "{location} discomfort frequently arises during {trigger} periods.",
                "Pain in the {location} tends to escalate with repeated {trigger} episodes.",
                "{location} pain is typically triggered by instances of {trigger}."
            ]
            for location in frequent_locations:
                triggers = defaultdict(int)
                for details in location_details[location]:
                    tokens = word_tokenize(details.lower())
                    for i, token in enumerate(tokens):
                        if token in activity_keywords:
                            context = ' '.join(tokens[max(0, i-2):i+2]).lower()
                            if 'after' in context or 'during' in context:
                                triggers[token] += 1
                logger.debug(f"Triggers for {location}: {dict(triggers)}")
                if not triggers:
                    logger.debug(f"No triggers found for {location}")
                    continue
                
                # Generate pattern for each trigger with format rotation
                for trigger, count in triggers.items():
                    prompt = f"""### Instruction: Generate a concise medical pain pattern description for pain in the {location} triggered by {trigger}. Use the following format (selected randomly or cyclically): {formats[format_index % len(formats)].format(location=location, trigger=trigger)} Output a single grammatically correct sentence that matches this format. Ensure the sentence is clear, professional, and includes both the location and the {trigger}. ### Response:"""
                    inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.model.device)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=80,  
                        do_sample=True,
                        temperature=0.8,   
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    # Decode and clean generated output
                    full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                    logger.debug(f"Raw Phi-2 output for {location}-{trigger}: {full_output}")

                    # Extract only the generated response portion
                    pattern_start = full_output.find("### Response:")
                    if pattern_start != -1:
                        generated = full_output[pattern_start + len("### Response:"):].strip()
                    else:
                        generated = full_output

                    # Clean up unwanted tokens
                    pattern = re.sub(r"[#]+", "", generated).strip().strip("'\"").rstrip(".") + "."

                    # Validate
                    if len(pattern.split()) < 5 or location.lower() not in pattern.lower() or trigger.lower() not in pattern.lower():
                        attern = formats[format_index % len(formats)].format(location=location, trigger=trigger)

                    patterns.append(pattern)

                    logger.debug(f"Generated pattern for {location}-{trigger}: {pattern}")
                    format_index += 1  # Move to next format

            logger.debug(f"Returning {len(result_logs)} logs and {len(patterns)} patterns")
            return Response({"logs": result_logs, "pattern": patterns}, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error in PainAnalysisView: {str(e)}", exc_info=True)
            return Response(
                {"error": f"Failed to analyze patterns: {str(e)}", "logs": [], "pattern": []},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )