

# Create your views here.

import os
from django.http import JsonResponse
from django.shortcuts import render
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from openai import model,processor

def transcribe(request):
    transcription = ""

    if request.method == 'POST':
        # Check if the audio file is present in the request
        if 'audioFile' not in request.FILES:
            return JsonResponse({'error': 'No audio file found'})
        
        # Access the audio file from the request
        audio_file = request.FILES['audioFile']

        # Read audio file
        input_audio, sampling_rate = torchaudio.load(audio_file)

        # Load model and processor
        

        # Convert audio to features using the Whisper processor
        input_features = processor(input_audio[0], sampling_rate=sampling_rate, return_tensors="pt").input_features

        # Generate token ids
        predicted_ids = model.generate(input_features)

        # Decode token ids to text
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        transcription = str(transcription[0])


    return render(request, 'transcribe.html', {'transcription': transcription})
