from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model.config.forced_decoder_ids = None

print("model loaded")
