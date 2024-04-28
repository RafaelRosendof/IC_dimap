import speech_recognition as sr
from transformers import BertTokenizer

# Load audio file
audio_file = "/home/rafael/audio/figas.wav"

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Load audio file
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)

# Convert speech to text
transcription = recognizer.recognize_google(audio_data)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize transcribed text
tokens = tokenizer.tokenize(transcription)

print(tokens)
