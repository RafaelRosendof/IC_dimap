
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Load pre-trained ASR model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Load audio file
audio_file = "/home/rafael/audio/teste.wav"
waveform, sample_rate = torchaudio.load(audio_file)



# Perform inference
inputs = tokenizer(waveform, return_tensors="pt", padding=True)

input_data = inputs.input_values.squeeze(0)
with torch.no_grad():
    logits = model(input_values=input_data).logits

# Decode token IDs to text
transcription = tokenizer.batch_decode(torch.argmax(logits, dim=-1))

# Tokenize the transcription if needed
print(transcription)