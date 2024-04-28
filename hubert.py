import librosa
import torch
from transformers import  HubertModel

# Load pre-trained model and tokenizer
model_name = "facebook/hubert-large-ls960-ft"
tokenizer = HubertTokenizer.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)

# Load audio file
audio_file = "your_audio_file.wav"
audio, sample_rate = librosa.load(audio_file, sr=None)

# Convert audio to chunks or frames
# You may need to adjust the chunk size and overlap based on your requirements
chunk_size = 1024
overlap = 256
chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size - overlap)]

# Tokenize each chunk and obtain embeddings
embeddings = []
for chunk in chunks:
    # Tokenize audio chunk
    inputs = tokenizer(chunk, return_tensors="pt", padding=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_values=inputs.input_values)

    # Extract embeddings from the last layer
    chunk_embeddings = outputs.last_hidden_state
    embeddings.append(chunk_embeddings)

# Concatenate embeddings from all chunks
final_embeddings = torch.cat(embeddings, dim=1)