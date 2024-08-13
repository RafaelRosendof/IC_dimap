import spaces 
import os
import csv
import torch
from transformers import AutoFeatureExtractor, WhisperModel, AutoModelForSpeechSeq2Seq
import numpy as np
import torchaudio
import librosa
import argparse

from modules import load_audio, MosPredictor, denorm

mos_ckp = "ckpt_mosa_net_plus"

print("Carregando os checkpoints... MOSA NET+")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MosPredictor().to(device)
model.eval() 
model.load_state_dict(torch.load(mos_ckp, map_location=device))

#Carregando o checkpoint do Whisper
print("Carregando o checkpoint do Whisper...")

extrator_feature = AutoFeatureExtractor.from_pretrained("openai/whisper-large-v3") #precisa do larger mesmo?
model_2 = WhisperModel.from_pretrained("openai/whisper-large-v3")
model_2 = model_2.to(device)

@spaces.GPU
def predito(wavefile:str):
    model.to(device)
    if device != model_2.device:
        model_2.to(device)

    print("Iniciando a predição...")

    wav = torchaudio.load(wavefile)[0]
    lps = torch.from_numpy(np.expand_dims(np.abs(librosa.stft(wav[0].detach().numpy(), n_fft = 512, hop_length=256, win_length=512)).T, axis=0))
    lps = lps.unsqueeze(1)

    # Whisper Feature tirando agora 
    audio = load_audio(wavefile)
    inputs = extrator_feature(audio, return_tensors="pt")
    input_features = inputs.input_features
    input_features = input_features.to(device)
    
    #Aqui é a inferência de fato
    with torch.no_grad():
        decoder_input_ids = torch.tensor([[1, 1]]) * model_2.config.decoder_start_token_id
        decoder_input_ids = decoder_input_ids.to(device)
        #ultima camada oculta
        last_hidden = model_2(input_features, decoder_input_ids=decoder_input_ids).encoder_last_hidden_state
        whisper_feat = last_hidden

    print("Model features shapes...")
    print(whisper_feat.shape)
    print(wav.shape)
    print(lps.shape)
    
    #predição
    wav = wav.to(device)
    lps = lps.to(device) 
    qualidade1 , intel1 , frame1 , frame2 = model(wav , lps , whisper_feat)
    qualidade2 = qualidade1.cpu().detach().numpy()[0]
    intel2 = intel1.cpu().detach().numpy()[0]
    

    #print("PREDIÇÕES")
#    qa_text = f" Audio: {wavefile}    Quality: {denorm(qualidade2)[0]:.2f}  Inteligibility: {intel2[0]:.2f}" 
#    print(qa_text) 
#    return qa_text

    print("PREDIÇÕES")
    qa_text = f"Quality: {denorm(qualidade2)[0]:.2f}, Inteligibility: {intel2[0]:.2f}" 
    print(qa_text) 
    return qa_text
'''  
def processaPasta(pasta:str , output_file: str):
    with open(output_file , "w") as f:
        for root , dirs , files in os.walk(pasta):
            for file in files:
                if file.endswith(".wav"):
                    caminho = os.path.join(root , file)
                    res = predito(caminho)
                    f.write(f"{file} : {res}\n")
 '''
 

def processaPasta(pasta: str, output_file: str):
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["File", "Quality", "Inteligibility"])  # Cabeçalho do CSV

        for root, dirs, files in os.walk(pasta):
            for file in files:
                if file.endswith(".wav"):
                    caminho = os.path.join(root, file)
                    relative_path = os.path.relpath(caminho, pasta)  # Caminho relativo
                    res = predito(caminho)
                    # Dividindo as predições em colunas para o CSV
                    qualidade, inteligibilidade = res.split(", ")
                    writer.writerow([relative_path, qualidade.split(": ")[1], inteligibilidade.split(": ")[1]])

   
def main():
    parser = argparse.ArgumentParser(description="Predição de qualidade e inteligibilidade de audio")
    parser.add_argument("--folder", type=str, help="Caminho da pasta com os arquivos de audio")
    parser.add_argument("--out", type=str, help="Caminho do arquivo de saída")
    args = parser.parse_args()

    if not os.path.exists(args.folder):
        raise FileNotFoundError(f"pasta {args.folder} não encontrado")

    processaPasta(args.folder , args.out) 

if __name__ == "__main__":
    main()