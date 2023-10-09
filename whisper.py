import whisper

model = whisper.load_model("base")

result = model.transcribe('/home/rafaelrosendo/IC_testes/test3.wav')

result["text"]

caminho_arquivo_txt = '/home/rafaelrosendo/IC_testes/results.txt'

with open(caminho_arquivo_txt, 'w', encoding='utf-8') as arquivo:
    arquivo.write(texto)

print(f"Texto salvo em '{caminho_arquivo_txt}'")