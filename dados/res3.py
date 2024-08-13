import os 
import csv
import argparse

def process(input_file, output_file):
    data = []  # Inicializa a lista para armazenar os dados
    with open(input_file, 'r') as infile:
        lines = infile.readlines()  # Lê todas as linhas do arquivo
        for line in lines:
            parts = line.split()  # Divide a linha em partes usando qualquer espaço em branco como separador
            
            # Garante que há partes suficientes para extrair o caminho e as métricas
            if len(parts) >= 6:
                path = parts[0]  # O caminho é a primeira parte
                quality = parts[2]  # A qualidade é a terceira parte
                inteligibility = parts[5]  # A inteligibilidade é a sexta parte

                data.append([path, quality, inteligibility])  # Adiciona os dados à lista
            else:
                print(f"Formato inesperado na linha: {line}")

    # Escreve os dados no arquivo CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Path', 'Quality', 'Inteligibility'])  # Cabeçalhos
        writer.writerows(data)
            
def main():
    print("Iniciando o processamento dos arquivos...\n")

    parser = argparse.ArgumentParser(description='Processa arquivos de entrada e remove o caminho do arquivo')
    parser.add_argument("--i" , type=str , help="Caminho do arquivo de entrada")
    parser.add_argument("--o" , type=str , help="Caminho do arquivo de saída")

    args = parser.parse_args()

    if not os.path.exists(args.i):
        print('Arquivo de entrada não encontrado')
        raise FileNotFoundError('Arquivo de entrada não encontrado')

    process(args.i , args.o)

if __name__ == "__main__":
    main()