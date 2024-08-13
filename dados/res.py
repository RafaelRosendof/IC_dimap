import os
import argparse

def process(input_file , output_file):
    with open(input_file , 'r') as infile , open(output_file , 'w') as outfile:
        for line in infile:
            pos = line.find('/data')
            if pos != -1:
                outfile.write(line[pos:] + '\n')
            else:
                outfile.write(line + '\n')

    
def main():
    print('Iniciando o processamento dos arquivos...') 
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