import os
import csv
import argparse
#precisa do pandas?
def extrair(input_file, output_file, filter_condition):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Escreve o cabeçalho no arquivo de saída
        headers = next(reader)
        writer.writerow(headers)
        
        for row in reader:
            if filter_condition in row[0]:  # Verifica se a condição está na coluna correta (no exemplo, coluna 4)
                writer.writerow(row)


            

def main():
    parser = argparse.ArgumentParser(description='Extrai dados de um arquivo CSV')
    parser.add_argument("--i" , type=str , help="Arquivo de entrada")
    parser.add_argument("--o" , type=str , help="Arquivo de saida")

    parser.add_argument("--filter", type=str, help="Condição para filtrar as linhas", required=True)

    args = parser.parse_args()

    if not os.path.exists(args.i):
        print("Arquivo de entrada não existe")
        exit(1)
        
    extrair(args.i, args.o, args.filter)


if __name__ == '__main__':
    main()
    