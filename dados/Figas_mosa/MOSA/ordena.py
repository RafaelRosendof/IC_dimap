import csv
import os
import argparse

#estou com dúvida como fazer a ordenação, mas acho que é algo parecido com isso
def ordenar(input_file , output_file , column):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        headers = next(reader)
        writer.writerow(headers)
        
        # Ordena as linhas pela coluna especificada
        sorted_rows = sorted(reader, key=lambda x: x[column])
        
        # Escreve as linhas ordenadas no arquivo de saída
        for row in sorted_rows:
            writer.writerow(row)

            
def main():
    print("Ordenando arquivo CSV")

    parser = argparse.ArgumentParser(description='Ordena um arquivo CSV')
    parser.add_argument("--i" , type=str , help="Arquivo de entrada")
    parser.add_argument("--o" , type=str , help="Arquivo de saida")
    parser.add_argument("--column" , type=int , help="Coluna para ordenar", required=True)

    args = parser.parse_args()

    ordenar(args.i , args.o , args.column)


if __name__ == '__main__':
    main()
