import csv
import os 
import argparse

#preciso concatenar essas 3 tabelas em uma só
def concat(file1 , file2 , file3 , output):
    with(open(file1 , 'r') , open(file2 , 'r') , open(file3 , 'r') , open(output , 'w' , newline='')) as (f1 , f2 , f3 , out):
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)
        reader3 = csv.reader(f3)
        writer = csv.writer(out)
        
        # Escreve o cabeçalho no arquivo de saída
        headers = next(reader1)
        writer.writerow(headers)
        
        for row in reader1:
            writer.writerow(row)
        
        for row in reader2:
            writer.writerow(row)
        
        for row in reader3:
            writer.writerow(row)