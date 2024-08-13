import os 
import csv
import argparse
#sklearn e outras bibliotecas com as métricas

def leitura(file1 , file2):
    with open(file1 , 'r') as f1 , open(file2 , 'r') as f2:
        read1 = csv.reader(f1)
        read2 = csv.reader(f2)

        #Aqui eu devo ler os arquivos e comparar os valores usando 
        #MÉTRICAS: MSE , LCC , SRCC , KTAU