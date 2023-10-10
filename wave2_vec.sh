#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --time=0-00:25          #Tempo máximo do job no formato DIAS-HORAS:MINUTOS
#SBATCH --partition=gpu
#SBATCH --exclusive

# Execute conda activate forge
# antes de submeter este script de job

## Parametros iniciais
XDG_RUNTIME_DIR=""
ipnport=$(shuf -i8000-9999 -n1)
ipnip=$(hostname -i)

# informando ao tch-rs que desejo compilar com cuda na versão 11.7
export TORCH_CUDA_VERSION=cu117

python wave2_vec.py
