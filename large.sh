#!/bin/bash
#SBATCH --job-name=LARGE-V3
#SBATCH --time=1-23:59          
#SBATCH --partition=gpu-8-v100
#SBATCH --gres=gpu:8

export TORCH_CUDA_VERSION=cu117
torchrun --nproc_per_node=4 CNN.py