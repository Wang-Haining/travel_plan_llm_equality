#!/bin/sh
#SBATCH --job-name=llama3.1_8b
##SBATCH --account=group-jasonclark
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0 CUDA/12.3.0
export HF_HOME=$HOME/models

. .venv/bin/activate
python -m run --num_runs 200 --model_name 'meta-llama/Meta-Llama-3.1-8B-Instruct' --batch_size 10