#!/bin/sh
#SBATCH --job-name=llama3_70b
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0 CUDA/12.3.0
export HF_HOME=$HOME/models

mkdir -p logs
. .venv/bin/activate
python -m run --num_runs 2000 --model_name 'meta-llama/Meta-Llama-3-70B-Instruct' --batch_size 1
