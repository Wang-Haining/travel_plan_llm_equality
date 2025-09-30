#!/bin/sh
#SBATCH --job-name=gemma2_27b
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
export HF_HOME=$HOME/models

. .venv/bin/activate
python -m run --num_runs 2000 --model_name 'google/gemma-2-27b-it' --batch_size 10