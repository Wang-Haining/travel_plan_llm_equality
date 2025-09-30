#!/bin/sh
#SBATCH --job-name=gpt5-mini
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
# cuda not required
# module load CUDA/12.2.0
export HF_HOME=$HOME/models

# bring in OPENAI_API_KEY from your shell init if needed
[ -f "$HOME/.bashrc" ] && . "$HOME/.bashrc"

. .venv/bin/activate
python -m run --num_runs 2000 --model_name 'gpt-5-mini-2025-08-07' --batch_size 10
