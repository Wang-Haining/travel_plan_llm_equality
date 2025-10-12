#!/bin/sh
#SBATCH --job-name=gpt5mini
##SBATCH --account=group-jasonclark
#SBATCH --partition=unsafe
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

# modules (cpu run; no cuda needed)
module purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0

# ensure api key exists in environment
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"

# activate venv
. .venv/bin/activate

# run (csv output, resumable). adjust num_runs or model as needed.
python -m run \
  --backend openai \
  --model_name gpt-5-mini-2025-08-07 \
  --num_runs 2000 \
  --batch_size 10
