#!/bin/bash -l
#SBATCH --job-name=oss_20b
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

set -euo pipefail

# --- env/modules ---
module purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0 CUDA/12.3.0
source "$HOME/.bashrc" 2>/dev/null || true

# --- paths and cache ---
mkdir -p logs
export HF_HOME="${HF_HOME:-$HOME/models}"                     # host hf cache
export IMG_FILE="${IMG_FILE:-$HOME/containers/vllm-gptoss.sif}"  # apptainer image with vLLM

# honor file-based HF token if env missing
export HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || true)}"

# --- python env ---
. .venv/bin/activate

# --- vLLM knobs ---
MODEL="${MODEL:-openai/gpt-oss-20b}"   # switch to openai/gpt-oss-20b if needed
HOST="127.0.0.1"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-10240}"
MAX_SEQS="${MAX_SEQS:-64}"
GPU_UTIL="${GPU_UTIL:-0.90}"

# --- choose a free port starting at $PORT ---
port_in_use() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:]$1\$" -q || false; }
choose_free_port() {
  local p="${1:-8000}"
  for _ in $(seq 0 100); do port_in_use "$p" || { echo "$p"; return; }; p=$((p+1)); done
  echo "${1:-8000}"
}
PORT="$(choose_free_port "$PORT")"
API="http://${HOST}:${PORT}"

command -v apptainer >/dev/null || { echo "[fatal] apptainer not found"; exit 90; }
[[ -f "$IMG_FILE" ]] || { echo "[fatal] image missing: $IMG_FILE"; exit 91; }

echo "[server] launching vLLM in apptainer for ${MODEL} at ${API}"

# start vLLM server inside apptainer container (OpenAI-compatible)
apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  --env HF_TOKEN="${HF_TOKEN}",PYTHONNOUSERSITE=1,PIP_USER=no,TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  "$IMG_FILE" bash -lc "
set -euo pipefail
python3 -m vllm.entrypoints.openai.api_server \
  --model '${MODEL}' \
  --served-model-name '${MODEL}' \
  --download-dir /root/.cache/huggingface \
  --host 0.0.0.0 --port ${PORT} \
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAX_SEQS} \
  --gpu-memory-utilization ${GPU_UTIL} \
  --enable-prefix-caching
" & SERVER_PID=$!

cleanup() {
  echo "[cleanup] stopping vLLM (pid ${SERVER_PID})"
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT

# wait until server is ready
echo "[server] waiting for ${API}/v1/models ..."
ATTEMPTS=600
for i in $(seq 1 $ATTEMPTS); do
  if curl -fsS "${API}/v1/models" >/dev/null 2>&1; then
    echo "[server] ready"
    break
  fi
  if [[ $i -eq $ATTEMPTS ]]; then
    echo "[fatal] vLLM failed to start"
    exit 92
  fi
  sleep 2
done

# tell the client where to find the OpenAI-compatible endpoint
export VLLM_BASE_URL="${API}/v1"

# --- run your study script against the OSS model served by vLLM ---
python -m run --num_runs 2000 --model_name 'openai/gpt-oss-20b' --batch_size 10

echo "[done]"
