#!/bin/bash -l
#SBATCH --job-name=vllm-oss120b
#SBATCH --partition=nextgen-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ---- environment (tempest) ----
module purge
module unload Python CUDA 2>/dev/null || true
module load OpenSSL/3 Python/3.12.3-GCCcore-13.3.0 CUDA/12.3.0

mkdir -p logs
export PYTHONUNBUFFERED=1

# activate venv if present
[[ -f ".venv/bin/activate" ]] && source .venv/bin/activate

# ---- model/server knobs ----
MODEL="${MODEL:-openai/gpt-oss-120b}"
TP="${TP:-2}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAXLEN="${MAXLEN:-10240}"
MAX_SEQS="${MAX_SEQS:-64}"

# ---- container & cache ----
HF_HOME="${HF_HOME:-$HOME/hf-cache}"
IMG_FILE="${IMG_FILE:-$HOME/containers/vllm-gptoss.sif}"
CUDA_BIND="/mnt/shared/moduleapps/eb/CUDA/12.3.0"
mkdir -p "$HF_HOME"

# ---- api endpoint ----
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8020}"

# honor file-based HF token if env missing
export HF_TOKEN="${HF_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || echo)}"

# ---- port helpers ----
port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -E "[:]$1\$" -q
  else
    netstat -ltn 2>/dev/null | grep -E "[: ]$1 " -q
  fi
}
choose_free_port() {
  p="${1:-8020}"
  for _ in $(seq 0 100); do
    port_in_use "$p" || { echo "$p"; return; }
    p=$((p+1))
  done
  echo "${1:-8020}"
}

PORT="$(choose_free_port "$PORT")"
API="http://${HOST}:${PORT}"

echo "[server] launching vLLM ${MODEL} → ${API} (tp=${TP}, maxlen=${MAXLEN}, max_seqs=${MAX_SEQS})"

command -v apptainer >/dev/null || { echo "[fatal] apptainer missing"; exit 90; }
[[ -f "$IMG_FILE" ]] || { echo "[fatal] image missing: $IMG_FILE"; exit 91; }

# ---- start vLLM server (OpenAI-compatible) ----
apptainer exec --nv --cleanenv \
  --bind "$HF_HOME":/root/.cache/huggingface \
  $( [[ -d "$CUDA_BIND" ]] && echo --bind "$CUDA_BIND":"$CUDA_BIND" ) \
  --env PYTHONNOUSERSITE=1,PIP_USER=no,TRANSFORMERS_NO_ADVISORY_WARNINGS=1,HF_TOKEN="${HF_TOKEN}",TORCH_COMPILE_DISABLE=1,PYTORCH_JIT=0 \
  "$IMG_FILE" bash --noprofile --norc -c "
set -euo pipefail
python3 -m pip install --no-cache-dir --force-reinstall 'tokenizers>=0.21,<0.22' >/dev/null 2>&1 || true
python3 -m vllm.entrypoints.openai.api_server \
  --model '${MODEL}' \
  --served-model-name 'openai/gpt-oss-120b' \
  --download-dir /root/.cache/huggingface \
  --tensor-parallel-size ${TP} \
  --gpu-memory-utilization ${GPU_UTIL} \
  --max-model-len ${MAXLEN} \
  --max-num-seqs ${MAX_SEQS} \
  --enable-prefix-caching \
  --enforce-eager \
  --host 0.0.0.0 \
  --port ${PORT}
" \
  >> "logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID:-$$}.server.out" \
  2>> "logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID:-$$}.server.err" &

SERVER_PID=$!

# ---- readiness probe ----
echo "[server] waiting for ${API}/v1/models ..."
MAX_ATTEMPTS=600
for attempt in $(seq 1 $MAX_ATTEMPTS); do
  if curl -fsS "${API}/v1/models" >/dev/null 2>&1; then
    echo "[server] ready (attempt ${attempt})"
    break
  fi
  if [[ "$attempt" -eq $MAX_ATTEMPTS ]]; then
    echo "[fatal] server failed to start after $((MAX_ATTEMPTS * 2)) seconds"
    tail -n 50 "logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID:-$$}.server.err" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    exit 92
  fi
  if [[ $((attempt % 15)) -eq 0 ]]; then
    echo "[server] still waiting... (attempt ${attempt}/${MAX_ATTEMPTS})"
  fi
  sleep 2
done

# ---- run travel fairness experiment (client) ----
export VLLM_BASE_URL="${API}/v1"

echo "[client] running travel_fairness.py with 200 generations on ${MODEL}"
python3 run.py \
  --model_name "${MODEL}" \
  --num_runs 200 \
  --batch_size 64 \
  --vllm_base_url "${VLLM_BASE_URL}" \
  --concurrency 64 \
  --request_timeout 120 \
  >> "logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID:-$$}.client.out" \
  2>> "logs/${SLURM_JOB_NAME}.${SLURM_JOB_ID:-$$}.client.err" \
  || CLIENT_RC=$?

CLIENT_RC="${CLIENT_RC:-0}"

# ---- graceful shutdown of server ----
if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "[cleanup] terminating vLLM server (PID ${SERVER_PID})..."
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for i in {1..10}; do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[cleanup] force killing vLLM server..."
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
fi

echo "[server] exited. client rc=${CLIENT_RC}"
exit "${CLIENT_RC}"
