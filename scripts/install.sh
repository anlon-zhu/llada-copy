#!/usr/bin/env bash
set -euo pipefail

### 1) Load Anaconda module
module load anaconda3/2024

### 2) Define your env prefix & name
ENV_PREFIX=/n/fs/vl/anlon/envs/llada
ENV_NAME=llada

### 3) Create the env if it doesn’t exist
if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "Creating Conda env at $ENV_PREFIX …"
  conda create -p "$ENV_PREFIX" python=3.10 pip -y
else
  echo "Conda env already exists at $ENV_PREFIX, skipping creation."
fi

### 4) Make sure Conda sees your shared‐envs dir & hook into it
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

### 5) Relocate pip’s cache + build dirs into your big‐space area
# (so pip never tries to scribble in your full $HOME)
PIP_CACHE_DIR="$ENV_PREFIX/pip_cache"
TMPDIR="$ENV_PREFIX/pip_build"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export PIP_CACHE_DIR
export TMPDIR

### 6) Upgrade pip & install everything, skipping the wheel‐cache
pip install --upgrade pip

if [[ ! -d "bigcode-evaluation-harness" ]]; then
  git clone https://github.com/bigcode-project/bigcode-evaluation-harness.git
fi
pip install -e bigcode-evaluation-harness

pip install --no-cache-dir \
  transformers==4.49.0 \
  torch \ 
  accelerate==0.34.2 \
  bitsandbytes \
  bitsandbytes-cuda128 \
  gradio \
  "huggingface_hub>=0.15.1,<0.29" \
  "peft" \
  antlr4-python3-runtime==4.11 \
  math_verify \
  sympy \
  hf_xet

pip install -e .

echo "✅ llada is ready"
