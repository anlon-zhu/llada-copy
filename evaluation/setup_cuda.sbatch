#!/bin/bash
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=01:00:00
#SBATCH --partition=pvl
#SBATCH --account=pvl
#SBATCH --output=logs/setup.log     
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=END,FAIL

#SBATCH --job-name=check_cuda

# go to your repo root
cd /n/fs/vl/anlon/llada-copy/

# set up conda
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

# ensure your model cache is writable
source scripts/cache_setup.sh

module avail cuda             # list available cuda modules
module load cudatoolkit/12.8

nvidia-smi | head -n 5

if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc not in PATH"
fi

TORCH_CUDA_VER=$(python - <<'EOF'
import torch
print(torch.version.cuda or "none")
EOF
)
echo "torch.version.cuda → $TORCH_CUDA_VER"

CUDA_SUFFIX=${TORCH_CUDA_VER//./}
if [[ -z "$TORCH_CUDA_VER" || "$TORCH_CUDA_VER" == "none" ]]; then
  echo "⚠️  torch.version.cuda is empty; skipping bitsandbytes install"
else
  PKG="bitsandbytes-cuda${CUDA_SUFFIX}"
  echo "Installing $PKG …"
  pip install --upgrade "$PKG"
fi

python - <<'EOF'
import bitsandbytes as bnb
import torch
print("bnb.__version__:", bnb.__version__)
print("bnb.cuda_support:", hasattr(bnb, 'cuda'))
print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
EOF

echo "✅  All done"
