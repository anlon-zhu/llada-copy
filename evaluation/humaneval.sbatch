#!/bin/bash
#SBATCH --job-name=eval_humaneval
#SBATCH --output=logs/%j_eval_humaneval.log     
#SBATCH --ntasks=1               
#SBATCH --nodes=1          
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=8 
#SBATCH --mem=64G                
#SBATCH --time=24:00:00    
#SBATCH --mail-user=az4244@princeton.edu
#SBATCH --mail-type=END,FAIL

# Load necessary modules (adjust based on your cluster setup)
export CONDA_ENVS_PATH=/n/fs/vl/anlon/envs
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh
conda activate llada

cd /n/fs/vl/anlon/llada-copy/evaluation
source ../scripts/cache_setup.sh

# Set required environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Configure accelerate for distributed training
accelerate config default --config_file config.yaml

# Run evaluations
accelerate launch --multi_gpu --num_processes=4 --main_process_port 0 \
    eval_llada.py \
    --tasks humaneval \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=256,steps=256,block_length=256

# accelerate launch --multi_gpu --num_processes=4 eval_llada.py \
#     --tasks mbpp \
#     --model llada_dist \
#     --confirm_run_unsafe_code \
#     --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024
