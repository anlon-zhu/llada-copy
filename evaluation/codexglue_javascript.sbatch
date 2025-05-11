#!/bin/bash
#SBATCH --job-name=eval_codexglue_javascript
#SBATCH --output=logs/%j_eval_codexglue_javascript.log     
#SBATCH --ntasks=1               
#SBATCH --nodes=1    
#SBATCH --gres=gpu:8
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
export TMPDIR=/tmp/$USER/llada_eval_tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR


# Configure accelerate for distributed training
accelerate config default --config_file config.yaml

# Run evaluations
accelerate launch --multi_gpu --num_processes=8 --main_process_port=29505 \
    eval_llada.py \
    --tasks code2text_javascript \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --limit 160 \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=128,steps=128,block_length=128 \
    --output_path ../results/codexglue_javascript.json