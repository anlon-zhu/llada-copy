#!/bin/bash
#SBATCH --job-name=eval_mbpp_instruct
#SBATCH --output=logs/%j_eval_mbpp_instruct.log     
#SBATCH --ntasks=1               
#SBATCH --nodes=1      
#SBATCH --partition=pvl
#SBATCH --account=pvl       
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
export TMPDIR=/tmp/$USER/llada_eval_tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p $TMPDIR


# Configure accelerate for distributed training
accelerate config default --config_file config.yaml

# Run evaluations
accelerate launch --multi_gpu --num_processes=4 --main_process_port=29503 \
    eval_llada.py \
    --tasks mbpp \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=1024,steps=1024,block_length=1024 \
    --output_path ../results/mbpp_instruct.json