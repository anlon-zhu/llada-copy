#!/bin/bash

# Set required environment variables
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export LOGLEVEL=INFO

cd /n/fs/vl/anlon/llada-copy/evaluation

# Run with minimal settings for testing
accelerate launch \
    eval_llada.py \
    --tasks humaneval \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=32,steps=8,block_length=32,num_samples=1 \
    --num_fewshot=0 \
    --limit=2
