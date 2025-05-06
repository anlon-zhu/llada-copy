# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# conditional generation benchmarks
accelerate launch eval_llada.py --tasks humaneval --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks mbpp --model llada_dist --confirm_run_unsafe_code --model_args model_path='GSAI-ML/LLaDA-8B-Base',gen_length=1024,steps=1024,block_length=1024

