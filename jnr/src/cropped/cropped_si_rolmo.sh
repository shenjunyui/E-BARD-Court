#!/bin/bash
#SBATCH -A uBS25_EcoGiu
#SBATCH -p boost_usr_prod
#SBATCH --time 23:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1       # 4 gpus per node out of 4
#SBATCH --mem=123000        # memory per node out of 494000MB (481GB)
#SBATCH --job-name=my_batch_job
#SBATCH --output=cropped_si_rolmo.out


# Set the environment variables for vLLM and Hugging Face caches
export VLLM_CACHE_ROOT="/leonardo_scratch/large/userexternal/ggiudici"
export HF_HOME="/leonardo_scratch/large/userexternal/ggiudici"
export VLLM_TORCH_COMPILE_CACHE="/leonardo_scratch/large/userexternal/ggiudici"

srun python cropped_si_rolmo.py