#!/bin/bash
#SBATCH -A uBS26_EcoGiu
#SBATCH -p boost_usr_prod
#SBATCH --time 02:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1       # 4 gpus per node out of 4
#SBATCH --mem=123000        # memory per node out of 494000MB (481GB)
#SBATCH --job-name=my_batch_job
#SBATCH --output=full_mi_qwen3bft.out

srun python full_mi_qwen3bft.py