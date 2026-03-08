#!/bin/bash
#SBATCH -A uBS25_EcoGiu
#SBATCH -p boost_usr_prod
#SBATCH --time 23:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1       # 4 gpus per node out of 4
#SBATCH --mem=123000        # memory per node out of 494000MB (481GB)
#SBATCH --job-name=my_batch_job
#SBATCH --output=train_rfdetr.out

# Set environment variables for a single-process distributed setup
export MASTER_ADDR=localhost  # Or use a placeholder like 127.0.0.1
export MASTER_PORT=29500      # A common default port, choose an unused one
export RANK=0                 # For a single process, rank is always 0
export WORLD_SIZE=1           # For a single process, world size is always 1

# If your script requires the local rank for per-GPU configuration (common for multi-GPU)
export LOCAL_RANK=0           # For a single GPU, local rank is 0

srun python train_rfdetr.py