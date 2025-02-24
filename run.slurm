#!/bin/bash
#SBATCH --job-name=shot-categorizer
#SBATCH --nodes=1
# Set the QoS
#SBATCH --qos=normal
# set 48h for job wall time limit
#SBATCH --time=2-00:00:00
# activate the requeue option
#SBATCH --requeue
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH -o /fsx/sayak/logs/%x-%j-train.out

set -xe

export NCCL_ASYNC_ERROR_HANDLING=1

srun --wait=60 --kill-on-bad-exit=1 accelerate launch --config_file=configs/accelerate-configs/ds.yaml accelerate_train.py \
    --config=configs/training-configs/config.yaml

echo "END TIME: $(date)"
