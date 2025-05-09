#!/bin/bash
#SBATCH --job-name=Llama_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=logs/deepspeed_%j.out
#SBATCH --error=logs/deepspeed_%j.err

# Ensure environment is correct
# Note that the GPU per node count and world size may need to be tweaked based on the server.
export NCCL_P2P_LEVEL=NVL
export GPUS_PER_NODE=7
export MASTER_ADDR=localhost
export MASTER_PORT=29993
export WORLD_SIZE=7
export RANK=0

# Explicitly mask GPUs BEFORE torch launches
#export CUDA_VISIBLE_DEVICES=1,3,4,5,6,7

# Run DeepSpeed through torchrun with visible GPU mask
srun python -m torch.distributed.run \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_PROCID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  "Master Control Program.py" --deepspeed ds_config.json