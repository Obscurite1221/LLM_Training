#!/bin/bash
#SBATCH --job-name=Llama_training
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks-per-node=1     # Number of processes per node
#SBATCH --gres=gpu:7            # GPUs per node
#SBATCH --cpus-per-task=12      # CPU cores per task
#SBATCH --time=12:00:00         # Max runtime
#SBATCH --output=logs/deepspeed_%j.out  # Stdout log file
#SBATCH --error=logs/deepspeed_%j.err   # Stderr log file

export NCCL_P2P_LEVEL=NVL
export GPUS_PER_NODE=7
export MASTER_ADDR=localhost
export MASTER_PORT=29993
export WORLD_SIZE=7
export RANK=0
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

# Run DeepSpeed
#sbatch deepspeed 'Master Control Program.py' \ 
srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT "Master Control Program.py" --deepspeed ds_config.json'
#torchrun --nproc_per_node=7 --nnodes=1 --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT 'Master Control Program.py'