#!/bin/bash
#SBATCH --job-name=DDPTraining       # Name of the job
#SBATCH --nodes=$1                   # Number of nodes specified by the user
#SBATCH --ntasks-per-node=1          # Ensure only one task runs per node
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --gres=gpu:1                 # Number of GPUs per node
#SBATCH --time=01:00:00              # Time limit hrs:min:sec (1 hour in this example)
#SBATCH --mem=4GB                    # Memory limit per GPU
#SBATCH --output=training_%j.log     # Output and error log name (%j expands to jobId)
#SBATCH --partition=gpu              # Specify the partition or queue

srun python3 multi_node.py --total_epochs 10 --save_every 2 --batch_size 32 --node_rank $SLURM_NODEID
