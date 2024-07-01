# Distributed Data Parallel (DDP)

## Introduction

Welcome to this lesson on Distributed Data Parallel (DDP). In this lesson, we will learn how to use DDP to train a model on multiple nodes with multiple GPUs.

## Environment

You just need an environment with scikit-learn and PyTorch installed. 

## Organization of the files

- `baseline.py`: This is the vanilla script to train a model on a single GPU
- `single_node.py`: This is the script to train a model on a single node with multiple GPUs
- `multi_node.py`: This is the script to train a model on multiple nodes with multiple GPUs
- `train_net.sh`: Bash script to send a SLURM job to train a model with multiple nodes

## How to run

```bash
sbatch train_net.sh 2  # for using 2 nodes
```

