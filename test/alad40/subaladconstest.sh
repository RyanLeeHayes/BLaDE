#!/bin/bash

#SBATCH --job-name=aladBLADEtest
#SBATCH -p gpu
#SBATCH -A rhayes1_lab_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --tasks-per-node=1

source /dfs8/rhayes1_lab/cyeh5/BLaDE/modules
export OMP_NUM_THREADS=1

/dfs8/rhayes1_lab/cyeh5/BLaDE/build/blade alad.inp


