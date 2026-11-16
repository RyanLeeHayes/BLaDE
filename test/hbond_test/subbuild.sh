#! /bin/bash

#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH--cpus-per-task=1
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -A rhayes1_lab_gpu

source /dfs8/rhayes1_lab/bin/CHARMM_EXE/modules

mpirun -np 1 /dfs8/rhayes1_lab/bin/CHARMM_EXE/gnu/charmm -i build_hbtest.inp


