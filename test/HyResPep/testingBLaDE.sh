#! /bin/bash

#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu
#SBATCH -A rhayes1_lab_gpu
#SBATCH --gres=gpu:1
source ../../modules
BLADEEXE=../../buildDD/blade

export OMP_NUM_THREADS=1
mpirun -np $SLURM_NTASKS $BLADEEXE comparenergy.inp
