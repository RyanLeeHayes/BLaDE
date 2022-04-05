#!/bin/bash
#SBATCH --job-name=profileblade
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -p gpu2080
#SBATCH --gres=gpu:1

export BLADEDIR=/home/rhaye/BLaDE
source $BLADEDIR/modules
export BLADEEXE=$BLADEDIR/build/blade

./bench.sh
