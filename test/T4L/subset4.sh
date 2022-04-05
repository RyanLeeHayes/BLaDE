#! /bin/bash

module load slurm
export nnodes=`cat nnodes`
export nreps=`cat nreps`
export nitt=1

# DEPEND="--dependency=afterok:"
# NICE="--nice=500"

for p in a
do

export ini=112
export i=${ini}$p
sbatch -A rhayes_gpu --time=720 --ntasks=$nreps --tasks-per-node=1 --cpus-per-task=$nnodes -p a100 --gres=gpu:$nnodes --export=ALL --array=1-1%1 $DEPEND $NICE ./runset4.sh

done
