#! /bin/bash


# Set slurm options for cluster
export SLURMOPTSMD="--time=1440 --ntasks=1 --tasks-per-node=1 --cpus-per-task=1 -A rhayes1_lab_gpu -p gpu  --gres=gpu:1 --export=ALL"
export SLURMOPTSPP="--time=1440 --ntasks=1 --tasks-per-node=1 --cpus-per-task=1 -p free-gpu --gres=gpu:1 --export=ALL"
export CHARMMDIR=/dfs8/rhayes1_lab/bin/CHARMM_EXE
source $CHARMMDIR/modules
export CHARMMEXEC=$CHARMMDIR/gnu/charmm 
source /dfs8/rhayes1_lab/cyeh5/ALF_NBias/env-alf/bin/activate
export CRITERIA=3.2e-4
# Run flattening
#DEPEND=""
PID=`sbatch --parsable $SLURMOPTSMD $DEPEND ./runflat.sh`

