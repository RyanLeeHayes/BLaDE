#! /bin/bash

#SBATCH -p free
#SBATCH --ntasks=1
source /dfs8/rhayes1_lab/bin/CHARMM_EXE/modules
mpirun -np $SLURM_NTASKS /dfs8/rhayes1_lab/bin/CHARMM_EXE/gnu/charmm -i build_chain_2.inp -o build_chain_2.out

