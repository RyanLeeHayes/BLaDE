#! /bin/bash

source ../modules
MSLDEXE=../build/blade

rm output_?.nvvp
export OMP_NUM_THREADS=1
# mpirun -np 1 --bind-to none --bynode nvprof -o output_%q{OMPI_COMM_WORLD_RANK}.nvvp $MSLDEXE input
nvprof -o output_0.nvvp $MSLDEXE input
