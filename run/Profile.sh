#! /bin/bash

source ../modules
MSLDEXE=../build/msld

# export OMP_NUM_THREADS=4
# mpirun -np 1 --bind-to none --bynode nvprof -o output_%q{OMPI_COMM_WORLD_RANK}.profile $MSLDEXE input
nvprof -o output_0.profile $MSLDEXE input
