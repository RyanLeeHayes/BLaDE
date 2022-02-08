#! /bin/bash

source ../modules
MSLDEXE=../build/blade

rm output_?.nsight
export OMP_NUM_THREADS=1
# mpirun -np 1 --bind-to none --bynode nvprof -o output_%q{OMPI_COMM_WORLD_RANK}.nvvp $MSLDEXE input
# nvprof -o output_0.nvvp $MSLDEXE input
# ncu -o output_0.nsight $MSLDEXE input
nsys profile -o output_0.nsight $MSLDEXE input
