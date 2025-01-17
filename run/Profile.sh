#! /bin/bash

source ../modules
MSLDEXE=../build/blade

rm output_?.nvvp output_?.nsight
export OMP_NUM_THREADS=1

# Old nvvp profiler
# mpirun -np 1 --bind-to none --bynode nvprof -o output_%q{OMPI_COMM_WORLD_RANK}.nvvp $MSLDEXE input
# nvprof -o output_0.nvvp $MSLDEXE input
# view profiles with nvvp

# New nsys profiler
# ncu -o output_0.nsight $MSLDEXE input
nsys profile -o output_0.nsight $MSLDEXE input
# use nsys-ui to view profiles
