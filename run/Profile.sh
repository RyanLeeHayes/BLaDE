#! /bin/bash

source ../modules
LADYEXE=../build/lady

rm output_?.profile
export OMP_NUM_THREADS=1
mpirun -np 2 --bind-to none --bynode nvprof -o output_%q{OMPI_COMM_WORLD_RANK}.profile $LADYEXE input
# nvprof -o output_0.profile $MSLDEXE input
