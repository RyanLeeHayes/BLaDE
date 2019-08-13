#! /bin/bash

source ../modules
MSLDEXE=../build/blade

# $MSLDEXE input
export OMP_NUM_THREADS=1
mpirun -np 3 --bind-to none --bynode $MSLDEXE input
