#! /bin/bash

source ../modules
LADYEXE=../build/lady

# $LADYEXE input
export OMP_NUM_THREADS=1
mpirun -np 2 --bind-to none --bynode $LADYEXE input
