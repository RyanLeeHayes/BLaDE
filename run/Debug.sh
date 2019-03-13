#! /bin/bash

source ../modules
module load gdb/8.2_gcc_6.4.0
MSLDEXE=../buildDebug/lady

echo $MSLDEXE input
# gdb $MSLDEXE
# cuda-gdb $MSLDEXE
mpirun -np 2 --bind-to none cuda-gdb $MSLDEXE
