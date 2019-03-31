#! /bin/bash

source ../modules
module load gdb/8.2_gcc_6.4.0
MSLDEXE=../buildDebug/lady

export OMP_NUM_THREADS=1
mpirun -np 1 --bind-to none $MSLDEXE input_debug
echo "cuda-gdb $MSLDEXE <PID>"
