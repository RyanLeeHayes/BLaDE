#! /bin/bash

source ../modules
module load gdb/8.2_gcc_6.4.0
MSLDEXE=../buildDebug/blade

echo "arrest $1" > input_debug
echo "stream input" >> input_debug

echo "cuda-gdb $MSLDEXE <PID>"
export OMP_NUM_THREADS=1
# mpirun -np 1 --bind-to none $MSLDEXE input_debug
$MSLDEXE input_debug
