#!/bin/bash

source charmm.sh

mpirun -np 1 -x OMP_NUM_THREADS=4 --bind-to none --bynode $CHARMMEXEC -i msld_flat.inp > output 2> error
