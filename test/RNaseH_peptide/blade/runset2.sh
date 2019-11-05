#!/bin/bash

source charmm.sh

export OMP_NUM_THREADS=1
$CHARMMEXEC msld_flat.inp > output 2> error
