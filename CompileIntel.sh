#! /bin/bash -l

# source modules
module load environments/intel_2018

mkdir buildIntel
cd buildIntel
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Release ../src
# cmake -DUNITS=AKMA -DPRECISION=DOUBLE -DCMAKE_BUILD_TYPE=Release ../src

make $1 blade
