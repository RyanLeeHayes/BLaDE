#! /bin/bash -l

source modules

mkdir build
cd build
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Release ../src
# cmake -DUNITS=AKMA -DPRECISION=DOUBLE -DCMAKE_BUILD_TYPE=Release ../src

make $1 blade
