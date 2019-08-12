#! /bin/bash -l

source modules

mkdir buildDebug
cd buildDebug
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Debug ../src
# cmake -DUNITS=AKMA -DPRECISION=DOUBLE -DCMAKE_BUILD_TYPE=Debug ../src

make $1 blade
