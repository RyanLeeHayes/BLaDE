#! /bin/bash -l

# rm -r ../build

# OPTLEVEL="-g -O3"

source modules

# export CC="gcc"
# export CXX="g++"

mkdir build
cd build
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Release ../src
# cmake -DUNITS=NMPS -DCMAKE_BUILD_TYPE=Release ../src

make $1 lady
