#! /bin/bash -l

# rm -r ../build

# OPTLEVEL="-g -O3"

source modules

# export CC="gcc"
# export CXX="g++"

mkdir build
cd build
cmake ../src

make $1 msld
