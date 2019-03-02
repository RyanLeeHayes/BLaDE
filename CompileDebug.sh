#! /bin/bash -l

# rm -r ../build

# OPTLEVEL="-g -O3"

source modules

# export CC="gcc"
# export CXX="g++"

mkdir buildDebug
cd buildDebug
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Debug ../src

make $1 msld
