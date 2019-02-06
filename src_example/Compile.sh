#! /bin/bash -l

# rm -r ../build

# OPTLEVEL="-g -O3"

module load gcc/4.9.4 cuda/7.0 openmpi/2.0.2-gcc-4.9.4
module load cmake/3.2.2

export CC="gcc"
export CXX="g++"

mkdir ../build
cd ../build
cmake ../src

make $1 md
