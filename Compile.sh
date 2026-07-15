#! /bin/bash -l

source modules

mkdir build
cd build
# useful flags:
# -DPRECISION=DOUBLE # Compile double precision
# -DRX=ON # Compile with replica exchange
# -DUNITS=AKMA # Compile with AKMA units (GROMACS is the alternative)
# -DPROFILE=ON # Compile with serial execution of all kernels for simpler profiling
# -DCMAKE_CUDA_ARCHITECTURES=70 # choose appropriately for your cuda hardware and compilers, but don't use below 60 unless you're willing to take a performance hit. Cuda 13 requires at least 75
# -DCMAKE_BUILD_TYPE=Release || -DCMAKE_BUILD_TYPE=Debug
cmake -DCMAKE_CUDA_ARCHITECTURES=70 -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Release ../src

# -j8 # compile with 8 threads
# VERBOSE=1 # Show all the compilation commands
make $1 blade
