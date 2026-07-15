#! /bin/bash -l

source modules

mkdir build
cd build
# useful flags:
# -DPRECISION=DOUBLE # Compile double precision
# -DRX=ON # Compile with replica exchange
# -DUNITS=AKMA # Compile with AKMA units (GROMACS is the alternative)
# -DPROFILE=ON # Compile with serial execution of all kernels for simpler profiling
# -DCMAKE_CUDA_ARCHITECTURES=70 # Should compile for sm_70 devices, but it's broken
# -DCMAKE_BUILD_TYPE=Release || -DCMAKE_BUILD_TYPE=Debug
cmake -DUNITS=AKMA -DCMAKE_BUILD_TYPE=Release ../src

# -j8 # compile with 8 threads
# VERBOSE=1 # Show all the compilation commands
make $1 blade
