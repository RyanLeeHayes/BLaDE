#! /bin/bash -l

source modules

## The LibTorch must be >= 2.10   (libtorch 2.6 didn't work)
## For CUDA 12.6, use the following command to download the LibTorch package:
# wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.10.0%2Bcu126.zip
# unzip libtorch-shared-with-deps-2.10.0+cu126.zip # the unzipped folder named libtorch will appear in working directory
## make sure to append libtorch/lib to LD_LIBRARY_PATH in running shell of blade 

mkdir build
cd build
cmake -DUNITS=AKMA -DWITH_TORCH=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/libtorch ../src

make $1 blade # NUMBER OF CORES FOR COMPILE AS ARGUMENT


