cmake_minimum_required(VERSION 3.0)
project(md)

file(GLOB_RECURSE sources *.cu *.h)

find_package(CUDA 7.0)

add_definitions(-DNOMPI)
# add_definitions(-DDOUBLEFLAG)
# set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -O3 -arch=sm_20)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g;-G")
# message("${sources}")
cuda_add_executable(md ${sources})
# target_compile_options(md PUBLIC -DNOMPI -g -O3 -arch=sm_20)
