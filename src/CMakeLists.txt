cmake_minimum_required(VERSION 3.0)
project(msld LANGUAGES CXX CUDA)

file(GLOB_RECURSE sources *.cxx *.cu *.h)

find_package(CUDA 10.0)

## Add compiler definition flags here
# add_definitions(-DCUDAGRAPH)

## Use for debugging kernels
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

# message("${sources}")
cuda_add_executable(msld ${sources})
target_include_directories(msld PUBLIC .)
# target_compile_options(md PUBLIC -DNOMPI -g -O3 -arch=sm_20)

# Stuff that didn't work:
# set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lcurand")
# add_definitions(-lcurand)
