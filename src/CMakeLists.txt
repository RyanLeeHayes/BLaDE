cmake_minimum_required(VERSION 3.0)
project(msld)

file(GLOB_RECURSE sources *.cxx *.cu *.h)

find_package(CUDA 10.0)

# add_definitions(-DNOMPI)
# add_definitions(-DDOUBLEFLAG)
# set (CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -O3 -arch=sm_20)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-g;-G")
# message("${sources}")
cuda_add_executable(msld ${sources})
target_include_directories(msld PUBLIC .)
# target_compile_options(md PUBLIC -DNOMPI -g -O3 -arch=sm_20)
