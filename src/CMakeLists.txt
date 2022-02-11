cmake_minimum_required(VERSION 3.0)
project(blade LANGUAGES CXX CUDA)

file(GLOB_RECURSE sources *.cxx *.cu *.h)

find_package(CUDA 10.0)
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")
# MPI: find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# option(UNITS "AKMA (CHARMM) or NMPS (GROMACS)" "NMPS")
set(UNITS "NMPS" CACHE STRING "AKMA (CHARMM) or NMPS (GROMACS)")
if (UNITS STREQUAL "AKMA")
  add_definitions(-DCHARMM_UNITS)
endif()

# option(PRECISION "FLOAT OR DOUBLE" "FLOAT")
set(PRECISION "FLOAT" CACHE STRING "FLOAT OR DOUBLE")
if (PRECISION STREQUAL "DOUBLE")
  add_definitions(-DDOUBLE)
endif()

## Add compiler definition flags here
# add_definitions(-DCUDAGRAPH)
# add_definitions(-DPROFILESERIAL)

## Use for debugging kernels
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "--use_fast_math")
# set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-lineinfo")

# # Use for double atomicAdd - only works on GTX1080, not on GTX980
# set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-gencode=arch=compute_60,code=sm_60")
# set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-gencode=arch=compute_60,code=sm_60")

# MPI: set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} -lcufft -lnvToolsExt")
# set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lcufft -lnvToolsExt")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lnvToolsExt")
# NOTE: Moved cufft later with target_link_libraries
# https://cmake.org/cmake/help/latest/module/FindCUDA.html
# May also want to consider adding CUDA_nvToolsExt_LIBRARY there...

# message("${sources}")
cuda_add_executable(blade ${sources})
# MPI: target_include_directories(blade PUBLIC ${MPI_CXX_INCLUDE_PATH} .)
target_include_directories(blade PUBLIC .)
# target_compile_options(md PUBLIC -DNOMPI -g -O3 -arch=sm_20)

target_link_libraries(blade ${CUDA_CUFFT_LIBRARIES})

# Stuff that didn't work:
# set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lcurand")
# add_definitions(-lcurand)
