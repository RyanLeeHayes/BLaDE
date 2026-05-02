cmake_minimum_required(VERSION 3.0)
project(blade LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

file(GLOB_RECURSE sources *.cxx *.cu *.h)

# ADDED: optional Torch switch, default OFF
option(WITH_TORCH "Build with LibTorch support" OFF)

# ADDED: Torch config block
if (WITH_TORCH)
  add_definitions(-DWITH_TORCH)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS OFF)
  find_package(Torch REQUIRED)
  # ADDED: explicit NVCC standard for old FindCUDA path
  list(APPEND CUDA_NVCC_FLAGS -std=c++17)
endif()

find_package(CUDA 10.0)
set (CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")

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
# add_definitions(-DPROFILESERIAL)

## Use for debugging kernels
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "--use_fast_math")
# set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-lineinfo")

# MPI: set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS} -lcufft -lnvToolsExt")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lnvToolsExt")

cuda_add_executable(blade ${sources})

# ADDED: enforce standard at target level too
if (WITH_TORCH)
  set_property(TARGET blade PROPERTY CXX_STANDARD 17)
  set_property(TARGET blade PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET blade PROPERTY CXX_EXTENSIONS OFF)
  set_property(TARGET blade PROPERTY CUDA_STANDARD 17)
  set_property(TARGET blade PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()

# MPI: target_include_directories(blade PUBLIC ${MPI_CXX_INCLUDE_PATH} .)
target_include_directories(blade PUBLIC .)

# ADDED: Torch include dirs only when Torch is enabled
if (WITH_TORCH)
  target_include_directories(blade PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

target_link_libraries(blade ${CUDA_CUFFT_LIBRARIES})

# ADDED: Torch link only when Torch is enabled
if (WITH_TORCH)
  target_link_libraries(blade ${TORCH_LIBRARIES})
endif()