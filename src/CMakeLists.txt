cmake_minimum_required(VERSION 3.18)
project(blade LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

file(GLOB_RECURSE sources *.cxx *.cu *.h)

# Parse options
option(WITH_TORCH "Build with LibTorch support" OFF)
option(RX "Build with replica exchange through MPI" OFF)
option(PROFILE "PROFILE=ON sets PROFILESERIAL for serial execution of kernels." OFF)
set(UNITS "NMPS" CACHE STRING "AKMA (CHARMM) or NMPS (GROMACS)")
set(PRECISION "FLOAT" CACHE STRING "FLOAT OR DOUBLE")

# ADDED: Torch config block
if (WITH_TORCH)
  add_definitions(-DWITH_TORCH)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
  set(CMAKE_CXX_EXTENSIONS OFF)
  find_package(Torch REQUIRED)
  # ADDED: explicit NVCC standard for old FindCUDA path
  # list(APPEND CMAKE_CUDA_FLAGS -std=c++17)
endif()

if (RX)
  find_package(MPI REQUIRED)
endif()

find_package(CUDAToolkit REQUIRED)
# useful nvcc flags:
# -lineinfo # use for ncu profiling
# -use_fast_math # gives a 9% performance boost
# -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -use_fast_math")
# set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lnvToolsExt") # not currently used

find_package(OpenMP REQUIRED)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# Apply options
if (UNITS STREQUAL "AKMA")
  add_definitions(-DCHARMM_UNITS)
endif()
if (PROFILE)
  add_definitions(-DPROFILESERIAL)
endif()
if (RX)
  add_definitions(-DREPLICAEXCHANGE)
endif()
if (PRECISION STREQUAL "DOUBLE")
  add_definitions(-DDOUBLE)
endif()

add_executable(blade ${sources})

# ADDED: enforce standard at target level too
if (WITH_TORCH)
  set_property(TARGET blade PROPERTY CXX_STANDARD 17)
  set_property(TARGET blade PROPERTY CXX_STANDARD_REQUIRED ON)
  set_property(TARGET blade PROPERTY CXX_EXTENSIONS OFF)
  set_property(TARGET blade PROPERTY CUDA_STANDARD 17)
  set_property(TARGET blade PROPERTY CUDA_STANDARD_REQUIRED ON)
endif()

if (RX)
  target_include_directories(blade PUBLIC ${MPI_CXX_INCLUDE_PATH} .)
else()
  target_include_directories(blade PUBLIC .)
endif()

# ADDED: Torch include dirs only when Torch is enabled
if (WITH_TORCH)
  target_include_directories(blade PUBLIC ${TORCH_INCLUDE_DIRS})
endif()

target_link_libraries(blade CUDA::curand CUDA::cufft)
target_link_libraries(blade ${OpenMP_EXE_LINKER_FLAGS})

if (RX)
  target_link_libraries(blade ${MPI_CXX_LIBRARIES})
endif()

# ADDED: Torch link only when Torch is enabled
if (WITH_TORCH)
  target_link_libraries(blade ${TORCH_LIBRARIES})
endif()
