cmake_minimum_required(VERSION 3.18)
project(blade LANGUAGES CXX CUDA)

# Modern CMake CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE sources *.cxx *.cu *.h)

# Modern CUDA toolkit discovery
find_package(CUDAToolkit 10.0 REQUIRED)

find_package(OpenMP REQUIRED)

# Unit system: AKMA (CHARMM) or NMPS (GROMACS)
set(UNITS "NMPS" CACHE STRING "AKMA (CHARMM) or NMPS (GROMACS)")
if (UNITS STREQUAL "AKMA")
  add_definitions(-DCHARMM_UNITS)
endif()

# Precision: FLOAT or DOUBLE
set(PRECISION "FLOAT" CACHE STRING "FLOAT OR DOUBLE")
if (PRECISION STREQUAL "DOUBLE")
  add_definitions(-DDOUBLE)
endif()

# Standalone build flag
# When ON (default), includes blade_log.cxx implementation (printf to stdout)
# When OFF (submodule in CHARMM), consumer provides blade_log implementation
option(BLADE_STANDALONE "Build standalone BLaDE with built-in logging" ON)
if(BLADE_STANDALONE)
  add_definitions(-DBLADE_STANDALONE)
endif()

# Create executable with modern CMake
add_executable(blade ${sources})

# Set CUDA architectures (adjust as needed for your GPU)
set_target_properties(blade PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86"
)

# CUDA compiler flags
target_compile_options(blade PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>
)

# Include directories
target_include_directories(blade PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link libraries
target_link_libraries(blade
    CUDA::cufft
    OpenMP::OpenMP_CXX
)

# nvToolsExt (NVTX) - try different target names for different CUDA versions
if(TARGET CUDA::nvToolsExt)
    target_link_libraries(blade CUDA::nvToolsExt)
elseif(TARGET CUDA::nvtx3)
    target_link_libraries(blade CUDA::nvtx3)
else()
    # NVTX3 is header-only in CUDA 10.0+, just need include path
    target_include_directories(blade PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
endif()

# Install rules
install(TARGETS blade DESTINATION bin)
