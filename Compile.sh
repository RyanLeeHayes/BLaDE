#!/bin/bash

# Build BLaDE - assumes environment (CUDA, compilers) is already configured
#
# Usage:
#   ./Compile.sh [options]
#
# Options:
#   -j N, --jobs N      Parallel build jobs (default: nproc)
#   --double            Use double precision (default: float)
#   --gromacs           Use NMPS/GROMACS units (default: AKMA/CHARMM)
#   --install [PREFIX]  Build and install (default PREFIX: /usr/local)
#   --clean             Remove build directory
#   --reconfigure       Force CMake reconfiguration
#   -h, --help          Show this help

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Defaults
UNITS="AKMA"
PRECISION="FLOAT"
JOBS=$(nproc)
DO_INSTALL=false
INSTALL_PREFIX="/usr/local"
DO_CLEAN=false
RECONFIGURE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --double)
            PRECISION="DOUBLE"
            shift
            ;;
        --gromacs)
            UNITS="NMPS"
            shift
            ;;
        --install)
            DO_INSTALL=true
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                INSTALL_PREFIX="$2"
                shift
            fi
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --reconfigure)
            RECONFIGURE=true
            shift
            ;;
        -h|--help)
            head -16 "$0" | tail -14
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Clean
if $DO_CLEAN; then
    rm -rf "${BUILD_DIR}"
    echo "Build directory removed"
    exit 0
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure if needed
if [ ! -f "Makefile" ] || $RECONFIGURE; then
    echo "Configuring: UNITS=${UNITS}, PRECISION=${PRECISION}"
    cmake -DUNITS="${UNITS}" \
          -DPRECISION="${PRECISION}" \
          -DBLADE_STANDALONE=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
          ../src
fi

# Build
echo "Building with ${JOBS} jobs..."
make -j"${JOBS}" blade

# Install if requested
if $DO_INSTALL; then
    echo "Installing to ${INSTALL_PREFIX}..."
    make install
fi
