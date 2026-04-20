#ifndef MAIN_REDUCTION_H
#define MAIN_REDUCTION_H

#include <cuda_runtime.h>
#include "main/defines.h"
#include "main/real3.h"

//=============================================================================
// Generic Multi-Block Reduction Utilities
//
// Extends existing single-block reductions in real3.h to handle arrays
// larger than 1024 elements (CUDA thread-per-block limit).
//
// Supports:
//   - real (scalars, e.g., energies)
//   - real3 (3D vectors, e.g., forces)
//   - real33 (3x3 tensors, e.g., virials)
//
// Usage:
//   ReductionWorkspace ws;
//   reduction_workspace_init<real3>(&ws, n, stream);
//   parallel_reduce(input, output, n, &ws, stream);
//   reduction_workspace_free(&ws);
//=============================================================================

// Workspace for multi-block reduction
struct ReductionWorkspace {
    void *blockResults_d;  // Intermediate block results
    void *blockResults2_d; // Alternate buffer for iterative reductions
    int maxBlocks;         // Allocated capacity
    size_t elementSize;    // Size of each element
};

__host__ inline int reduction_round_up_warp(int n)
{
    int threads = ((n + 31) >> 5) << 5;
    return (threads > 0) ? threads : 32;
}

//-----------------------------------------------------------------------------
// Type traits for reduction operations
//-----------------------------------------------------------------------------

template <typename T>
struct ReductionTraits {
    // Default: no specialization - will cause compile error if used with unsupported type
};

// Specialization for real (scalar)
template <>
struct ReductionTraits<real> {
    __device__ __host__ static real identity() { return 0; }

    __device__ static real combine(real a, real b) { return a + b; }

    __device__ static real warp_reduce_sum(real val) {
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        return val;
    }

    __device__ static void atomic_add(real *addr, real val) {
#ifdef DOUBLE
        atomicAdd(addr, val);
#else
        atomicAdd(addr, val);
#endif
    }
};

// Specialization for real_e (double precision energy)
template <>
struct ReductionTraits<real_e> {
    __device__ __host__ static real_e identity() { return 0; }

    __device__ static real_e combine(real_e a, real_e b) { return a + b; }

    __device__ static real_e warp_reduce_sum(real_e val) {
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        return val;
    }

    __device__ static void atomic_add(real_e *addr, real_e val) {
        atomicAdd(addr, val);
    }
};

// Specialization for real3 (3D vector)
template <>
struct ReductionTraits<real3> {
    __device__ __host__ static real3 identity() {
        return make_real3(0, 0, 0);
    }

    __device__ static real3 combine(real3 a, real3 b) {
        return make_real3(a.x + b.x, a.y + b.y, a.z + b.z);
    }

    __device__ static real3 warp_reduce_sum(real3 val) {
        // Reduce each component independently
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, 16);
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, 8);
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, 4);
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, 2);
        val.x += __shfl_down_sync(0xFFFFFFFF, val.x, 1);

        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, 16);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, 8);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, 4);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, 2);
        val.y += __shfl_down_sync(0xFFFFFFFF, val.y, 1);

        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, 16);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, 8);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, 4);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, 2);
        val.z += __shfl_down_sync(0xFFFFFFFF, val.z, 1);

        return val;
    }

    __device__ static void atomic_add(real3 *addr, real3 val) {
        atomicAdd(&addr->x, val.x);
        atomicAdd(&addr->y, val.y);
        atomicAdd(&addr->z, val.z);
    }
};

// Specialization for real33 (3x3 tensor, e.g., virial)
template <>
struct ReductionTraits<real33> {
    __device__ __host__ static real33 identity() {
        real33 z;
        z.a = make_real3(0, 0, 0);
        z.b = make_real3(0, 0, 0);
        return z;
    }

    __device__ static real33 combine(real33 a, real33 b) {
        real33 c;
        c.a = make_real3(a.a.x + b.a.x, a.a.y + b.a.y, a.a.z + b.a.z);
        c.b = make_real3(a.b.x + b.b.x, a.b.y + b.b.y, a.b.z + b.b.z);
        return c;
    }

    __device__ static real33 warp_reduce_sum(real33 val) {
        // Reduce all 6 components (symmetric 3x3 stored as two real3)
        val.a.x += __shfl_down_sync(0xFFFFFFFF, val.a.x, 16);
        val.a.x += __shfl_down_sync(0xFFFFFFFF, val.a.x, 8);
        val.a.x += __shfl_down_sync(0xFFFFFFFF, val.a.x, 4);
        val.a.x += __shfl_down_sync(0xFFFFFFFF, val.a.x, 2);
        val.a.x += __shfl_down_sync(0xFFFFFFFF, val.a.x, 1);

        val.a.y += __shfl_down_sync(0xFFFFFFFF, val.a.y, 16);
        val.a.y += __shfl_down_sync(0xFFFFFFFF, val.a.y, 8);
        val.a.y += __shfl_down_sync(0xFFFFFFFF, val.a.y, 4);
        val.a.y += __shfl_down_sync(0xFFFFFFFF, val.a.y, 2);
        val.a.y += __shfl_down_sync(0xFFFFFFFF, val.a.y, 1);

        val.a.z += __shfl_down_sync(0xFFFFFFFF, val.a.z, 16);
        val.a.z += __shfl_down_sync(0xFFFFFFFF, val.a.z, 8);
        val.a.z += __shfl_down_sync(0xFFFFFFFF, val.a.z, 4);
        val.a.z += __shfl_down_sync(0xFFFFFFFF, val.a.z, 2);
        val.a.z += __shfl_down_sync(0xFFFFFFFF, val.a.z, 1);

        val.b.x += __shfl_down_sync(0xFFFFFFFF, val.b.x, 16);
        val.b.x += __shfl_down_sync(0xFFFFFFFF, val.b.x, 8);
        val.b.x += __shfl_down_sync(0xFFFFFFFF, val.b.x, 4);
        val.b.x += __shfl_down_sync(0xFFFFFFFF, val.b.x, 2);
        val.b.x += __shfl_down_sync(0xFFFFFFFF, val.b.x, 1);

        val.b.y += __shfl_down_sync(0xFFFFFFFF, val.b.y, 16);
        val.b.y += __shfl_down_sync(0xFFFFFFFF, val.b.y, 8);
        val.b.y += __shfl_down_sync(0xFFFFFFFF, val.b.y, 4);
        val.b.y += __shfl_down_sync(0xFFFFFFFF, val.b.y, 2);
        val.b.y += __shfl_down_sync(0xFFFFFFFF, val.b.y, 1);

        val.b.z += __shfl_down_sync(0xFFFFFFFF, val.b.z, 16);
        val.b.z += __shfl_down_sync(0xFFFFFFFF, val.b.z, 8);
        val.b.z += __shfl_down_sync(0xFFFFFFFF, val.b.z, 4);
        val.b.z += __shfl_down_sync(0xFFFFFFFF, val.b.z, 2);
        val.b.z += __shfl_down_sync(0xFFFFFFFF, val.b.z, 1);

        return val;
    }

    __device__ static void atomic_add(real33 *addr, real33 val) {
        atomicAdd(&addr->a.x, val.a.x);
        atomicAdd(&addr->a.y, val.a.y);
        atomicAdd(&addr->a.z, val.a.z);
        atomicAdd(&addr->b.x, val.b.x);
        atomicAdd(&addr->b.y, val.b.y);
        atomicAdd(&addr->b.z, val.b.z);
    }
};

//-----------------------------------------------------------------------------
// Workspace management
//-----------------------------------------------------------------------------

template <typename T>
__host__ inline
void reduction_workspace_init(ReductionWorkspace *ws, int maxElements, cudaStream_t stream)
{
    const int blockSize = BLUP;
    ws->maxBlocks = (maxElements + blockSize - 1) / blockSize;
    ws->elementSize = sizeof(T);
    cudaMalloc(&ws->blockResults_d, ws->maxBlocks * sizeof(T));
    cudaMalloc(&ws->blockResults2_d, ws->maxBlocks * sizeof(T));
}

__host__ inline
void reduction_workspace_free(ReductionWorkspace *ws)
{
    if (ws->blockResults_d) cudaFree(ws->blockResults_d);
    if (ws->blockResults2_d) cudaFree(ws->blockResults2_d);
    ws->blockResults_d = NULL;
    ws->blockResults2_d = NULL;
}

//-----------------------------------------------------------------------------
// Phase 1: Block-level reduction kernel
// Each block reduces its portion and stores result in blockResults
//-----------------------------------------------------------------------------

template <typename T>
__global__ void reduce_phase1_kernel(T *input, T *blockResults, int n)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input (or identity if out of bounds)
    T val = (gid < n) ? input[gid] : ReductionTraits<T>::identity();

    // Warp-level reduction
    val = ReductionTraits<T>::warp_reduce_sum(val);

    // Store warp results to shared memory
    if ((tid & 31) == 0) {
        sdata[tid >> 5] = val;
    }
    __syncthreads();

    // Final reduction across warps (first warp only)
    if (tid < 32) {
        int numWarps = (blockDim.x + 31) >> 5;
        val = (tid < numWarps) ? sdata[tid] : ReductionTraits<T>::identity();
        val = ReductionTraits<T>::warp_reduce_sum(val);

        if (tid == 0) {
            blockResults[blockIdx.x] = val;
        }
    }
}

//-----------------------------------------------------------------------------
// Phase 2: Final reduction kernel (single block)
// Reduces block results to final output
//-----------------------------------------------------------------------------

template <typename T>
__global__ void reduce_phase2_kernel(T *blockResults, T *output, int numBlocks)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    int tid = threadIdx.x;

    // Load block result (or identity if out of bounds)
    T val = (tid < numBlocks) ? blockResults[tid] : ReductionTraits<T>::identity();

    // Warp-level reduction
    val = ReductionTraits<T>::warp_reduce_sum(val);

    // Store warp results to shared memory
    if ((tid & 31) == 0) {
        sdata[tid >> 5] = val;
    }
    __syncthreads();

    // Final reduction across warps
    if (tid < 32) {
        int numWarps = (blockDim.x + 31) >> 5;
        val = (tid < numWarps) ? sdata[tid] : ReductionTraits<T>::identity();
        val = ReductionTraits<T>::warp_reduce_sum(val);

        if (tid == 0) {
            *output = val;
        }
    }
}

//-----------------------------------------------------------------------------
// Single-block reduction kernel (for small arrays)
//-----------------------------------------------------------------------------

template <typename T>
__global__ void reduce_single_block_kernel(T *input, T *output, int n)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    int tid = threadIdx.x;

    // Load input
    T val = (tid < n) ? input[tid] : ReductionTraits<T>::identity();

    // Warp-level reduction
    val = ReductionTraits<T>::warp_reduce_sum(val);

    // Store warp results
    if ((tid & 31) == 0) {
        sdata[tid >> 5] = val;
    }
    __syncthreads();

    // Final reduction
    if (tid < 32) {
        int numWarps = (n + 31) >> 5;
        val = (tid < numWarps) ? sdata[tid] : ReductionTraits<T>::identity();
        val = ReductionTraits<T>::warp_reduce_sum(val);

        if (tid == 0) {
            *output = val;
        }
    }
}

//-----------------------------------------------------------------------------
// Host API: Multi-block reduction
//-----------------------------------------------------------------------------

template <typename T>
__host__ void parallel_reduce(T *d_input, T *d_output, int n,
                              ReductionWorkspace *ws, cudaStream_t stream)
{
    const int blockSize = BLUP;

    if (n <= blockSize) {
        // Single block - direct reduction
        int threads = reduction_round_up_warp(n);
        int numWarps = threads >> 5;
        reduce_single_block_kernel<T><<<1, threads, numWarps * sizeof(T), stream>>>
            (d_input, d_output, n);
    } else {
        T *input = d_input;
        T *output = reinterpret_cast<T*>(ws->blockResults_d);
        T *alternate = reinterpret_cast<T*>(ws->blockResults2_d);
        int currentCount = n;
        int numWarps1 = blockSize >> 5;

        while (currentCount > blockSize) {
            int numBlocks = (currentCount + blockSize - 1) / blockSize;
            reduce_phase1_kernel<T><<<numBlocks, blockSize, numWarps1 * sizeof(T), stream>>>
                (input, output, currentCount);

            input = output;
            output = alternate;
            alternate = input;
            currentCount = numBlocks;
        }

        int threads = reduction_round_up_warp(currentCount);
        int numWarps = threads >> 5;
        reduce_single_block_kernel<T><<<1, threads, numWarps * sizeof(T), stream>>>
            (input, d_output, currentCount);
    }
}

//-----------------------------------------------------------------------------
// Convenience: Reduction with atomic accumulation to existing value
// (Matches the pattern used in real3.h for energy accumulation)
//-----------------------------------------------------------------------------

template <typename T>
__global__ void reduce_atomic_kernel(T *input, T *global_output, int n)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input
    T val = (gid < n) ? input[gid] : ReductionTraits<T>::identity();

    // Warp-level reduction
    val = ReductionTraits<T>::warp_reduce_sum(val);

    // Store warp results
    if ((tid & 31) == 0) {
        sdata[tid >> 5] = val;
    }
    __syncthreads();

    // Final reduction and atomic add
    if (tid < 32) {
        int numWarps = (blockDim.x + 31) >> 5;
        val = (tid < numWarps) ? sdata[tid] : ReductionTraits<T>::identity();
        val = ReductionTraits<T>::warp_reduce_sum(val);

        if (tid == 0) {
            ReductionTraits<T>::atomic_add(global_output, val);
        }
    }
}

// Host API for atomic reduction (accumulates to existing value)
template <typename T>
__host__ void parallel_reduce_atomic(T *d_input, T *d_global_output, int n,
                                     cudaStream_t stream)
{
    const int blockSize = BLUP;
    int numBlocks = (n + blockSize - 1) / blockSize;
    int numWarps = blockSize >> 5;

    reduce_atomic_kernel<T><<<numBlocks, blockSize, numWarps * sizeof(T), stream>>>
        (d_input, d_global_output, n);
}

#endif // MAIN_REDUCTION_H
