#ifndef MAIN_SCAN_H
#define MAIN_SCAN_H

#include <cuda_runtime.h>
#include "main/defines.h"

// Platform-independent logging - implementation provided by build system
#include "main/blade_log.h"

//=============================================================================
// Multi-Block Parallel Prefix Sum (Scan) Primitives
//
// Supports two algorithms selectable at runtime:
//   SCAN_BLELLOCH (0): Work-efficient O(n) work, O(log n) steps
//   SCAN_HILLIS_STEELE (1): Work-inefficient O(n log n) work, lower latency
//
// Usage:
//   ScanWorkspace ws;
//   scan_workspace_init(&ws, n, stream);
//   parallel_inclusive_scan(input, output, n, &ws, stream, algorithm);
//   scan_workspace_free(&ws);
//=============================================================================

// Algorithm selection constants
#define SCAN_AUTO         -1  // Auto-select via benchmarking
#define SCAN_BLELLOCH      0  // Work-efficient O(n) work
#define SCAN_HILLIS_STEELE 1  // Low-latency O(n log n) work

// Workspace for multi-block scan operations
struct ScanWorkspace {
    int *blockSums_d;      // Block sums for hierarchical scan
    int *blockSums2_d;     // Second-level block sums (for very large arrays)
    int maxBlocks;         // Allocated capacity
    int maxBlocks2;        // Second-level capacity
};

//-----------------------------------------------------------------------------
// Workspace management
//-----------------------------------------------------------------------------

__host__ inline
void scan_workspace_init(ScanWorkspace *ws, int maxElements, cudaStream_t stream)
{
    const int blockSize = BLUP;
    ws->maxBlocks = (maxElements + blockSize - 1) / blockSize;
    ws->maxBlocks2 = (ws->maxBlocks + blockSize - 1) / blockSize;

    cudaMalloc(&ws->blockSums_d, ws->maxBlocks * sizeof(int));
    if (ws->maxBlocks > blockSize) {
        cudaMalloc(&ws->blockSums2_d, ws->maxBlocks2 * sizeof(int));
    } else {
        ws->blockSums2_d = NULL;
    }
}

__host__ inline
void scan_workspace_free(ScanWorkspace *ws)
{
    if (ws->blockSums_d) cudaFree(ws->blockSums_d);
    if (ws->blockSums2_d) cudaFree(ws->blockSums2_d);
    ws->blockSums_d = NULL;
    ws->blockSums2_d = NULL;
}

//-----------------------------------------------------------------------------
// Blelloch (work-efficient) single-block scan kernel
// Exclusive scan, then add input for inclusive result
//-----------------------------------------------------------------------------

template <typename T>
__global__ void scan_blelloch_single_kernel(T *input, T *output, int n)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    int tid = threadIdx.x;

    // Load input to shared memory
    sdata[tid] = (tid < n) ? input[tid] : 0;
    T original = sdata[tid];
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }

    // Clear last element for exclusive scan
    if (tid == blockDim.x - 1) {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            T temp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += temp;
        }
        __syncthreads();
    }

    // Write inclusive scan result (exclusive + original)
    if (tid < n) {
        output[tid] = sdata[tid] + original;
    }
}

//-----------------------------------------------------------------------------
// Hillis-Steele (work-inefficient but low latency) single-block scan kernel
//-----------------------------------------------------------------------------

template <typename T>
__global__ void scan_hillis_steele_single_kernel(T *input, T *output, int n)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);
    T *sdata2 = reinterpret_cast<T*>(smem) + blockDim.x;

    int tid = threadIdx.x;

    // Load input
    sdata[tid] = (tid < n) ? input[tid] : 0;
    __syncthreads();

    // Hillis-Steele inclusive scan
    T *src = sdata;
    T *dst = sdata2;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            dst[tid] = src[tid] + src[tid - stride];
        } else {
            dst[tid] = src[tid];
        }
        __syncthreads();
        // Swap buffers
        T *tmp = src; src = dst; dst = tmp;
    }

    // Write result
    if (tid < n) {
        output[tid] = src[tid];
    }
}

//-----------------------------------------------------------------------------
// Phase 1: Block-local scan with block sum extraction (Blelloch)
//-----------------------------------------------------------------------------

template <typename T>
__global__ void scan_phase1_blelloch_kernel(T *input, T *output, T *blockSums, int n)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load to shared memory
    sdata[tid] = (gid < n) ? input[gid] : 0;
    T original = sdata[tid];
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }

    // Save block sum before clearing
    if (tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = sdata[tid];
        sdata[tid] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            T temp = sdata[index - stride];
            sdata[index - stride] = sdata[index];
            sdata[index] += temp;
        }
        __syncthreads();
    }

    // Write inclusive result
    if (gid < n) {
        output[gid] = sdata[tid] + original;
    }
}

//-----------------------------------------------------------------------------
// Phase 1: Block-local scan with block sum extraction (Hillis-Steele)
//-----------------------------------------------------------------------------

template <typename T>
__global__ void scan_phase1_hillis_steele_kernel(T *input, T *output, T *blockSums, int n)
{
    extern __shared__ char smem[];
    T *sdata = reinterpret_cast<T*>(smem);
    T *sdata2 = reinterpret_cast<T*>(smem) + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input
    sdata[tid] = (gid < n) ? input[gid] : 0;
    __syncthreads();

    // Hillis-Steele inclusive scan
    T *src = sdata;
    T *dst = sdata2;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid >= stride) {
            dst[tid] = src[tid] + src[tid - stride];
        } else {
            dst[tid] = src[tid];
        }
        __syncthreads();
        T *tmp = src; src = dst; dst = tmp;
    }

    // Save block sum (last element of this block's scan)
    if (tid == blockDim.x - 1) {
        blockSums[blockIdx.x] = src[tid];
    }

    // Write result
    if (gid < n) {
        output[gid] = src[tid];
    }
}

//-----------------------------------------------------------------------------
// Phase 3: Add scanned block sums to each block's results
//-----------------------------------------------------------------------------

template <typename T>
__global__ void scan_phase3_kernel(T *output, T *blockSums, int n)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // Block 0 doesn't need adjustment; other blocks add prefix
    if (blockIdx.x > 0 && gid < n) {
        output[gid] += blockSums[blockIdx.x - 1];
    }
}

//-----------------------------------------------------------------------------
// Host API: Multi-block inclusive scan
//-----------------------------------------------------------------------------

template <typename T>
__host__ void parallel_inclusive_scan_impl(T *d_input, T *d_output, int n,
                                           ScanWorkspace *ws, cudaStream_t stream,
                                           int algorithm)
{
    const int blockSize = BLUP;
    int numBlocks = (n + blockSize - 1) / blockSize;

    if (numBlocks == 1) {
        // Single block - use direct single-block kernel
        int actualSize = n;
        // Round up to power of 2 for Blelloch algorithm
        int paddedSize = 1;
        while (paddedSize < actualSize) paddedSize *= 2;
        if (paddedSize > 1024) paddedSize = 1024;

        if (algorithm == SCAN_HILLIS_STEELE) {
            scan_hillis_steele_single_kernel<T><<<1, paddedSize, 2*paddedSize*sizeof(T), stream>>>
                (d_input, d_output, n);
        } else {
            scan_blelloch_single_kernel<T><<<1, paddedSize, paddedSize*sizeof(T), stream>>>
                (d_input, d_output, n);
        }
        return;
    }

    // Multi-block: Three-phase hierarchical scan

    // Phase 1: Block-local scans, extract block sums
    if (algorithm == SCAN_HILLIS_STEELE) {
        scan_phase1_hillis_steele_kernel<T><<<numBlocks, blockSize, 2*blockSize*sizeof(T), stream>>>
            (d_input, d_output, ws->blockSums_d, n);
    } else {
        scan_phase1_blelloch_kernel<T><<<numBlocks, blockSize, blockSize*sizeof(T), stream>>>
            (d_input, d_output, ws->blockSums_d, n);
    }

    // Phase 2: Scan block sums (recursive for very large arrays)
    if (numBlocks <= blockSize) {
        // Block sums fit in single block
        int paddedSize = 1;
        while (paddedSize < numBlocks) paddedSize *= 2;
        if (paddedSize > 1024) paddedSize = 1024;

        if (algorithm == SCAN_HILLIS_STEELE) {
            scan_hillis_steele_single_kernel<T><<<1, paddedSize, 2*paddedSize*sizeof(T), stream>>>
                (ws->blockSums_d, ws->blockSums_d, numBlocks);
        } else {
            scan_blelloch_single_kernel<T><<<1, paddedSize, paddedSize*sizeof(T), stream>>>
                (ws->blockSums_d, ws->blockSums_d, numBlocks);
        }
    } else {
        // Need second-level scan for block sums
        int numBlocks2 = (numBlocks + blockSize - 1) / blockSize;

        // Scan block sums with second-level workspace
        if (algorithm == SCAN_HILLIS_STEELE) {
            scan_phase1_hillis_steele_kernel<T><<<numBlocks2, blockSize, 2*blockSize*sizeof(T), stream>>>
                (ws->blockSums_d, ws->blockSums_d, ws->blockSums2_d, numBlocks);
        } else {
            scan_phase1_blelloch_kernel<T><<<numBlocks2, blockSize, blockSize*sizeof(T), stream>>>
                (ws->blockSums_d, ws->blockSums_d, ws->blockSums2_d, numBlocks);
        }

        // Scan second-level block sums (should fit in single block now)
        int paddedSize = 1;
        while (paddedSize < numBlocks2) paddedSize *= 2;
        if (paddedSize > 1024) paddedSize = 1024;

        if (algorithm == SCAN_HILLIS_STEELE) {
            scan_hillis_steele_single_kernel<T><<<1, paddedSize, 2*paddedSize*sizeof(T), stream>>>
                (ws->blockSums2_d, ws->blockSums2_d, numBlocks2);
        } else {
            scan_blelloch_single_kernel<T><<<1, paddedSize, paddedSize*sizeof(T), stream>>>
                (ws->blockSums2_d, ws->blockSums2_d, numBlocks2);
        }

        // Add second-level sums to first-level
        scan_phase3_kernel<T><<<numBlocks2, blockSize, 0, stream>>>
            (ws->blockSums_d, ws->blockSums2_d, numBlocks);
    }

    // Phase 3: Add scanned block sums to each block
    scan_phase3_kernel<T><<<numBlocks, blockSize, 0, stream>>>
        (d_output, ws->blockSums_d, n);
}

// Convenience wrapper using int type
__host__ inline
void parallel_inclusive_scan_int(int *d_input, int *d_output, int n,
                                 ScanWorkspace *ws, cudaStream_t stream,
                                 int algorithm = SCAN_BLELLOCH)
{
    parallel_inclusive_scan_impl<int>(d_input, d_output, n, ws, stream, algorithm);
}

//-----------------------------------------------------------------------------
// Warp-level inclusive scan (for use within kernels, up to 32 elements)
//-----------------------------------------------------------------------------

__device__ inline
int warp_inclusive_scan(int val)
{
    int lane = threadIdx.x & 31;
    val += (lane >= 1)  * __shfl_up_sync(0xFFFFFFFF, val, 1);
    val += (lane >= 2)  * __shfl_up_sync(0xFFFFFFFF, val, 2);
    val += (lane >= 4)  * __shfl_up_sync(0xFFFFFFFF, val, 4);
    val += (lane >= 8)  * __shfl_up_sync(0xFFFFFFFF, val, 8);
    val += (lane >= 16) * __shfl_up_sync(0xFFFFFFFF, val, 16);
    return val;
}

__device__ inline
int warp_exclusive_scan(int val)
{
    int inclusive = warp_inclusive_scan(val);
    return inclusive - val;
}

#endif // MAIN_SCAN_H
