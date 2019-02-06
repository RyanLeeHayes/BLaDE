#ifndef MD_RNA_IONS_CUDA_UTIL_H
#define MD_RNA_IONS_CUDA_UTIL_H

#include "defines.h"

__device__ static inline
void reduce(int N, real* a,real* A)
{
  int i;

  for (i=1; i<N; i*=2) {
    if (threadIdx.x%(2*i)==0 && threadIdx.x+i<N) {
      a[threadIdx.x]+=a[threadIdx.x+i];
    }
    __syncthreads();
  }

  if (A) {
    if (threadIdx.x==0) {
      realAtomicAdd(A,a[0]);
    }
  }
}

#endif
