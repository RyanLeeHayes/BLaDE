/* include MTGP host helper functions */
#include <curand_mtgp32_host.h>
/* include MTGP pre-computed parameter sets */
#include <curand_mtgp32dc_p_11213.h>

#include "rng/rng_gpu.h"

void RngGPU::setup()
{
  long long seed=time(NULL);

  // 200 is limit, each of the 200 states can have up to 256 threads
  cudaMalloc((void**)&devStates, 200*sizeof(curandStateMtgp32_t));
  /* Allocate space for MTGP kernel parameters */
  cudaMalloc((void**)&devParams, sizeof(mtgp32_kernel_params_t));
  
  /* Reformat from predefined parameter sets to kernel format, */
  /* and copy kernel parameters to device memory               */
  curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devParams);
  /* Initialize one state per thread block */
  curandMakeMTGP32KernelState(devStates, 
    mtgp32dc_params_fast_11213, devParams, 200, seed);
}

__global__ void kernel_normal(curandStateMtgp32 *state,int n,real *p)
{
  int i;
  for (i=256*blockIdx.x+threadIdx.x; i<n; i+=200) {
    p[i]=curand_normal(&state[blockIdx.x]);
  }
}

__global__ void kernel_uniform(curandStateMtgp32 *state,int n,real *p)
{
  int i;
  for (i=256*blockIdx.x+threadIdx.x; i<n; i+=200) {
    p[i]=curand_uniform(&state[blockIdx.x]);
  }
}

// Generate n random numbers in the pointer p
void RngGPU::rand_normal(int n,real *p)
{
  int nblocks=(n+256-1)/256;
  nblocks=(nblocks>200)?200:nblocks;
  kernel_normal<<<nblocks,256>>>(devStates,n,p);
}

// Generate n random numbers in the pointer p
void RngGPU::rand_uniform(int n,real *p)
{
  int nblocks=(n+256-1)/256;
  nblocks=(nblocks>200)?200:nblocks;
  kernel_uniform<<<nblocks,256>>>(devStates,n,p);
}
