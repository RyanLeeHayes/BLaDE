// If this is a CHARMM compilation
#ifdef BLADE_IN_CHARMM

#include "rng_gpu.h"
#include <ctime>
#include <cstdio>
#include <cstdlib>

static const char *curandGetErrorString(curandStatus_t error)
{
  switch (error)
    {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";

    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";

    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";

    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";

    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";

    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";

    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";

    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";

    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";

    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
    }

  return "<unknown>";
}

#define curandCheck(stmt) do {                                           \
        curandStatus_t err = stmt;                                       \
        if (err != CURAND_STATUS_SUCCESS) {                              \
	  fprintf(stderr, "Error running %s in file %s, function %s, line %d\n", \
                 #stmt, __FILE__, __FUNCTION__, __LINE__);               \
	  fprintf(stderr, "Error string: %s\n", curandGetErrorString(err)); \
	  exit(1);						         \
        }                                                                \
    } while(0)

RngGPU::RngGPU()
{
  unsigned long long seed = time(NULL);

  /* Create pseudo-random number generator */
  curandCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));
  
  /* Set seed */
  curandCheck(curandSetPseudoRandomGeneratorSeed(gen, seed));

  rngStream=NULL;
  curandCheck(curandSetStream(gen,rngStream));
}

RngGPU::~RngGPU()
{
    curandCheck(curandDestroyGenerator(gen));
}

// Generate n random numbers in the pointer p
void RngGPU::rand_normal(int n,real *p,cudaStream_t s)
{
  if (rngStream!=s) {
    rngStream=s;
    curandCheck(curandSetStream(gen,rngStream));
  }

  curandCheck(curandGenerateNormal(gen, p, n, 0.0f, 1.0f));
}

// Generate n random numbers in the pointer p
void RngGPU::rand_uniform(int n,real *p,cudaStream_t s)
{
  if (rngStream!=s) {
    rngStream=s;
    curandCheck(curandSetStream(gen,rngStream));
  }

  curandCheck(curandGenerateUniform(gen, p, n));
}

#else

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
  for (i=256*blockIdx.x+threadIdx.x; i<((n+255)&0xFFFFFF00); i+=200*256) {
    real result=curand_normal(&state[blockIdx.x]); // Whole block generates
    if (i<n) p[i]=result; // Only requested threads store it
  }
}

__global__ void kernel_uniform(curandStateMtgp32 *state,int n,real *p)
{
  int i;
  for (i=256*blockIdx.x+threadIdx.x; i<((n+255)&0xFFFFFF00); i+=200*256) {
    real result=curand_uniform(&state[blockIdx.x]); // Whole block generates
    if (i<n) p[i]=result; // Only requested threads store it
  }
}

// Generate n random numbers in the pointer p
void RngGPU::rand_normal(int n,real *p,cudaStream_t s)
{
  int nblocks=(n+256-1)/256;
  nblocks=(nblocks>200)?200:nblocks;
  kernel_normal<<<nblocks,256,0,s>>>(devStates,n,p);
}

// Generate n random numbers in the pointer p
void RngGPU::rand_uniform(int n,real *p,cudaStream_t s)
{
  int nblocks=(n+256-1)/256;
  nblocks=(nblocks>200)?200:nblocks;
  kernel_uniform<<<nblocks,256,0,s>>>(devStates,n,p);
}

#endif
