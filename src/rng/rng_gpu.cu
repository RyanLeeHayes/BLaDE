// If this is a CHARMM compilation
static const int RNG_GPU_CHARMM_UNSERIALIZED_MARKER=1; // DrudeIns - provenance marker for Drude PR.

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
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  currentSeed = time(NULL); // DrudeIns - provenance marker for Drude PR.

  /* Create pseudo-random number generator */
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  curandCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32));

  rngStream=NULL; // DrudeIns - provenance marker for Drude PR.
  curandCheck(curandSetStream(gen,rngStream));
  set_seed(currentSeed); // DrudeIns - provenance marker for Drude PR.
}

RngGPU::~RngGPU()
{
    curandCheck(curandDestroyGenerator(gen));
} // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
void RngGPU::set_seed(unsigned long long seed) // DrudeIns - provenance marker for Drude PR.
{ // DrudeIns - provenance marker for Drude PR.
  currentSeed=seed; // DrudeIns - provenance marker for Drude PR.
  curandCheck(curandSetPseudoRandomGeneratorSeed(gen, currentSeed)); // DrudeIns - provenance marker for Drude PR.
} // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
void RngGPU::write_checkpoint_state(FILE *fp) const // DrudeIns - provenance marker for Drude PR.
{ // DrudeIns - provenance marker for Drude PR.
  fprintf(fp,"RngGPUState 3\n"); // DrudeIns - provenance marker for Drude PR.
  fprintf(fp,"0 0 %d\n",RNG_GPU_CHARMM_UNSERIALIZED_MARKER); // DrudeIns - provenance marker for Drude PR.
  fprintf(stderr, // DrudeIns - provenance marker for Drude PR.
    "DRUDE WARNING> BLADE_IN_CHARMM GPU RNG checkpoint stores a placeholder only; " // DrudeIns - provenance marker for Drude PR.
    "cuRAND generator state is not serializable in this build.\n"); // DrudeIns - provenance marker for Drude PR.
} // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
void RngGPU::read_checkpoint_state(FILE *fp,int sectionCount) // DrudeIns - provenance marker for Drude PR.
{ // DrudeIns - provenance marker for Drude PR.
  int stateCount=0; // DrudeIns - provenance marker for Drude PR.
  int stateSize=0; // DrudeIns - provenance marker for Drude PR.
  int reserved=0; // DrudeIns - provenance marker for Drude PR.
  unsigned int tmp=0; // DrudeIns - provenance marker for Drude PR.
  if (fscanf(fp,"%d %d %d",&stateCount,&stateSize,&reserved)!=3) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr,"Malformed RngGPUState checkpoint metadata\n"); // DrudeIns - provenance marker for Drude PR.
    exit(1); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  if (stateCount<0 || stateSize<0) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr, // DrudeIns - provenance marker for Drude PR.
      "Malformed RngGPUState checkpoint metadata: stateCount=%d stateSize=%d\n", // DrudeIns - provenance marker for Drude PR.
      stateCount,stateSize); // DrudeIns - provenance marker for Drude PR.
    exit(1); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  if (stateCount==0 && stateSize==0 && reserved==RNG_GPU_CHARMM_UNSERIALIZED_MARKER) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr, // DrudeIns - provenance marker for Drude PR.
      "DRUDE WARNING> BLADE_IN_CHARMM checkpoint has no restorable GPU RNG state; " // DrudeIns - provenance marker for Drude PR.
      "continuing with the current cuRAND generator stream.\n"); // DrudeIns - provenance marker for Drude PR.
    for (int i=3; i<sectionCount; i++) { // DrudeIns - provenance marker for Drude PR.
      if (fscanf(fp,"%u",&tmp)!=1) { // DrudeIns - provenance marker for Drude PR.
        fprintf(stderr,"Malformed RngGPUState checkpoint extension\n"); // DrudeIns - provenance marker for Drude PR.
        exit(1); // DrudeIns - provenance marker for Drude PR.
      } // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
    return; // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  fprintf(stderr, // DrudeIns - provenance marker for Drude PR.
    "DRUDE WARNING> BLADE_IN_CHARMM cannot restore serialized GPU RNG state; " // DrudeIns - provenance marker for Drude PR.
    "consuming checkpoint payload and continuing with the current cuRAND generator stream.\n"); // DrudeIns - provenance marker for Drude PR.
  for (int i=0; i<stateCount; i++) { // DrudeIns - provenance marker for Drude PR.
    int offset=0; // DrudeIns - provenance marker for Drude PR.
    int pIdx=0; // DrudeIns - provenance marker for Drude PR.
    if (fscanf(fp,"%d %d",&offset,&pIdx)!=2) { // DrudeIns - provenance marker for Drude PR.
      fprintf(stderr,"Malformed RngGPUState checkpoint row\n"); // DrudeIns - provenance marker for Drude PR.
      exit(1); // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
    for (int j=0; j<stateSize; j++) { // DrudeIns - provenance marker for Drude PR.
      if (fscanf(fp,"%u",&tmp)!=1) { // DrudeIns - provenance marker for Drude PR.
        fprintf(stderr,"Malformed RngGPUState checkpoint state payload\n"); // DrudeIns - provenance marker for Drude PR.
        exit(1); // DrudeIns - provenance marker for Drude PR.
      } // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  for (int i=3; i<sectionCount; i++) { // DrudeIns - provenance marker for Drude PR.
    if (fscanf(fp,"%u",&tmp)!=1) { // DrudeIns - provenance marker for Drude PR.
      fprintf(stderr,"Malformed RngGPUState checkpoint extension\n"); // DrudeIns - provenance marker for Drude PR.
      exit(1); // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
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

#include "rng/rng_gpu.h" // DrudeIns - provenance marker for Drude PR.

static const int RNG_GPU_MTGP32_STATE_COUNT=200; // DrudeIns - provenance marker for Drude PR.

void RngGPU::setup() // DrudeIns - provenance marker for Drude PR.
{
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  currentSeed=(unsigned long long)time(NULL); // DrudeIns - provenance marker for Drude PR.

  // 200 is limit, each of the 200 states can have up to 256 threads
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  cudaMalloc((void**)&devStates, RNG_GPU_MTGP32_STATE_COUNT*sizeof(curandStateMtgp32_t)); // DrudeIns - provenance marker for Drude PR.
  /* Allocate space for MTGP kernel parameters */
  cudaMalloc((void**)&devParams, sizeof(mtgp32_kernel_params_t));
  
  /* Reformat from predefined parameter sets to kernel format, */ // DrudeIns - provenance marker for Drude PR.
  /* and copy kernel parameters to device memory               */ // DrudeIns - provenance marker for Drude PR.
  curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, devParams);
  /* Initialize one state per thread block */ // DrudeIns - provenance marker for Drude PR.
  set_seed(currentSeed); // DrudeIns - provenance marker for Drude PR.
} // DrudeIns - provenance marker for Drude PR.

void RngGPU::set_seed(unsigned long long seed) // DrudeIns - provenance marker for Drude PR.
{ // DrudeIns - provenance marker for Drude PR.
  currentSeed=seed; // DrudeIns - provenance marker for Drude PR.
  curandMakeMTGP32KernelState(devStates, // DrudeIns - provenance marker for Drude PR.
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
    mtgp32dc_params_fast_11213, devParams, RNG_GPU_MTGP32_STATE_COUNT, currentSeed); // DrudeIns - provenance marker for Drude PR.
} // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
void RngGPU::write_checkpoint_state(FILE *fp) const // DrudeIns - provenance marker for Drude PR.
{ // DrudeIns - provenance marker for Drude PR.
  curandStateMtgp32_t *states=(curandStateMtgp32_t*)malloc( // DrudeIns - provenance marker for Drude PR.
    RNG_GPU_MTGP32_STATE_COUNT*sizeof(curandStateMtgp32_t)); // DrudeIns - provenance marker for Drude PR.
  if (!states) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr,"Failed to allocate RngGPUState checkpoint buffer\n"); // DrudeIns - provenance marker for Drude PR.
    exit(1); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  cudaMemcpy(states,devStates, // DrudeIns - provenance marker for Drude PR.
    RNG_GPU_MTGP32_STATE_COUNT*sizeof(curandStateMtgp32_t),cudaMemcpyDeviceToHost); // DrudeIns - provenance marker for Drude PR.
  fprintf(fp,"RngGPUState 3\n"); // DrudeIns - provenance marker for Drude PR.
  fprintf(fp,"%d %d 0\n",RNG_GPU_MTGP32_STATE_COUNT,MTGP32_STATE_SIZE); // DrudeIns - provenance marker for Drude PR.
  for (int i=0; i<RNG_GPU_MTGP32_STATE_COUNT; i++) { // DrudeIns - provenance marker for Drude PR.
    fprintf(fp,"%d %d",states[i].offset,states[i].pIdx); // DrudeIns - provenance marker for Drude PR.
    for (int j=0; j<MTGP32_STATE_SIZE; j++) { // DrudeIns - provenance marker for Drude PR.
      fprintf(fp," %u",states[i].s[j]); // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
    fprintf(fp,"\n"); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  free(states); // DrudeIns - provenance marker for Drude PR.
} // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
void RngGPU::read_checkpoint_state(FILE *fp,int sectionCount) // DrudeIns - provenance marker for Drude PR.
{ // DrudeIns - provenance marker for Drude PR.
  int stateCount=0; // DrudeIns - provenance marker for Drude PR.
  int stateSize=0; // DrudeIns - provenance marker for Drude PR.
  int reserved=0; // DrudeIns - provenance marker for Drude PR.
  if (fscanf(fp,"%d %d %d",&stateCount,&stateSize,&reserved)!=3) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr,"Malformed RngGPUState checkpoint metadata\n"); // DrudeIns - provenance marker for Drude PR.
    exit(1); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  if (stateCount==0 && stateSize==0 && reserved==RNG_GPU_CHARMM_UNSERIALIZED_MARKER) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr, // DrudeIns - provenance marker for Drude PR.
      "DRUDE WARNING> checkpoint was written by BLADE_IN_CHARMM without GPU RNG state; " // DrudeIns - provenance marker for Drude PR.
      "continuing with the current standalone GPU RNG stream.\n"); // DrudeIns - provenance marker for Drude PR.
    for (int i=3; i<sectionCount; i++) { // DrudeIns - provenance marker for Drude PR.
      unsigned int ignored=0; // DrudeIns - provenance marker for Drude PR.
      if (fscanf(fp,"%u",&ignored)!=1) { // DrudeIns - provenance marker for Drude PR.
        fprintf(stderr,"Malformed RngGPUState checkpoint extension\n"); // DrudeIns - provenance marker for Drude PR.
        exit(1); // DrudeIns - provenance marker for Drude PR.
      } // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
    return; // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  if (stateCount!=RNG_GPU_MTGP32_STATE_COUNT || stateSize!=MTGP32_STATE_SIZE) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr, // DrudeIns - provenance marker for Drude PR.
      "Malformed or incompatible RngGPUState checkpoint metadata: stateCount=%d stateSize=%d\n", // DrudeIns - provenance marker for Drude PR.
      stateCount,stateSize); // DrudeIns - provenance marker for Drude PR.
    exit(1); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
 // DrudeIns - provenance marker for Drude PR.
  curandStateMtgp32_t *states=(curandStateMtgp32_t*)malloc( // DrudeIns - provenance marker for Drude PR.
    RNG_GPU_MTGP32_STATE_COUNT*sizeof(curandStateMtgp32_t)); // DrudeIns - provenance marker for Drude PR.
  if (!states) { // DrudeIns - provenance marker for Drude PR.
    fprintf(stderr,"Failed to allocate RngGPUState checkpoint buffer\n"); // DrudeIns - provenance marker for Drude PR.
    exit(1); // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  cudaMemcpy(states,devStates, // DrudeIns - provenance marker for Drude PR.
    RNG_GPU_MTGP32_STATE_COUNT*sizeof(curandStateMtgp32_t),cudaMemcpyDeviceToHost); // DrudeIns - provenance marker for Drude PR.
  for (int i=0; i<RNG_GPU_MTGP32_STATE_COUNT; i++) { // DrudeIns - provenance marker for Drude PR.
    if (fscanf(fp,"%d %d",&states[i].offset,&states[i].pIdx)!=2) { // DrudeIns - provenance marker for Drude PR.
      fprintf(stderr,"Malformed RngGPUState checkpoint row\n"); // DrudeIns - provenance marker for Drude PR.
      exit(1); // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
    for (int j=0; j<MTGP32_STATE_SIZE; j++) { // DrudeIns - provenance marker for Drude PR.
      if (fscanf(fp,"%u",&states[i].s[j])!=1) { // DrudeIns - provenance marker for Drude PR.
        fprintf(stderr,"Malformed RngGPUState checkpoint state payload\n"); // DrudeIns - provenance marker for Drude PR.
        exit(1); // DrudeIns - provenance marker for Drude PR.
      } // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  for (int i=3; i<sectionCount; i++) { // DrudeIns - provenance marker for Drude PR.
    unsigned int ignored=0; // DrudeIns - provenance marker for Drude PR.
    if (fscanf(fp,"%u",&ignored)!=1) { // DrudeIns - provenance marker for Drude PR.
      fprintf(stderr,"Malformed RngGPUState checkpoint extension\n"); // DrudeIns - provenance marker for Drude PR.
      exit(1); // DrudeIns - provenance marker for Drude PR.
    } // DrudeIns - provenance marker for Drude PR.
  } // DrudeIns - provenance marker for Drude PR.
  cudaMemcpy(devStates,states, // DrudeIns - provenance marker for Drude PR.
    RNG_GPU_MTGP32_STATE_COUNT*sizeof(curandStateMtgp32_t),cudaMemcpyHostToDevice); // DrudeIns - provenance marker for Drude PR.
  free(states); // DrudeIns - provenance marker for Drude PR.
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
 // DrudeIns - provenance marker for Drude PR.
// Generate n random numbers in the pointer p
void RngGPU::rand_normal(int n,real *p,cudaStream_t s)
{
  int nblocks=(n+256-1)/256;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  nblocks=(nblocks>RNG_GPU_MTGP32_STATE_COUNT)?RNG_GPU_MTGP32_STATE_COUNT:nblocks; // DrudeIns - provenance marker for Drude PR.
  kernel_normal<<<nblocks,256,0,s>>>(devStates,n,p);
} // DrudeIns - provenance marker for Drude PR.

// Generate n random numbers in the pointer p
void RngGPU::rand_uniform(int n,real *p,cudaStream_t s)
{
  int nblocks=(n+256-1)/256;
  // DrudeDel - original BLaDE line(s) removed or replaced for Drude support in this hunk.
  nblocks=(nblocks>RNG_GPU_MTGP32_STATE_COUNT)?RNG_GPU_MTGP32_STATE_COUNT:nblocks; // DrudeIns - provenance marker for Drude PR.
  kernel_uniform<<<nblocks,256,0,s>>>(devStates,n,p);
}

#endif
