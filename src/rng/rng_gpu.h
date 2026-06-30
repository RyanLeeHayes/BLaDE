#ifndef RNG_RNG_GPU_H
#define RNG_RNG_GPU_H

// If this is a CHARMM compilation
#ifdef BLADE_IN_CHARMM

#include <curand.h>
#include <stdio.h> // DrudeIns - provenance marker for Drude PR.
#include "main/defines.h"

// See
// https://docs.nvidia.com/cuda/curand
// for details

class RngGPU
{
private:
  curandGenerator_t gen;
  cudaStream_t rngStream;
  unsigned long long currentSeed; // DrudeIns - provenance marker for Drude PR.
  
public:

  RngGPU(); // (unnsigned long long seed)
  ~RngGPU();

  void set_seed(unsigned long long seed); // DrudeIns - provenance marker for Drude PR.
  void write_checkpoint_state(FILE *fp) const; // DrudeIns - provenance marker for Drude PR.
  void read_checkpoint_state(FILE *fp,int sectionCount); // DrudeIns - provenance marker for Drude PR.

  // Generate n random numbers in the pointer p
  void rand_normal(int n,real *p,cudaStream_t s);
  void rand_uniform(int n,real *p,cudaStream_t s);
};

#else

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <stdlib.h> // DrudeIns - provenance marker for Drude PR.
#include <stdio.h> // DrudeIns - provenance marker for Drude PR.
#include <time.h>

#include "main/defines.h"

// See https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview for details

class RngGPU
{
  public:
  curandStateMtgp32_t *devStates;
  mtgp32_kernel_params_t *devParams;
  unsigned long long currentSeed; // DrudeIns - provenance marker for Drude PR.

  RngGPU() // (unnsigned long long seed)
  {
    setup();
  }

  ~RngGPU()
  {
    cudaFree(devStates);
    cudaFree(devParams);
  }

  void setup();
  void set_seed(unsigned long long seed); // DrudeIns - provenance marker for Drude PR.
  void write_checkpoint_state(FILE *fp) const; // DrudeIns - provenance marker for Drude PR.
  void read_checkpoint_state(FILE *fp,int sectionCount); // DrudeIns - provenance marker for Drude PR.
  // Generate n random numbers in the pointer p
  void rand_normal(int n,real *p,cudaStream_t s);
  void rand_uniform(int n,real *p,cudaStream_t s);
};

#endif

#endif
