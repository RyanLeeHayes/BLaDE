#ifndef UPDATE_UPDATE_H
#define UPDATE_UPDATE_H

// See https://aip.scitation.org/doi/abs/10.1063/1.1420460 for barostat
// Nevermind, this barostat is better:
// dx.doi.org/10.1016/j.cplett.2003.12.039
//
// See also for integrator/bond constraints/thermostat
// https://pubs.acs.org/doi/10.1021/jp411770f

#include <cuda_runtime.h>

#include "main/defines.h"

// Forward declarations
class System;

struct LeapParms1
{
  real dt;
  real gamma;
  real kT;
};

struct LeapParms2
{
  real sqrta; // a=exp(-gamma*dt)
  real noise; // sqrt((1-a)*kT) - still need to divide by sqrt(m)
  real fscale; // 0.5*b*dt; b=sqrt(tanh(0.5*gamma*dt)/(0.5*gamma*dt));
};

struct LeapState
{
  int N;
  real *x;
  real *v;
  real *f;
  real *ism; // 1/sqrt(m)
  real *random;
};

class Update {
  public:
  struct LeapParms1 *leapParms1;
  struct LeapParms2 *leapParms2;
  struct LeapState *leapState;

  cudaStream_t updateStream;
  cudaEvent_t updateComplete;
#ifdef CUDAGRAPH
  cudaGraph_t updateGraph;
  cudaGraphExec_t updateGraphExec;
#endif

  Update();
  ~Update();

  void initialize(System *system);
  void update(int step,System *system);
  void finalize();

  // void calcABCD(real slambda,real* A,real* B,real* C,real* D);
  struct LeapParms1* alloc_leapparms1(real dt,real gamma,real T);
  struct LeapParms2* alloc_leapparms2(real dt,real gamma,real T);
  struct LeapState* alloc_leapstate(System *system);
};

__global__ void update_VO(struct LeapState ls,struct LeapParms2 lp);
__global__ void update_OV(struct LeapState ls,struct LeapParms2 lp);
__global__ void update_R(struct LeapState ls,struct LeapParms2 lp);
__global__ void reset_F(struct LeapState ls);

#endif
