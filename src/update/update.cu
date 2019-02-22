#include <math.h>
#include <stdlib.h>

#include "update/update.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/structure.h"
#include "system/potential.h"
#include "rng/rng_gpu.h"



// Class constructors
Update::Update()
{
  leapParms1=NULL;
  leapParms2=NULL;
  leapState=NULL;
}

Update::~Update()
{
  if (leapParms1) free(leapParms1);
  if (leapParms2) free(leapParms2);
  if (leapState) free(leapState);
}



void Update::initialize(System *system)
{
  if (system->update->leapParms1) free(system->update->leapParms1);
  if (system->update->leapParms2) free(system->update->leapParms2);
  if (system->update->leapState) free(system->update->leapState);

  system->update->leapParms1=alloc_leapparms1(system->run->dt,system->run->gamma,system->run->T);
  system->update->leapParms2=alloc_leapparms2(system->run->dt,system->run->gamma,system->run->T);
  system->update->leapState=alloc_leapstate(system);

  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP>>>(*leapState);
  system->state->send_position();
  system->state->send_velocity();
  system->state->send_invsqrtMass();

  cudaStreamCreate(&updateStream);
  // cudaEventCreate(&updateComplete);
#ifdef CUDAGRAPH
  cudaStreamBeginCapture(updateStream);
  // https://pubs.acs.org/doi/10.1021/jp411770f equation 7

  // Get Gaussian distributed random numbers
  system->state->rngGPU->rand_normal(2*leapState->N,leapState->random,updateStream);

  // equation 7f&g - after force calculation
  // KERNEL
  update_VO<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);
  // grab velocity if you want it here, but apply bond constriants...
  // equation 7a&b - after force calculation
  // KERNEL
  update_OV<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);
// NYI constrain velocities here

  // equation 7c&e
  update_R<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);

  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState);
  cudaStreamEndCapture(updateStream,&updateGraph);
  cudaGraphInstantiate(&updateGraphExec,updateGraph,NULL,NULL,0);
#endif
}

void Update::update(int step,System *system)
{
  // cudaStreamWaitEvent(updateStream,system->potential->forceComplete,0);
#ifndef CUDAGRAPH
  // https://pubs.acs.org/doi/10.1021/jp411770f equation 7

  // Get Gaussian distributed random numbers
  system->state->rngGPU->rand_normal(2*leapState->N,leapState->random,updateStream);

  // equation 7f&g - after force calculation
  // KERNEL
  update_VO<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);
  // grab velocity if you want it here, but apply bond constriants...
  // equation 7a&b - after force calculation
  // KERNEL
  update_OV<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);
// NYI constrain velocities here

  // equation 7c&e
  update_R<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);

  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState);
#else
  cudaGraphLaunch(updateGraphExec,updateStream);
#endif
  // cudaEventRecord(updateComplete,updateStream);
  // cudaStreamWaitEvent(system->potential->bondedStream[0],updateComplete,0);
}

void Update::finalize()
{
#ifdef CUDAGRAPH
  cudaGraphExecDestroy(updateGraphExec);
  cudaGraphDestroy(updateGraph);
#endif
}

__global__ void update_VO(struct LeapState ls,struct LeapParms2 lp)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i < ls.N) {
    // Force is dU/dx by convention in this program, not -dU/dx
    ls.v[i]=ls.v[i]-lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
    ls.v[i]=lp.sqrta*ls.v[i]+lp.noise*ls.random[i];
  }
}

__global__ void update_OV(struct LeapState ls,struct LeapParms2 lp)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i < ls.N) {
    // Force is dU/dx by convention in this program, not -dU/dx
    ls.v[i]=lp.sqrta*ls.v[i]+lp.noise*ls.random[ls.N+i];
    ls.v[i]=ls.v[i]-lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
  }
}

__global__ void update_R(struct LeapState ls,struct LeapParms2 lp)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i < ls.N) {
    ls.x[i]=ls.x[i]+2*lp.fscale*ls.v[i]; // hamiltonian changes half way through this update
  }
}

__global__ void reset_F(struct LeapState ls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  
  if (i < ls.N) {
    ls.f[i]=0;
  }
}

struct LeapParms1* Update::alloc_leapparms1(real dt,real gamma,real T)
{
  struct LeapParms1 *lp;
  real kT=kB*T;

  lp=(struct LeapParms1*) malloc(sizeof(struct LeapParms1));

  // Integrator from https://pubs.acs.org/doi/10.1021/jp411770f
  lp->dt=dt;
  lp->gamma=gamma;
  lp->kT=kT;

  return lp;
}

struct LeapParms2* Update::alloc_leapparms2(real dt,real gamma,real T)
{
  struct LeapParms2 *lp;
  real kT=kB*T;
  real a=exp(-gamma*dt);
  real b=sqrt(tanh(0.5*gamma*dt)/(0.5*gamma*dt));

  lp=(struct LeapParms2*) malloc(sizeof(struct LeapParms2));

  // Integrator from https://pubs.acs.org/doi/10.1021/jp411770f
  lp->sqrta=sqrt(a);
  lp->noise=sqrt((1-a)*kT);
  lp->fscale=0.5*b*dt;

  return lp;
}

struct LeapState* Update::alloc_leapstate(System *system)
{
  struct LeapState *ls;

  ls=(struct LeapState*) malloc(sizeof(struct LeapState));

  ls->N=3*system->state->atomCount;
  ls->x=(real*)system->state->position_d;
  ls->v=(real*)system->state->velocity_d;
  ls->f=(real*)system->state->force_d;
  ls->ism=(real*)system->state->invsqrtMass_d;
  ls->random=(real*)system->state->random_d;
  return ls;
}
