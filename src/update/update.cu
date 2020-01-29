#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "system/system.h"
#include "system/state.h"
#include "msld/msld.h"
#include "run/run.h"
#include "system/potential.h"
#include "rng/rng_gpu.h"
#include "holonomic/holonomic.h"
#include "update/pressure.h"
#include "update/rex.h"

#include "main/real3.h"



// See https://aip.scitation.org/doi/abs/10.1063/1.1420460 for barostat
// Nevermind, this barostat is better:
// dx.doi.org/10.1016/j.cplett.2003.12.039
//
// See also for integrator/bond constraints/thermostat
// https://pubs.acs.org/doi/10.1021/jp411770f

// Molecular dynamics simulations of water and biomolecules with a Monte Carlo constant pressure algorithm
// Johan A...vist *, Petra Wennerstro...m, Martin Nervall, Sinisa Bjelic, Bj...rn O. Brandsdal

/*
// Declarations for header
#ifdef CUDAGRAPH
  cudaGraph_t updateGraph, updateLambdaGraph;
  cudaGraphExec_t updateGraphExec, updateLambdaGraphExec;
#endif
void State::initialize(System *system)
{
#ifdef CUDAGRAPH
  cudaStreamBeginCapture(updateStream);
  system->state->rngGPU->rand_normal(2*leapState->N,leapState->random,updateStream);
  update_VO<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);
  update_OV<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);
  update_R<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);
  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState);
  cudaStreamEndCapture(updateStream,&updateGraph);
  cudaGraphInstantiate(&updateGraphExec,updateGraph,NULL,NULL,0);

  cudaStreamBeginCapture(updateLambdaStream);
  system->state->rngGPU->rand_normal(2*lambdaLeapState->N,lambdaLeapState->random,updateLambdaStream);
  system->msld->calc_thetaForce_from_lambdaForce(updateLambdaStream);
  update_VO<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState,*lambdaLeapParms2);
  update_OV<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState,*lambdaLeapParms2);
  update_R<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState,*lambdaLeapParms2);
  system->msld->calc_lambda_from_theta(updateLambdaStream);
  reset_F<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState);
  cudaStreamEndCapture(updateLambdaStream,&updateLambdaGraph);
  cudaGraphInstantiate(&updateLambdaGraphExec,updateLambdaGraph,NULL,NULL,0);
#endif
}
// Calling the graph
  cudaGraphLaunch(updateGraphExec,r->updateStream);
// Cleaning up the graphs
#ifdef CUDAGRAPH
  cudaGraphExecDestroy(updateGraphExec);
  cudaGraphDestroy(updateGraph);
#endif
*/

__global__ void update_V(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    // Force is dU/dx by convention in this program, not -dU/dx
    ls.v[i]=ls.v[i]-lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
  }
}

__global__ void update_VV(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    // Force is dU/dx by convention in this program, not -dU/dx
    real v=ls.v[i]-lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
    ls.v[i]=v-lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
  }
}

__global__ void update_VhbpR(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2,real *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    // Force is dU/dx by convention in this program, not -dU/dx
    real v=ls.v[i];
    real x=ls.x[i];
    v-=lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
    if (bx) bx[i]=x;
    x+=lp.fscale*v;
    ls.v[i]=v;
    ls.x[i]=x;
  }
}

__global__ void update_VVhbpR(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2,real *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    // Force is dU/dx by convention in this program, not -dU/dx
    real v=ls.v[i];
    real x=ls.x[i];
    // v-=lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
    // v-=lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i];
    v-=2*lp.fscale*ls.ism[i]*ls.ism[i]*ls.f[i]; // Let's just do both those lines in one step.
    if (bx) bx[i]=x;
    x+=lp.fscale*v;
    ls.v[i]=v;
    ls.x[i]=x;
  }
}

__global__ void kinetic_energy_kernel(struct LeapState ls,real *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  if (i<ls.N) {
    if (energy) {
      lEnergy=ls.v[i]/ls.ism[i];
      lEnergy*=0.5*lEnergy;
    }
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

__global__ void update_R(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    ls.x[i]=ls.x[i]+lp.fscale*ls.v[i];
  }
}

__global__ void update_OO(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    real v=lp.sqrta*ls.v[i]+lp.noise*ls.ism[i]*ls.random[i];
    // Hamiltonian changes here
    ls.v[i]=lp.sqrta*v+lp.noise*ls.ism[i]*ls.random[ls.N+i];
  }
}

__global__ void update_OOhbpR(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2,real *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    real v=ls.v[i];
    real x=ls.x[i];
    v=lp.sqrta*v+lp.noise*ls.ism[i]*ls.random[i];
    // Hamiltonian changes here
    v=lp.sqrta*v+lp.noise*ls.ism[i]*ls.random[ls.N+i];
    if (bx) bx[i]=x;
    x+=lp.fscale*v;
    ls.v[i]=v;
    ls.x[i]=x;
  }
}



void State::update(int step,System *system)
{
  Run *r=system->run;

  if (system->run->freqNPT>0 && (system->run->step%system->run->freqNPT)==0) {
    pressure_coupling(system);
  }

#ifdef REPLICAEXCHANGE
  if (system->run->freqREx>0 && (system->run->step%system->run->freqREx)==0) {
    replica_exchange(system);
  }
#endif

  cudaStreamWaitEvent(r->updateStream,r->forceComplete,0);
  if (system->id==0) {
  // https://pubs.acs.org/doi/10.1021/jp411770f equation 7
  // Use VRORV
  // https://royalsocietypublishing.org/doi/10.1098/rspa.2016.0138
  // More detailed citation

  // Resolve lambda forces
  system->msld->calc_thetaForce_from_lambdaForce(r->updateStream,system);

  if ((system->run->step%system->run->freqNRG)==0) {
    // Update V from previous step
    update_V<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2);
    // Velocity Constraint
    holonomic_velocity(system);
    // Kinetic Energy
    kinetic_energy_kernel<<<(leapState->N+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,r->updateStream>>>(*leapState,energy_d+eekinetic);
    // Update V for current step
    // update_V<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2);
    update_VhbpR<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2,positionCons_d);
  } else {
    // Update V from previous step and for current step
    // update_VV<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2);
    update_VVhbpR<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2,positionCons_d);
  }
  // Velocity Constraint
  // holonomic_velocity(system); // superfluous, I think
  // Get Gaussian distributed random numbers // done in calc_force now
  // system->rngGPU->rand_normal(2*leapState->N,leapState->random,r->updateStream);
  // Update spatial coordinates
  // holonomic_backup_position(leapState,positionCons_d,r->updateStream); // done in VVhbpR
  // update_R<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2); // done in VVhbpR
  // Position Constraint
  holonomic_position(system);
  // Apply random drift
  // update_OO<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2);
  update_OOhbpR<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2,positionCons_d);
  // Velocity Constraint
  // holonomic_velocity(system); // superfluous, I think
  // Update spatial coordinates
  // holonomic_backup_position(leapState,positionCons_d,r->updateStream); // done in OOhbpR
  // update_R<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2); // done in OOhbpR
  // Position Constraint
  holonomic_position(system);
  // Project lambdas
  system->msld->calc_lambda_from_theta(r->updateStream,system);
  }
}
