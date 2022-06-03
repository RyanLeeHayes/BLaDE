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
// http://dx.doi.org/10.1098/rspa.2016.0138

// Molecular dynamics simulations of water and biomolecules with a Monte Carlo constant pressure algorithm
// Johan A...vist *, Petra Wennerstro...m, Martin Nervall, Sinisa Bjelic, Bj...rn O. Brandsdal

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
    if (isfinite(ls.ism[i])) {
      // Force is dU/dx by convention in this program, not -dU/dx
      ls.v[i]=ls.v[i]-lp.halfdt*ls.ism[i]*ls.ism[i]*ls.f[i];
      // if (!(ls.v[i] < 100 && ls.v[i] > -100)) printf("Crashing V i=%d, v=%f, f=%f, x=%f\n",i,ls.v[i],ls.f[i],ls.x[i]); // DEBUG
    }
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
    if (isfinite(ls.ism[i])) {
      // Force is dU/dx by convention in this program, not -dU/dx
      real_v v=ls.v[i]-lp.halfdt*ls.ism[i]*ls.ism[i]*ls.f[i];
      ls.v[i]=v-lp.halfdt*ls.ism[i]*ls.ism[i]*ls.f[i];
      // if (!(ls.v[i] < 100 && ls.v[i] > -100)) printf("Crashing VV i=%d, v=%f, f=%f, x=%f\n",i,ls.v[i],ls.f[i],ls.x[i]); // DEBUG
    }
  }
}

__global__ void update_VhbpR(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2,real_x *bx)
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
    real_v v=ls.v[i];
    real_x x=ls.x[i];
    v-=lp.halfdt*ls.ism[i]*ls.ism[i]*ls.f[i];
    if (bx) bx[i]=x;
    if (isfinite(ls.ism[i])) {
      x+=lp.halfdt*v;
      ls.v[i]=v;
      // if (!(ls.v[i] < 100 && ls.v[i] > -100)) printf("Crashing VhbpR i=%d, v=%f, f=%f, x=%f\n",i,ls.v[i],ls.f[i],ls.x[i]); // DEBUG
      ls.x[i]=x;
    }
  }
}

__global__ void update_VVhbpR(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2,real_x *bx)
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
    real_v v=ls.v[i];
    real_x x=ls.x[i];
    v-=2*lp.halfdt*ls.ism[i]*ls.ism[i]*ls.f[i];
    if (bx) bx[i]=x;
    if (isfinite(ls.ism[i])) {
      x+=lp.halfdt*v;
      ls.v[i]=v;
      // if (!(ls.v[i] < 100 && ls.v[i] > -100)) printf("Crashing VVhbpR i=%d, v=%f, f=%f, x=%f\n",i,ls.v[i],ls.f[i],ls.x[i]); // DEBUG
      ls.x[i]=x;
    }
  }
}

__global__ void kinetic_energy_kernel(struct LeapState ls,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  if (i<ls.N) {
    if (energy) {
      if (isfinite(ls.ism[i])) {
        lEnergy=ls.v[i]/ls.ism[i];
        lEnergy*=((real)0.5)*lEnergy;
      }
    }
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

__global__ void update_hbpR(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2,real_x *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    real_x x=ls.x[i];
    if (bx) bx[i]=x;
    if (isfinite(ls.ism[i])) {
      // if (!(ls.v[i] < 100 && ls.v[i] > -100)) printf("Crashing hbpR i=%d, v=%f, f=%f, x=%f\n",i,ls.v[i],ls.f[i],ls.x[i]); // DEBUG
      ls.x[i]=x+lp.halfdt*ls.v[i];
    }
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
    if (isfinite(ls.ism[i])) {
      ls.v[i]=lp.friction*ls.v[i]+lp.noise*ls.ism[i]*ls.random[i];
      // if (!(ls.v[i] < 100 && ls.v[i] > -100)) printf("Crashing OO i=%d, v=%f, r=%f, x=%f\n",i,ls.v[i],ls.random[i],ls.x[i]); // DEBUG
    }
  }
}

__global__ void update_OOhbpR(struct LeapState ls,struct LeapParms2 lp1,struct LeapParms2 lp2,real_x *bx)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct LeapParms2 lp;

  if (i < ls.N1) {
    lp=lp1;
  } else {
    lp=lp2;
  }

  if (i < ls.N) {
    real_v v=ls.v[i];
    real_x x=ls.x[i];
    v=lp.friction*v+lp.noise*ls.ism[i]*ls.random[i];
    if (bx) bx[i]=x;
    if (isfinite(ls.ism[i])) {
      x+=lp.halfdt*v;
      ls.v[i]=v;
      // if (!(ls.v[i] < 100 && ls.v[i] > -100)) printf("Crashing OOhbpR i=%d, v=%f, r=%f, x=%f\n",i,ls.v[i],ls.random[i],ls.x[i]); // DEBUG
      ls.x[i]=x;
    }
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

  // cudaStreamWaitEvent(r->updateStream,r->forceComplete,0);
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
    // holonomic_velocity(system);
    // update_hbpR<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2,positionCons_d);
    update_VhbpR<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2,positionCons_d);
  } else {
    // Update V from previous step and for current step
    // update_VV<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2);
    // holonomic_velocity(system);
    // update_hbpR<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2,positionCons_d);
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
  // holonomic_velocity(system); // superfluous, I think
  // update_hbpR<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(*leapState,*leapParms2,*lambdaLeapParms2,positionCons_d);
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


__global__ void set_fd_kernel(int N,real *p_fd,real_x *p_d)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N) {
    p_fd[i]=p_d[i];
  }
}

void State::set_fd(System *system)
{
  int N=3*atomCount+2*lambdaCount;

  // Call elsewhere
  // orthBox_f.x=orthBox.x;
  // orthBox_f.y=orthBox.y;
  // orthBox_f.z=orthBox.z;

  if ((void*)positionBuffer_fd != (void*)positionBuffer_d) {
    set_fd_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(N,positionBuffer_fd,positionBuffer_d);
  }
}


void State::kinetic_energy(System *system)
{
  if (system->id==0) {
    kinetic_energy_kernel<<<(leapState->N+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32,0>>>(*leapState,energy_d+eekinetic);
  }
}
