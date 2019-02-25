#include <math.h>
#include <stdlib.h>

#include "update/update.h"
#include "system/system.h"
#include "system/state.h"
#include "msld/msld.h"
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

  lambdaLeapParms1=NULL;
  lambdaLeapParms2=NULL;
  lambdaLeapState=NULL;

#ifdef PROFILESERIAL
  updateStream=0;
  updateLambdaStream=0;
#else
  cudaStreamCreate(&updateStream);
  cudaStreamCreate(&updateLambdaStream);
#endif
  cudaEventCreate(&updateComplete);
  cudaEventCreate(&updateLambdaComplete);
}

Update::~Update()
{
  if (leapParms1) free(leapParms1);
  if (leapParms2) free(leapParms2);
  if (leapState) free(leapState);

  if (lambdaLeapParms1) free(lambdaLeapParms1);
  if (lambdaLeapParms2) free(lambdaLeapParms2);
  if (lambdaLeapState) free(lambdaLeapState);

#ifndef PROFILESERIAL
  cudaStreamDestroy(updateStream);
  cudaStreamDestroy(updateLambdaStream);
#endif
  cudaEventDestroy(updateComplete);
  cudaEventDestroy(updateLambdaComplete);
}



void Update::initialize(System *system)
{
  if (leapParms1) free(leapParms1);
  if (leapParms2) free(leapParms2);
  if (leapState) free(leapState);

  leapParms1=alloc_leapparms1(system->run->dt,system->run->gamma,system->run->T);
  leapParms2=alloc_leapparms2(system->run->dt,system->run->gamma,system->run->T);
  leapState=alloc_leapstate(
    3*system->state->atomCount,
    (real*)system->state->position_d,
    (real*)system->state->velocity_d,
    (real*)system->state->force_d,
    (real*)system->state->invsqrtMass_d,
    (real*)system->state->random_d);

  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP>>>(*leapState);
  system->state->send_position();
  system->state->send_velocity();
  system->state->send_invsqrtMass();

  if (lambdaLeapParms1) free(lambdaLeapParms1);
  if (lambdaLeapParms2) free(lambdaLeapParms2);
  if (lambdaLeapState) free(lambdaLeapState);

  lambdaLeapParms1=alloc_leapparms1(system->run->dt,system->msld->gamma,system->run->T);
  lambdaLeapParms2=alloc_leapparms2(system->run->dt,system->msld->gamma,system->run->T);
  lambdaLeapState=alloc_leapstate(
    system->msld->blockCount,
    system->msld->theta_d,
    system->msld->thetaVelocity_d,
    system->msld->thetaForce_d,
    system->msld->thetaInvsqrtMass_d,
    system->msld->thetaRandom_d);

  reset_F<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP>>>(*lambdaLeapState);
  system->msld->send_real(system->msld->theta_d,system->msld->theta);
  system->msld->send_real(system->msld->thetaVelocity_d,system->msld->thetaVelocity);
  system->msld->send_real(system->msld->thetaInvsqrtMass_d,system->msld->thetaInvsqrtMass);

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

void Update::update(int step,System *system)
{
  cudaStreamWaitEvent(updateStream,system->run->forceComplete,0);
  cudaStreamWaitEvent(updateLambdaStream,system->run->forceComplete,0);
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
  cudaEventRecord(updateComplete,updateStream);
  cudaStreamWaitEvent(system->run->masterStream,updateComplete,0);

#ifndef CUDAGRAPH
  system->state->rngGPU->rand_normal(2*lambdaLeapState->N,lambdaLeapState->random,updateLambdaStream);
  system->msld->calc_thetaForce_from_lambdaForce(updateLambdaStream);
  update_VO<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState,*lambdaLeapParms2);
  update_OV<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState,*lambdaLeapParms2);
  update_R<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState,*lambdaLeapParms2);
  system->msld->calc_lambda_from_theta(updateLambdaStream);

  reset_F<<<(lambdaLeapState->N+BLUP-1)/BLUP,BLUP,0,updateLambdaStream>>>(*lambdaLeapState);
#else
  cudaGraphLaunch(updateLambdaGraphExec,updateLambdaStream);
#endif

  cudaEventRecord(updateLambdaComplete,updateLambdaStream);
  cudaStreamWaitEvent(system->run->masterStream,updateLambdaComplete,0);

  cudaEventRecord(system->run->updateComplete,system->run->masterStream);
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

struct LeapState* Update::alloc_leapstate(int N,real *x,real *v,real *f,real *ism,real *random)
{
  struct LeapState *ls;

  ls=(struct LeapState*) malloc(sizeof(struct LeapState));

  ls->N=N;
  ls->x=x;
  ls->v=v;
  ls->f=f;
  ls->ism=ism;
  ls->random=random;
  return ls;
}
