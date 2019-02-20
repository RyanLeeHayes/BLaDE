#include <math.h>
#include <stdlib.h>

#include "system/system.h"
#include "update/update.h"

void Update::initialize(System *system)
{
  int i,j;

  if (system->update->leapParms1) free(system->update->leapParms1);
  if (system->update->leapParms2) free(system->update->leapParms2);
  if (system->update->leapState) free(system->update->leapState);

  system->update->leapParms1=alloc_leapparms1(system->run->dt,system->run->gamma,system->run->T);
  system->update->leapParms2=alloc_leapparms2(system->run->dt,system->run->gamma,system->run->T);
  system->update->leapState=alloc_leapstate(system);

  for (i=0; i<system->state->atomCount; i++) {
    for (j=0; j<3; j++) {
      system->state->mass[i][j]=system->structure->mass[i];
      system->state->invsqrtMass[i][j]=1.0/sqrt(system->state->mass[i][j]);
    }
  }

  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP>>>(*leapState);
  system->state->send_position();
  system->state->send_velocity();
  system->state->send_invsqrtMass();

  cudaStreamCreate(&updateStream);
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
#warning "Constrain velocities here"

  // equation 7c&e
  update_R<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState,*leapParms2);

  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP,0,updateStream>>>(*leapState);
  cudaStreamEndCapture(updateStream,&updateGraph);
  cudaGraphInstantiate(&updateGraphExec,updateGraph,NULL,NULL,0);
#endif
}

void Update::update(int step,System *system)
{
#ifndef CUDAGRAPH
  // https://pubs.acs.org/doi/10.1021/jp411770f equation 7

  // Get Gaussian distributed random numbers
  system->state->rngGPU->rand_normal(2*leapState->N,leapState->random,updateStream);

  // equation 7f&g - after force calculation
  // KERNEL
  update_VO<<<(leapState->N+BLUP-1)/BLUP,BLUP>>>(*leapState,*leapParms2);
  // grab velocity if you want it here, but apply bond constriants...
  // equation 7a&b - after force calculation
  // KERNEL
  update_OV<<<(leapState->N+BLUP-1)/BLUP,BLUP>>>(*leapState,*leapParms2);
#warning "Constrain velocities here"

  // equation 7c&e
  update_R<<<(leapState->N+BLUP-1)/BLUP,BLUP>>>(*leapState,*leapParms2);

  reset_F<<<(leapState->N+BLUP-1)/BLUP,BLUP>>>(*leapState);
#else
  cudaGraphLaunch(updateGraphExec,updateStream);
#endif
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
    ls.v[i]=lp.sqrta*ls.v[i]+lp.noise*ls.random[i];
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

#warning "Better served by more general function"
__global__ void reset_F(struct LeapState ls)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  
  if (i < ls.N) {
    ls.f[i]=0;
  }
}
/*
void Update::calcABCD(real slambda,real* A,real* B,real* C,real* D)
{
  real sign,shlambda;
  sign = (slambda>0 ? 1.0 : -1.0);

  if ((slambda/sign)>0.12) {
    // V noise variance
    A[0]=sign*(exp(slambda)-1);
    // related to conditional variance
    B[0]=slambda*(exp(slambda)-1)-4*(exp(slambda/2)-1)*(exp(slambda/2)-1);
    // X noise variance
    C[0]=sign*(slambda-3+4*exp(-slambda/2)-exp(-slambda));
    // VX noise correlation
    D[0]=sign*(2-exp(slambda/2)-exp(-slambda/2));
  } else {
    shlambda=slambda/2;
    // Max As error: 2e-14 at shlambda=0.025; error(0.1)=3e-10
    // As=sign*(2*shlambda+2*shlambda^2+(4./3)*shlambda^3+(2./3)*shlambda^4+(4./15)*shlambda^5+(4./45)*shlambda^6+(8./315)*shlambda^7);
    A[0]=(8./315); // shl^7
    A[0]=(4./45)+A[0]*shlambda; // shl^6
    A[0]=(4./15)+A[0]*shlambda; // shl^5
    A[0]=(2./3)+A[0]*shlambda; // shl^4
    A[0]=(4./3)+A[0]*shlambda; // shl^3
    A[0]=2+A[0]*shlambda; // shl^2
    A[0]=2+A[0]*shlambda; // shl
    A[0]=sign*A[0]*shlambda;
    // Max Bs error: 2e-11 at shlambda=0.08; error(0.1)=1e-10
    // Bs=(1./3)*shlambda^4+(1./3)*shlambda^5+(17./90)*shlambda^6+(7./90)*shlambda^7+(43./1680)*shlambda^8+(107./15120)*shlambda^9+(769./453600)*shlambda^10;
    B[0]=(769./453600); // shl^10;
    B[0]=(107./15120)+B[0]*shlambda; // shl^9
    B[0]=(43./1680)+B[0]*shlambda; // shl^8
    B[0]=(7./90)+B[0]*shlambda; // shl^7
    B[0]=(17./90)+B[0]*shlambda; // shl^6
    B[0]=(1./3)+B[0]*shlambda; // shl^5
    B[0]=(1./3)+B[0]*shlambda; // shl^4
    B[0]=B[0]*shlambda*shlambda*shlambda*shlambda;
    // Max Ds error: 2e-14 at shlambda=0.12; error(0.1)=5e-14
    // Ds=sign*(-shlambda^2-(1./12)*shlambda^4-(1./360)*shlambda^6-(1./20160)*shlambda^8);
    D[0]=-(1./20160); // shl^8
    D[0]=-(1./360)+D[0]*shlambda*shlambda; // shl^6
    D[0]=-(1./12)+D[0]*shlambda*shlambda; // shl^4
    D[0]=-1+D[0]*shlambda*shlambda; // shl^2
    D[0]=sign*D[0]*shlambda*shlambda;
    // Best combined accuracy 1e-11 at shlambda=0.06, from semilogy(shlambda,max(max(abs(Ds1-Ds2)./Ds2,abs(Bs1-Bs2)./Bs2),abs(As1-As2)./As2))
  }
}

struct LeapParms* Update::alloc_leapparms(real dt,real gamma,real T)
{
  struct LeapParms *lp;
  real lambda;
  real As, Bs, Cs, Ds;
  real Ar, Br, Cr, Dr;
  real kT=kB*T;
  real mDiff=kT/gamma;

  lp=(struct LeapParms*) malloc(sizeof(struct LeapParms));

  // integrator from W. F. Van Gunsteren and H. J. C. Berendsen (1988)
  lp->dt=dt;
  lp->gamma=gamma;
  lp->kT=kT;
  // lp->mDiff=kT/gamma; // Diffusion * Mass
  mDiff=kT/gamma; // Diffusion * Mass
  lambda=dt*gamma;
  // [t to t+dt/2]
  // Ar prop svv^2; Br=Ar*Cr-Dr^2; Cr prop sxx^2; Dr prop sxv^2
  calcABCD(lambda,&As,&Bs,&Cs,&Ds);
  // Eq 3.19
  lp->SigmaVsVs=sqrt(mDiff*gamma*As);
  lp->XsVsCoeff=Ds/(gamma*As); // Equal to SigmaXsVs^2/SigmaVsVs^2
  lp->CondSigmaXsXs=sqrt((mDiff/gamma)*Bs/As); // Equal to sqrt(SigmaXsXs^2-SigmaXsVs^4/SigmaVsVs^2)
  // [t+dt/2 to t+dt]
  calcABCD(-lambda,&Ar,&Br,&Cr,&Dr);
  lp->SigmaVrVr=sqrt(mDiff*gamma*Ar);
  lp->XrVrCoeff=Dr/(gamma*Ar); // Equal to SigmaXrVr^2/SigmaVrVr^2
  lp->CondSigmaXrXr=sqrt((mDiff/gamma)*Br/Ar); // Equal to sqrt(SigmaXrXr^2-SigmaXrVr^4/SigmaVrVr^2)

  // // Made up
  // lp->vhalf_vscale=exp(-lambda/2);
  // lp->vhalf_ascale=dt*(1-exp(-lambda/2))/lambda;
  // lp->vhalf_Vsscale=-exp(-lambda/2);

  // Eq 3.6
  lp->v_vscale=exp(-lambda);
  lp->v_ascale=dt*Ar/lambda;
  lp->v_Vsscale=-exp(-lambda);

  // Eq 3.10
  lp->x_vscale=dt*exp(lambda/2)*Ar/lambda;
  lp->x_Xrscale=-1;

  return lp;
}
*/

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
