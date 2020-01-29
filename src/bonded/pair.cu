#include <cuda_runtime.h>
#include <math.h>

#include "bonded.h"
#include "main/defines.h"
#include "system/system.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

#include "main/real3.h"



#ifdef DOUBLE
#define fasterfc erfc
#else
// Directly from CHARMM source code, charmm/source/domdec_gpu/gpu_utils.h
#warning "From CHARMM, not fully compatible"
static __forceinline__ __device__ float __internal_fmad(float a, float b, float c)
{
#if __CUDA_ARCH__ >= 200
  return __fmaf_rn (a, b, c);
#else // __CUDA_ARCH__ >= 200
  return a * b + c;
#endif // __CUDA_ARCH__ >= 200
}

// Following inline functions are copied from PMEMD CUDA implementation.
// Credit goes to:
/*             Scott Le Grand (NVIDIA)             */
/*               Duncan Poole (NVIDIA)             */
/*                Ross Walker (SDSC)               */
//
// Faster ERFC approximation courtesy of Norbert Juffa. NVIDIA Corporation
static __forceinline__ __device__ float fasterfc(float a)
{
  /* approximate log(erfc(a)) with rel. error < 7e-9 */
  float t, x = a;
  t =                       (float)-1.6488499458192755E-006;
  t = __internal_fmad(t, x, (float)2.9524665006554534E-005);
  t = __internal_fmad(t, x, (float)-2.3341951153749626E-004);
  t = __internal_fmad(t, x, (float)1.0424943374047289E-003);
  t = __internal_fmad(t, x, (float)-2.5501426008983853E-003);
  t = __internal_fmad(t, x, (float)3.1979939710877236E-004);
  t = __internal_fmad(t, x, (float)2.7605379075746249E-002);
  t = __internal_fmad(t, x, (float)-1.4827402067461906E-001);
  t = __internal_fmad(t, x, (float)-9.1844764013203406E-001);
  t = __internal_fmad(t, x, (float)-1.6279070384382459E+000);
  t = t * x;
  return exp2f(t);
}
#endif

// From charmm/source/domdec/enbfix14_kernel.inc:985, vfswitch code (not normal vfswitch??? 1-4s are different)
// r*fpair=6*c6*r^-6 - 12*c12*r^-12                                                 r<rs
// r*fpair=6*c6*r^-6*(rc^3-r^3)/(rc^3-rs^3) - 12*c12*r^-12*(rc^6-r^6)/(rc^6-rs^6)   rs<r<rc
// lE=integral(fpair)
// Use charmm shortcuts to calculate those
__device__ void function_pair(Nb14Potential pp,Cutoffs rc,real r,real *fpair,real *lE,bool calcEnergy)
{
  real rinv=1/r;

  if (r<rc.rCut) {
    real rCut3=rc.rCut*rc.rCut*rc.rCut;
    real rCut6=rCut3*rCut3;
    real rSwitch3=rc.rSwitch*rc.rSwitch*rc.rSwitch;
    real rSwitch6=rSwitch3*rSwitch3;

    real k6=rCut3/(rCut3-rSwitch3);
    real k12=rCut6/(rCut6-rSwitch6);

    real rinv3=rinv*rinv*rinv;
    real rinv6=rinv3*rinv3;

    real A6=(r<rc.rSwitch?1:k6);
    real A12=(r<rc.rSwitch?1:k12);
    real B6=(r<rc.rSwitch?0:1/rCut3);
    real B12=(r<rc.rSwitch?0:1/rCut6);

    fpair[0]=6*pp.c6*A6*(rinv3-B6)*rinv3*rinv-12*pp.c12*A12*(rinv6-B12)*rinv6*rinv;
    if (calcEnergy) {
      real dv6=-1/rCut6;
      real dv12=-1/(rCut6*rCut6);

      real CC6=(r<rc.rSwitch?dv6:0);
      real CC12=(r<rc.rSwitch?dv12:0);
      real rinv3_B6_sq=(rinv3-B6)*(rinv3-B6);
      real rinv6_B12_sq=(rinv6-B12)*(rinv6-B12);

      lE[0]=pp.c12*(A12*rinv6_B12_sq+CC12)-pp.c6*(A6*rinv3_B6_sq+CC6);
    }

    real br=rc.betaEwald*r;
    real kqq=kELECTRIC*pp.qxq;

    real erfcrinv=fasterfc(br)*rinv;
    // fpair[0]+=kqq*(-erfcf(br)*rinv-(2/sqrt(M_PI))*rc.betaEwald*expf(-br*br))*rinv;
    fpair[0]+=kqq*(-erfcrinv-((real)1.128379167095513)*rc.betaEwald*expf(-br*br))*rinv;
    if (calcEnergy) {
      lE[0]+=kqq*erfcrinv;
    }
  }
}

__device__ void function_pair(NbExPotential pp,Cutoffs rc,real r,real *fpair,real *lE,bool calcEnergy)
{
  real rinv=1/r;
  real br=rc.betaEwald*r;
  real kqq=kELECTRIC*pp.qxq;

#warning "No nan guard"
  // fpair[0]=kqq*(erff(br)*rinv-(2/sqrt(M_PI))*rc.betaEwald*expf(-br*br))*rinv;
  fpair[0]=kqq*(erff(br)*rinv-((real)1.128379167095513)*rc.betaEwald*expf(-br*br))*rinv;
  if (calcEnergy) {
    lE[0]=-kqq*erff(br)*rinv;
  }
}


template <class PairPotential,bool useSoftCore>
__global__ void getforce_pair_kernel(int pairCount,PairPotential *pairs,Cutoffs cutoffs,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  PairPotential pp;
  real fpair;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj;
  int b[2];
  real l[2]={1,1};
  real rEff,dredr,dredll; // Soft core stuff

  if (i<pairCount) {
    // Geometry
    pp=pairs[i];
    ii=pp.idx[0];
    jj=pp.idx[1];
    xi=position[ii];
    xj=position[jj];
    dr=real3_subpbc(xi,xj,box);
    r=real3_mag(dr);

    // Scaling
    b[0]=0xFFFF & pp.siteBlock[0];
    b[1]=0xFFFF & pp.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
      }
    }

    rEff=r;
    if (useSoftCore) {
      dredr=1; // d(rEff) / d(r)
      dredll=0; // d(rEff) / d(lixljtmp)
      if (b[0]) {
        real rSoft=SOFTCORERADIUS*(1-l[0]*l[1]);
        if (r<rSoft) {
          real rdivrs=r/rSoft;
          rEff=1-((real)0.5)*rdivrs;
          rEff=rEff*rdivrs*rdivrs*rdivrs+((real)0.5);
          dredr=3-2*rdivrs;
          dredr*=rdivrs*rdivrs;
          dredll=rEff-dredr*rdivrs;
          dredll*=-SOFTCORERADIUS;
          rEff*=rSoft;
        }
      }
    }

    // Interaction
    function_pair(pp,cutoffs,rEff,&fpair,&lEnergy, b[0] || energy);
    fpair*=l[0]*l[1];

    // Lambda force
    if (useSoftCore) {
      if (b[0]) {
        atomicAdd(&lambdaForce[b[0]],l[1]*(lEnergy+fpair*dredll));
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]],l[0]*(lEnergy+fpair*dredll));
        }
      }
    } else {
      if (b[0]) {
        atomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
        }
      }
    }

    // Spatial force
    if (useSoftCore) {
      fpair*=dredr;
    }
    at_real3_scaleinc(&force[ii], fpair/r,dr);
    at_real3_scaleinc(&force[jj],-fpair/r,dr);
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l[0]*l[1];
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void getforce_nb14(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N=p->nb14Count;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eenb14]==false) return;

  if (N==0) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eenb14;
  }

  if (system->msld->useSoftCore14) {
    getforce_pair_kernel <Nb14Potential,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->nb14s_d,system->run->cutoffs,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  } else {
    getforce_pair_kernel <Nb14Potential,false> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->nb14s_d,system->run->cutoffs,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
  }
}

void getforce_nbex(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N=p->nbexCount;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (N==0) return;

  if (r->calcTermFlag[eenbrecipexcl]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecipexcl;
  }

  // Never use soft cores for nbex, they're already soft.
  getforce_pair_kernel <NbExPotential,false> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->nbexs_d,system->run->cutoffs,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,s->lambda_d,s->lambdaForce_d,pEnergy);
}
