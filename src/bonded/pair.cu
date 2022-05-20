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
// #warning "From CHARMM, not fully compatible"
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
__device__ void function_pair(Nb14Potential pp,Cutoffs rc,real r,real *fpair,real *lE,bool calcEnergy,bool usevdWSwitch,bool usePME)
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

    if(usevdWSwitch){
      real c2ofnb=rc.rCut*rc.rCut;
      real c2onnb=rc.rSwitch*rc.rSwitch;
      real rul3=(c2ofnb-c2onnb)*(c2ofnb-c2onnb)*(c2ofnb-c2onnb);
      real rul12=12/rul3;
      real rijl=c2onnb - r * r;
      real riju=c2ofnb - r * r;
      real fsw=(r<rc.rSwitch?1:riju*riju*(riju-3*rijl)/rul3);
      real dfsw=(r<rc.rSwitch?0:rijl*riju*rul12);
      fpair[0]=fsw*(6*pp.c6-12*pp.c12*rinv6)*rinv6*rinv	\
        +dfsw*(pp.c12*rinv6-pp.c6)*rinv6;
      if (calcEnergy){lE[0]=fsw*(pp.c12*rinv6-pp.c6)*rinv6;}
    } else {
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
    }
    
    real kqq=kELECTRIC*pp.qxq;
    if (usePME) {
      real br=rc.betaEwald*r;

      real erfcrinv=fasterfc(br)*rinv;
      // fpair[0]+=kqq*(-erfcf(br)*rinv-(2/sqrt(M_PI))*rc.betaEwald*expf(-br*br))*rinv;
      fpair[0]+=kqq*(-erfcrinv-((real)1.128379167095513)*rc.betaEwald*expf(-br*br))*rinv;
      if (calcEnergy) {
        lE[0]+=kqq*erfcrinv;
      }
    } else {
      real roff2=rc.rCut*rc.rCut;
      real ron2=rc.rSwitch*rc.rSwitch;
      real ginv=1/((roff2-ron2)*(roff2-ron2)*(roff2-ron2));
      real Aconst=roff2*roff2*(roff2-3*ron2)*ginv;
      real Bconst=6*roff2*ron2*ginv;
      real Cconst=-(ron2+roff2)*ginv;
      real Dconst=2*ginv/5;
      real dvc=8*(ron2*roff2*(rc.rCut-rc.rSwitch)-(roff2*roff2*rc.rCut-ron2*ron2*rc.rSwitch)/5)*ginv;
      real r2=r*r;
      real r3=r2*r;
      real r5=r3*r2;
      fpair[0]+=(r<=rc.rSwitch)?
        -kqq*rinv*rinv:
        -kqq*rinv*(Aconst*rinv+Bconst*r+3*Cconst*r3+5*Dconst*r5);
      if (calcEnergy) {
        lE[0]+=(r<=rc.rSwitch)?
          kqq*(rinv+dvc):
          kqq*(Aconst*(rinv-1/rc.rCut)+Bconst*(rc.rCut-r)+Cconst*(roff2*rc.rCut-r3)+Dconst*(roff2*roff2*rc.rCut-r5));
      }
    }
  }
}
  

__device__ void function_pair(NbExPotential pp,Cutoffs rc,real r,real *fpair,real *lE,bool calcEnergy,bool usevdWSwitch,bool usePME)
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


template <bool flagBox,class PairPotential,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
__global__ void getforce_pair_kernel(int pairCount,PairPotential *pairs,Cutoffs cutoffs,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real_e *energy)
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
    dr=real3_subpbc<flagBox>(xi,xj,box);
    r=real3_mag<real>(dr);

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
    function_pair(pp,cutoffs,rEff,&fpair,&lEnergy, b[0] || energy,usevdWSwitch,usePME);
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

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,bool usePME,typename box_type>
void getforce_nb14TTTT(System *system,box_type box,bool calcEnergy)
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

  getforce_pair_kernel <flagBox,Nb14Potential,useSoftCore,usevdWSwitch,usePME> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->nb14s_d,system->run->cutoffs,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy);
}

template <bool flagBox,bool useSoftCore,bool usevdWSwitch,typename box_type>
void getforce_nb14TTT(System *system,box_type box,bool calcEnergy)
{
  if (system->run->usePME) {
    getforce_nb14TTTT<flagBox,useSoftCore,usevdWSwitch,true>(system,box,calcEnergy);
  } else {
    getforce_nb14TTTT<flagBox,useSoftCore,usevdWSwitch,false>(system,box,calcEnergy);
  }
}

template <bool flagBox,bool useSoftCore,typename box_type>
void getforce_nb14TT(System *system,box_type box,bool calcEnergy)
{
  if (!system->run->vfSwitch) {
    getforce_nb14TTT<flagBox,useSoftCore,true>(system,box,calcEnergy);
  } else {
    getforce_nb14TTT<flagBox,useSoftCore,false>(system,box,calcEnergy);
  }
}

template <bool flagBox,typename box_type>
void getforce_nb14T(System *system,box_type box,bool calcEnergy)
{
  if (system->msld->useSoftCore14) {
    getforce_nb14TT<flagBox,true>(system,box,calcEnergy);
  } else {
    getforce_nb14TT<flagBox,false>(system,box,calcEnergy);
  }
}

void getforce_nb14(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_nb14T<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_nb14T<false>(system,system->state->orthBox_f,calcEnergy);
  }
}



template <bool flagBox,typename box_type>
void getforce_nbexT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N=p->nbexCount;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (N==0) return;

  if (r->usePME==false) return;
  if (r->calcTermFlag[eenbrecipexcl]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecipexcl;
  }

  // Never use soft cores for nbex, they're already soft.
  getforce_pair_kernel <flagBox,NbExPotential,false,false,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->nbexs_d,system->run->cutoffs,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy);
}

void getforce_nbex(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_nbexT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_nbexT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
