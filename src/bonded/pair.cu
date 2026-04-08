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
// vdwMethod: 0=VSWITCH, 1=VFSWITCH, 2=VSHIFT
__device__ void function_pair(Nb14Potential pp,Cutoffs rc,real r,real *fpair,real *lE,bool calcEnergy,int vdwMethod,int elecMethod)
{
  real rinv=1/r;

  if (r<rc.rCut) {
    real rinv3=rinv*rinv*rinv;
    real rinv6=rinv3*rinv3;

    if (vdwMethod == 2) {  // VSHIFT
      // Potential shift - energy zero at cutoff, forces discontinuous
      real r2 = r*r;
      real r5 = r2*r2*r;
      real r6 = r2*r2*r2;
      real rinv7 = rinv6*rinv;
      real rinv12 = rinv6*rinv6;
      real rinv13 = rinv12*rinv;
      real rCut2 = rc.rCut*rc.rCut;
      real rCut6 = rCut2*rCut2*rCut2;
      real rCut12 = rCut6*rCut6;
      real rCut18 = rCut12*rCut6;
      real roffinv6 = 1/rCut6;
      real roffinv12 = 1/rCut12;
      real roffinv18 = 1/rCut18;

      fpair[0] = 6*pp.c6*(rinv7 - r5*roffinv12) -
                 12*pp.c12*(rinv13 + r5*roffinv18);

      if (calcEnergy) {
        lE[0] = pp.c12*(rinv12 + 2*r6*roffinv18 - 3*roffinv12) -
                pp.c6*(rinv6 + r6*roffinv12 - 2*roffinv6);
      }
    } else if (vdwMethod == 1) {  // VFSWITCH
      // Force switching
      real rCut3=rc.rCut*rc.rCut*rc.rCut;
      real rCut6=rCut3*rCut3;
      real rSwitch3=rc.rSwitch*rc.rSwitch*rc.rSwitch;
      real rSwitch6=rSwitch3*rSwitch3;

      real k6=rCut3/(rCut3-rSwitch3);
      real k12=rCut6/(rCut6-rSwitch6);

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
    } else {  // vdwMethod == 0, VSWITCH
      // Potential switching
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
    }

    real kqq=kELECTRIC*pp.qxq;
    if (elecMethod==1) { // PME
      real br=rc.betaEwald*r;

      real erfcrinv=(fasterfc(br)+pp.e14fac-1)*rinv;
      // fpair[0]+=kqq*(-erfcf(br)*rinv-(2/sqrt(M_PI))*rc.betaEwald*expf(-br*br))*rinv;
      fpair[0]+=kqq*(-erfcrinv-((real)1.128379167095513)*rc.betaEwald*expf(-br*br))*rinv;
      if (calcEnergy) {
        lE[0]+=kqq*erfcrinv;
      }
    } else if (elecMethod==0) { // FSWITCH
      real kqqe14=kqq*pp.e14fac;
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
        -kqqe14*rinv*rinv:
        -kqqe14*rinv*(Aconst*rinv+Bconst*r+3*Cconst*r3+5*Dconst*r5);
      if (calcEnergy) {
        lE[0]+=(r<=rc.rSwitch)?
          kqqe14*(rinv+dvc):
          kqqe14*(Aconst*(rinv-1/rc.rCut)+Bconst*(rc.rCut-r)+Cconst*(roff2*rc.rCut-r3)+Dconst*(roff2*roff2*rc.rCut-r5));
      }
    } else if (elecMethod==2) { // FSHIFT
      real roff=rc.rCut;
      real roffinv=1/roff;
      real roffinv2=roffinv*roffinv;
      real kqqe14=kqq*pp.e14fac;
      fpair[0]+=-kqqe14*(rinv*rinv-roffinv2);
      if (calcEnergy) {
        lE[0]+=kqqe14*(rinv-2*roffinv+r*roffinv2);
      }
    }
  }
}
  

__device__ void function_pair(NbExPotential pp,Cutoffs rc,real r,real *fpair,real *lE,bool calcEnergy,int vdwMethod,int elecMethod)
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


template <bool flagBox,class PairPotential,bool useSoftCore,int vdwMethod,int elecMethod,typename box_type>
__global__ void getforce_pair_kernel(int pairCount,PairPotential *pairs,Cutoffs cutoffs,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real_e *energy,int msldEwaldType=0)
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
  real rEff,dredr,dredll,pairScale=1; // Soft core stuff

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

    bool sameBlock = (b[0] == b[1] && b[0] != 0);
    pairScale = l[0] * l[1];
    if (!useSoftCore && sameBlock) {
      if (msldEwaldType <= 1) {
        // Mode 0/ON same-block exclusions scale linearly with lambda.
        pairScale = l[0];
      } else if (msldEwaldType == 2) {
        // Mode EX same-block exclusions use squared lambda scaling.
        pairScale = l[0] * l[0];
      }
    }

    // Interaction
    function_pair(pp,cutoffs,rEff,&fpair,&lEnergy, b[0] || energy,vdwMethod,elecMethod);
    fpair *= pairScale;

    // Lambda force - with PMEL mode-specific handling for NbEx
    if (useSoftCore) {
      if (b[0]) {
        atomicAdd(&lambdaForce[b[0]],l[1]*(lEnergy+fpair*dredll));
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]],l[0]*(lEnergy+fpair*dredll));
        }
      }
    } else {
      // PMEL mode-specific lambda force calculation
      // msldEwaldType: 0=not specified (treat as ON), 1=ON, 2=EX, 3=NN
      if (msldEwaldType <= 1 && sameBlock) {
        // Mode 0/ON same-block: single lambda derivative (dE/dlambda = E)
        if (b[0]) atomicAdd(&lambdaForce[b[0]], lEnergy);
      } else if (msldEwaldType == 2 && sameBlock) {
        // Mode EX same-block: squared scaling derivative (d(lambda^2*E)/dlambda = 2*lambda*E)
        if (b[0]) atomicAdd(&lambdaForce[b[0]], 2*l[0]*lEnergy);
      } else {
        // Standard case: Mode 0/ON/EX cross-block, Mode NN, or nb14
        // Derivative of lambda_i * lambda_j * E gives lambda_j*E to i, lambda_i*E to j
        if (b[0]) {
          atomicAdd(&lambdaForce[b[0]], l[1]*lEnergy);
          if (b[1]) {
            atomicAdd(&lambdaForce[b[1]], l[0]*lEnergy);
          }
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

  // Energy, if requested - with the same PMEL mode-specific scaling used for forces
  if (energy) {
    lEnergy *= pairScale;
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,bool useSoftCore,int vdwMethod,int elecMethod,typename box_type>
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

  getforce_pair_kernel <flagBox,Nb14Potential,useSoftCore,vdwMethod,elecMethod> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->nb14s_d,system->run->cutoffs,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy);
}

template <bool flagBox,bool useSoftCore,typename box_type>
void getforce_nb14TTT(System *system,box_type box,bool calcEnergy)
{
  int elec = (int)system->run->elecMethod;
  int vdw = (int)system->run->vdwMethod;
  // Dispatch to 9 combinations: 3 elec × 3 VDW
  if (elec==0) {
    if (vdw==0) getforce_nb14TTTT<flagBox,useSoftCore,0,0>(system,box,calcEnergy);      // VSWITCH + FSWITCH
    else if (vdw==1) getforce_nb14TTTT<flagBox,useSoftCore,1,0>(system,box,calcEnergy); // VFSWITCH + FSWITCH
    else if (vdw==2) getforce_nb14TTTT<flagBox,useSoftCore,2,0>(system,box,calcEnergy); // VSHIFT + FSWITCH
  } else if (elec==1) {
    if (vdw==0) getforce_nb14TTTT<flagBox,useSoftCore,0,1>(system,box,calcEnergy);      // VSWITCH + PME
    else if (vdw==1) getforce_nb14TTTT<flagBox,useSoftCore,1,1>(system,box,calcEnergy); // VFSWITCH + PME
    else if (vdw==2) getforce_nb14TTTT<flagBox,useSoftCore,2,1>(system,box,calcEnergy); // VSHIFT + PME
  } else if (elec==2) {
    if (vdw==0) getforce_nb14TTTT<flagBox,useSoftCore,0,2>(system,box,calcEnergy);      // VSWITCH + FSHIFT
    else if (vdw==1) getforce_nb14TTTT<flagBox,useSoftCore,1,2>(system,box,calcEnergy); // VFSWITCH + FSHIFT
    else if (vdw==2) getforce_nb14TTTT<flagBox,useSoftCore,2,2>(system,box,calcEnergy); // VSHIFT + FSHIFT
  }
}

template <bool flagBox,bool useSoftCore,typename box_type>
void getforce_nb14TT(System *system,box_type box,bool calcEnergy)
{
  getforce_nb14TTT<flagBox,useSoftCore>(system,box,calcEnergy);
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
void getforce_nbexT(System *system,box_type box,bool calcEnergy,int msldEwaldType)
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
  // vdwMethod=0 (not used for nbex), elecMethod=1 (PME)
  // Pass msldEwaldType for PMEL mode-specific handling (0=not specified/ON, 1=ON, 2=EX, 3=NN)
  getforce_pair_kernel <flagBox,NbExPotential,false,0,1> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->nbexs_d,system->run->cutoffs,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy,msldEwaldType);
}

void getforce_nbex(System *system,bool calcEnergy)
{
  // Get PMEL mode from MSLD module
  int msldEwaldType = system->msld ? system->msld->msldEwaldType : 0;
  if (system->state->typeBox) {
    getforce_nbexT<true>(system,system->state->tricBox_f,calcEnergy,msldEwaldType);
  } else {
    getforce_nbexT<false>(system,system->state->orthBox_f,calcEnergy,msldEwaldType);
  }
}
