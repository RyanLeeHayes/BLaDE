#include <cuda_runtime.h>
#include <math.h>

#include "restrain.h"
#include "main/defines.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

#include "main/real3.h"

template <bool flagBox,typename box_type>
__global__ void getforce_noe_kernel(int noeCount,struct NoePotential *noes,real3 *position,real3_f *force,box_type box,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  NoePotential noep;
  real r,r_r0;
  real3 dr;
  real fnoe=0;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj;
  
  if (i<noeCount) {
    // Geometry
    noep=noes[i];
    xi=position[noep.i];
    xj=position[noep.j];
    dr=real3_subpbc<flagBox>(xi,xj,box);
    r=real3_mag<real>(dr);
    if (r<noep.rmin) {
      r_r0=r-noep.rmin;
      fnoe=noep.kmin*r_r0;
      if (energy) lEnergy=((real)0.5)*fnoe*r_r0;
    } else if (r>noep.rmax) {
      r_r0=r-noep.rmax;
      if (noep.rswitch>0 && r_r0>noep.rswitch) {
        real bswitch=(noep.rpeak-noep.rswitch)/noep.nswitch*pow(noep.rswitch,noep.nswitch+1);
        real aswitch=0.5*noep.rswitch*noep.rswitch-noep.rpeak*noep.rswitch-noep.rswitch*(noep.rpeak-noep.rswitch)/noep.nswitch;
        fnoe=noep.kmax*(noep.rpeak-bswitch*pow(r_r0,-noep.nswitch-1));
        if (energy) lEnergy=noep.kmax*(aswitch+bswitch*pow(r_r0,-noep.nswitch)+noep.rpeak*r_r0);
      } else {
        fnoe=noep.kmax*r_r0;
        if (energy) lEnergy=((real)0.5)*fnoe*r_r0;
      }
    }
    // Spatial force
    at_real3_scaleinc(&force[noep.i], fnoe/r,dr);
    at_real3_scaleinc(&force[noep.j],-fnoe/r,dr);
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_noeT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eebias]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eebias;
  }

  N=p->noeCount;
  if (N>0) getforce_noe_kernel<flagBox><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->noes_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,pEnergy);
}

void getforce_noe(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_noeT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_noeT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

template <bool flagBox,typename box_type>
__global__ void getforce_harm_kernel(int harmCount,struct HarmonicPotential *harms,real3 *position,real3_f *force,box_type box,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii;
  real r2;
  real3 dr;
  HarmonicPotential hp;
  real krnm2;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,x0;
  
  if (i<harmCount) {
    // Geometry
    hp=harms[i];
    ii=hp.idx;
    xi=position[ii];
    x0=hp.r0;
// NOTE #warning "Unprotected division"
    dr=real3_subpbc<flagBox>(xi,x0,box);
    r2=real3_mag2<real>(dr);
    krnm2=(r2 ? (hp.k*pow(r2,((real)0.5)*hp.n-1)) : 0); // NaN guard it
    
    if (energy) {
      lEnergy=krnm2*r2;
    }
    at_real3_scaleinc(&force[ii], hp.n*krnm2,dr);
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_harmT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eebias]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eebias;
  }

  N=p->harmCount;
  if (N>0) getforce_harm_kernel<flagBox><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->harms_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,pEnergy);
}

void getforce_harm(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_harmT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_harmT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
