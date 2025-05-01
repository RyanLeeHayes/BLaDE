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

template <bool flagBox,class DiRestPotential,bool soft,typename box_type>
__global__ void getforce_dihedralRestraint_kernel(int diRestCount,DiRestPotential *diRests,real3 *position,real3_f *force,box_type box,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk,ll;
  DiRestPotential dr;
  real rjk;
  real3 drij,drjk,drkl;
  real3 mvec,nvec;
  real3 mvecnorm,nvecnorm;
  real mvecmag,nvecmag,bhatmag;
  real mvinv,nvinv,bhinv;
  real3 normcross;
  real3 bhatnorm;
  real phi;
  real cosp,sinp;
  real minv2,ninv2,rjkinv2;
  real p,q;
  real3 fi,fj,fk,fl;
  real fphir=0;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj,xk,xl;

  if (i<diRestCount) {
    // Geometry
    dr=diRests[i];
    ii=dr.idx[0];
    jj=dr.idx[1];
    kk=dr.idx[2];
    ll=dr.idx[3];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    xl=position[ll];

    drij=real3_subpbc<flagBox>(xj,xi,box);
    drjk=real3_subpbc<flagBox>(xk,xj,box);
    drkl=real3_subpbc<flagBox>(xl,xk,box);
    mvec=real3_cross(drij,drjk);
    mvecmag=real3_mag<real>(mvec);
    nvec=real3_cross(drjk,drkl);
    nvecmag=real3_mag<real>(nvec);
    bhatmag=real3_mag<real>(drjk);
    mvinv=1.0/mvecmag;
    nvinv=1.0/nvecmag;
    bhinv=1.0/bhatmag;
    mvecnorm = (mvecmag > 1e-8) ? real3_scale<real3>(mvinv, mvec) : real3_reset<real3>();
    nvecnorm = (nvecmag > 1e-8) ? real3_scale<real3>(nvinv, nvec) : real3_reset<real3>();
    bhatnorm = (bhatmag > 1e-8) ? real3_scale<real3>(bhinv, drjk) : real3_reset<real3>();
    normcross=real3_cross(mvecnorm,nvecnorm);
    sinp=real3_dot<real>(bhatnorm,normcross);
    cosp=real3_dot<real>(mvecnorm,nvecnorm);
    phi=atan2f(sinp,cosp);


    // Interaction
    function_torsion(dr,phi,&fphir,&lEnergy,energy);

    //for now no Lambda force
    // Spatial force
// NOTE #warning "Division and sqrt in kernel"
    minv2=1/(real3_mag2<real>(mvec));
    ninv2=1/(real3_mag2<real>(nvec));
    rjk=sqrt(real3_mag2<real>(drjk));
    rjkinv2=1/(rjk*rjk);
    fi=real3_scale<real3>(-fphir*rjk*minv2,mvec);
    at_real3_inc(&force[ii], fi);

    fk=real3_scale<real3>(-fphir*rjk*ninv2,nvec);
    p=real3_dot<real>(drij,drjk)*rjkinv2;
    q=real3_dot<real>(drkl,drjk)*rjkinv2;
    fj=real3_scale<real3>(-p,fi);
    real3_scaleinc(&fj,-q,fk);
    fl=real3_scale<real3>(-1,fk);
    at_real3_inc(&force[ll], fl);

    real3_dec(&fk,fj);
    at_real3_inc(&force[kk], fk);

    real3_dec(&fj,fi);
    at_real3_inc(&force[jj], fj);
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_diRestT(System *system,box_type box,bool calcEnergy)
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

  N=p->diRestCount;
  if (N>0) getforce_dihedralRestraint_kernel <flagBox,DiRestPotential,false> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->diRests_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,pEnergy);
}

void getforce_diRest(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_diRestT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_diRestT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

__device__ void function_torsion(DiRestPotential dr,real phi,real *fphi,real *lE,bool calcEnergy)
{
  real dphi;
  if (dr.nphi>0) {
    dphi=dr.nphi*phi-dr.phi0;
    dphi-=(2*((real)M_PI))*floor((dphi+((real)M_PI))/(2*((real)M_PI)));
    fphi[0]=dr.kphi*dr.nphi*sinf(dphi);
    if (calcEnergy) {
      lE[0]=dr.kphi*(1-cosf(dphi));
    }
  }
  else {
    dphi=phi-dr.phi0;
    dphi-=(2*((real)M_PI))*floor((dphi+((real)M_PI))/(2*((real)M_PI)));
    fphi[0]=dr.kphi*dphi;
    if (calcEnergy) {
      lE[0]=((real)0.5)*dr.kphi*dphi*dphi;
    }
  }
}



