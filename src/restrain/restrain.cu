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

  if (r->calcTermFlag[eenoe]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eenoe;
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

  if (r->calcTermFlag[eeharmonic]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eeharmonic;
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

// Bond Restraint
template <bool flagBox,typename box_type>
__global__ void getforce_bondRestraint_kernel(int boRestCount,struct BoRestPotential *boRests,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  BoRestPotential br;
  real fbond;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj;
  int b;
  real l=1;
  
  if (i<boRestCount) {
    // Geometry
    br=boRests[i];
    ii=br.idx[0];
    jj=br.idx[1];
    xi=position[ii];
    xj=position[jj];
// NOTE #warning "Unprotected division"
    dr=real3_subpbc<flagBox>(xi,xj,box);
// NOTE #warning "Unprotected sqrt"
    r=real3_mag<real>(dr);
    
    // Scaling
    b=br.block;
    if (b) {
      l=lambda[b];
    }

    // interaction
    fbond=br.kr*(r-br.r0);
    if (b || energy) {
      lEnergy=((real)0.5)*br.kr*(r-br.r0)*(r-br.r0);
    }
    fbond*=l;

    // Lambda force
    if (b) {
      atomicAdd(&lambdaForce[b],lEnergy);
    }

    // Spatial force
// NOTE #warning "division in kernel"
    at_real3_scaleinc(&force[ii], fbond/r,dr);
    at_real3_scaleinc(&force[jj],-fbond/r,dr);
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l;
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_boRestT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eemmfp]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eemmfp;
  }

  N=p->boRestCount;
  if (N>0) getforce_bondRestraint_kernel <flagBox> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->boRests_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy);
}

void getforce_boRest(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_boRestT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_boRestT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

// Angle Restraint
template <bool flagBox,typename box_type>
__global__ void getforce_angleRestraint_kernel(int anRestCount,struct AnRestPotential *anRests,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk;
  AnRestPotential ar;
  real3 drij,drkj;
  real t;
  real dotp, mcrop;
  real3 crop;
  real3 fi,fj,fk;
  real fangle;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi, xj, xk;
  int b;
  real l=1;

  if (i<anRestCount) {
    // Geometry
    ar=anRests[i];
    ii=ar.idx[0];
    jj=ar.idx[1];
    kk=ar.idx[2];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    
    drij=real3_subpbc<flagBox>(xi,xj,box);
    drkj=real3_subpbc<flagBox>(xk,xj,box);
    dotp=real3_dot<real>(drij,drkj);
    crop=real3_cross(drij,drkj); // c = a x b
    mcrop=real3_mag<real>(crop);
    t=atan2f(mcrop,dotp);

    // Scaling
    b=ar.block;
    if (b) {
      l=lambda[b];
    }

    // Interaction
    fangle=ar.kt*(t-ar.t0);
    if (b || energy) {
      lEnergy=((real)0.5)*ar.kt*(t-ar.t0)*(t-ar.t0);
    }
    fangle*=l;

    // Lambda force
    if (b) {
      atomicAdd(&lambdaForce[b],lEnergy);
    }

    // Spatial force
    fi=real3_cross(drij,crop);
// NOTE #warning "division on kernel, was using realRecip before."
    real3_scaleself(&fi, fangle/(mcrop*real3_mag2<real>(drij)));
    at_real3_inc(&force[ii], fi);
    fk=real3_cross(drkj,crop);
    real3_scaleself(&fk,-fangle/(mcrop*real3_mag2<real>(drkj)));
    at_real3_inc(&force[kk], fk);
    fj=real3_add(fi,fk);
    real3_scaleself(&fj,-1);
    at_real3_inc(&force[jj], fj);
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l;
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_anRestT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eemmfp]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eemmfp;
  }

  N=p->anRestCount;
  if (N>0) getforce_angleRestraint_kernel <flagBox> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->anRests_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy);
}

void getforce_anRest(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_anRestT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_anRestT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

// Dihedral Restraint
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

template <bool flagBox,typename box_type>
__global__ void getforce_dihedralRestraint_kernel(int diRestCount,DiRestPotential *diRests,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real_e *energy)
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
  int b;
  real l=1;

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
    b=dr.block;
    if (b) {
      l=lambda[b];
    }

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
    function_torsion(dr,phi,&fphir,&lEnergy, b || energy);

    // Lambda force
    if (b) {
      atomicAdd(&lambdaForce[b],lEnergy);
      fphir*=l;
    }

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
    lEnergy*=l;
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

  if (r->calcTermFlag[eemmfp]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eemmfp;
  }

  N=p->diRestCount;
  if (N>0) getforce_dihedralRestraint_kernel <flagBox> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->diRests_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy);
}

void getforce_diRest(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_diRestT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_diRestT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}



