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

/*
// In case we need global variables to save time uploading arguments
__device__ int N_bond;
__device__ struct_bondparms *bondparms;
__device__ struct_bondblock bondblock;
__device__ real *sp_box;
__device__ real *sp_Gt;

void upload_bonded_d(
  int N_b,struct_bondparms* h_bondparms,struct_bondblock* h_bondblock,
  struct_atoms h_at,real* h_box,real* h_Gt)
{
  cudaMemcpyToSymbol(N_bond, &N_b, sizeof(int), size_t(0),cudaMemcpyHostToDevice);
  if (N_b) {
    cudaMemcpyToSymbol(bondparms, &h_bondparms, sizeof(struct_bondparms*), size_t(0),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(bondblock, &h_bondblock[0], sizeof(struct_bondblock), size_t(0),cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(sp_at, &h_at, sizeof(struct_atoms), size_t(0),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(sp_box, &h_box, sizeof(real*), size_t(0),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(sp_Gt, &h_Gt, sizeof(real*), size_t(0),cudaMemcpyHostToDevice);
}
*/



// NYI - Try inlining the fdihe, fimp, fnb14, fnbex potentials...

// getforce_bond_kernel<<<(N+BLBO-1)/BLBO,BLBO,0,p->bondedStream>>>(N,p->bonds,s->position_d,s->force_d,s->box,m->lambda_d,m->lambdaForce_d,NULL);
template <bool flagBox,bool soft,typename box_type>
__global__ void getforce_bond_kernel(int bond12Count,int bondCount,struct BondPotential *bonds,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real softAlpha,real softExp,real_e *energy)
{
// NYI - maybe energy should be a double
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  real r;
  real3 dr;
  BondPotential bp;
  real fbond;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj;
  int b[2];
  real l[2]={1,1};
  
  if (i<bondCount) {
    // Geometry
    bp=bonds[i];
    ii=bp.idx[0];
    jj=bp.idx[1];
    xi=position[ii];
    xj=position[jj];
// NOTE #warning "Unprotected division"
    dr=real3_subpbc<flagBox>(xi,xj,box);
// NOTE #warning "Unprotected sqrt"
    r=real3_mag<real>(dr);
    
    // Scaling
    b[0]=0xFFFF & bp.siteBlock[0];
    b[1]=0xFFFF & bp.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
      }
    }

    if (soft) {
      // interaction
      fbond=bp.kb*(r-bp.b0);
      real lambdaExpM1=pow(l[0]*l[1],softExp-1);
      real lambdaExp=lambdaExpM1*l[0]*l[1];
      real softFactor=1/(1+(1-lambdaExp)*softAlpha*(r-bp.b0)*(r-bp.b0));
      real dsfdle=softFactor*softFactor*softAlpha*(r-bp.b0)*(r-bp.b0);
      real dsfdr=-2*softFactor*softFactor*(1-lambdaExp)*softAlpha*(r-bp.b0);
      if (b[0] || energy) {
        lEnergy=((real)0.5)*bp.kb*(r-bp.b0)*(r-bp.b0);
      }
      fbond=lambdaExp*(fbond*softFactor+lEnergy*dsfdr);
      real flambda=lEnergy*softExp*lambdaExpM1*(softFactor+lambdaExp*dsfdle);
      lEnergy*=lambdaExpM1*softFactor; // last factor of l[0]*l[1] shows up later

      // Lambda force
      if (b[0]) {
        atomicAdd(&lambdaForce[b[0]],l[1]*flambda);
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]],l[0]*flambda);
        }
      }

      // Spatial force
// NOTE #warning "division in kernel"
      at_real3_scaleinc(&force[ii], fbond/r,dr);
      at_real3_scaleinc(&force[jj],-fbond/r,dr);
    } else {
      // interaction
      fbond=bp.kb*(r-bp.b0);
      if (b[0] || energy) {
        lEnergy=((real)0.5)*bp.kb*(r-bp.b0)*(r-bp.b0);
      }
      fbond*=l[0]*l[1];

      // Lambda force
      if (b[0]) {
        atomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
        }
      }

      // Spatial force
// NOTE #warning "division in kernel"
      at_real3_scaleinc(&force[ii], fbond/r,dr);
      at_real3_scaleinc(&force[jj],-fbond/r,dr);
    }
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l[0]*l[1];
    if (blockIdx.x*blockDim.x<bond12Count) {
      real_sum_reduce((i<bond12Count?lEnergy:0),sEnergy,energy);
      __syncthreads();
    }
    if (blockIdx.x*blockDim.x+blockDim.x>bond12Count) {
      real_sum_reduce((i<bond12Count?0:lEnergy),sEnergy,energy+eeurey-eebond);
    }
  }
}

template <bool flagBox,typename box_type>
void getforce_bondT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softAlpha=1/(system->msld->softBondRadius*system->msld->softBondRadius);
  real softExp=system->msld->softBondExponent;
  int N12,N;
  struct BondPotential *bonds;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eebond]==false && r->calcTermFlag[eeurey]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eebond;
  }

  N12=(r->calcTermFlag[eebond]?p->bond12Count:0);
  N=N12+(r->calcTermFlag[eeurey]?p->bond13Count:0);
  bonds=p->bonds_d+(p->bond12Count-N12);
  if (N>0) getforce_bond_kernel<flagBox,false><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N12,N,bonds,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,0,1,pEnergy);
  N=p->softBondCount;
  N12=(r->calcTermFlag[eebond]?p->softBond12Count:0);
  N=N12+(r->calcTermFlag[eeurey]?p->softBond13Count:0);
  bonds=p->softBonds_d+(p->softBond12Count-N12);
  if (N>0) getforce_bond_kernel<flagBox,true><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N12,N,bonds,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,softAlpha,softExp,pEnergy);
}

void getforce_bond(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_bondT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_bondT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}



// getforce_angle_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->angles_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <bool flagBox,bool soft,typename box_type>
__global__ void getforce_angle_kernel(int angleCount,struct AnglePotential *angles,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real softExp,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk;
  AnglePotential ap;
  real3 drij,drkj;
  real t;
  real dotp, mcrop;
  real3 crop;
  real3 fi,fj,fk;
  real fangle;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi, xj, xk;
  int b[2];
  real l[2]={1,1};

  if (i<angleCount) {
    // Geometry
    ap=angles[i];
    ii=ap.idx[0];
    jj=ap.idx[1];
    kk=ap.idx[2];
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
    b[0]=0xFFFF & ap.siteBlock[0];
    b[1]=0xFFFF & ap.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
      }
    }

    // Interaction
    fangle=ap.kangle*(t-ap.angle0);
    if (b[0] || energy) {
      lEnergy=((real)0.5)*ap.kangle*(t-ap.angle0)*(t-ap.angle0);
    }
    if (soft) {
      fangle*=pow(l[0]*l[1],softExp);
    } else {
      fangle*=l[0]*l[1];
    }

    // Lambda force
    if (soft) {
      lEnergy*=softExp*pow(l[0]*l[1],softExp-1);
    }
    if (b[0]) {
      atomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        atomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }
    if (soft) {
      lEnergy/=softExp;
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
    lEnergy*=l[0]*l[1];
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_angleT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eeangle]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eeangle;
  }

  N=p->angleCount;
  if (N>0) getforce_angle_kernel<flagBox,false><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->angles_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,1,pEnergy);
  N=p->softAngleCount;
  if (N>0) getforce_angle_kernel<flagBox,true><<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->softAngles_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,softExp,pEnergy);
}

void getforce_angle(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_angleT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_angleT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}



__device__ void function_torsion(DihePotential dp,real phi,real *fphi,real *lE,bool calcEnergy)
{
  real dphi;

  dphi=dp.ndih*phi-dp.dih0;
  fphi[0]=-dp.kdih*dp.ndih*sinf(dphi);
  if (calcEnergy) {
    lE[0]=dp.kdih*(cosf(dphi)+1);
  }
}

__device__ void function_torsion(ImprPotential ip,real phi,real *fphi,real *lE,bool calcEnergy)
{
  real dphi;

  dphi=phi-ip.imp0;
  dphi-=(2*((real)M_PI))*floor((dphi+((real)M_PI))/(2*((real)M_PI)));
  fphi[0]=ip.kimp*dphi;
  if (calcEnergy) {
    lE[0]=((real)0.5)*ip.kimp*dphi*dphi;
  }
}


// getforce_dihe_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->dihes_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <bool flagBox,class TorsionPotential,bool soft,typename box_type>
__global__ void getforce_torsion_kernel(int torsionCount,TorsionPotential *torsions,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real softExp,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk,ll;
  TorsionPotential tp;
  real rjk;
  real3 drij,drjk,drkl;
  real3 mvec,nvec;
  real phi,sign,ipr;
  real cosp,sinp;
  real3 dsinp;
  real minv2,ninv2,rjkinv2;
  real p,q;
  real3 fi,fj,fk,fl;
  real ftorsion;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj,xk,xl;
  int b[2];
  real l[2]={1,1};

  if (i<torsionCount) {
    // Geometry
    tp=torsions[i];
    ii=tp.idx[0];
    jj=tp.idx[1];
    kk=tp.idx[2];
    ll=tp.idx[3];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    xl=position[ll];

    drij=real3_subpbc<flagBox>(xi,xj,box);
    drjk=real3_subpbc<flagBox>(xj,xk,box);
    drkl=real3_subpbc<flagBox>(xk,xl,box);
    mvec=real3_cross(drij,drjk);
    nvec=real3_cross(drjk,drkl);
    dsinp=real3_cross(mvec,nvec);
    sinp=real3_mag<real>(dsinp);
    cosp=real3_dot<real>(mvec,nvec);
    phi=atan2f(sinp,cosp);
    ipr=real3_dot<real>(drij,nvec);
    sign=(ipr > 0.0) ? -1.0 : 1.0; // Opposite of gromacs because m and n are opposite
    phi=sign*phi;

    // Scaling
    b[0]=0xFFFF & tp.siteBlock[0];
    b[1]=0xFFFF & tp.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
      }
    }

    // Interaction
    function_torsion(tp,phi,&ftorsion,&lEnergy, b[0] || energy);
    if (soft) {
      ftorsion*=pow(l[0]*l[1],softExp);
    } else {
      ftorsion*=l[0]*l[1];
    }

    // Lambda force
    if (soft) {
      lEnergy*=softExp*pow(l[0]*l[1],softExp-1);
    }
    if (b[0]) {
      atomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        atomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }
    if (soft) {
      lEnergy/=softExp;
    }

    // Spatial force
// NOTE #warning "Division and sqrt in kernel"
    minv2=1/(real3_mag2<real>(mvec));
    ninv2=1/(real3_mag2<real>(nvec));
    rjk=sqrt(real3_mag2<real>(drjk));
    rjkinv2=1/(rjk*rjk);
    fi=real3_scale<real3>(-ftorsion*rjk*minv2,mvec);
    at_real3_inc(&force[ii], fi);

    fk=real3_scale<real3>(-ftorsion*rjk*ninv2,nvec);
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
    lEnergy*=l[0]*l[1];
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_diheT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eedihe]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eedihe;
  }

  N=p->diheCount;
  if (N>0) getforce_torsion_kernel <flagBox,DihePotential,false> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->dihes_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,1,pEnergy);
  N=p->softDiheCount;
  if (N>0) getforce_torsion_kernel <flagBox,DihePotential,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->softDihes_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,softExp,pEnergy);
}

void getforce_dihe(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_diheT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_diheT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

template <bool flagBox,typename box_type>
void getforce_imprT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eeimpr]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eeimpr;
  }

  N=p->imprCount;
  if (N>0) getforce_torsion_kernel <flagBox,ImprPotential,false> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->imprs_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,1,pEnergy);
  N=p->softImprCount;
  if (N>0) getforce_torsion_kernel <flagBox,ImprPotential,true> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->softImprs_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,softExp,pEnergy);
}

void getforce_impr(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_imprT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_imprT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}



// getforce_cmap_kernel<<<(2*N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->cmaps_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <bool flagBox,bool soft,typename box_type>
__global__ void getforce_cmap_kernel(int cmapCount,struct CmapPotential *cmaps,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real softExp,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk,ll;
  CmapPotential cp;
  real rjk;
  real3 drij,drjk,drkl;
  real3 mvec,nvec;
  real phi,sign,ipr;
  real rPhi[2]; // remainders, one for each angle
  real cosp,sinp;
  real3 dsinp;
  real minv2,ninv2,rjkinv2;
  real p,q;
  real3 fi,fj,fk,fl;
  int lastBit;
  int binPhi[2];
  int cmapBin;
  real invSpace;
  real fcmapPhi[2]; // one for each angle
  real fcmapPhiColumn[2]; // one for each angle
  real fcmap;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj,xk,xl;
  int b[3];
  real l[3]={1,1,1};

  lastBit=threadIdx.x&1;
  if (i<2*cmapCount) { // Threads work in pairs
    // Geometry
    cp=cmaps[i>>1];
    ii=cp.idx[0+4*lastBit];
    jj=cp.idx[1+4*lastBit];
    kk=cp.idx[2+4*lastBit];
    ll=cp.idx[3+4*lastBit];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    xl=position[ll];

    drij=real3_subpbc<flagBox>(xi,xj,box);
    drjk=real3_subpbc<flagBox>(xj,xk,box);
    drkl=real3_subpbc<flagBox>(xk,xl,box);
    mvec=real3_cross(drij,drjk);
    nvec=real3_cross(drjk,drkl);
    dsinp=real3_cross(mvec,nvec);
    sinp=real3_mag<real>(dsinp);
    cosp=real3_dot<real>(mvec,nvec);
    phi=atan2f(sinp,cosp);
    ipr=real3_dot<real>(drij,nvec);
    sign=(ipr > 0.0) ? -1.0 : 1.0; // Opposite of gromacs because m and n are opposite
    phi=sign*phi;

    // Scaling
    b[0]=0xFFFF & cp.siteBlock[0];
    b[1]=0xFFFF & cp.siteBlock[1];
    b[2]=0xFFFF & cp.siteBlock[2];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
        if (b[2]) {
          l[2]=lambda[b[2]];
        }
      }
    }

    // Interaction
      // get phi and psi (both called phi), phi[0] is on even threads, phi[1] on odd
    rPhi[lastBit]=phi;
  }
    rPhi[1-lastBit]=__shfl_xor_sync(0xFFFFFFFF,phi,1);
  if (i<2*cmapCount) { // Avoid hang
      // Get the remainders within each box
    invSpace=cp.ngrid*(1/(2*((real) M_PI)));
    rPhi[0]*=invSpace;
    binPhi[0]=((int) floor(rPhi[0]));
    rPhi[0]-=binPhi[0];
    binPhi[0]+=cp.ngrid/2;
    binPhi[0]+=(binPhi[0]>=cp.ngrid?-cp.ngrid:0);
    binPhi[0]+=(binPhi[0]<0?cp.ngrid:0);
    rPhi[1]*=invSpace;
    binPhi[1]=((int) floor(rPhi[1]));
    rPhi[1]-=binPhi[1];
    binPhi[1]+=cp.ngrid/2;
    binPhi[1]+=(binPhi[1]>=cp.ngrid?-cp.ngrid:0);
    binPhi[1]+=(binPhi[1]<0?cp.ngrid:0);
    cmapBin=cp.ngrid*binPhi[0]+binPhi[1];
      // compute forces (and energy)
    fcmapPhiColumn[0]=3*cp.kcmapPtr[cmapBin][3][2+lastBit];
    fcmapPhiColumn[1]=  cp.kcmapPtr[cmapBin][3][2+lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=2*cp.kcmapPtr[cmapBin][2][2+lastBit];
    fcmapPhiColumn[1]+=  cp.kcmapPtr[cmapBin][2][2+lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=cp.kcmapPtr[cmapBin][1][2+lastBit];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][1][2+lastBit];

    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][0][2+lastBit];

    if (b[0] || energy) {
      lEnergy=rPhi[1]*rPhi[1]*fcmapPhiColumn[1];
    }
    fcmapPhi[0]=rPhi[1]*rPhi[1]*fcmapPhiColumn[0];
    fcmapPhi[1]=(2+lastBit)*rPhi[1]*fcmapPhiColumn[1];
    fcmapPhi[1]*=(lastBit?rPhi[1]:1);

    fcmapPhiColumn[0]=3*cp.kcmapPtr[cmapBin][3][lastBit];
    fcmapPhiColumn[1]=  cp.kcmapPtr[cmapBin][3][lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=2*cp.kcmapPtr[cmapBin][2][lastBit];
    fcmapPhiColumn[1]+=  cp.kcmapPtr[cmapBin][2][lastBit];

    fcmapPhiColumn[0]*=rPhi[0];
    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[0]+=cp.kcmapPtr[cmapBin][1][lastBit];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][1][lastBit];

    fcmapPhiColumn[1]*=rPhi[0];
    fcmapPhiColumn[1]+=cp.kcmapPtr[cmapBin][0][lastBit];

    if (b[0] || energy) {
      lEnergy+=fcmapPhiColumn[1];
      lEnergy*=(lastBit?rPhi[1]:1);
      // Put all energy on first thread of pair
      // NOHANG lEnergy+=__shfl_xor_sync(0xFFFFFFFF,lEnergy,1);
      // NOHANG lEnergy=(lastBit?0:lEnergy);
    }
    fcmapPhi[0]+=fcmapPhiColumn[0];
    fcmapPhi[0]*=(lastBit?rPhi[1]:1);
    fcmapPhi[1]+=lastBit*fcmapPhiColumn[1];

      // Put partner's force in fcmap for exchange
    fcmap=fcmapPhi[1-lastBit];
  }
    fcmap=__shfl_xor_sync(0xFFFFFFFF,fcmap,1);
  if (i<2*cmapCount) { // Avoid hang
      // Add own force
    fcmap+=fcmapPhi[lastBit];
    fcmap*=invSpace;
    if (soft) {
      fcmap*=pow(l[0]*l[1]*l[2],softExp);
    } else {
      fcmap*=l[0]*l[1]*l[2];
    }

    // Lambda force
    if (soft) {
      lEnergy*=softExp*pow(l[0]*l[1]*l[2],softExp-1);
    }
    // NOHANG if (lastBit==0) { // First partner has full energy
      if (b[0]) {
        atomicAdd(&lambdaForce[b[0]],l[1]*l[2]*lEnergy);
        if (b[1]) {
          atomicAdd(&lambdaForce[b[1]],l[0]*l[2]*lEnergy);
          if (b[2]) {
            atomicAdd(&lambdaForce[b[2]],l[0]*l[1]*lEnergy);
          }
        }
      }
    // NOHANG }
    if (soft) {
      lEnergy/=softExp;
    }

    // Spatial force
// NOTE #warning "Division and sqrt in kernel"
    minv2=1/(real3_mag2<real>(mvec));
    ninv2=1/(real3_mag2<real>(nvec));
    rjk=sqrt(real3_mag2<real>(drjk));
    rjkinv2=1/(rjk*rjk);
    fi=real3_scale<real3>(-fcmap*rjk*minv2,mvec);
    at_real3_inc(&force[ii], fi);

    fk=real3_scale<real3>(-fcmap*rjk*ninv2,nvec);
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
    lEnergy*=l[0]*l[1]*l[2];
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_cmapT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  real softExp=system->msld->softNotBondExponent;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eecmap]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eecmap;
  }

  N=p->cmapCount;
  if (N>0) getforce_cmap_kernel<flagBox,false><<<(2*N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->cmaps_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,1,pEnergy);
  N=p->softCmapCount;
  if (N>0) getforce_cmap_kernel<flagBox,true><<<(2*N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,p->softCmaps_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,softExp,pEnergy);
}

void getforce_cmap(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_cmapT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_cmapT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
