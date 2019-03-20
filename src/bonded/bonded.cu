#include <cuda_runtime.h>
#include <math.h>

#include "bonded.h"
#include "main/defines.h"
#include "system/system.h"
#include "system/state.h"
#include "msld/msld.h"
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
__global__ void getforce_bond_kernel(int bondCount,struct BondPotential *bonds,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
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
    dr=real3_subpbc(xi,xj,box);
// NOTE #warning "Unprotected sqrt"
    r=real3_mag(dr);
    
    // Scaling
    b[0]=0xFFFF & bp.siteBlock[0];
    b[1]=0xFFFF & bp.siteBlock[1];
    if (b[0]) {
      l[0]=lambda[b[0]];
      if (b[1]) {
        l[1]=lambda[b[1]];
      }
    }

    // interaction
    fbond=bp.kb*(r-bp.b0);
    if (b[0] || energy) {
      lEnergy=0.5*bp.kb*(r-bp.b0)*(r-bp.b0);
    }
    fbond*=l[0]*l[1];

    // Lambda force
    if (b[0]) {
      realAtomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        realAtomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }

    // Spatial force
// NOTE #warning "division in kernel"
    at_real3_scaleinc(&force[ii], fbond/r,dr);
    at_real3_scaleinc(&force[jj],-fbond/r,dr);
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l[0]*l[1];
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void getforce_bond(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  int N=p->bondCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (N==0) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eebond;
  }

  getforce_bond_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->bonds_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
}



// getforce_angle_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->angles_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
__global__ void getforce_angle_kernel(int angleCount,struct AnglePotential *angles,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
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
    
    drij=real3_subpbc(xi,xj,box);
    drkj=real3_subpbc(xk,xj,box);
    dotp=real3_dot(drij,drkj);
    crop=real3_cross(drij,drkj); // c = a x b
    mcrop=real3_mag(crop);
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
      lEnergy=0.5*ap.kangle*(t-ap.angle0)*(t-ap.angle0);
    }
    fangle*=l[0]*l[1];

    // Lambda force
    if (b[0]) {
      realAtomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        realAtomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }

    // Spatial force
    fi=real3_cross(drij,crop);
// NOTE #warning "division on kernel, was using realRecip before."
    real3_scaleself(&fi, fangle/(mcrop*real3_mag2(drij)));
    at_real3_inc(&force[ii], fi);
    fk=real3_cross(drkj,crop);
    real3_scaleself(&fk,-fangle/(mcrop*real3_mag2(drkj)));
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

void getforce_angle(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  int N=p->angleCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (N==0) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eeangle;
  }

  getforce_angle_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->angles_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
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
  dphi-=(2*M_PI)*floor((dphi+M_PI)/(2*M_PI));
  fphi[0]=ip.kimp*dphi;
  if (calcEnergy) {
    lE[0]=0.5*ip.kimp*dphi*dphi;
  }
}


// getforce_dihe_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->dihes_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
template <class TorsionPotential>
__global__ void getforce_torsion_kernel(int torsionCount,TorsionPotential *torsions,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
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

    drij=real3_subpbc(xi,xj,box);
    drjk=real3_subpbc(xj,xk,box);
    drkl=real3_subpbc(xk,xl,box);
    mvec=real3_cross(drij,drjk);
    nvec=real3_cross(drjk,drkl);
    dsinp=real3_cross(mvec,nvec);
    sinp=real3_mag(dsinp);
    cosp=real3_dot(mvec,nvec);
    phi=atan2f(sinp,cosp);
    ipr=real3_dot(drij,nvec);
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
    ftorsion*=l[0]*l[1];

    // Lambda force
    if (b[0]) {
      realAtomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        realAtomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }

    // Spatial force
// NOTE #warning "Division and sqrt in kernel"
    minv2=1/(real3_mag2(mvec));
    ninv2=1/(real3_mag2(nvec));
    rjk=sqrt(real3_mag2(drjk));
    rjkinv2=1/(rjk*rjk);
    fi=real3_scale(-ftorsion*rjk*minv2,mvec);
    at_real3_inc(&force[ii], fi);

    fk=real3_scale(-ftorsion*rjk*ninv2,nvec);
    p=real3_dot(drij,drjk)*rjkinv2;
    q=real3_dot(drkl,drjk)*rjkinv2;
    fj=real3_scale(-p,fi);
    real3_scaleinc(&fj,-q,fk);
    fl=real3_scale(-1,fk);
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

void getforce_dihe(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  int N=p->diheCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (N==0) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eedihe;
  }

  getforce_torsion_kernel <DihePotential> <<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->dihes_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
}

void getforce_impr(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  int N=p->imprCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (N==0) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eeimpr;
  }

  getforce_torsion_kernel <ImprPotential> <<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->imprs_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
}



// getforce_cmap_kernel<<<(2*N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->cmaps_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
__global__ void getforce_cmap_kernel(int cmapCount,struct CmapPotential *cmaps,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
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

  if (i<2*cmapCount) { // Threads work in pairs
    // Geometry
    cp=cmaps[i>>1];
    lastBit=threadIdx.x&1;
    ii=cp.idx[0+4*lastBit];
    jj=cp.idx[1+4*lastBit];
    kk=cp.idx[2+4*lastBit];
    ll=cp.idx[3+4*lastBit];
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];
    xl=position[ll];

    drij=real3_subpbc(xi,xj,box);
    drjk=real3_subpbc(xj,xk,box);
    drkl=real3_subpbc(xk,xl,box);
    mvec=real3_cross(drij,drjk);
    nvec=real3_cross(drjk,drkl);
    dsinp=real3_cross(mvec,nvec);
    sinp=real3_mag(dsinp);
    cosp=real3_dot(mvec,nvec);
    phi=atan2f(sinp,cosp);
    ipr=real3_dot(drij,nvec);
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
    rPhi[1-lastBit]=__shfl_xor_sync(0xFFFFFFFF,phi,1);
      // Get the remainders within each box
    invSpace=cp.ngrid*(1/(2*((float) M_PI)));
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
      lEnergy+=__shfl_xor_sync(0xFFFFFFFF,lEnergy,1);
      lEnergy=(lastBit?0:lEnergy);
    }
    fcmapPhi[0]+=fcmapPhiColumn[0];
    fcmapPhi[0]*=(lastBit?rPhi[1]:1);
    fcmapPhi[1]+=lastBit*fcmapPhiColumn[1];

      // Put partner's force in fcmap for exchange
    fcmap=fcmapPhi[1-lastBit];
    fcmap=__shfl_xor_sync(0xFFFFFFFF,fcmap,1);
      // Add own force
    fcmap+=fcmapPhi[lastBit];
    fcmap*=invSpace;
    fcmap*=l[0]*l[1]*l[2];

    // Lambda force
    if (lastBit==0) { // First partner has full energy
      if (b[0]) {
        realAtomicAdd(&lambdaForce[b[0]],l[1]*l[2]*lEnergy);
        if (b[1]) {
          realAtomicAdd(&lambdaForce[b[1]],l[0]*l[2]*lEnergy);
          if (b[2]) {
            realAtomicAdd(&lambdaForce[b[2]],l[0]*l[1]*lEnergy);
          }
        }
      }
    }

    // Spatial force
// NOTE #warning "Division and sqrt in kernel"
    minv2=1/(real3_mag2(mvec));
    ninv2=1/(real3_mag2(nvec));
    rjk=sqrt(real3_mag2(drjk));
    rjkinv2=1/(rjk*rjk);
    fi=real3_scale(-fcmap*rjk*minv2,mvec);
    at_real3_inc(&force[ii], fi);

    fk=real3_scale(-fcmap*rjk*ninv2,nvec);
    p=real3_dot(drij,drjk)*rjkinv2;
    q=real3_dot(drkl,drjk)*rjkinv2;
    fj=real3_scale(-p,fi);
    real3_scaleinc(&fj,-q,fk);
    fl=real3_scale(-1,fk);
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

void getforce_cmap(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  int N=p->cmapCount;
  int shMem=0;
  real *pEnergy=NULL;

  if (N==0) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eecmap;
  }

  getforce_cmap_kernel<<<(2*N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->cmaps_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
}
