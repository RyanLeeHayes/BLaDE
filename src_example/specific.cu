
#include "specific.h"

#include "defines.h"

#include "md.h"
#include "topol.h"
#include "state.h"
#include "parms.h"
#include "atoms.h"
#include "sort.h"
#include "vec.h"
#include "error.h"

#include "cuda_util.h"

#include <math.h>
// #include <assert.h>
#define OPTIMIZE_SPECIFIC

__device__ int N_bond;
__device__ struct_bondparms *bondparms;
__device__ struct_bondblock bondblock;
__device__ int N_angle;
__device__ struct_angleparms *angleparms;
__device__ struct_angleblock angleblock;
__device__ int N_dih;
__device__ struct_dihparms *dihparms;
__device__ struct_dihblock dihblock;
__device__ int N_pair1;
__device__ struct_pair1parms *pair1parms;
__device__ int N_pair6;
__device__ struct_pair6parms *pair6parms;
__device__ struct_atoms sp_at;
__device__ real *sp_box;
__device__ real *sp_Gt;


__device__ void function_bond(struct_bondparms bp,real r,real *E,real *f_r,real *ddE)
{
  real k0,r0;

  k0=bp.k0;
  r0=bp.r0;

  E[0]=0.5*k0*(r-r0)*(r-r0);
  f_r[0]=k0*(r-r0);
  if (ddE) {
    ddE[0]=k0;
  }
}

// __device__ inline void getforce_bond_di(int Noff,real3* sharebuf,int N,struct_bondparms* bondparms,struct_bondblock bondblock,struct_atoms at,real* box,real* Gt)
__device__ inline void getforce_bond_di(int Noff,real3* sharebuf,int N)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  int N_loc;
  int locid[2];
  // __shared__ real3 x[2*BB];
  // __shared__ real3 f[2*BB];
  ufreal3 *x=(ufreal3*) sharebuf;
  real3 *f=sharebuf;
  int j;
  int ii,jj;
  real r;
  vec dr;
  struct_bondparms bp;
  real lGt;
  real fbond;
  ufreal3 xi,xj;
  
  N_loc=bondblock.N_local[offBlockIdx];
  for (j=0; j<2; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      locid[j]=bondblock.local[2*BB*offBlockIdx+BB*j+threadIdx.x];
      x[BB*j+threadIdx.x]=((ufreal3*) sp_at.fixx)[locid[j]];
    }
  }

  __syncthreads();

  if (i<N) {
    bp=bondparms[i];
    ii=bp.i;
    jj=bp.j;
    xi=x[ii];
    xj=x[jj];
    // vec_subquick(at.x+DIM3*ii,at.x+DIM3*jj,box,dr);
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,dr);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r=vec_mag(dr);
    function_bond(bp,r,&lGt,&fbond,NULL);
  }

  __syncthreads();

  for (j=0; j<2; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      f[BB*j+threadIdx.x]=make_real3(0.0,0.0,0.0);
    }
  }

  __syncthreads();

  if (i<N) {
    at_vec_scaleinc((real*) (&f[ii]), fbond/r,dr);
    at_vec_scaleinc((real*) (&f[jj]),-fbond/r,dr);
  }

  __syncthreads();

  if (i<N) {
    realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
  }

  for (j=0; j<2; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      at_vec_inc(sp_at.f+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
    }
  }
}

__device__ inline void gethessian_bond_di(int Noff,real3* sharebuf,int N,struct_bias bias)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  int N_loc;
  int locid[2];
  // __shared__ real3 x[2*BB];
  // __shared__ real3 f[2*BB];
  ufreal3 *x=(ufreal3*) sharebuf;
  real3 *f=sharebuf;
  __shared__ real3 dotwggU[2*BB];
  int j;
  int ii,jj;
  real r;
  vec dr;
  struct_bondparms bp;
  real lGt;
  real fbond;
  real ddE;
  vec fi;
  ufreal3 xi,xj;
  // real dBdFQ=bias.dBdFQ[0];
  real dBdFQ=1.0;
  
  N_loc=bondblock.N_local[offBlockIdx];
  for (j=0; j<2; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      locid[j]=bondblock.local[2*BB*offBlockIdx+BB*j+threadIdx.x];
      x[BB*j+threadIdx.x]=((ufreal3*) sp_at.fixx)[locid[j]];
      dotwggU[BB*j+threadIdx.x]=((real3*) bias.dotwggU)[locid[j]];
    }
  }

  __syncthreads();

  if (i<N) {
    bp=bondparms[i];
    ii=bp.i;
    jj=bp.j;
    xi=x[ii];
    xj=x[jj];
    // vec_subquick(at.x+DIM3*ii,at.x+DIM3*jj,box,dr);
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,dr);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r=vec_mag(dr);
    function_bond(bp,r,&lGt,&fbond,&ddE);
    vec dotwggUij;
    vec_subtract(((real*) (&dotwggU[ii])),((real*) (&dotwggU[jj])),dotwggUij);
    vec_scale(fi,dBdFQ*(ddE-fbond/r)*vec_dot(dotwggUij,dr)/(r*r),dr);
    vec_scaleinc(fi,dBdFQ*fbond/r,dotwggUij);
  }

  __syncthreads();

  for (j=0; j<2; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      f[BB*j+threadIdx.x]=make_real3(0.0,0.0,0.0);
    }
  }

  __syncthreads();

  if (i<N) {
    at_vec_scaleinc((real*) (&f[ii]), 1.0,fi);
    at_vec_scaleinc((real*) (&f[jj]),-1.0,fi);
  }

  __syncthreads();

  // if (i<N) {
  //   realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
  // }

  for (j=0; j<2; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      // at_vec_inc(sp_at.f+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
      at_vec_inc(bias.gFQ+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
    }
  }
}

// __global__ void getforce_bond_d(int N,struct_bondparms* bondparms,struct_bondblock bondblock,struct_atoms at,real* box,real* Gt)
__global__ void getforce_bond_d()
{
  __shared__ real3 sharebuf[4*BB];
  getforce_bond_di(0,sharebuf,N_bond);
}


void getforce_bond(struct_md* md)
{
  int N=md->parms->N_bond;
  // cudaStream_t strm=md->state->stream_bond;
  cudaStream_t strm=md->state->stream_default;
  // getforce_bond_d <<< (N+BB-1)/BB, BB, 0, strm >>> (N,md->parms->bondparms,md->parms->bondblock[0],md->state->atoms[0],md->state->box,md->state->Gt);
  getforce_bond_d <<< (N+BB-1)/BB, BB, 0, strm >>> ();
  // cudaEventCreateWithFlags(&md->state->event_bdone,cudaEventDisableTiming);
  // cudaEventRecord(md->state->event_bdone,strm);
}


__global__ void gethessian_bond_d(struct_bias bias)
{
  __shared__ real3 sharebuf[4*BB];
  gethessian_bond_di(0,sharebuf,N_bond,bias);
}


void gethessian_bond(struct_md* md)
{
  int N=md->parms->N_bond;
  cudaStream_t strm=md->state->stream_default;
  gethessian_bond_d <<< (N+BB-1)/BB, BB, 0, strm >>> (md->state->bias[0]);
}


__device__ void function_angle(struct_angleparms ap,real t,real *E,real *dE,real *ddE)
{
  real k0,t0;

  k0=ap.k0;
  t0=ap.t0;

  E[0]=0.5*k0*(t-t0)*(t-t0);
  dE[0]=k0*(t-t0);
  if (ddE) {
    ddE[0]=k0;
  }
}


// __device__ inline void getforce_angle_di(int Noff,real3* sharebuf,int N,struct_angleparms* angleparms,struct_angleblock angleblock,struct_atoms at,real* box,real* Gt)
__device__ inline void getforce_angle_di(int Noff,real3* sharebuf,int N)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  int N_loc;
  int locid[3];
  // __shared__ real3 x[3*BB];
  // __shared__ real3 f[3*BB];
  ufreal3 *x=(ufreal3*) sharebuf;
  // real3 *f=sharebuf+3*BB;
  real3 *f=sharebuf;
  int j;
  int ii,jj,kk;
  struct_angleparms ap;
  // real rij,rkj;
  vec drij,drkj;
  real t;
  real dotp, mcrop;
  vec crop;
  vec fi,fj,fk;
  real lGt;
  real dE;
  ufreal3 xi, xj, xk;

  N_loc=angleblock.N_local[offBlockIdx];
  for (j=0; j<3; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      locid[j]=angleblock.local[3*BB*offBlockIdx+BB*j+threadIdx.x];
      x[BB*j+threadIdx.x]=((ufreal3*) sp_at.fixx)[locid[j]];
    }
  }

  __syncthreads();

  if (i<N) {
    ap=angleparms[i];
    ii=ap.i;
    jj=ap.j;
    kk=ap.k;
    xi=x[ii];
    xj=x[jj];
    xk=x[kk];
    
    // vec_subquick(at.x+DIM3*ii,at.x+DIM3*jj,box,drij);
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,drij);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,drij);
    // vec_subquick(at.x+DIM3*kk,at.x+DIM3*jj,box,drkj);
    // vec_subquick((real*) &xk,(real*) &xj,sp_box,drkj);
    fixvec_sub((unsignedfixreal*) &xk,(unsignedfixreal*) &xj,sp_at.fixreal2real,drkj);
    // rij=vec_mag(drij);
    // rkj=vec_mag(drkj);
    dotp=vec_dot(drij,drkj);
    vec_cross(drij,drkj,crop); // c = a x b
    mcrop=vec_mag(crop);
    t=atan2(mcrop,dotp);
    function_angle(ap,t,&lGt,&dE,NULL);
    vec_cross(drij,crop,fi);
    vec_scaleself(fi, dE*realRecip(mcrop*vec_mag2(drij)));
    vec_cross(drkj,crop,fk);
    vec_scaleself(fk,-dE*realRecip(mcrop*vec_mag2(drkj)));
  }

  __syncthreads();

  // for (j=threadIdx.x; j<N_loc; j+=BB) {
  //   f[j]=make_real3(0.0,0.0,0.0);
  // }
  for (j=0; j<3; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      f[BB*j+threadIdx.x]=make_real3(0.0,0.0,0.0);
    }
  }

  __syncthreads();

  if (i<N) {
    at_vec_inc((real*) (&f[ii]),fi);
    at_vec_inc((real*) (&f[kk]),fk);
    for (j=0; j<DIM3; j++) {
      fj[j]=-fi[j]-fk[j];
    }
    at_vec_inc((real*) (&f[jj]),fj);
  }

  __syncthreads();

  if (i<N) {
    realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
  }

  // for (j=threadIdx.x; j<N_loc; j+=BB) {
  //   at_vec_inc(sp_at.f+DIM3*locid[j/BB],(real*) (&f[j]));
  // }
  for (j=0; j<3; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      at_vec_inc(sp_at.f+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
    }
  }
}


__device__ inline void gethessian_angle_di(int Noff,real3* sharebuf,int N,struct_bias bias)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  int N_loc;
  int locid[3];
  // __shared__ real3 x[3*BB];
  // __shared__ real3 f[3*BB];
  ufreal3 *x=(ufreal3*) sharebuf;
  // real3 *f=sharebuf+3*BB;
  real3 *f=sharebuf;
  __shared__ real3 dotwggU[3*BB];
  int j;
  int ii,jj,kk;
  struct_angleparms ap;
  real rij2,rkj2;
  vec drij,drkj;
  real t;
  real dotp, mcrop;
  vec crop;
  vec fi,fj,fk;
  vec dTi,dTk;
  real lGt;
  real dE;
  real ddE;
  ufreal3 xi, xj, xk;
  // real dBdFQ=bias.dBdFQ[0];
  real dBdFQ=1.0;

  N_loc=angleblock.N_local[offBlockIdx];
  for (j=0; j<3; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      locid[j]=angleblock.local[3*BB*offBlockIdx+BB*j+threadIdx.x];
      x[BB*j+threadIdx.x]=((ufreal3*) sp_at.fixx)[locid[j]];
      dotwggU[BB*j+threadIdx.x]=((real3*) bias.dotwggU)[locid[j]];
    }
  }

  __syncthreads();

  if (i<N) {
    ap=angleparms[i];
    ii=ap.i;
    jj=ap.j;
    kk=ap.k;
    xi=x[ii];
    xj=x[jj];
    xk=x[kk];
    
    // vec_subquick(at.x+DIM3*ii,at.x+DIM3*jj,box,drij);
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,drij);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,drij);
    // vec_subquick(at.x+DIM3*kk,at.x+DIM3*jj,box,drkj);
    // vec_subquick((real*) &xk,(real*) &xj,sp_box,drkj);
    fixvec_sub((unsignedfixreal*) &xk,(unsignedfixreal*) &xj,sp_at.fixreal2real,drkj);
    rij2=vec_mag2(drij);
    rkj2=vec_mag2(drkj);
    dotp=vec_dot(drij,drkj);
    vec_cross(drij,drkj,crop); // c = a x b
    mcrop=vec_mag(crop);
    t=atan2(mcrop,dotp);
    function_angle(ap,t,&lGt,&dE,&ddE);
    vec_cross(drij,crop,dTi);
    vec_scaleself(dTi, realRecip(mcrop*vec_mag2(drij)));
    vec_cross(drkj,crop,dTk);
    vec_scaleself(dTk,-realRecip(mcrop*vec_mag2(drkj)));

    vec dotwggUij,dotwggUkj;
    vec_subtract(((real*) (&dotwggU[ii])),((real*) (&dotwggU[jj])),dotwggUij);
    vec_subtract(((real*) (&dotwggU[kk])),((real*) (&dotwggU[jj])),dotwggUkj);

    vec_scale(fi,dBdFQ*ddE*(vec_dot(dotwggUij,dTi)+vec_dot(dotwggUkj,dTk)),dTi);
    vec_scale(fk,dBdFQ*ddE*(vec_dot(dotwggUij,dTi)+vec_dot(dotwggUkj,dTk)),dTk);

    // gigiT
    vec_scaleinc(fi,dBdFQ*dE*(-(dotp/mcrop)*vec_dot(dotwggUij,drij)/(rij2*rij2)-vec_dot(dotwggUij,dTi)/rij2),drij);
    vec_scaleinc(fi,dBdFQ*dE*(-(dotp/mcrop)*vec_dot(dotwggUij,dTi)-vec_dot(dotwggUij,drij)/rij2),dTi);
    vec_scaleinc(fi,dBdFQ*dE*((dotp/mcrop)/rij2),dotwggUij);

    // gkgkT
    vec_scaleinc(fk,dBdFQ*dE*(-(dotp/mcrop)*vec_dot(dotwggUkj,drkj)/(rkj2*rkj2)-vec_dot(dotwggUkj,dTk)/rkj2),drkj);
    vec_scaleinc(fk,dBdFQ*dE*(-(dotp/mcrop)*vec_dot(dotwggUkj,dTk)-vec_dot(dotwggUkj,drkj)/rkj2),dTk);
    vec_scaleinc(fk,dBdFQ*dE*((dotp/mcrop)/rkj2),dotwggUkj);

    // gigkT
    vec_scaleinc(fi,dBdFQ*dE*((1/mcrop)*vec_dot(dotwggUkj,drij)/rij2),drij);
    vec_scaleinc(fi,dBdFQ*dE*((1/mcrop)*vec_dot(dotwggUkj,dTi)*rij2),dTi);
    vec_scaleinc(fi,dBdFQ*dE*(-(1/mcrop)),dotwggUkj);

    // gkgiT
    vec_scaleinc(fk,dBdFQ*dE*((1/mcrop)*vec_dot(dotwggUij,drkj)/rkj2),drkj);
    vec_scaleinc(fk,dBdFQ*dE*((1/mcrop)*vec_dot(dotwggUij,dTk)*rkj2),dTk);
    vec_scaleinc(fk,dBdFQ*dE*(-(1/mcrop)),dotwggUij);
  }

  __syncthreads();

  // for (j=threadIdx.x; j<N_loc; j+=BB) {
  //   f[j]=make_real3(0.0,0.0,0.0);
  // }
  for (j=0; j<3; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      f[BB*j+threadIdx.x]=make_real3(0.0,0.0,0.0);
    }
  }

  __syncthreads();

  if (i<N) {
    at_vec_inc((real*) (&f[ii]),fi);
    at_vec_inc((real*) (&f[kk]),fk);
    for (j=0; j<DIM3; j++) {
      fj[j]=-fi[j]-fk[j];
    }
    at_vec_inc((real*) (&f[jj]),fj);
  }

  __syncthreads();

  // if (i<N) {
  //   realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
  // }

  // for (j=threadIdx.x; j<N_loc; j+=BB) {
  //   at_vec_inc(sp_at.f+DIM3*locid[j/BB],(real*) (&f[j]));
  // }
  for (j=0; j<3; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      // at_vec_inc(sp_at.f+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
      at_vec_inc(bias.gFQ+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
    }
  }
}


// __global__ void getforce_angle_d(int N,struct_angleparms* angleparms,struct_angleblock angleblock,struct_atoms at,real* box,real* Gt)
__global__ void getforce_angle_d()
{
  // __shared__ real3 sharebuf[6*BB];
  __shared__ real3 sharebuf[3*BB];
  getforce_angle_di(0,sharebuf,N_angle);
}


void getforce_angle(struct_md* md)
{
  int N=md->parms->N_angle;
  // cudaStream_t strm=md->state->stream_angle;
  cudaStream_t strm=md->state->stream_default;
  // getforce_angle_d <<< (N+BB-1)/BB, BB, 0, strm >>> (N,md->parms->angleparms,md->parms->angleblock[0],md->state->atoms[0],md->state->box,md->state->Gt);
  getforce_angle_d <<< (N+BB-1)/BB, BB, 0, strm >>> ();
  // cudaEventCreateWithFlags(&md->state->event_adone,cudaEventDisableTiming);
  // cudaEventRecord(md->state->event_adone,strm);
}


__global__ void gethessian_angle_d(struct_bias bias)
{
  __shared__ real3 sharebuf[3*BB];
  gethessian_angle_di(0,sharebuf,N_angle,bias);
}


void gethessian_angle(struct_md* md)
{
  int N=md->parms->N_angle;
  cudaStream_t strm=md->state->stream_default;
  gethessian_angle_d <<< (N+BB-1)/BB, BB, 0, strm >>> (md->state->bias[0]);
}


__device__ void function_dih(struct_dihparms dip,real phi,real* E,real* dE,real* ddE)
{
  real k0,p0,n;
  real k02,p02,n2;
  real dp;

  k0=dip.k0;
  p0=dip.p0;
  n=dip.n;
  k02=dip.k02;
  p02=dip.p02;
  n2=dip.n2;
  if (n==0) {
    dp=phi-p0;
    dp-=(2*M_PI)*floor((dp+M_PI)/(2*M_PI));
    E[0]=0.5*k0*dp*dp;
    dE[0]=k0*dp;
    if (ddE) {
      ddE[0]=k0;
    }
  } else {
    dp=n*phi-p0;
    // lGt=k0*cos(dp);
    // dGdp=-k0*n*sin(dp);
    E[0]=k0*cosreal(dp);
    dE[0]=-k0*n*sinreal(dp);
    if (ddE) {
      ddE[0]=-k0*n*n*cosreal(dp);
    }
    // Second dihedral if appropriate
    if (n2 != 0) {
      dp=n2*phi-p02;
      // lGt+=k02*cos(dp);
      // dGdp+=-k02*n2*sin(dp);
      E[0]+=k02*cosreal(dp);
      dE[0]+=-k02*n2*sinreal(dp);
      if (ddE) {
        ddE[0]+=-k02*n2*n2*cosreal(dp);
      }
    }
  }
}


// __device__ inline void getforce_dih_di(int Noff,real3* sharebuf,int N,struct_dihparms* dihparms,struct_dihblock dihblock,struct_atoms at,real* box,real* Gt)
__device__ inline void getforce_dih_di(int Noff,real3* sharebuf,int N)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  int N_loc;
  int locid[4];
  // __shared__ real3 x[4*BB];
  // __shared__ real3 f[4*BB];
  ufreal3 *x=(ufreal3*) sharebuf;
  // real3 *f=sharebuf+4*BB;
  real3 *f=sharebuf;
  int j,ii,jj,kk,ll;
  struct_dihparms dip;
  real rjk;
  vec drij,drjk,drkl;
  // vec mvec,nvec,uvec,vvec,svec;
  vec mvec,nvec;
  // real phi,sign,ipr,dp,dGdp;
  real phi,sign,ipr,dGdp;
  real cosp,sinp;
  vec dsinp;
  // real mmag2,nmag2;
  real minv2,ninv2,rjkinv2;
  // real a,b,p,q;
  real p,q;
  vec fi,fj,fk,fl;
  real lGt;
  ufreal3 xi,xj,xk,xl;

  N_loc=dihblock.N_local[offBlockIdx];
  // for (ii=threadIdx.x; ii<N_loc; ii+=BB) {
  //   locid[ii/BB]=dihblock.local[4*BB*offBlockIdx+ii];
  //   x[ii]=((real3*) sp_at.x)[locid[ii/BB]];
  //   // f[ii]=make_real3(0.0,0.0,0.0);
  // }
  for (j=0; j<4; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      locid[j]=dihblock.local[4*BB*offBlockIdx+BB*j+threadIdx.x];
      x[BB*j+threadIdx.x]=((ufreal3*) sp_at.fixx)[locid[j]];
    }
  }

  __syncthreads();

  if (i<N) {
    dip=dihparms[i];
    ii=dip.i;
    jj=dip.j;
    kk=dip.k;
    ll=dip.l;
/* GROMACS
// inside dih_angle
    *t1 = pbc_rvec_sub(pbc, xi, xj, r_ij);
    *t2 = pbc_rvec_sub(pbc, xk, xj, r_kj);
    *t3 = pbc_rvec_sub(pbc, xk, xl, r_kl);

    cprod(r_ij, r_kj, m);
    cprod(r_kj, r_kl, n);
    phi     = gmx_angle(m, n);
    ipr     = iprod(r_ij, n);
    (*sign) = (ipr < 0.0) ? -1.0 : 1.0;
    phi     = (*sign)*phi;

// inside dih_do_fup
    iprm  = iprod(m, m);
    iprn  = iprod(n, n);
    nrkj2 = iprod(r_kj, r_kj);
    toler = nrkj2*GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        nrkj_1 = gmx_invsqrt(nrkj2);
        nrkj_2 = nrkj_1*nrkj_1;
        nrkj   = nrkj2*nrkj_1;
        a      = -ddphi*nrkj/iprm;
        svmul(a, m, f_i);
        b     = ddphi*nrkj/iprn;
        svmul(b, n, f_l);
        p     = iprod(r_ij, r_kj);
        p    *= nrkj_2;
        q     = iprod(r_kl, r_kj);
        q    *= nrkj_2;
        svmul(p, f_i, uvec);
        svmul(q, f_l, vvec);
        rvec_sub(uvec, vvec, svec);
        rvec_sub(f_i, svec, f_j);
        rvec_add(f_l, svec, f_k);
        rvec_inc(f[i], f_i);
        rvec_dec(f[j], f_j);
        rvec_dec(f[k], f_k);
        rvec_inc(f[l], f_l);
    }
*/

    xi=x[ii];
    xj=x[jj];
    xk=x[kk];
    xl=x[ll];
    // vec_subquick(at.x+DIM3*ii,at.x+DIM3*jj,box,drij);
    // vec_subquick(at.x+DIM3*jj,at.x+DIM3*kk,box,drjk);
    // vec_subquick(at.x+DIM3*kk,at.x+DIM3*ll,box,drkl);
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,drij);
    // vec_subquick((real*) &xj,(real*) &xk,sp_box,drjk);
    // vec_subquick((real*) &xk,(real*) &xl,sp_box,drkl);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,drij);
    fixvec_sub((unsignedfixreal*) &xj,(unsignedfixreal*) &xk,sp_at.fixreal2real,drjk);
    fixvec_sub((unsignedfixreal*) &xk,(unsignedfixreal*) &xl,sp_at.fixreal2real,drkl);
    vec_cross(drij,drjk,mvec);
    vec_cross(drjk,drkl,nvec);
    // mmag=vec_mag(mvec);
    // nmag=vec_mag(nvec);
    // phi=acos(vec_dot(mvec,nvec)/(mmag*nmag)); // Need sign still
    // Use gmx_angle style instead, acos is unstable
    vec_cross(mvec,nvec,dsinp);
    sinp=vec_mag(dsinp);
    cosp=vec_dot(mvec,nvec);
    phi=atan2(sinp,cosp);
    ipr=vec_dot(drij,nvec);
    sign=(ipr > 0.0) ? -1.0 : 1.0; // Opposite of gromacs because m and n are opposite
    phi=sign*phi;

    function_dih(dip,phi,&lGt,&dGdp,NULL);

/*
    mmag2=vec_mag2(mvec);
    nmag2=vec_mag2(nvec);
    // rjk=vec_mag(drjk);
    rjk=__fsqrt_rn(vec_mag2(drjk));
    // a=-dGdp*rjk/(mmag*mmag);
    // a=dGdp*rjk/(mmag*mmag);
    a=dGdp*rjk*realRecip(mmag2);
    vec_scale(fi,a,mvec);
    // b=dGdp*rjk/(nmag*nmag);
    // b=-dGdp*rjk/(nmag*nmag);
    b=-dGdp*rjk*realRecip(nmag2);
    vec_scale(fl,b,nvec);
    p=-vec_dot(drij,drjk)*realRecip(rjk*rjk);
    q=-vec_dot(drkl,drjk)*realRecip(rjk*rjk);
    vec_scale(uvec,p,fi);
    vec_scale(vvec,q,fl);
    vec_subtract(uvec,vvec,svec);
    vec_subtract(fi,svec,fj);
    vec__add(fl,svec,fk);
*/
    minv2=realRecip(vec_mag2(mvec));
    ninv2=realRecip(vec_mag2(nvec));
    rjk=sqrtreal(vec_mag2(drjk));
    rjkinv2=realRecip(rjk*rjk);
    vec_scale(fi,-dGdp*rjk*minv2,mvec);
    vec_scale(fk,-dGdp*rjk*ninv2,nvec);
    p=vec_dot(drij,drjk)*rjkinv2;
    q=vec_dot(drkl,drjk)*rjkinv2;
    vec_scale(fj,-p,fi);
    vec_scaleinc(fj,-q,fk);
    vec_scale(fl,-1,fk);
    vec_dec(fk,fj);
    vec_dec(fj,fi);
  }

  __syncthreads();

  // for (j=threadIdx.x; j<N_loc; j+=BB) {
  //   f[j]=make_real3(0.0,0.0,0.0);
  // }
  for (j=0; j<4; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      f[BB*j+threadIdx.x]=make_real3(0.0,0.0,0.0);
    }
  }

  __syncthreads();

  if (i<N) {
/*
    at_vec_dec((real*) (&f[ii]),fi);
    at_vec_inc((real*) (&f[jj]),fj);
    at_vec_inc((real*) (&f[kk]),fk);
    at_vec_dec((real*) (&f[ll]),fl);
*/
    at_vec_inc((real*) (&f[ii]),fi);
    at_vec_inc((real*) (&f[jj]),fj);
    at_vec_inc((real*) (&f[kk]),fk);
    at_vec_inc((real*) (&f[ll]),fl);
  }

  __syncthreads();

  if (i<N) {
    realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
  }

  // for (ii=threadIdx.x; ii<N_loc; ii+=BB) {
  //   at_vec_inc(sp_at.f+DIM3*locid[ii/BB],(real*) (&f[ii]));
  // }
  for (j=0; j<4; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      at_vec_inc(sp_at.f+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
    }
  }
}


__device__ inline void gethessian_dih_di(int Noff,real3* sharebuf,int N,struct_bias bias)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  int N_loc;
  int locid[4];
  // __shared__ real3 x[4*BB];
  // __shared__ real3 f[4*BB];
  ufreal3 *x=(ufreal3*) sharebuf;
  // real3 *f=sharebuf+4*BB;
  real3 *f=sharebuf;
  __shared__ real3 dotwggU[4*BB];
  int j,ii,jj,kk,ll;
  struct_dihparms dip;
  real rjk;
  vec drij,drjk,drkl;
  // vec mvec,nvec,uvec,vvec,svec;
  vec mvec,nvec,svec;
  // real phi,sign,ipr,dp,dGdp;
  real phi,sign,ipr,dGdp,ddE;
  real cosp,sinp;
  vec dsinp;
  // real mmag2,nmag2;
  real minv2,ninv2,rjkinv2;
  // real a,b,p,q;
  real p,q;
  vec fi,fj,fk,fl;
  real lGt;
  ufreal3 xi,xj,xk,xl;
  // real dBdFQ=bias.dBdFQ[0];
  real dBdFQ=1.0;

  N_loc=dihblock.N_local[offBlockIdx];
  // for (ii=threadIdx.x; ii<N_loc; ii+=BB) {
  //   locid[ii/BB]=dihblock.local[4*BB*offBlockIdx+ii];
  //   x[ii]=((real3*) sp_at.x)[locid[ii/BB]];
  //   // f[ii]=make_real3(0.0,0.0,0.0);
  // }
  for (j=0; j<4; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      locid[j]=dihblock.local[4*BB*offBlockIdx+BB*j+threadIdx.x];
      x[BB*j+threadIdx.x]=((ufreal3*) sp_at.fixx)[locid[j]];
      dotwggU[BB*j+threadIdx.x]=((real3*) bias.dotwggU)[locid[j]];
    }
  }

  __syncthreads();

  if (i<N) {
    dip=dihparms[i];
    ii=dip.i;
    jj=dip.j;
    kk=dip.k;
    ll=dip.l;
/* GROMACS
// inside dih_angle
    *t1 = pbc_rvec_sub(pbc, xi, xj, r_ij);
    *t2 = pbc_rvec_sub(pbc, xk, xj, r_kj);
    *t3 = pbc_rvec_sub(pbc, xk, xl, r_kl);

    cprod(r_ij, r_kj, m);
    cprod(r_kj, r_kl, n);
    phi     = gmx_angle(m, n);
    ipr     = iprod(r_ij, n);
    (*sign) = (ipr < 0.0) ? -1.0 : 1.0;
    phi     = (*sign)*phi;

// inside dih_do_fup
    iprm  = iprod(m, m);
    iprn  = iprod(n, n);
    nrkj2 = iprod(r_kj, r_kj);
    toler = nrkj2*GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        nrkj_1 = gmx_invsqrt(nrkj2);
        nrkj_2 = nrkj_1*nrkj_1;
        nrkj   = nrkj2*nrkj_1;
        a      = -ddphi*nrkj/iprm;
        svmul(a, m, f_i);
        b     = ddphi*nrkj/iprn;
        svmul(b, n, f_l);
        p     = iprod(r_ij, r_kj);
        p    *= nrkj_2;
        q     = iprod(r_kl, r_kj);
        q    *= nrkj_2;
        svmul(p, f_i, uvec);
        svmul(q, f_l, vvec);
        rvec_sub(uvec, vvec, svec);
        rvec_sub(f_i, svec, f_j);
        rvec_add(f_l, svec, f_k);
        rvec_inc(f[i], f_i);
        rvec_dec(f[j], f_j);
        rvec_dec(f[k], f_k);
        rvec_inc(f[l], f_l);
    }
*/

    xi=x[ii];
    xj=x[jj];
    xk=x[kk];
    xl=x[ll];
    // vec_subquick(at.x+DIM3*ii,at.x+DIM3*jj,box,drij);
    // vec_subquick(at.x+DIM3*jj,at.x+DIM3*kk,box,drjk);
    // vec_subquick(at.x+DIM3*kk,at.x+DIM3*ll,box,drkl);
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,drij);
    // vec_subquick((real*) &xj,(real*) &xk,sp_box,drjk);
    // vec_subquick((real*) &xk,(real*) &xl,sp_box,drkl);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,drij);
    fixvec_sub((unsignedfixreal*) &xj,(unsignedfixreal*) &xk,sp_at.fixreal2real,drjk);
    fixvec_sub((unsignedfixreal*) &xk,(unsignedfixreal*) &xl,sp_at.fixreal2real,drkl);
    vec_cross(drij,drjk,mvec);
    vec_cross(drjk,drkl,nvec);
    // mmag=vec_mag(mvec);
    // nmag=vec_mag(nvec);
    // phi=acos(vec_dot(mvec,nvec)/(mmag*nmag)); // Need sign still
    // Use gmx_angle style instead, acos is unstable
    vec_cross(mvec,nvec,dsinp);
    sinp=vec_mag(dsinp);
    cosp=vec_dot(mvec,nvec);
    phi=atan2(sinp,cosp);
    ipr=vec_dot(drij,nvec);
    sign=(ipr > 0.0) ? -1.0 : 1.0; // Opposite of gromacs because m and n are opposite
    phi=sign*phi;

    function_dih(dip,phi,&lGt,&dGdp,&ddE);

    vec dPi,dPj,dPk;

    minv2=realRecip(vec_mag2(mvec));
    ninv2=realRecip(vec_mag2(nvec));
    rjk=sqrtreal(vec_mag2(drjk));
    rjkinv2=realRecip(rjk*rjk);
    vec_scale(dPi,-rjk*minv2,mvec);
    vec_scale(dPk,-rjk*ninv2,nvec);
    p=vec_dot(drij,drjk)*rjkinv2;
    q=vec_dot(drkl,drjk)*rjkinv2;
    vec_scale(dPj,-p,fi);
    vec_scaleinc(dPj,-q,fk);
    // vec_scale(fl,-1,fk);
    // vec_dec(fk,fj);
    // vec_dec(fj,fi);

    vec dotwggUij,dotwggUjk,dotwggUkl;
    vec_subtract(((real*) (&dotwggU[ii])),((real*) (&dotwggU[jj])),dotwggUij);
    vec_subtract(((real*) (&dotwggU[jj])),((real*) (&dotwggU[kk])),dotwggUjk);
    vec_subtract(((real*) (&dotwggU[kk])),((real*) (&dotwggU[ll])),dotwggUkl);

    vec_scale(fi,dBdFQ*ddE*(vec_dot(dotwggUij,dPi)+vec_dot(dotwggUjk,dPj)+vec_dot(dotwggUkl,dPk)),dPi);
    vec_scale(fj,dBdFQ*ddE*(vec_dot(dotwggUij,dPi)+vec_dot(dotwggUjk,dPj)+vec_dot(dotwggUkl,dPk)),dPj);
    vec_scale(fk,dBdFQ*ddE*(vec_dot(dotwggUij,dPi)+vec_dot(dotwggUjk,dPj)+vec_dot(dotwggUkl,dPk)),dPk);

    // gigiP
    vec_cross(drjk,mvec,svec);
    vec_scaleinc(fi,dBdFQ*dGdp*(-minv2)*vec_dot(dotwggUij,dPi),svec);
    vec_scaleinc(fi,dBdFQ*dGdp*(-minv2)*vec_dot(dotwggUij,svec),dPi);
    // gjgjP preview
    vec_scaleinc(fj,dBdFQ*dGdp*(-rjkinv2*rjkinv2)*vec_dot(dotwggUjk,dPi),svec);

    // gjgiP and gigjP
    vec_cross(drij,mvec,svec);
    vec_scaleinc(fi,dBdFQ*dGdp*(minv2)*vec_dot(dotwggUjk,dPi),svec);
    vec_scaleinc(fj,dBdFQ*dGdp*(minv2)*vec_dot(dotwggUij,dPi),svec);
    vec_scaleinc(fi,dBdFQ*dGdp*(minv2)*vec_dot(dotwggUjk,svec),dPi);
    vec_scaleinc(fj,dBdFQ*dGdp*(minv2)*vec_dot(dotwggUij,svec),dPi);
    vec_scaleinc(fi,dBdFQ*dGdp*(rjkinv2)*vec_dot(dotwggUjk,drjk),dPi);
    vec_scaleinc(fj,dBdFQ*dGdp*(rjkinv2)*vec_dot(dotwggUij,dPi),drjk);

    // gkgiP=gigkP=0

    // gjgjP
    vec_scaleinc(fj,dBdFQ*dGdp*(-p*minv2)*vec_dot(dotwggUjk,dPi),svec);
    vec_scaleinc(fj,dBdFQ*dGdp*(-p*minv2)*vec_dot(dotwggUjk,svec),dPi);
    vec_cross(drkl,nvec,svec);
    vec_scaleinc(fj,dBdFQ*dGdp*(q*ninv2)*vec_dot(dotwggUjk,dPk),svec);
    vec_scaleinc(fj,dBdFQ*dGdp*(q*ninv2)*vec_dot(dotwggUjk,svec),dPk);

    // gjgkP and gkgjP
    vec_scaleinc(fj,dBdFQ*dGdp*(-ninv2)*vec_dot(dotwggUkl,dPk),svec);
    vec_scaleinc(fk,dBdFQ*dGdp*(-ninv2)*vec_dot(dotwggUjk,dPk),svec);
    vec_scaleinc(fj,dBdFQ*dGdp*(-ninv2)*vec_dot(dotwggUkl,svec),dPk);
    vec_scaleinc(fk,dBdFQ*dGdp*(-ninv2)*vec_dot(dotwggUjk,svec),dPk);
    vec_scaleinc(fj,dBdFQ*dGdp*(rjkinv2)*vec_dot(dotwggUkl,dPk),drjk);
    vec_scaleinc(fk,dBdFQ*dGdp*(rjkinv2)*vec_dot(dotwggUjk,drjk),dPk);

    // gkgkP
    vec_cross(drjk,nvec,svec);
    vec_scaleinc(fk,dBdFQ*dGdp*(ninv2)*vec_dot(dotwggUkl,dPk),svec);
    vec_scaleinc(fk,dBdFQ*dGdp*(ninv2)*vec_dot(dotwggUkl,svec),dPk);
    // gjgjP finish
    vec_scaleinc(fj,dBdFQ*dGdp*(rjkinv2*rjkinv2)*vec_dot(dotwggUjk,dPk),svec);

    vec_scale(fl,-1,fk);
    vec_dec(fk,fj);
    vec_dec(fj,fi);
  }

  __syncthreads();

  // for (j=threadIdx.x; j<N_loc; j+=BB) {
  //   f[j]=make_real3(0.0,0.0,0.0);
  // }
  for (j=0; j<4; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      f[BB*j+threadIdx.x]=make_real3(0.0,0.0,0.0);
    }
  }

  __syncthreads();

  if (i<N) {
/*
    at_vec_dec((real*) (&f[ii]),fi);
    at_vec_inc((real*) (&f[jj]),fj);
    at_vec_inc((real*) (&f[kk]),fk);
    at_vec_dec((real*) (&f[ll]),fl);
*/
    at_vec_inc((real*) (&f[ii]),fi);
    at_vec_inc((real*) (&f[jj]),fj);
    at_vec_inc((real*) (&f[kk]),fk);
    at_vec_inc((real*) (&f[ll]),fl);
  }

  __syncthreads();

  // if (i<N) {
  //   realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
  // }

  // for (ii=threadIdx.x; ii<N_loc; ii+=BB) {
  //   at_vec_inc(sp_at.f+DIM3*locid[ii/BB],(real*) (&f[ii]));
  // }
  for (j=0; j<4; j++) {
    if (BB*j+threadIdx.x<N_loc) {
      // at_vec_inc(sp_at.f+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
      at_vec_inc(bias.gFQ+DIM3*locid[j],(real*) (&f[BB*j+threadIdx.x]));
    }
  }
}


// __global__ void getforce_dih_d(int N,struct_dihparms* dihparms,struct_dihblock dihblock,struct_atoms at,real* box,real* Gt)
__global__ void getforce_dih_d()
{
  // __shared__ real3 sharebuf[8*BB];
  __shared__ real3 sharebuf[4*BB];
  getforce_dih_di(0,sharebuf,N_dih);
}


void getforce_dih(struct_md* md)
{
  int N=md->parms->N_dih;
  // cudaStream_t strm=md->state->stream_dih;
  cudaStream_t strm=md->state->stream_default;
  // getforce_dih_d <<< (N+BB-1)/BB, BB, 0, strm >>> (N,md->parms->dihparms,md->parms->dihblock[0],md->state->atoms[0],md->state->box,md->state->Gt);
  getforce_dih_d <<< (N+BB-1)/BB, BB, 0, strm >>> ();
  // cudaEventCreateWithFlags(&md->state->event_ddone,cudaEventDisableTiming);
  // cudaEventRecord(md->state->event_ddone,strm);
}


__global__ void gethessian_dih_d(struct_bias bias)
{
  __shared__ real3 sharebuf[4*BB];
  gethessian_dih_di(0,sharebuf,N_dih,bias);
}


void gethessian_dih(struct_md* md)
{
  int N=md->parms->N_dih;
  cudaStream_t strm=md->state->stream_default;
  gethessian_dih_d <<< (N+BB-1)/BB, BB, 0, strm >>> (md->state->bias[0]);
}


__device__ void function_pair1(struct_pair1parms pp,real r,real *E,real *f_r,real *ddE)
{
  real C6,C12;
  real ir,ir2,ir6,ir12;

  C6=pp.C6;
  C12=pp.C12;

  ir=realRecip(r);
  ir2=ir*ir;
  ir6=ir2*ir2*ir2;
  ir12=ir6*ir6;
  E[0]=C12*ir12-C6*ir6;
  f_r[0]=-12*C12*ir12*ir+6*C6*ir6*ir;
  if (ddE) {
    ddE[0]=156*C12*ir12*ir2-42*C6*ir6*ir2;
  }
}


// __device__ inline void getforce_pair1_di(int Noff,int N,struct_pair1parms* pair1parms,struct_atoms at,real* box,real* Gt)
__device__ inline void getforce_pair1_di(int Noff,int N)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  // __shared__ real sGt[BB]; // Gt[blockDim.x];
  int ii,jj;
  struct_pair1parms pp;
  real r;
  vec dr;
  real lGt;
  real f;
  ufreal3 xi,xj;
  
  if (i<N) {
    pp=pair1parms[i];
    ii=pp.i;
    jj=pp.j;
    xi=((ufreal3*) sp_at.fixx)[ii];
    xj=((ufreal3*) sp_at.fixx)[jj];
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,dr);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r=vec_mag(dr);
    function_pair1(pp,r,&lGt,&f,NULL);
    realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
    at_vec_scaleinc(sp_at.f+DIM3*ii, f/r,dr);
    at_vec_scaleinc(sp_at.f+DIM3*jj,-f/r,dr);
  }
}


__device__ inline void gethessian_pair1_di(int Noff,int N,struct_bias bias)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  // __shared__ real sGt[BB]; // Gt[blockDim.x];
  int ii,jj;
  struct_pair1parms pp;
  real r;
  vec dr;
  real lGt;
  real f;
  real ddE;
  ufreal3 xi,xj;
  // real dBdFQ=bias.dBdFQ[0];
  real dBdFQ=1.0;
  
  if (i<N) {
    pp=pair1parms[i];
    ii=pp.i;
    jj=pp.j;
    xi=((ufreal3*) sp_at.fixx)[ii];
    xj=((ufreal3*) sp_at.fixx)[jj];
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,dr);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r=vec_mag(dr);
    function_pair1(pp,r,&lGt,&f,&ddE);
    // realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
    vec fi,dotwggUij;
    vec_subtract(bias.dotwggU+DIM3*ii,bias.dotwggU+DIM3*jj,dotwggUij);
    vec_scale(fi,dBdFQ*(ddE-f/r)*vec_dot(dotwggUij,dr)/(r*r),dr);
    vec_scaleinc(fi,dBdFQ*f/r,dotwggUij);
    // at_vec_scaleinc(sp_at.f+DIM3*ii, 1.0,fi);
    // at_vec_scaleinc(sp_at.f+DIM3*jj,-1.0,fi);
    at_vec_scaleinc(bias.gFQ+DIM3*ii, 1.0,fi);
    at_vec_scaleinc(bias.gFQ+DIM3*jj,-1.0,fi);
  }
}


// __global__ void getforce_pair1_d(int N,struct_pair1parms* pair1parms,struct_atoms at,real* box,real* Gt)
__global__ void getforce_pair1_d()
{
  getforce_pair1_di(0,N_pair1);
}


void getforce_pair1(struct_md* md)
{
  int N=md->parms->N_pair1;
  // cudaStream_t strm=md->state->stream_pair1;
  cudaStream_t strm=md->state->stream_default;
  // getforce_pair1_d <<< (N+BB-1)/BB, BB, 0, strm >>> (N,md->parms->pair1parms,md->state->atoms[0],md->state->box,md->state->Gt);
  getforce_pair1_d <<< (N+BB-1)/BB, BB, 0, strm >>> ();
  // cudaEventCreateWithFlags(&md->state->event_p1done,cudaEventDisableTiming);
  // cudaEventRecord(md->state->event_p1done,strm);
}


__global__ void gethessian_pair1_d(struct_bias bias)
{
    gethessian_pair1_di(0,N_pair1,bias);
}


void gethessian_pair1(struct_md* md)
{
  int N=md->parms->N_pair1;
  cudaStream_t strm=md->state->stream_default;
  if (N > 0) {
    gethessian_pair1_d <<< (N+BB-1)/BB, BB, 0, strm >>> (md->state->bias[0]);
  }
}


__device__ inline static real umbrella_switch_tgt( const struct_umbrella* const umbrella,
                                   int step )
{
  real efac = exp( -step / umbrella->Q_steps ) ;
  real tgt  = umbrella->Q_ref * ( 1-efac ) + umbrella->Q_start * efac ;
  return tgt ;
}


// sqrt(Lorentzian) umbrella
__device__ void function_pair8_t3(struct_pair8parms pp,real r,real *E,real *f_r,real *ddE,real *dddE)
{
  real A,mu,sig;
  real elon, elonsq, F;
  real lorentz,sqrtlorentz;
  real dFoF,ddFoF,dddFoF;

  A=pp.amp;
  mu=pp.mu;
  sig=pp.sigma;

  elon=(r-mu)/sig;
  elonsq=elon*elon;
  lorentz=1/(elonsq+1);
  sqrtlorentz=sqrt(lorentz);
  F=A*sqrtlorentz;
  dFoF=-elon*lorentz/sig;
  ddFoF=(2*elonsq-1)*lorentz*lorentz/(sig*sig);
  dddFoF=(-6*elonsq*elon+9*elon)*lorentz*lorentz*lorentz/(sig*sig*sig);
  E[0]=F;
  f_r[0]=dFoF*F;
  if (ddE) {
    ddE[0]=ddFoF*F;
  }
  if (dddE) {
    dddE[0]=dddFoF*F;
  }
}


// Lorentzian umbrella
__device__ void function_pair8_t2(struct_pair8parms pp,real r,real *E,real *f_r,real *ddE,real *dddE)
{
  real A,mu,sig;
  real elon, elonsq, F;
  real lorentz;
  real dFoF,ddFoF,dddFoF;

  A=pp.amp;
  mu=pp.mu;
  sig=pp.sigma;

  elon=(r-mu)/sig;
  elonsq=elon*elon;
  lorentz=1/(elonsq+1);
  F=A*lorentz;
  dFoF=-2*elon*lorentz/sig;
  ddFoF=(6*elonsq-2)*lorentz*lorentz/(sig*sig);
  dddFoF=(-24*elonsq*elon+24*elon)*lorentz*lorentz*lorentz/(sig*sig*sig);
  E[0]=F;
  f_r[0]=dFoF*F;
  if (ddE) {
    ddE[0]=ddFoF*F;
  }
  if (dddE) {
    dddE[0]=dddFoF*F;
  }
}


// Full Gaussian umbrella
__device__ void function_pair8_t1(struct_pair8parms pp,real r,real *E,real *f_r,real *ddE,real *dddE)
{
  real A,mu,sig;
  real elon, elonsq, twosigsq, F;
  real dFoF,ddFoF;

  A=pp.amp;
  mu=pp.mu;
  sig=pp.sigma;

  // if (r<mu) {
  //   F=A;
  //   ddFoF=0;
  //   dFoF=0;
  // } else {
    elon=r-mu;
    elonsq=elon*elon;
    twosigsq=2*sig*sig;
    F=A*exp(-elonsq/twosigsq);
    ddFoF=(-2.0/twosigsq);
    dFoF=elon*ddFoF;
  // }
  E[0]=F;
  f_r[0]=dFoF*F;
  if (ddE) {
    ddE[0]=(dFoF*dFoF+ddFoF)*F;
  }
  if (dddE) {
    dddE[0]=(dFoF*dFoF*dFoF+3*ddFoF*dFoF)*F;
  }
}


__device__ void function_pair8(int type,struct_pair8parms pp,real r,real *E,real *f_r,real *ddE,real *dddE)
{
  if (type==1) {
    function_pair8_t1(pp,r,E,f_r,ddE,dddE);
  } else if (type==2) {
    function_pair8_t2(pp,r,E,f_r,ddE,dddE);
  } else if (type==3) {
    function_pair8_t3(pp,r,E,f_r,ddE,dddE);
  }
}


__global__ void getforce_pair8_d(int step,int N,struct_pair8parms *pair8parms,struct_umbrella umbrella)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  struct_pair8parms pp;
  real k_Q, Q_0, F_Q;
  __shared__ real Q_local[1024]; // 1024 max umbrella size
  real Q_global;
  real r2, r;
  vec dr;
  real fbond;
  ufreal3 xi,xj;

  if (i<N) {
    pp=pair8parms[i];
    ii=pp.i;
    jj=pp.j;

    xi=((ufreal3*) sp_at.fixx)[ii];
    xj=((ufreal3*) sp_at.fixx)[jj];
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r2=vec_mag2(dr);
    r=sqrt(r2);

    function_pair8(umbrella.type,pp,r,&Q_local[i],&fbond,NULL,NULL);
  }

  __syncthreads();
  reduce(N,Q_local,NULL);
  Q_global=Q_local[0];

  k_Q=umbrella.k_umb;
  Q_0=umbrella_switch_tgt(&umbrella,step);

  F_Q=k_Q*(Q_global-Q_0);

  if (i==0 && (step % umbrella.freq_out)==0) {
    umbrella.Q[0]=Q_global;
    umbrella.EQ[0]=0.5*k_Q*(Q_global-Q_0)*(Q_global-Q_0);
  }

  if (i<N) {
    fbond*=F_Q;

    at_vec_scaleinc(sp_at.f+DIM3*ii, fbond/r,dr);
    at_vec_scaleinc(sp_at.f+DIM3*jj,-fbond/r,dr);
  }
}


__global__ void getforce2_pair8_d(int N,struct_pair8parms *pair8parms,struct_bias bias)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  struct_pair8parms pp;
  __shared__ real Q_local[1024]; // 1024 max umbrella size
  real r2, r;
  vec dr;
  vec fi;
  vec gQ;
  real fbond;
  real ddE;
  ufreal3 xi,xj;

  if (i<N) {
    pp=pair8parms[i];
    ii=pp.i;
    jj=pp.j;

    xi=((ufreal3*) sp_at.fixx)[ii];
    xj=((ufreal3*) sp_at.fixx)[jj];
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r2=vec_mag2(dr);
    r=sqrt(r2);

    function_pair8(bias.type,pp,r,&Q_local[i],&fbond,&ddE,NULL);
  }

  __syncthreads();
  reduce(N,Q_local,bias.Q);

  if (i<N) {
    at_vec_scaleinc(bias.gQ+DIM3*ii, fbond/r,dr);
    at_vec_scaleinc(bias.gQ+DIM3*jj,-fbond/r,dr);
  }
// #warning "Turned g(gQ/gQgQ) off"
/**/
  // del squared of Q (ggQ)
  __syncthreads();
  if (i<N) {
    Q_local[i]=2*(ddE+2*fbond/r);
  }
  __syncthreads();
  reduce(N,Q_local,bias.ggQ);

  // gQ.ggQ.gQ
  __syncthreads();
  if (i<N) {
    vec_subtract(bias.gQ+DIM3*ii,bias.gQ+DIM3*jj,gQ);
    vec_scale(fi,(ddE-fbond/r)*vec_dot(gQ,dr)/r2,dr);
    vec_scaleinc(fi,fbond/r,gQ);
    at_vec_scaleinc(bias.ggQgQ+DIM3*ii, 1.0,fi);
    at_vec_scaleinc(bias.ggQgQ+DIM3*jj,-1.0,fi);

    Q_local[i]=vec_dot(gQ,fi);
  }
  __syncthreads();
  reduce(N,Q_local,bias.gQggQgQ);  
/**/
}


__global__ void gethessian2_pair8_d(int N,struct_pair8parms *pair8parms,struct_bias bias,struct_atoms Qdrive)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj;
  struct_pair8parms pp;
  __shared__ real Q_local[1024]; // 1024 max umbrella size
  real r2, r;
  vec dr;
  vec fi;
  vec dotwggQ;
  real fbond;
  real ddE;
  real dddE;
  ufreal3 xi,xj;
  real dBdQ=bias.dBdQ[0];
  // real dBdFQ=bias.dBdFQ[0];
  real dBdFQ=1.0;
  real k;

  if (i<N) {
    pp=pair8parms[i];
    ii=pp.i;
    jj=pp.j;

    xi=((ufreal3*) sp_at.fixx)[ii];
    xj=((ufreal3*) sp_at.fixx)[jj];
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r2=vec_mag2(dr);
    r=sqrt(r2);

    function_pair8(bias.type,pp,r,&Q_local[i],&fbond,&ddE,&dddE);
  }

  #warning "Hardcoded Qdrive spring coefficient"
  k=0.002; // CONST
  dBdQ+=k*(bias.Q[0]-Qdrive.x[0]);
  if (i==0) {
    Qdrive.f[0]=k*(Qdrive.x[0]-bias.Q[0]); // driving Q minus actual Q
  }

  // Already computed
  // __syncthreads();
  // reduce(N,Q_local,Q);
  if (i<N) {
    // Bias on Q
    vec_scale(fi,dBdQ*fbond/r,dr); // a[i]=b*c[i]

    at_vec_scaleinc(sp_at.f+DIM3*ii, 1.0,fi);
    at_vec_scaleinc(sp_at.f+DIM3*jj,-1.0,fi);

    // Bias on FQ, gUgQ/gQgQ term
    vec_subtract(bias.dotwggQ+DIM3*ii,bias.dotwggQ+DIM3*jj,dotwggQ); // c[i]=a[i]-b[i]
    // vec_scaleinc(fi,dBdFQ*(ddE-fbond/r)*vec_dot(dotwggQ,dr)/r2,dr);
    vec_scale(fi,dBdFQ*(ddE-fbond/r)*vec_dot(dotwggQ,dr)/r2,dr);
    vec_scaleinc(fi,dBdFQ*fbond/r,dotwggQ);
// #warning "Turned g(gQ/gQgQ) off"
/**/
    // Bias on FQ g(gQ/gQgQ) term
    // del squared of Q (ggQ)
    real negkT=-sp_at.lp.kT;
    real gQgQ=bias.gQgQ[0];
    real ggQ=bias.ggQ[0];
    vec_scaleinc(fi,dBdFQ*negkT*2*(dddE/r+2*ddE/r2-2*fbond/(r*r2))/gQgQ,dr);

    vec gQ;
    vec_subtract(bias.gQ+DIM3*ii,bias.gQ+DIM3*jj,gQ);
    real dr_gQ=vec_dot(gQ,dr);
    vec_scaleinc(fi,dBdFQ*negkT*(-2*ggQ/(gQgQ*gQgQ))*(ddE-fbond/r)*dr_gQ/r2,dr);
    vec_scaleinc(fi,dBdFQ*negkT*(-2*ggQ/(gQgQ*gQgQ))*fbond/r,gQ);

    // gQ.ggQ.gQ
    real gQggQgQ=bias.gQggQgQ[0];
    vec_scaleinc(fi,dBdFQ*negkT*(8*gQggQgQ/(gQgQ*gQgQ*gQgQ))*(ddE-fbond/r)*dr_gQ/r2,dr);
    vec_scaleinc(fi,dBdFQ*negkT*(8*gQggQgQ/(gQgQ*gQgQ*gQgQ))*fbond/r,gQ);

    vec_scaleinc(fi,dBdFQ*negkT*(-2/(gQgQ*gQgQ))*(dddE-3*ddE/r+3*fbond/r2)*dr_gQ*dr_gQ/(r*r2),dr);
    vec_scaleinc(fi,dBdFQ*negkT*(-2/(gQgQ*gQgQ))*(2*ddE-2*fbond/r)*dr_gQ/r2,gQ);
    vec_scaleinc(fi,dBdFQ*negkT*(-2/(gQgQ*gQgQ))*(ddE-fbond/r)*vec_dot(gQ,gQ)/r2,dr);

    vec ggQgQ;
    vec_subtract(bias.ggQgQ+DIM3*ii,bias.ggQgQ+DIM3*jj,ggQgQ);
    vec_scaleinc(fi,dBdFQ*negkT*(-4/(gQgQ*gQgQ))*(ddE-fbond/r)*vec_dot(ggQgQ,dr)/r2,dr);
    vec_scaleinc(fi,dBdFQ*negkT*(-4/(gQgQ*gQgQ))*fbond/r,ggQgQ);
/**/
    // at_vec_scaleinc(sp_at.f+DIM3*ii, 1.0,fi);
    // at_vec_scaleinc(sp_at.f+DIM3*jj,-1.0,fi);
    at_vec_scaleinc(bias.gFQ+DIM3*ii, 1.0,fi);
    at_vec_scaleinc(bias.gFQ+DIM3*jj,-1.0,fi);
  }
}


void getforce_pair8(struct_md* md)
{
  int N_pair8=md->parms->N_pair8;
  struct_pair8parms *pair8parms=md->parms->pair8parms;

  if (N_pair8>1024) {
    char message[MAXLENGTH];
    sprintf(message,"smoggpu not designed for umbrellas of more than 1024 coordinates. Current umbrella has %d coordinates.\n",N_pair8);
    fatalerror(message);
  }

  if (md->parms->umbrella) {
    getforce_pair8_d <<< 1, N_pair8, 0, md->state->stream_default >>> (md->state->step,N_pair8,pair8parms,md->parms->umbrella[0]);
  }
  if (md->parms->hessian) {
    getforce2_pair8_d <<< 1, N_pair8, 0, md->state->stream_default >>> (N_pair8,pair8parms,md->state->bias[0]);
  }
}


void gethessian_pair8(struct_md* md)
{
  int N_pair8=md->parms->N_pair8;
  struct_pair8parms *pair8parms=md->parms->pair8parms;

  gethessian2_pair8_d <<< 1, N_pair8, 0, md->state->stream_default >>> (N_pair8,pair8parms,md->state->bias[0],md->state->Qdrive[0]);
}

#ifdef UNIMPLEMENTED
void getforce_pair5(struct_md* md)
{
  int i,ii,jj,imin,imax;
  struct_pair5parms *pair5parms=md->parms->pair5parms;
  real eps,r0,sigma;
  // real r,r2;
  real r;
  real Gauss,fGauss;
  vec dr;
  real Gt=0;
  real *x=md->state->xhost;
  real *f;
  int ID,NID;
  gmx_cycles_t start;
  #pragma omp master
    start=gmx_cycles_read();
  
  ID=0; // omp_get_thread_num();
  NID=1; // omp_get_max_threads();
  imin=(md->parms->N_pair5 * ID)/NID;
  imax=(md->parms->N_pair5 * (ID+1))/NID;
  f=md->state->atoms->f;

  for (i=imin; i<imax; i++) {
    ii=pair5parms[i].i;
    jj=pair5parms[i].j;
    eps=pair5parms[i].eps;
    r0=pair5parms[i].r0;
    sigma=pair5parms[i].sigma;
    vec_subquick(x+DIM3*ii,x+DIM3*jj,md->state->box,dr);
    r=vec_mag(dr);
    // r2=r*r;
    Gauss=-eps*exp(-0.5*((r-r0)*(r-r0))/(sigma*sigma));
    if (r0==0) {
      fGauss=Gauss/(sigma*sigma);
    } else {
      fGauss=((r-r0)/r)*Gauss/(sigma*sigma);
    }
    Gt+=Gauss;
    // assert(f[DIM3*ii]>-1e4);
    // assert(f[DIM3*ii]<1e4);
    vec_scaleinc(f+DIM3*ii,-fGauss,dr);
    vec_scaleinc(f+DIM3*jj, fGauss,dr);
    // assert(f[DIM3*ii]>-1e4);
    // assert(f[DIM3*ii]<1e4);
  }

  md->state->Gt[0]+=Gt;
  md->state->Gts[eE_pair5]+=Gt;
  // gmx_cycles_t start;
  // #pragma omp master
  //   start=gmx_cycles_read();
  #pragma omp master
    md->times->force_pair+=(gmx_cycles_read()-start);
}
#endif

__device__ void function_pair6(struct_pair6parms pp,real r,real *E,real *f_r,real *ddE)
{
  real eps,r0,sigma,eps12;
  real r2,r6,r12;
  real s2;
  real Repel,Gauss,fRepel,fGauss,ddRepel,ddGauss;

  eps=pp.eps;
  r0=pp.r0;
  sigma=pp.sigma;
  eps12=pp.eps12;

  r2=r*r;
  r6=r2*r2*r2;
  r12=r6*r6;
  s2=sigma*sigma;
  Repel=eps12/r12;
  fRepel=-12.0*Repel/r;
  ddRepel=-13.0*fRepel/r;
  Gauss=exp(-0.5*((r-r0)*(r-r0))/s2);
  fGauss=-(r-r0)*Gauss/s2;
  ddGauss=-(r-r0)*fGauss/s2-Gauss/s2;
  E[0]=Repel-eps*Gauss-Repel*Gauss;
  f_r[0]=fRepel*(1-Gauss)-(eps+Repel)*fGauss;
  if (ddE) {
    ddE[0]=ddRepel*(1-Gauss)-(eps+Repel)*ddGauss-2*fRepel*fGauss;
  }
}


// __device__ inline void getforce_pair6_di(int Noff,int N,struct_pair6parms* pair6parms,struct_atoms at,real* box,real* Gt)
__device__ inline void getforce_pair6_di(int Noff,int N)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  // __shared__ real sGt[BB]; // Gt[blockDim.x];
  int ii,jj;
  struct_pair6parms pp;
  real r;
  vec dr;
  real lGt;
  real f;
  ufreal3 xi,xj;
  
  if (i<N) {
    pp=pair6parms[i];
    ii=pp.i;
    jj=pp.j;
    xi=((ufreal3*) sp_at.fixx)[ii];
    xj=((ufreal3*) sp_at.fixx)[jj];
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,dr);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r=vec_mag(dr);
    function_pair6(pp,r,&lGt,&f,NULL);
    realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
    at_vec_scaleinc(sp_at.f+DIM3*ii, f/r,dr);
    at_vec_scaleinc(sp_at.f+DIM3*jj,-f/r,dr);
  }
}


__device__ inline void gethessian_pair6_di(int Noff,int N,struct_bias bias)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  // __shared__ real sGt[BB]; // Gt[blockDim.x];
  int ii,jj;
  struct_pair6parms pp;
  real r;
  vec dr;
  real lGt;
  real f;
  real ddE;
  ufreal3 xi,xj;
  // real dBdFQ=bias.dBdFQ[0];
  real dBdFQ=1.0;
  
  if (i<N) {
    pp=pair6parms[i];
    ii=pp.i;
    jj=pp.j;
    xi=((ufreal3*) sp_at.fixx)[ii];
    xj=((ufreal3*) sp_at.fixx)[jj];
    // vec_subquick((real*) &xi,(real*) &xj,sp_box,dr);
    fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj,sp_at.fixreal2real,dr);
    r=vec_mag(dr);
    function_pair6(pp,r,&lGt,&f,&ddE);
    // realAtomicAdd(&sp_Gt[threadIdx.x],lGt);
    vec fi,dotwggUij;
    vec_subtract(bias.dotwggU+DIM3*ii,bias.dotwggU+DIM3*jj,dotwggUij);
    vec_scale(fi,dBdFQ*(ddE-f/r)*vec_dot(dotwggUij,dr)/(r*r),dr);
    vec_scaleinc(fi,dBdFQ*f/r,dotwggUij);
    // at_vec_scaleinc(sp_at.f+DIM3*ii, 1.0,fi);
    // at_vec_scaleinc(sp_at.f+DIM3*jj,-1.0,fi);
    at_vec_scaleinc(bias.gFQ+DIM3*ii, 1.0,fi);
    at_vec_scaleinc(bias.gFQ+DIM3*jj,-1.0,fi);
  }
}


// __global__ void getforce_pair6_d(int N,struct_pair6parms* pair6parms,struct_atoms at,real* box,real* Gt)
__global__ void getforce_pair6_d()
{
  getforce_pair6_di(0,N_pair6);
}


void getforce_pair6(struct_md* md)
{
  int N=md->parms->N_pair6;
  // cudaStream_t strm=md->state->stream_pair6;
  cudaStream_t strm=md->state->stream_default;
  // getforce_pair6_d <<< (N+BB-1)/BB, BB, 0, strm >>> (N,md->parms->pair6parms,md->state->atoms[0],md->state->box,md->state->Gt);
  if (N > 0) {
    getforce_pair6_d <<< (N+BB-1)/BB, BB, 0, strm >>> ();
  }
  // cudaEventCreateWithFlags(&md->state->event_p1done,cudaEventDisableTiming);
  // cudaEventRecord(md->state->event_p1done,strm);
}


__global__ void gethessian_pair6_d(struct_bias bias)
{
  gethessian_pair6_di(0,N_pair6,bias);
}


void gethessian_pair6(struct_md* md)
{
  int N=md->parms->N_pair6;
  cudaStream_t strm=md->state->stream_default;
  if (N > 0) {
    gethessian_pair6_d <<< (N+BB-1)/BB, BB, 0, strm >>> (md->state->bias[0]);
  }
}

#ifdef UNIMPLEMENTED
void getforce_pair7(struct_md* md)
{
  int i,ii,jj,imin,imax;
  struct_pair7parms *pair7parms=md->parms->pair7parms;
  real eps,r01,sigma1,r02,sigma2,eps12;
  real r,r2,r6,r12;
  real R, F, G ;
  real AF, AG, AFG, FR, GR, FGR ;
  real ffac, gfac, rfac ;
  real fval ;
  vec dr;
  real Gt=0;
  real *x=md->state->xhost;
  real *f;
  int ID,NID;
  gmx_cycles_t start;
  #pragma omp master
    start=gmx_cycles_read();

  ID=0; // omp_get_thread_num();
  NID=1; // omp_get_max_threads();
  imin=(md->parms->N_pair7 * ID)/NID;
  imax=(md->parms->N_pair7 * (ID+1))/NID;
  f=md->state->atoms->f;

  for (i=imin; i<imax; i++) {
    ii=pair7parms[i].i;
    jj=pair7parms[i].j;
    eps=pair7parms[i].eps;
    r01=pair7parms[i].r01;
    sigma1=pair7parms[i].sigma1;
    r02=pair7parms[i].r02;
    sigma2=pair7parms[i].sigma2;
    eps12=pair7parms[i].eps12;
    vec_subquick(x+DIM3*ii,x+DIM3*jj,md->state->box,dr);
    r=vec_mag(dr);
    r2=r*r;
    r6=r2*r2*r2;
    r12=r6*r6;
    R = eps12/r12;
    F = exp( -0.5*((r-r01)*(r-r01))/(sigma1*sigma1) );
    G = exp( -0.5*((r-r02)*(r-r02))/(sigma2*sigma2) );

    AF  = eps * F;
    AG  = eps * G;
    AFG = AF * G;
    FR  = F * R;
    GR  = G * R;
    FGR = F * GR;

    ffac = -(r-r01) / (sigma1*sigma1);
    gfac = -(r-r02) / (sigma2*sigma2);
    rfac = -12.0 / r;

    fval = ffac * AF + gfac * AG - ( ffac + gfac ) * AFG  \
         + ( ffac + rfac ) * FR + ( gfac + rfac ) * GR    \
         - ( ffac + gfac + rfac ) * FGR - rfac * R;
    fval /= r ;

    Gt += AFG - AF - AG + FGR - FR - GR + R;
    // assert(f[DIM3*ii]>-1e4);
    // assert(f[DIM3*ii]<1e4);
    vec_scaleinc(f+DIM3*ii,-(fval),dr);
    vec_scaleinc(f+DIM3*jj, (fval),dr);
    // assert(f[DIM3*ii]>-1e4);
    // assert(f[DIM3*ii]<1e4);
  }

  md->state->Gt[0]+=Gt;
  md->state->Gts[eE_pair7]+=Gt;
  // gmx_cycles_t start;
  // #pragma omp master
  //   start=gmx_cycles_read();
  #pragma omp master
    md->times->force_pair+=(gmx_cycles_read()-start);
}
#endif

// getforce_bonded_d(md->parms->N_bond,md->parms->bondparms,md->parms->bondblock[0],md->parms->N_angle,md->parms->angleparms,md->parms->angleblock[0],md->parms->N_dih,md->parms->dihparms,md->parms->dihblock[0],md->parms->N_pair1,md->parms->pair1parms,md->state->atoms[0],md->state->box,md->state->Gt);
// __global__ void getforce_bonded_d(int N_b,struct_bondparms* bondparms,struct_bondblock bondblock,int N_a,struct_angleparms* angleparms,struct_angleblock angleblock,int N_d,struct_dihparms* dihparms,struct_dihblock dihblock,int N_p1,struct_pair1parms* pair1parms,struct_atoms at,real* box,real* Gt)
__global__ void getforce_bonded_d()
{
  __shared__ real3 sharebuf[4*BB];
  int Nblock[5];
  Nblock[0]=0;
  Nblock[1]=Nblock[0]+(N_bond+BB-1)/BB;
  Nblock[2]=Nblock[1]+(N_angle+BB-1)/BB;
  Nblock[3]=Nblock[2]+(N_dih+BB-1)/BB;
  Nblock[4]=Nblock[3]+(N_pair1+BB-1)/BB;

  if (blockIdx.x<Nblock[1]) {
    getforce_bond_di(Nblock[0],sharebuf,N_bond);
  } else if (blockIdx.x<Nblock[2]) {
// #warning "Turned off angles and dihedrals"
    getforce_angle_di(Nblock[1],sharebuf,N_angle);
  } else if (blockIdx.x<Nblock[3]) {
    getforce_dih_di(Nblock[2],sharebuf,N_dih);
  } else if (blockIdx.x<Nblock[4]) {
    getforce_pair1_di(Nblock[3],N_pair1);
  }
}


__global__ void gethessian_bonded_d(struct_bias bias)
{
  __shared__ real3 sharebuf[4*BB];
  int Nblock[5];
  Nblock[0]=0;
  Nblock[1]=Nblock[0]+(N_bond+BB-1)/BB;
  Nblock[2]=Nblock[1]+(N_angle+BB-1)/BB;
  Nblock[3]=Nblock[2]+(N_dih+BB-1)/BB;
  Nblock[4]=Nblock[3]+(N_pair1+BB-1)/BB;

  if (blockIdx.x<Nblock[1]) {
    gethessian_bond_di(Nblock[0],sharebuf,N_bond,bias);
  } else if (blockIdx.x<Nblock[2]) {
// #warning "Turned off angles and dihedrals"
    gethessian_angle_di(Nblock[1],sharebuf,N_angle,bias);
  } else if (blockIdx.x<Nblock[3]) {
    gethessian_dih_di(Nblock[2],sharebuf,N_dih,bias);
  } else if (blockIdx.x<Nblock[4]) {
    gethessian_pair1_di(Nblock[3],N_pair1,bias);
  }
}


void upload_bonded_d(
  int N_b,struct_bondparms* h_bondparms,struct_bondblock* h_bondblock,
  int N_a,struct_angleparms* h_angleparms,struct_angleblock* h_angleblock,
  int N_d,struct_dihparms* h_dihparms,struct_dihblock* h_dihblock,
  int N_p1,struct_pair1parms* h_pair1parms,
  int N_p6,struct_pair6parms* h_pair6parms,
  struct_atoms h_at,real* h_box,real* h_Gt)
{
  cudaMemcpyToSymbol(N_bond, &N_b, sizeof(int), size_t(0),cudaMemcpyHostToDevice);
  if (N_b) {
    cudaMemcpyToSymbol(bondparms, &h_bondparms, sizeof(struct_bondparms*), size_t(0),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(bondblock, &h_bondblock[0], sizeof(struct_bondblock), size_t(0),cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(N_angle, &N_a, sizeof(int), size_t(0),cudaMemcpyHostToDevice);
  if (N_a) {
    cudaMemcpyToSymbol(angleparms, &h_angleparms, sizeof(struct_angleparms*), size_t(0),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(angleblock, &h_angleblock[0], sizeof(struct_angleblock), size_t(0),cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(N_dih, &N_d, sizeof(int), size_t(0),cudaMemcpyHostToDevice);
  if (N_d) {
    cudaMemcpyToSymbol(dihparms, &h_dihparms, sizeof(struct_dihparms*), size_t(0),cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dihblock, &h_dihblock[0], sizeof(struct_dihblock), size_t(0),cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(N_pair1, &N_p1, sizeof(int), size_t(0),cudaMemcpyHostToDevice);
  if (N_p1) {
    cudaMemcpyToSymbol(pair1parms, &h_pair1parms, sizeof(struct_pair1parms*), size_t(0),cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(N_pair6, &N_p6, sizeof(int), size_t(0),cudaMemcpyHostToDevice);
  if (N_p6) {
    cudaMemcpyToSymbol(pair6parms, &h_pair6parms, sizeof(struct_pair6parms*), size_t(0),cudaMemcpyHostToDevice);
  }
  cudaMemcpyToSymbol(sp_at, &h_at, sizeof(struct_atoms), size_t(0),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(sp_box, &h_box, sizeof(real*), size_t(0),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(sp_Gt, &h_Gt, sizeof(real*), size_t(0),cudaMemcpyHostToDevice);
}
