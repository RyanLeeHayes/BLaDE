#include <cuda_runtime.h>
#include <math.h>

#include "bonded.h"
#include "main/defines.h"
#include "system/system.h"
#include "system/state.h"
#include "msld/msld.h"
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
#warning "Unprotected division"
    dr=real3_subpbc(xi,xj,box);
#warning "Unprotected sqrt"
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

    // Lambda force
    if (b[0]) {
      realAtomicAdd(&lambdaForce[b[0]],l[1]*lEnergy);
      if (b[1]) {
        realAtomicAdd(&lambdaForce[b[1]],l[0]*lEnergy);
      }
    }

    // Spatial force
#warning "division in kernel"
    at_real3_scaleinc(&force[ii], fbond/r,dr);
    at_real3_scaleinc(&force[jj],-fbond/r,dr);
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l[0]*l[1];
    lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,1);
    lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,2);
    lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,4);
    lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,8);
    lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,16);
    __syncthreads();
    if ((0x1F & threadIdx.x)==0) {
      sEnergy[threadIdx.x>>5]=lEnergy;
    }
    __syncthreads();
    if (threadIdx.x < (BLBO>>5)) {
      lEnergy=sEnergy[threadIdx.x];
      lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,1);
      lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,2);
      lEnergy+=__shfl_up_sync(0xFFFFFFFF,lEnergy,4);
    }
    if (threadIdx.x==0) {
      realAtomicAdd(energy,lEnergy);
    }
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


  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d;
  }
  getforce_bond_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->bonds_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
}

#ifdef COMMENTED
WORKING HERE

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

#endif
