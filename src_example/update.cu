
#include "update.h"

#include "defines.h"

#include "force.h"
#include "mersenne.h"
#include "md.h"
#include "atoms.h"
#include "state.h"
#include "parms.h"
#include "leapparms.h"
#include "times.h"
#include "nblist.h"
#include "vec.h"

// for debugging
#include <stdio.h>

__device__ struct_atoms up_at;

__device__ void leapfrog_md(struct_atoms at,struct_leapparms lp)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i >= at.Ni && i < at.Nf) {
    at.v[i]=at.v[i]-at.f[i]*lp.dt/at.m[i];
    // at.x[i]+=at.v[i]*lp.dt;
    if (at.fixx == NULL) {
      at.x[i]+=at.v[i]*lp.dt;
    } else {
      at.fixx[i]+= (fixreal) floor(( at.v[i]*lp.dt )*at.real2fixreal[i%DIM3]+0.5);
      at.x[i]=at.fixx[i]*at.fixreal2real[i%DIM3];
    }
  }

  if (i < at.N) {
    at.f[i]=0;
  }
}


__global__ void update_md()
{
  leapfrog_md(up_at,up_at.lp);
}


// __global__ void leapfrog(struct_mtstate* mtstate,struct_atoms at,struct_leapparms lp,real* box)
__device__ inline void leapfrog_di(int Noff,real* nu,struct_atoms at,struct_leapparms lp)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  real Vs, Vr, Xr, Xs;

  if (i >= at.Ni && i < at.Nf) {
    // Compute noise
    // rand_normal(nu,&(mtstate[i]));
    // Vs=SigmaVsVs*nu(:,:,1);
    Vs=at.Vs_delay[i];
    at.Vs_delay[i]=lp.SigmaVsVs*nu[i]*at.misqrt[i];
    // Xs=CondSigmaXsXs*nu(:,:,2)+XsVsCoeff*[Vs(2:end,:);zeros([1,N])];
    // Correlate Xs with Velocity noise Vs from next leapfrog step
    Xs=lp.CondSigmaXsXs*nu[at.N+i]*at.misqrt[i]+lp.XsVsCoeff*at.Vs_delay[i];
    // rand_normal(nu,&(mtstate[i]));
    // Vr=SigmaVrVr*nu(:,:,3);
    Vr=lp.SigmaVrVr*nu[2*at.N+i]*at.misqrt[i];
    // Xr=CondSigmaXrXr*nu(:,:,4)+XrVrCoeff*Vr;
    Xr=lp.CondSigmaXrXr*nu[3*at.N+i]*at.misqrt[i]+lp.XrVrCoeff*Vr;
  
    // Update v from t-0.5*dt to t+0.5*dt
    at.v[i]=lp.v_vscale*at.v[i]-lp.v_ascale*at.f[i]*(at.misqrt[i]*at.misqrt[i])+lp.v_Vsscale*Vs+Vr;
    // Update x from t to t+dt
    if (at.fixx == NULL) {
      at.x[i]+=lp.x_vscale*at.v[i]+lp.x_Xrscale*Xr+Xs;
    } else {
      at.fixx[i]+= (fixreal) floor(( lp.x_vscale*at.v[i]+lp.x_Xrscale*Xr+Xs )*at.real2fixreal[i%DIM3]+0.5);
      at.x[i]=at.fixx[i]*at.fixreal2real[i%DIM3];
    }
  }

  if (i<at.N) {
    at.f[i]=0;
  }
}


__global__ void leapfrog_d(real* nu,struct_atoms at,struct_leapparms lp)
{
  leapfrog_di(0,nu,at,lp);
}


static
void incstep(struct_md* md)
{
  md->state->step++;
}


// __global__ void update_d(int step,struct_atoms at,struct_leapparms lp,real* box,struct_mtstate mtsact,struct_mtstate mtsinact)
__global__ void update_d(int step)
{
  int N=up_at.N;
  int Nblocks=(N+BU-1)/BU;
  int mti=4*(step % (MT_N/4)); // Index within the mt array (0 to MT_N, 4 each step)
  int mtact=(step/(MT_N/4))%2; // Two mtstates (0 and 1), this is the active one
  int mtinact=(mtact+1)%2;
  
  /*if (blockIdx.x<Nblocks) {
    leapfrog_di(0,up_mts[mtact].gauss+N*mti,up_at,up_lp,up_box);
  } else if (blockIdx.x<2*Nblocks) {
    // Current step is working on mts[mtact], mti to mti+4. Generate new ints for the current step, well use them to generate gaussians in MT_N/4 steps. Generate new gaussians for the opposite step mts[mtinact]. We'll use those Gaussians in MT_N/4 steps.
    genrand_int32_part_di(Nblocks,up_mts[mtinact],up_mts[mtact], mti,mti+2);
  } else if (blockIdx.x<3*Nblocks) {
    genrand_int32_part_di(2*Nblocks,up_mts[mtinact],up_mts[mtact], mti+2,mti+4);
  } else if (blockIdx.x<4*Nblocks) {
    genrand_gauss_part_di(3*Nblocks,up_mts[mtinact],mti,mti+2);
  } else if (blockIdx.x<5*Nblocks) {
    genrand_gauss_part_di(4*Nblocks,up_mts[mtinact],mti+2,mti+4);
  }*/
  if (blockIdx.x<Nblocks) {
    leapfrog_di(0,up_at.mts[mtact].gauss+N*mti,up_at,up_at.lp);
  } else if (blockIdx.x<2*Nblocks) {
    // Current step is working on mts[mtact], mti to mti+4. Generate new ints for the current step, well use them to generate gaussians in MT_N/4 steps. Generate new gaussians for the opposite step mts[mtinact]. We'll use those Gaussians in MT_N/4 steps.
    genrand_int32_part_di(Nblocks,up_at.mts[mtinact],up_at.mts[mtact], mti,mti+4);
  } else if (blockIdx.x<3*Nblocks) {
    genrand_gauss_part_di(2*Nblocks,up_at.mts[mtinact],mti,mti+4);
  }
}


__global__ void Qupdate_d(struct_atoms at,int step)
{
  int N=at.N;
  int Nblocks=(N+BU-1)/BU;
  int mti=4*(step % (MT_N/4)); // Index within the mt array (0 to MT_N, 4 each step)
  int mtact=(step/(MT_N/4))%2; // Two mtstates (0 and 1), this is the active one
  int mtinact=(mtact+1)%2;
  
  if (blockIdx.x<Nblocks) {
    leapfrog_di(0,at.mts[mtact].gauss+N*mti,at,at.lp);
  } else if (blockIdx.x<2*Nblocks) {
    genrand_int32_part_di(Nblocks,at.mts[mtinact],at.mts[mtact], mti,mti+4);
  } else if (blockIdx.x<3*Nblocks) {
    genrand_gauss_part_di(2*Nblocks,at.mts[mtinact],mti,mti+4);
  }
}


// void upload_update_d(struct_atoms h_at,struct_leapparms h_lp,real* h_box,struct_mtstate* h_mts)
void upload_update_d(struct_atoms h_at)
{
  cudaMemcpyToSymbol(up_at, &h_at, sizeof(struct_atoms), size_t(0),cudaMemcpyHostToDevice);
  // cudaMemcpyToSymbol(up_lp, &h_lp, sizeof(struct_leapparms), size_t(0),cudaMemcpyHostToDevice);
  // cudaMemcpyToSymbol(up_box, &h_box, sizeof(real*), size_t(0),cudaMemcpyHostToDevice);
  // cudaMemcpyToSymbol(*up_mts, h_mts, 2*sizeof(struct_mtstate), size_t(0),cudaMemcpyHostToDevice);
}


void update(struct_md* md)
{
  cudaStream_t strm=md->state->stream_default;

  md->times->start=gmx_cycles_read();

  // #warning "Turned on energy conservation"
  update_d <<< 3*((md->state->atoms->N+BU-1)/BU), BU, 0, strm >>> (md->state->step);
  // #warning "Turned off Qdrive update"
  Qupdate_d <<< 3*((1+BU-1)/BU), BU, 0, strm >>> (md->state->Qdrive[0],md->state->step);
  // update_md <<< ((md->state->atoms->N+BU-1)/BU), BU, 0, strm >>> ();


  if ((md->state->step+1) % md->parms->t_output == 0) {
    resetenergy(md);
    // resetforce_d <<< 1, BU, 0, strm >>> (BU,md->state->Gt);
    // resetforce_d <<< 1, BU, 0, strm >>> (BU,md->state->Kt);
  }


  md->times->update+=(gmx_cycles_read()-md->times->start);

  incstep(md); // barrier wrapped

  // Unnecessary performance hit
  // cudaDeviceSynchronize(); // WARNING - unnecessary?
}


void resetstep(struct_md* md)
{
  md->state->step=0;
}


void resetcumcycles(struct_md* md)
{
  // int ID;
  // ID=0; // omp_get_thread_num();
  /*md->state->nlelec->cycles_sort[ID]=0;
  md->state->nlelec->cycles_check[ID]=0;
  md->state->nlelec->cycles_force[ID]=0;
  md->state->nlother->cycles_sort[ID]=0;
  md->state->nlother->cycles_check[ID]=0;
  md->state->nlother->cycles_force[ID]=0;
  md->state->nlelec->cumcycles_sort[ID]=0;
  md->state->nlelec->cumcycles_check[ID]=0;
  md->state->nlelec->cumcycles_force[ID]=0;
  md->state->nlother->cumcycles_sort[ID]=0;
  md->state->nlother->cumcycles_check[ID]=0;
  md->state->nlother->cumcycles_force[ID]=0;
  */
  fprintf(stderr,"Not reseting anything %s line %d\n",__FILE__,__LINE__);
}

