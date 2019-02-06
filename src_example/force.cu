
// #include <omp.h>

#include "force.h"

#include "sort.h"
#include "leapparms.h"
#include "specific.h"
#include "unspecific.h"
#include "nblist.h"
#include "times.h"
#include "parms.h"
#include "atoms.h"
#include "md.h"
#include "state.h"
#include "files.h"
#include "adapt.h"
#include "error.h"

#include "cuda_util.h"


__global__
void reduce_energy_d(real* Et)
{
  __shared__ real sEt[BU];

  sEt[threadIdx.x]=Et[threadIdx.x];

  __syncthreads();
  reduce(BU,sEt,Et);
}

__global__
void reduce_fma_energy_d(real* Et,real mul,real add)
{
  __shared__ real sEt[BU];

  sEt[threadIdx.x]=Et[threadIdx.x];
  if (threadIdx.x==0) {
    sEt[threadIdx.x]+=add;
  }
  sEt[threadIdx.x]*=mul;

  __syncthreads();
  reduce(BU,sEt,Et);
}

__global__
void getkinetic_d(struct_atoms at,real* Kt)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lKt;

  if (i < at.N) {
    lKt=0.5*at.m[i]*at.v[i]*at.v[i];
    realAtomicAdd(&Kt[threadIdx.x],lKt);
  }
}


__global__
void getkineticunit_d(struct_atoms at,real* Kt,real unit)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lKt;

  if (i >= at.Ni && i < at.Nf) {
    lKt=0.5*unit*at.m[i]*at.v[i]*at.v[i];
    realAtomicAdd(&Kt[threadIdx.x],lKt);
  }
}


static
void getkinetic(struct_md *md)
{
  cudaStream_t strm=md->state->stream_kinetic;

  getkinetic_d <<< (md->state->atoms->N+BU-1)/BU, BU, 0, strm >>> (md->state->atoms[0],md->state->Kt);
  cudaEventCreateWithFlags(&md->state->event_kdone,cudaEventDisableTiming);
  cudaEventRecord(md->state->event_kdone,strm);
}


__global__
void get_dotgrads_d(int N,real* gU,real* gQ,real* gUgQ,real* gQgQ)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ real buf[BU];

  if (i<N) {
    buf[threadIdx.x]=gU[i]*gQ[i];
  } else {
    buf[threadIdx.x]=0;
  }

  __syncthreads();
  reduce(BU,buf,gUgQ);

  __syncthreads();

  if (i<N) {
    buf[threadIdx.x]=gQ[i]*gQ[i];
  } else {
    buf[threadIdx.x]=0;
  }

  __syncthreads();
  reduce(BU,buf,gQgQ);

}  


__global__
void get_dotwgg_d(struct_atoms at,struct_bias bias)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i==0) {
    bias.FQ[0]=(bias.gUgQ[0]-at.lp.kTeff*(bias.ggQ[0]-2*bias.gQggQgQ[0]/bias.gQgQ[0]))/bias.gQgQ[0];
  }

  if (i<at.N) {
    bias.dotwggU[i]=bias.gQ[i]/bias.gQgQ[0];
    bias.dotwggQ[i]=(at.f[i]-2*bias.gQ[i]*bias.gUgQ[0]/bias.gQgQ[0])/bias.gQgQ[0];
  }
}


__global__ void add_energy(real* A,real a)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    realAtomicAdd(A,a);
  }
}


__global__
void get_dotgQgFQ_d(struct_atoms at,struct_bias bias)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ real buf[BU];

  if (i<at.N) {
    buf[threadIdx.x]=bias.dBdFQ[0]*bias.gQ[i]*bias.gFQ[i]/bias.gQgQ[0];
    at.f[i]+=bias.dBdFQ[0]*bias.gFQ[i];
  } else {
    buf[threadIdx.x]=0;
  }

  __syncthreads();
  reduce(BU,buf,bias.gQgFQ);

    
}


void get_FQ(struct_md* md)
{
  struct_atoms *at=md->state->atoms;
  cudaStream_t strm=md->state->stream_default;
  struct_bias *bias=md->state->bias;
  // real FQ,U;

  get_dotgrads_d <<< (at->N+BU-1)/BU, BU, 0, strm >>> (at->N,at->f,bias->gQ,bias->gUgQ,bias->gQgQ);
  // get_dotgrads_d <<< (at->N+BU-1)/BU, BU, 0, strm >>> (at->N,at->f,bias->gQ,bias->gUgQ);
  // get_dotgrads_d <<< (at->N+BU-1)/BU, BU, 0, strm >>> (at->N,bias->gQ,bias->gQ,bias->gQgQ);
/*
  cudaMemcpy(bias->Qhost,bias->Q,sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(bias->gUgQhost,bias->gUgQ,sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(bias->gQgQhost,bias->gQgQ,sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(bias->ggQhost,bias->ggQ,sizeof(real),cudaMemcpyDeviceToHost);
  cudaMemcpy(bias->gQggQgQhost,bias->gQggQgQ,sizeof(real),cudaMemcpyDeviceToHost);
  fprintf(md->files->fps[eF_log],"gUgQ,gQgQ,ggQ,gQggQgQ,gUgQ/gQgQ = %g, %g, %g, %g, %g\n",bias->gUgQhost[0],bias->gQgQhost[0],bias->ggQhost[0],bias->gQggQgQhost[0],bias->gUgQhost[0]/bias->gQgQhost[0]);

  FQ=(bias->gUgQhost[0]-md->parms->kT*(bias->ggQhost[0]-2*bias->gQggQgQhost[0]/bias->gQgQhost[0]))/bias->gQgQhost[0];

  bias->FQhost[0]=FQ;
  // fprintf(md->files->fps[eF_log],"Q,FQ = %g, %g\n",bias->Qhost[0],bias->FQhost[0]);
  cudaMemcpy(bias->FQ,bias->FQhost,sizeof(real),cudaMemcpyHostToDevice);
*/
  get_dotwgg_d <<< (at->N+BU-1)/BU, BU, 0, strm >>> (at[0],bias[0]);
/*
  // real k=md->parms->kT/(4.7074*4.7074); // kT/sigma^2 observed
  // real A=4*md->parms->kT; // 4 kT target depth
  // real s2=A/k;
  real k=md->parms->kT/(6.3*6.3); // kT/sigma^2 observed
  real A=10*md->parms->kT; // 5 kT target depth
  real s2=A/k;
  FQ-=-1; // Adjust mean of distribution
  U=A*exp(-0.5*FQ*FQ/s2);
  #warning "Turned dBdFQ off"
  // U*=0;
  bias->dBdFQ=-FQ/s2*U;
  add_energy <<< 1,1 >>> (md->state->Gt,U);
*/
// #warning "Turn off adaptive biasing"
  if (md->state->step % md->parms->t_ns == 0)
    update_history(md);

  get_metadynamics_bias(md);

  if (md->state->step % md->parms->t_ns == 0) {
    cudaMemcpy(bias->Qhost,bias->Q,sizeof(real),cudaMemcpyDeviceToHost);
    cudaMemcpy(bias->FQhost,bias->FQ,sizeof(real),cudaMemcpyDeviceToHost);
    cudaMemcpy(bias->Bhost,bias->B,sizeof(real),cudaMemcpyDeviceToHost);
    fprintf(md->files->fps[eF_log],"Q,FQ,B = %g, %g, %g\n",bias->Qhost[0],bias->FQhost[0],bias->Bhost[0]);

    cudaMemcpy(md->state->Qdrivehost,md->state->Qdrive->x,sizeof(real),cudaMemcpyDeviceToHost);
    fprintf(md->files->fps[eF_log],"Q,Qd = %g, %g\n",bias->Qhost[0],md->state->Qdrivehost[0]);

#warning "Added extra debugging crap that will slow down simulation"
    cudaMemcpy(bias->gUgQhost,bias->gUgQ,sizeof(real),cudaMemcpyDeviceToHost);
    cudaMemcpy(bias->gQgQhost,bias->gQgQ,sizeof(real),cudaMemcpyDeviceToHost);
    cudaMemcpy(bias->ggQhost,bias->ggQ,sizeof(real),cudaMemcpyDeviceToHost);
    cudaMemcpy(bias->gQggQgQhost,bias->gQggQgQ,sizeof(real),cudaMemcpyDeviceToHost);
//     fprintf(md->files->fps[eF_log],"gUgQ,gQgQ,ggQ,gQggQgQ,gUgQ/gQgQ = %g, %g, %g, %g, %g\n",bias->gUgQhost[0],bias->gQgQhost[0],bias->ggQhost[0],bias->gQggQgQhost[0],bias->gUgQhost[0]/bias->gQgQhost[0]);
    fprintf(md->files->fps[eF_log],"FQ,FQ_pot,FQ_ent = %g, %g, %g\n",
      bias->FQhost[0],
      bias->gUgQhost[0]/bias->gQgQhost[0],
      -md->parms->kT*(bias->ggQhost[0]-2*bias->gQggQgQhost[0]/bias->gQgQhost[0])/bias->gQgQhost[0]);
  }
}


__global__ void resetforce_d(int N,real* f)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N) {
    f[i]=0;
  }
}

__global__ void resetenergy_d(int N,real* E1,real* E2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N) {
    E1[i]=0;
    E2[i]=0;
  }
}


__global__ void resetbias_d(int N,struct_bias bias)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<N) {
    bias.gQ[i]=0;
    bias.gFQ[i]=0;
    // bias.dotwggU[i]=0;
    // bias.dotwggQ[i]=0;
    bias.ggQgQ[i]=0;
  }

  // if (i<3) {
  //   bias.adapt_Q_FQ[i]=0;
  // }

  if (i==0) {
    bias.gUgQ[0]=0;
    bias.gQgQ[0]=0;
    bias.Q[0]=0;
    bias.ggQ[0]=0;
    bias.gQggQgQ[0]=0;
    bias.B[0]=0;
    bias.dBdQ[0]=0;
    bias.dBdFQ[0]=0;
    bias.gQgFQ[0]=0;
  }
}

void resetforce(struct_md* md)
{
  cudaStream_t strm=md->state->stream_default;
  // struct_bias *bias=md->state->bias;

  resetforce_d <<< (md->state->atoms->N+BU-1)/BU, BU, 0, strm >>> (md->state->atoms->N,md->state->atoms->f);
}

void resetenergy(struct_md* md)
{
  cudaStream_t strm=md->state->stream_default;
  // struct_bias *bias=md->state->bias;

  resetenergy_d <<< 1, BU, 0, strm >>> (BU,md->state->Gt,md->state->Kt);
}

void resetbias(struct_md* md)
{
  cudaStream_t strm=md->state->stream_default;
  struct_bias *bias=md->state->bias;

  if (md->parms->hessian==1) {
    resetbias_d <<< (md->state->atoms->N+BU-1)/BU, BU, 0, strm >>> (md->state->atoms->N,bias[0]);
  }
}

void getforce(struct_md* md)
{
  cudaStream_t strm_d=md->state->stream_default;
  md->times->start=gmx_cycles_read();

  cudaEventRecord(md->state->event_updone,strm_d);

  {
    int Nblock=(md->parms->N_bond+BB-1)/BB+
               (md->parms->N_angle+BB-1)/BB+
               (md->parms->N_dih+BB-1)/BB+
               (md->parms->N_pair1+BB-1)/BB;
    if (Nblock) {
      getforce_bonded_d <<< Nblock, BB, 0, strm_d >>> ();
    }
  }
  getforce_pair6(md);
#ifdef UNIMPLEMENTED
  getforce_pair5(md);
  getforce_pair7(md);
#endif
  getforce_pair8(md); // umbrella potential
  if (md->state->step % md->parms->t_output == 0) {
    cudaStreamWaitEvent(md->state->stream_kinetic,md->state->event_updone,0);
    getkinetic(md); // put inside sumforce if using halfstep velocity
    cudaStreamWaitEvent(strm_d, md->state->event_kdone,0);
  }

  cudaStreamWaitEvent(md->state->stream_nonbonded,md->state->event_updone,0);
// #warning "Turned of nonbonded forces"
  getforce_other(md);
  cudaStreamWaitEvent(strm_d, md->state->event_nbdone,0);

  if (md->state->step % md->parms->t_output == 0) {
    if (md->parms->kbgo==0) {
      reduce_energy_d <<< 1, BU, 0, strm_d >>> (md->state->Gt);
      reduce_energy_d <<< 1, BU, 0, strm_d >>> (md->state->Kt);
    } else {
      reduce_fma_energy_d <<< 1, BU, 0, strm_d >>> (md->state->Gt,1/4.186,md->parms->kbgo_dih0);
      reduce_fma_energy_d <<< 1, BU, 0, strm_d >>> (md->state->Kt,1/4.186,0.0);
    }
  }
  md->times->force+=(gmx_cycles_read()-md->times->start);
}

void gethessian(struct_md* md)
{
  cudaStream_t strm_d=md->state->stream_default;

  cudaEventRecord(md->state->event_biasdone,strm_d);

  gethessian_pair8(md);

#warning "Hardcoded Hessian off"
/*
  // gethessian_bond(md);
  // // #warning "Turned off angles and dihedrals"
  // gethessian_angle(md);
  // gethessian_dih(md);
  // gethessian_pair1(md);
  {
    int Nblock=(md->parms->N_bond+BB-1)/BB+
               (md->parms->N_angle+BB-1)/BB+
               (md->parms->N_dih+BB-1)/BB+
               (md->parms->N_pair1+BB-1)/BB;
    if (Nblock) {
      gethessian_bonded_d <<< Nblock, BB, 0, strm_d >>> (md->state->bias[0]);
    }
  }

  gethessian_pair6(md);

  cudaStreamWaitEvent(md->state->stream_nonbonded,md->state->event_biasdone,0);
// #warning "Turned of nonbonded forces"
  gethessian_other(md);
  cudaStreamWaitEvent(strm_d, md->state->event_nbhdone,0);

  get_dotgQgFQ_d <<< (md->state->atoms->N+BU-1)/BU, BU, 0, strm_d >>> (md->state->atoms[0],md->state->bias[0]);
  if (md->state->step % md->parms->t_ns == 0) {
    cudaMemcpy(md->state->bias->gQgFQhost,md->state->bias->gQgFQ,sizeof(real),cudaMemcpyDeviceToHost);
    fprintf(md->files->fps[eF_log],"kTgQgFQ/gQgQ = %g\n",md->state->bias->gQgFQhost[0]);
  }
*/
}
