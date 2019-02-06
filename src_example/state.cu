
#include "state.h"

#include "defines.h"

#include "mersenne.h"
#include "files.h"
#include "atoms.h"
#include "parms.h"
#include "nblist.h"
#include "topol.h"
#include "adapt.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


static
__global__
void set_masses(struct_atoms at,real m)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<at.N) {
    at.m[i]=m;
    at.misqrt[i]=1.0/sqrt(at.m[i]);
  }
}


static
void set_masses_kbgo(int N,char* topfile,struct_atoms at)
{
  FILE *fp;
  double m_d;
  real *tmp_m, m;
  int i,ii;
  char s[MAXLENGTH];

  tmp_m=(real*) calloc(3*N,sizeof(real));

  fp=fopen(topfile,"r");
  
  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (sscanf(s,"MASS %d G%d %lg",&i,&ii,&m_d)==3) {
      m=m_d;
      tmp_m[3*(i-1)+0]=m;
      tmp_m[3*(i-1)+1]=m;
      tmp_m[3*(i-1)+2]=m;
    }
  }
  fclose(fp);

  cudaMemcpy(at.m,tmp_m,3*N*sizeof(real),cudaMemcpyHostToDevice);

  for (i=0; i<3*N; i++) {
    tmp_m[i]=1.0/sqrt(tmp_m[i]);
  }

  cudaMemcpy(at.misqrt,tmp_m,3*N*sizeof(real),cudaMemcpyHostToDevice);

  free(tmp_m);
}

/*
static
__global__
void set_position(struct_atoms at,real xi)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<at.N) {
    at.x[i]=xi;
  }
}
*/

static
__global__
void set_velocities(struct_atoms at)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  struct_mtstate mts=at.mts[1];
  real kT=at.lp.kT;
  real *nu=mts.gauss;

  if (i<at.N) {
    at.v[i]=sqrt(kT/at.m[i])*nu[i];
    // at.Vs_delay[i]=lp.SigmaVsVs*nu[at.N+i]*at.misqrt[i];
    at.Vs_delay[i]=0;
  }
}


struct_bias* alloc_bias(struct_md* md,int N)
{
  struct_bias *bias;

  bias=(struct_bias*) malloc(sizeof(struct_bias));

  bias->Qhost=(real*) malloc(sizeof(real));
  bias->gUgQhost=(real*) malloc(sizeof(real));
  bias->gQgQhost=(real*) malloc(sizeof(real));
  bias->ggQhost=(real*) malloc(sizeof(real));
  bias->gQggQgQhost=(real*) malloc(sizeof(real));
  bias->FQhost=(real*) malloc(sizeof(real));
  bias->Bhost=(real*) malloc(sizeof(real));
  bias->dBdQhost=(real*) malloc(sizeof(real));
  bias->dBdFQhost=(real*) malloc(sizeof(real));
  bias->gQgFQhost=(real*) malloc(sizeof(real));
  cudaMalloc(&(bias->Q),BU*sizeof(real));
  cudaMalloc(&(bias->gUgQ),BU*sizeof(real));
  cudaMalloc(&(bias->gQgQ),BU*sizeof(real));
  cudaMalloc(&(bias->ggQ),BU*sizeof(real));
  cudaMalloc(&(bias->gQggQgQ),BU*sizeof(real));
  cudaMalloc(&(bias->FQ),BU*sizeof(real));
  cudaMalloc(&(bias->B),BU*sizeof(real));
  cudaMalloc(&(bias->dBdQ),BU*sizeof(real));
  cudaMalloc(&(bias->dBdFQ),BU*sizeof(real));
  cudaMalloc(&(bias->gQgFQ),BU*sizeof(real));

  cudaMalloc(&(bias->gQ),N*sizeof(real));
  cudaMalloc(&(bias->gFQ),N*sizeof(real));
  cudaMalloc(&(bias->dotwggU),N*sizeof(real));
  cudaMalloc(&(bias->dotwggQ),N*sizeof(real));
  cudaMalloc(&(bias->ggQgQ),N*sizeof(real));

  // bias->Nhistory=md->parms->steplim/md->parms->t_ns;
  bias->Nhistory=md->parms->stepmeta/md->parms->t_ns;
  bias->ihistory=0;
  cudaMalloc(&(bias->history),bias->Nhistory*6*sizeof(real));
  bias->history_buffer=(real*) malloc(bias->Nhistory*6*sizeof(real));

  // reads nothing if file can't be opened
  read_metadynamics_bias(bias,md); // reallocates history and history_buffer

  bias->type=md->parms->umbrella_function;
  bias->A=md->parms->bias_A;
  bias->sigQ=md->parms->bias_sigQ;
  bias->sigFQ=md->parms->bias_sigFQ;

  return bias;
}


void free_bias(struct_bias* bias)
{
  free(bias->Qhost);
  free(bias->gUgQhost);
  free(bias->gQgQhost);
  free(bias->ggQhost);
  free(bias->gQggQgQhost);
  free(bias->FQhost);
  free(bias->Bhost);
  free(bias->dBdQhost);
  free(bias->dBdFQhost);
  free(bias->gQgFQhost);
  cudaFree(bias->Q);
  cudaFree(bias->gUgQ);
  cudaFree(bias->gQgQ);
  cudaFree(bias->ggQ);
  cudaFree(bias->gQggQgQ);
  cudaFree(bias->FQ);
  cudaFree(bias->B);
  cudaFree(bias->dBdQ);
  cudaFree(bias->dBdFQ);
  cudaFree(bias->gQgFQ);

  cudaFree(bias->gQ);
  cudaFree(bias->gFQ);
  cudaFree(bias->dotwggU);
  cudaFree(bias->dotwggQ);
  cudaFree(bias->ggQgQ);

  cudaFree(bias->history);
  free(bias->history_buffer);

  free(bias);
}


struct_state* alloc_state(struct_md* md)
{
  struct_state *state;
  int N,Nbuf;
  // int id=0;
  // int i;
  // int *iitype;
  // real **rc2;
  real outputproj;
  cudaStream_t strm;

  state=(struct_state*) malloc(sizeof(struct_state));

  // cudaStreamCreate(&state->stream_random);
  cudaStreamCreate(&state->stream_bond);
  cudaStreamCreate(&state->stream_angle);
  cudaStreamCreate(&state->stream_dih);
  cudaStreamCreate(&state->stream_pair1);
  cudaStreamCreate(&state->stream_kinetic);
  cudaStreamCreate(&state->stream_nonbonded);
  // cudaStreamCreate(&state->stream_elec);
  // cudaStreamCreate(&state->stream_update);
  cudaStreamCreate(&state->stream_default);

  cudaEventCreateWithFlags(&state->event_updone,cudaEventDisableTiming);
  cudaEventCreateWithFlags(&state->event_biasdone,cudaEventDisableTiming);
  cudaEventCreateWithFlags(&state->event_nbdone,cudaEventDisableTiming);
  cudaEventCreateWithFlags(&state->event_nbhdone,cudaEventDisableTiming);
  cudaEventCreateWithFlags(&state->event_fdone,cudaEventDisableTiming);

  strm=state->stream_default;

  state->step=0;
  state->walltimeleft=1;

  cudaMalloc(&(state->box),DIM3*sizeof(real));
  state->boxhost=(real*) calloc(3,sizeof(real));
  state->boxhost[0]=state->boxhost[1]=state->boxhost[2]=md->parms->arg_box;
  cudaMemcpy(state->box,state->boxhost,DIM3*sizeof(real),cudaMemcpyHostToDevice);

  // Count atoms in file
  if (md->parms->kbgo==0) {
    state->N_all=count_gro(md->parms->arg_grofile);
  } else {
    state->N_all=count_pdb(md->parms->arg_grofile);
  }

  state->abortflag=0;

  // state->mtstate=alloc_mtstate(strm,time(NULL),DIM3*state->N_all); // CONST

  N=DIM3*state->N_all;
  Nbuf=DIM3*0; // number of stationary atoms
  state->atoms=alloc_atoms(N,Nbuf,N,md->parms->leapparms[0],strm,1);
  state->bufhost=(real*) calloc(N,sizeof(real));
  // state->atoms->shift=state->shift;
  if (md->parms->kbgo==0) {
    fprintf(stderr,"Warning: not reading masses from topology\n");
    set_masses <<< (N+BU-1)/BU, BU, 0, strm >>> (state->atoms[0],1); // CONST
  } else {
    set_masses_kbgo(state->N_all,md->parms->arg_topfile,state->atoms[0]);
  }
  set_velocities <<< (N+BU-1)/BU, BU, 0, strm >>> (state->atoms[0]);

  state->xhost=(real*) calloc(N,sizeof(real));

  state->floatx=(float*) calloc(state->N_all,DIM3*sizeof(float));
  state->res_id=(int*) calloc(state->N_all,sizeof(int));
  state->res_name=(char(*)[5]) calloc(state->N_all,sizeof(char[5]));
  state->atom_id=(int*) calloc(state->N_all,sizeof(int));
  state->atom_name=(char(*)[5]) calloc(state->N_all,sizeof(char[5]));

  // Read atoms in file
  if (md->parms->kbgo==0) {
    read_gro(md->parms->arg_grofile,state);
  } else {
    read_pdb(md->parms->arg_grofile,state);
  }

  cudaMemcpy(state->atoms->x, state->xhost, state->N_all*DIM3*sizeof(real), cudaMemcpyHostToDevice);

  N=DIM3*state->N_all;
  setfloat2fix <<< (N+BU-1)/BU, BU, 0, strm >>> (state->atoms[0],state->box);
  float2fix <<< (N+BU-1)/BU, BU, 0, strm >>> (state->atoms[0]);
  fix2float <<< (N+BU-1)/BU, BU, 0, strm >>> (state->atoms[0]);

  // Set up driving particle
  N=1;
  state->Qdrive=alloc_atoms(N,0,N,md->parms->Qleapparms[0],strm,0); // Last 0 means no pbc
  #warning "Hardcoded Qdrive mass"
  set_masses <<< (N+BU-1)/BU, BU, 0, strm >>> (state->Qdrive[0],0.8); // CONST
  set_velocities <<< (N+BU-1)/BU, BU, 0, strm >>> (state->Qdrive[0]);
  state->Qdrivehost=(real*) calloc(N,sizeof(real));
  #warning "Hardcoded Qrive initial position"
  // state->Qdrivehost[0]=450; // CONST
  state->Qdrivehost[0]=200; // CONST
  cudaMemcpy(state->Qdrive->x, state->Qdrivehost, N*sizeof(real), cudaMemcpyHostToDevice);
  
  // check outputguard
  outputproj=0;
  fprintf(stderr,"Warning, output estimation unimplemented\n");
  if (outputproj>md->parms->outputguard) {
    char message[MAXLENGTH];
    int messageoffset;
    messageoffset=sprintf(message,"projected output of %g bytes\nis greater than %g byte limit\n",outputproj,md->parms->outputguard);
    sprintf(message+messageoffset,"\nIf you are sure this is what you want, set\noutputguard = %g in your mdp file\n",2.0*outputproj);
    fatalerror(message);
  } else {
    fprintf(stderr,"Projected output is %g bytes\n",outputproj);
  }

  state->Gthost=(real*) malloc(sizeof(real));
  state->Kthost=(real*) malloc(sizeof(real));
  cudaMalloc(&(state->Gt),BU*sizeof(real));
  cudaMalloc(&(state->Kt),BU*sizeof(real));

  // if (md->parms->hessian) {
  state->bias=alloc_bias(md,state->atoms->N);
  // } // allocate it anyways, we just won't use it.

  alloc_nblist(state->N_all,TLJ,&(state->nblist_H4H),&(state->nblist_H4D));

  return state;
}


void free_state(struct_state* state)
{
  free(state->xhost);
  free(state->floatx);
  free_atoms(state->atoms);
  free(state->bufhost);
  free(state->Gthost);
  free(state->Kthost);
  cudaFree(state->Gt);
  cudaFree(state->Kt);
  free_bias(state->bias);
  free_nblist(state->nblist_H4H,state->nblist_H4D);
  free(state);
}

