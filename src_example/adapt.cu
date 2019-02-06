#include "adapt.h"
#include "defines.h"
#include "md.h"
#include "state.h"
#include "parms.h"

#include "cuda_util.h"

void update_history(struct_md* md)
{
  int ihistory;
  real history[6];

  if (md->state->step < md->parms->stepmeta) {
    // ihistory=md->state->step/md->parms->t_ns;
    ihistory=md->state->bias->ihistory;

    history[0]=md->state->bias->Qhost[0];
    history[1]=md->state->bias->FQhost[0];
    history[2]=md->state->bias->A;
    history[3]=1/(md->state->bias->sigQ*md->state->bias->sigQ);
    history[4]=1/(md->state->bias->sigFQ*md->state->bias->sigFQ);
    history[5]=-1; // Gaussian

    cudaMemcpy(&(md->state->bias->history[6*ihistory]),history,6*sizeof(real),cudaMemcpyHostToDevice);
    // md->state->bias->ihistory=ihistory+1;
    md->state->bias->ihistory++;
  }
}

__global__
void get_metadynamics_bias_d(struct_bias bias)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real Q0,FQ0,dQ,dFQ;
  real A,is2Q,is2FQ;
  real type;
  real Bias;
  __shared__ real B[BU];
  __shared__ real dBdQ[BU];
  __shared__ real dBdFQ[BU];

  if (i<bias.ihistory) {
    Q0=bias.history[6*i+0];
    FQ0=bias.history[6*i+1];
    A=bias.history[6*i+2];
    is2Q=bias.history[6*i+3];
    is2FQ=bias.history[6*i+4];
    type=bias.history[6*i+5];
    dQ=bias.Q[0]-Q0;
    dFQ=bias.FQ[0]-FQ0;
    if (type==-1) {
      Bias=A*exp(-0.5*dQ*dQ*is2Q)*exp(-0.5*dFQ*dFQ*is2FQ);
      B[threadIdx.x]=Bias;
      dBdQ[threadIdx.x]=-dQ*is2Q*Bias;
      dBdFQ[threadIdx.x]=-dFQ*is2FQ*Bias;
    } else if (type==1) {
      Bias=A*exp(-0.5*dQ*dQ*is2Q); // *dFQ;
      B[threadIdx.x]=Bias*dFQ;
      dBdQ[threadIdx.x]=-dQ*is2Q*Bias*dFQ;
      dBdFQ[threadIdx.x]=Bias;
    } else if (type==2) {
      Bias=A*exp(-0.5*dQ*dQ*is2Q)*dFQ; // *dFQ;
      B[threadIdx.x]=Bias*dFQ;
      dBdQ[threadIdx.x]=-dQ*is2Q*Bias*dFQ;
      dBdFQ[threadIdx.x]=2*Bias;
    }
  } else {
    B[threadIdx.x]=0;
    dBdQ[threadIdx.x]=0;
    dBdFQ[threadIdx.x]=0;
  }

  __syncthreads();

  reduce(BU,B,bias.B);
  reduce(BU,dBdQ,bias.dBdQ);
  reduce(BU,dBdFQ,bias.dBdFQ);
}

void get_metadynamics_bias(struct_md* md)
{
  struct_bias *bias=md->state->bias;

  if (bias->ihistory>0) {
    get_metadynamics_bias_d <<< (bias->ihistory+BU-1)/BU, BU >>> (bias[0]);
  }
}

void read_metadynamics_bias(struct_bias* bias,struct_md* md)
{
  int i;
  double buffer[6];
  FILE *fp;

  fp=fopen(md->parms->arg_biasfile,"r");

  if (fp) {
    i=0;
    while (fscanf(fp,"%lg %lg %lg %lg %lg %lg\n",&buffer[0],&buffer[1],&buffer[2],&buffer[3],&buffer[4],&buffer[5])==6) {
      i++;
    }
    bias->Nhistory+=i;

    if (bias->history)
      cudaFree(bias->history);
    cudaMalloc(&(bias->history),bias->Nhistory*6*sizeof(real));
    if (bias->history_buffer)
      free(bias->history_buffer);
    bias->history_buffer=(real*) malloc(bias->Nhistory*6*sizeof(real));

    fclose(fp);
    fp=fopen(md->parms->arg_biasfile,"r");

    while (fscanf(fp,"%lg %lg %lg %lg %lg %lg\n",&buffer[0],&buffer[1],&buffer[2],&buffer[3],&buffer[4],&buffer[5])==6) {
      for (i=0; i<3; i++) {
        bias->history_buffer[6*bias->ihistory+i]=buffer[i];
      }
      for (i=3; i<5; i++) {
        bias->history_buffer[6*bias->ihistory+i]=1/(buffer[i]*buffer[i]);
      }
      bias->history_buffer[6*bias->ihistory+5]=buffer[5];
      bias->ihistory++;
    }
    fclose(fp);

    cudaMemcpy(bias->history,bias->history_buffer,6*bias->ihistory*sizeof(real),cudaMemcpyHostToDevice);
  }
}

void print_metadynamics_bias(struct_md* md)
{
  struct_bias *bias=md->state->bias;
  int i;
  FILE *fp;
  char fnm[MAXLENGTH];

  cudaMemcpy(bias->history_buffer,bias->history,6*bias->ihistory*sizeof(real),cudaMemcpyDeviceToHost);

  sprintf(fnm,"bias.%d.%d.dat",md->parms->id,md->parms->phase);
  fp=fopen(fnm,"w");

  for (i=0; i<bias->ihistory; i++) {
    fprintf(fp,"%g %g %g %g %g %g\n",
      bias->history_buffer[6*i+0],
      bias->history_buffer[6*i+1],
      bias->history_buffer[6*i+2],
      1/sqrt(bias->history_buffer[6*i+3]),
      1/sqrt(bias->history_buffer[6*i+4]),
      bias->history_buffer[6*i+5]);
  }
  fclose(fp);
}
