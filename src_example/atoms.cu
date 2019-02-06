
#include "atoms.h"
#include "leapparms.h"
#include "mersenne.h"

#include "defines.h"

#include <stdlib.h>


struct_atoms* alloc_atoms(int N,int Ni,int Nf,struct_leapparms lp,cudaStream_t strm,int pbcflag)
{
  struct_atoms *atoms;

  atoms=(struct_atoms*) malloc(sizeof(struct_atoms));

  atoms->N=N;
  atoms->Ni=Ni;
  atoms->Nf=Nf;
  cudaMalloc(&(atoms->x),N*sizeof(real));
  cudaMalloc(&(atoms->f),N*sizeof(real));
  cudaMalloc(&(atoms->v),N*sizeof(real));
  cudaMalloc(&(atoms->vhalf),N*sizeof(real));
  cudaMalloc(&(atoms->m),N*sizeof(real));
  cudaMalloc(&(atoms->misqrt),N*sizeof(real));
  cudaMalloc(&(atoms->Vs_delay),N*sizeof(real));

  if (pbcflag) {
    cudaMalloc(&(atoms->fixx),N*sizeof(unsigned fixreal));
    cudaMalloc(&(atoms->fixreal2real),DIM3*sizeof(real));
    cudaMalloc(&(atoms->real2fixreal),DIM3*sizeof(real));
  } else {
    atoms->fixx=NULL;
    atoms->fixreal2real=NULL;
    atoms->real2fixreal=NULL;
  }

  atoms->lp=lp;
  // alloc_mtstate(atoms->mts,strm,time(NULL),DIM3*N);
  alloc_mtstate(atoms->mts,strm,time(NULL),N); // already multiplied by 3

  return atoms;
}


void free_atoms(struct_atoms* atoms)
{
  cudaFree(atoms->x);
  cudaFree(atoms->f);
  cudaFree(atoms->v);
  cudaFree(atoms->vhalf);
  cudaFree(atoms->m);
  cudaFree(atoms->misqrt);
  cudaFree(atoms->Vs_delay);

  if (atoms->fixx) {
    cudaFree(atoms->fixx);
    cudaFree(atoms->fixreal2real);
    cudaFree(atoms->real2fixreal);
  }

  free_mtstate(atoms->mts,atoms->N);

  free(atoms);
}


__global__ void setfloat2fix(struct_atoms at,real* box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<DIM3) {
    at.real2fixreal[i]=FREAL_MAX/box[i];
    at.fixreal2real[i]=box[i]/FREAL_MAX;
  }
}


__global__ void float2fix(struct_atoms at)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real fixx;

  if (i<at.N) {
    fixx=at.x[i]*at.real2fixreal[i%DIM3];
    at.fixx[i]=(unsignedfixreal) (fixx-FREAL_MAX*floor(fixx/FREAL_MAX));
  }
}


__global__ void fix2float(struct_atoms at)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  if (i<at.N) {
    at.x[i]=at.fixx[i]*at.fixreal2real[i%DIM3];
  }
}
