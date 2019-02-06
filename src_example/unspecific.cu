#include "unspecific.h"

#include "defines.h"

#include "md.h"
#include "state.h"
#include "parms.h"
#include "atoms.h"
#include "nblist.h"
#include "vec.h"
#include "topol.h"
#include "error.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <assert.h>

// Other stuff
__device__ real k12;
__device__ real rc2;

__device__ struct_atoms un_at;
__device__ real *un_Gt;

__device__ struct_nbparms *un_nbparms;

__device__ void function_nb(real r,real ir,real* E,real* F,real* ddE)
{
  real ir2,ir4,ir8;
  real x,dx,S,dS,ddS;
  real rawE,rawF,rawH;

  E[0]=0;
  F[0]=0;
  if (ddE) {
    ddE[0]=0;
  }
#warning "Rehardcoded cutoffs in function_nb"
  // if (r*r<=rc2)
  if (r<=0.7) {
    if (r>0.6) {
      dx=1/(0.7-0.6);
      x=(r-0.6)*dx;
      S=1-6*x*x*x*x*x+15*x*x*x*x-10*x*x*x;
      dS=(-30*x*x*x*x+60*x*x*x-30*x*x)*dx;
      ddS=(-120*x*x*x+180*x*x-60*x)*dx*dx;
    } else {
      S=1;
      dS=0;
      ddS=0;
    }
    // Inverting with division is rate limitting. Use this approximate solution instead
    ir2=ir*ir;
    ir4=ir2*ir2;
    ir8=ir4*ir4;
    rawE=k12*(ir4*ir8);
    rawF=-12*rawE*ir;
    rawH=-13*rawF*ir;
    E[0] = rawE*S;
    F[0] = rawF*S + rawE*dS;
    if (ddE) {
      // ddE[0]=-13*F[0]*ir;
      ddE[0] = rawH*S + 2*rawF*dS + rawE*ddS;
    }
  }
}


__global__ void getforce_other_simple_d(struct_nblist H4D)
{
  ufreal3 xi;
  real3 fi;
  int tileOfBlockIdx=threadIdx.x/TLJ;
  int tileIdx=tileOfBlockIdx+blockIdx.x*(BNB/TLJ);
  int threadOfTileIdx=(threadIdx.x % TLJ);
  __shared__ ufreal3 xj[BNB];
  __shared__ real3 fj[BNB];
  unsigned int nlisti;

  int ii,jj;
  real lGt=0;

  if (tileIdx < H4D.Ntiles) {
    struct_ij ij;
    int i0,iN,j0,jN;

    ij=H4D.ijtile[tileIdx];

    i0=H4D.iblock[ij.i];
    iN=H4D.iblock[ij.i+1]-i0;
    j0=H4D.iblock[ij.j];
    jN=H4D.iblock[ij.j+1]-j0;

    if (threadOfTileIdx < iN) {
      ii=H4D.bin[i0+threadOfTileIdx].i;
      nlisti=H4D.nlist[blockIdx.x*BNB+threadIdx.x];
      xi=((ufreal3*) un_at.fixx)[ii];
      fi=make_real3(0.0,0.0,0.0);
      // assert((i0!=j0) || (jN>1) || (nlisti==0));
    } else {
      ii=-1;
    }

    if (threadOfTileIdx < jN) {
      if (i0==j0) {
        jj=ii;
        xj[threadIdx.x]=xi;
      } else {
        jj=H4D.bin[j0+threadOfTileIdx].i;
        xj[threadIdx.x]=((ufreal3*) un_at.fixx)[jj];
      }
      fj[threadIdx.x]=make_real3(0.0,0.0,0.0);
    } else {
      jj=-1;
    }

    // vec_inc((real*) (&xi),H4D.shift[tileIdx]); // allows use of vec_subtract instead of vec_subshift_ns
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles) {
    int i,j;
    real r,ir;
    real Eij;
    real Fij;
    vec dr;

    // if (threadOfTileIdx < iN)
    if (ii>=0) {
      for (i=0; i<TLJ; i++) {
        j=(threadOfTileIdx+i) & (TLJ-1);
        // if (j<jN) // j>=jN are only excluded via neighbor list
        if (nlisti & (1<<j)) {
          j+=tileOfBlockIdx*TLJ;
          // vec_subshift_ns((real*) (&xi),(real*) (&xj[tileOfBlockIdx*TNB+j]),shift,dr);
          // vec_subtract((real*) (&xi),(real*) (&xj[j]),dr);
          fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj[j],un_at.fixreal2real,dr);
          r=vec_mag(dr);
          ir=realRecip(r); // 1 us for exact division. See defines.h
          function_nb(r,ir,&Eij,&Fij,NULL);
          lGt+=Eij;
          vec_scaleinc((real*) (&fi),    Fij*ir,dr);
          vec_scaleinc((real*) (&fj[j]),-Fij*ir,dr);
        }
      }
      if (un_Gt) {
        realAtomicAdd(&un_Gt[threadIdx.x],lGt);
      }
    }
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles) {
    // if (threadOfTileIdx < iN)
    if (ii>=0) {
      // if (i0==j0)
      if (ii==jj) {
        at_vec_inc((real*) (&fj[threadIdx.x]),(real*) (&fi));
      } else {
        at_vec_inc(un_at.f+DIM3*ii,(real*) (&fi));
      }
    }

    // if (threadOfTileIdx < jN)
    if (jj>=0) {
      at_vec_inc(un_at.f+DIM3*jj,(real*) (&fj[threadIdx.x]));
    }
  }
}

__global__ void gethessian_other_simple_d(struct_nblist H4D,struct_bias bias)
{
  ufreal3 xi;
  real3 fi;
  real3 dotwggUi;
  int tileOfBlockIdx=threadIdx.x/TLJ;
  int tileIdx=tileOfBlockIdx+blockIdx.x*(BNB/TLJ);
  int threadOfTileIdx=(threadIdx.x % TLJ);
  __shared__ ufreal3 xj[BNB];
  __shared__ real3 fj[BNB];
  __shared__ real3 dotwggUj[BNB];
  unsigned int nlisti;
  // real dBdFQ=bias.dBdFQ[0];
  real dBdFQ=1.0;

  int ii,jj;
  real lGt=0;

  if (tileIdx < H4D.Ntiles) {
    struct_ij ij;
    int i0,iN,j0,jN;

    ij=H4D.ijtile[tileIdx];

    i0=H4D.iblock[ij.i];
    iN=H4D.iblock[ij.i+1]-i0;
    j0=H4D.iblock[ij.j];
    jN=H4D.iblock[ij.j+1]-j0;

    if (threadOfTileIdx < iN) {
      ii=H4D.bin[i0+threadOfTileIdx].i;
      nlisti=H4D.nlist[blockIdx.x*BNB+threadIdx.x];
      xi=((ufreal3*) un_at.fixx)[ii];
      dotwggUi=((real3*) bias.dotwggU)[ii];
      fi=make_real3(0.0,0.0,0.0);
      // assert((i0!=j0) || (jN>1) || (nlisti==0));
    } else {
      ii=-1;
    }

    if (threadOfTileIdx < jN) {
      if (i0==j0) {
        jj=ii;
        xj[threadIdx.x]=xi;
        dotwggUj[threadIdx.x]=dotwggUi;
      } else {
        jj=H4D.bin[j0+threadOfTileIdx].i;
        xj[threadIdx.x]=((ufreal3*) un_at.fixx)[jj];
        dotwggUj[threadIdx.x]=((real3*) bias.dotwggU)[jj];
      }
      fj[threadIdx.x]=make_real3(0.0,0.0,0.0);
    } else {
      jj=-1;
    }

    // vec_inc((real*) (&xi),H4D.shift[tileIdx]); // allows use of vec_subtract instead of vec_subshift_ns
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles) {
    int i,j;
    real r,ir;
    real Eij;
    real Fij;
    real ddEij;
    vec dr;
    vec fij,dotwggUij;

    // if (threadOfTileIdx < iN)
    if (ii>=0) {
      for (i=0; i<TLJ; i++) {
        j=(threadOfTileIdx+i) & (TLJ-1);
        // if (j<jN) // j>=jN are only excluded via neighbor list
        if (nlisti & (1<<j)) {
          j+=tileOfBlockIdx*TLJ;
          // vec_subshift_ns((real*) (&xi),(real*) (&xj[tileOfBlockIdx*TNB+j]),shift,dr);
          // vec_subtract((real*) (&xi),(real*) (&xj[j]),dr);
          fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj[j],un_at.fixreal2real,dr);
          r=vec_mag(dr);
          ir=realRecip(r); // 1 us for exact division. See defines.h
          function_nb(r,ir,&Eij,&Fij,&ddEij);
          // lGt+=Eij;
          vec_subtract((real*) &dotwggUi,(real*) &dotwggUj[j],dotwggUij);
          vec_scale(fij,dBdFQ*(ddEij-Fij*ir)*vec_dot(dotwggUij,dr)*(ir*ir),dr);
          vec_scaleinc(fij,dBdFQ*Fij*ir,dotwggUij);
          vec_scaleinc((real*) (&fi),    1.0,fij);
          vec_scaleinc((real*) (&fj[j]),-1.0,fij);
        }
      }
      if (un_Gt) {
        realAtomicAdd(&un_Gt[threadIdx.x],lGt);
      }
    }
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles) {
    // if (threadOfTileIdx < iN)
    if (ii>=0) {
      // if (i0==j0)
      if (ii==jj) {
        at_vec_inc((real*) (&fj[threadIdx.x]),(real*) (&fi));
      } else {
        // at_vec_inc(un_at.f+DIM3*ii,(real*) (&fi));
        at_vec_inc(bias.gFQ+DIM3*ii,(real*) (&fi));
      }
    }

    // if (threadOfTileIdx < jN)
    if (jj>=0) {
      // at_vec_inc(un_at.f+DIM3*jj,(real*) (&fj[threadIdx.x]));
      at_vec_inc(bias.gFQ+DIM3*jj,(real*) (&fj[threadIdx.x]));
    }
  }
}


__global__ void getforce_other_kbgo_d(struct_nblist H4D)
{
  ufreal3 xi;
  real3 fi;
  real epsi, rmini;
  int tileOfBlockIdx=threadIdx.x/TLJ;
  int tileIdx=tileOfBlockIdx+blockIdx.x*(BNB/TLJ);
  int threadOfTileIdx=(threadIdx.x % TLJ);
  __shared__ ufreal3 xj[BNB];
  __shared__ real3 fj[BNB];
  __shared__ real epsj[BNB];
  __shared__ real rminj[BNB];
  unsigned int nlisti;

  int ii,jj;
  real lGt=0;

  if (tileIdx < H4D.Ntiles) {
    struct_ij ij;
    int i0,iN,j0,jN;

    ij=H4D.ijtile[tileIdx];

    i0=H4D.iblock[ij.i];
    iN=H4D.iblock[ij.i+1]-i0;
    j0=H4D.iblock[ij.j];
    jN=H4D.iblock[ij.j+1]-j0;

    if (threadOfTileIdx < iN) {
      ii=H4D.bin[i0+threadOfTileIdx].i;
      nlisti=H4D.nlist[blockIdx.x*BNB+threadIdx.x];
      xi=((ufreal3*) un_at.fixx)[ii];
      epsi=un_nbparms[ii].eps;
      rmini=un_nbparms[ii].rmin;
      fi=make_real3(0.0,0.0,0.0);
      // assert((i0!=j0) || (jN>1) || (nlisti==0));
    } else {
      ii=-1;
    }

    if (threadOfTileIdx < jN) {
      if (i0==j0) {
        jj=ii;
        xj[threadIdx.x]=xi;
        epsj[threadIdx.x]=epsi;
        rminj[threadIdx.x]=rmini;
      } else {
        jj=H4D.bin[j0+threadOfTileIdx].i;
        xj[threadIdx.x]=((ufreal3*) un_at.fixx)[jj];
        epsj[threadIdx.x]=un_nbparms[jj].eps;
        rminj[threadIdx.x]=un_nbparms[jj].rmin;
      }
      fj[threadIdx.x]=make_real3(0.0,0.0,0.0);
    } else {
      jj=-1;
    }

    // vec_inc((real*) (&xi),H4D.shift[tileIdx]); // allows use of vec_subtract instead of vec_subshift_ns
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles) {
    int i,j;
    real r2;
    real Eij;
    real Fij;
    real ir2,ir6,ir12;
    vec dr;
    real epsij;
    real rminij;

    // if (threadOfTileIdx < iN)
    if (ii>=0) {
      for (i=0; i<TLJ; i++) {
        j=(threadOfTileIdx+i) & (TLJ-1);
        // if (j<jN) // j>=jN are only excluded via neighbor list
        if (nlisti & (1<<j)) {
          j+=tileOfBlockIdx*TLJ;
          // vec_subshift_ns((real*) (&xi),(real*) (&xj[tileOfBlockIdx*TNB+j]),shift,dr);
          // vec_subtract((real*) (&xi),(real*) (&xj[j]),dr);
          fixvec_sub((unsignedfixreal*) &xi,(unsignedfixreal*) &xj[j],un_at.fixreal2real,dr);
          r2=vec_mag2(dr);
          if (r2<=rc2) {
            epsij=sqrt(epsi*epsj[j]);
            rminij=0.5*(rmini+rminj[j]);
            // Inverting with division is rate limitting. Use this approximate solution instead
            ir2=rminij*rminij*realRecip(r2); // 1 us for exact division. See defines.h
            ir6=ir2*ir2*ir2;
            ir12=ir6*ir6;
            Eij=epsij*(ir12-2.0*ir6);
            lGt+=Eij;
            // Fij=12*k12*(ir2*ir4*ir8);
            Fij=12.0*epsij*(ir12-ir6)*realRecip(r2);
            // Gt+=Eij;
            vec_scaleinc((real*) (&fi),   -Fij,dr);
            vec_scaleinc((real*) (&fj[j]), Fij,dr);
          }
        }
      }
      if (un_Gt) {
        realAtomicAdd(&un_Gt[threadIdx.x],lGt);
      }
    }
  }

  __syncthreads();

  if (tileIdx<H4D.Ntiles) {
    // if (threadOfTileIdx < iN)
    if (ii>=0) {
      // if (i0==j0)
      if (ii==jj) {
        at_vec_inc((real*) (&fj[threadIdx.x]),(real*) (&fi));
      } else {
        at_vec_inc(un_at.f+DIM3*ii,(real*) (&fi));
      }
    }

    // if (threadOfTileIdx < jN)
    if (jj>=0) {
      at_vec_inc(un_at.f+DIM3*jj,(real*) (&fj[threadIdx.x]));
    }
  }
}

  
void getforce_other(struct_md* md)
{
  cudaStream_t stream_nb=md->state->stream_nonbonded;
  struct_nblist H4D=md->state->nblist_H4D[0];

  if (md->parms->kbgo) {
    getforce_other_kbgo_d <<< (H4D.Ntiles+BNB/TLJ-1)/(BNB/TLJ), BNB, 0, stream_nb >>> (H4D);
  } else {
    getforce_other_simple_d <<< (H4D.Ntiles+BNB/TLJ-1)/(BNB/TLJ), BNB, 0, stream_nb >>> (H4D);
  }

  cudaEventRecord(md->state->event_nbdone,md->state->stream_nonbonded);
}


void gethessian_other(struct_md* md)
{
  cudaStream_t stream_nb=md->state->stream_nonbonded;
  struct_nblist H4D=md->state->nblist_H4D[0];

  if (md->parms->kbgo) {
    char message[MAXLENGTH];
    sprintf(message,"Fatal error: no hessian set up for kgbo nonbonded interactions\n");
    fatalerror(message);
  } else {
    // gethessian_other_simple_d <<< (H4D.Ntiles+BNB/TLJ-1)/(BNB/TLJ), BNB, 0, stream_nb >>> (H4D,md->state->bias[0]);
    gethessian_other_simple_d <<< (H4D.Ntiles+BNB/TLJ-1)/(BNB/TLJ), BNB, 0, stream_nb >>> (H4D,md->state->bias[0]);
  }

  cudaEventRecord(md->state->event_nbhdone,md->state->stream_nonbonded);
}


void upload_other_d(real h_k12,real h_rc2,struct_atoms h_at,struct_nblist h_H4D,real* h_Gt)
{
  cudaMemcpyToSymbol(k12, &h_k12, sizeof(real), size_t(0),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(rc2, &h_rc2, sizeof(real), size_t(0),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(un_at, &h_at, sizeof(struct_atoms), size_t(0),cudaMemcpyHostToDevice);
  // cudaMemcpyToSymbol(H4D, &h_H4D, sizeof(struct_nblist), size_t(0),cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(un_Gt, &h_Gt, sizeof(real*), size_t(0),cudaMemcpyHostToDevice);
}


void upload_kbgo_d(struct_nbparms *h_nbparms)
{
  cudaMemcpyToSymbol(un_nbparms,&h_nbparms,sizeof(struct_nbparms*),size_t(0),cudaMemcpyHostToDevice);
}
