#include "nbsort.h"

#include "nblist.h"
#include "error.h"

#include <stdio.h>


void swap_bins(struct_bin *bins,int i1,int i2)
{
  struct_bin buffer;
  buffer=bins[i1];
  bins[i1]=bins[i2];
  bins[i2]=buffer;
}


int compare_bins(struct_bin *bins,int i1,int i2)
{
  struct_bin b1=bins[i1];
  struct_bin b2=bins[i2];
  if (b2.i<0) {
    return 0;
  } else if (b1.i<0) {
    return 1;
  }
  if (b1.bx<b2.bx) {
    return 0;
  } else if (b1.bx>b2.bx) {
    return 1;
  }
  if (b1.by<b2.by) {
    return 0;
  } else if (b1.by>b2.by) {
    return 1;
  }
  if (b1.z<b2.z) {
    return 0;
  } else if (b1.z>b2.z) {
    return 1;
  }
  if (b1.i<b2.i) {
    return 0;
  } else if (b1.i>b2.i) {
    return 1;
  }
  // If all else fails
  return 0;
}

__device__ inline
int compare_bins_d(struct_bin *bins,int i1,int i2)
{
  struct_bin b1=bins[i1];
  struct_bin b2=bins[i2];
  int cmp=0;
  cmp+=16*((b1.i<0)-(b2.i<0));
  cmp+=8*((b1.bx>b2.bx)-(b1.bx<b2.bx));
  cmp+=4*((b1.by>b2.by)-(b1.by<b2.by));
  cmp+=2*((b1.z>b2.z)-(b1.z<b2.z));
  cmp+=1*((b1.i>b2.i)-(b1.i<b2.i));
  return (cmp>0);
}


__device__ inline
int compare_binds_d(struct_bin *bins,int *binds,int i1,int i2)
{
  struct_bin b1=bins[binds[i1]];
  struct_bin b2=bins[binds[i2]];
  int cmp=0;
  cmp+=16*((b1.i<0)-(b2.i<0));
  cmp+=8*((b1.bx>b2.bx)-(b1.bx<b2.bx));
  cmp+=4*((b1.by>b2.by)-(b1.by<b2.by));
  cmp+=2*((b1.z>b2.z)-(b1.z<b2.z));
  cmp+=1*((b1.i>b2.i)-(b1.i<b2.i));
  return (cmp>0);
}


int sift_bins(struct_bin *bins,int i1,int i2)
{
  int imid=i1+(i2-1-i1)/2;
  int q1,q2;
  swap_bins(bins,imid,i2-1); // place middle at end (to make already sorted lists faster)
  imid=i2-1;
  i2--;
  q1=compare_bins(bins,i1,imid);
  q2=compare_bins(bins,imid,i2-1);
  while (i1<i2-1) {
    if (q1==0) {
      i1++;
      q1=compare_bins(bins,i1,imid);
    } else if (q2==0) {
      i2--;
      q2=compare_bins(bins,imid,i2-1);
    } else {
      swap_bins(bins,i1,i2-1);
      q1=0;
      q2=0;
    }
  }
  if (q1==0) {
    swap_bins(bins,i1+1,imid);
    return i1+1;
  } else {
    swap_bins(bins,i1,imid);
    return i1;
  }
}


void quicksort_bins(struct_bin *bins,int i1,int i2)
{
  if (i1<i2-1) {
    int imid=sift_bins(bins,i1,i2);
    quicksort_bins(bins,i1,imid);
    quicksort_bins(bins,imid+1,i2);
  }
}


// bitonic sort, see http://www.cs.rutgers.edu/~venugopa/parallel_summer2012/bitonic_overview.html
__global__
void bitonic_init_d(struct_nblist H4D)
{
  int N;
  int idx,idx0,idx1;
  int s,smax;
  int lsb,msb;
  int i0,i1;
  int dir,cmp;
  // struct_bin tmp;
  int tmpd;
  __shared__ struct_bin bin[2*BNB];
  __shared__ int bind[2*BNB];
  
  N=H4D.Nbit;
  
  for (i0=threadIdx.x; i0<(N-2*BNB*blockIdx.x) && i0<2*BNB; i0+=BNB) {
    bin[i0]=H4D.bin[i0+2*BNB*blockIdx.x];
    bind[i0]=i0;
  }

  idx=BNB*blockIdx.x+threadIdx.x;

  __syncthreads();

  for (smax=1; smax <= BNB; smax*=2) {
    dir=(idx & smax);
    for (s=smax; s>=1; s/=2) {
      lsb=(idx&(s-1));
      msb=idx-lsb;
      idx0=2*msb+lsb;
      idx1=idx0+s;
      i0=idx0-2*BNB*blockIdx.x;
      i1=idx1-2*BNB*blockIdx.x;
      if (idx1<N) {
        // cmp=compare_bins_d(bin,i0,i1);
        cmp=compare_binds_d(bin,bind,i0,i1);
        if (cmp == (!dir)) {
          // tmp=bin[i0];
          // bin[i0]=bin[i1];
          // bin[i1]=tmp;
          tmpd=bind[i0];
          bind[i0]=bind[i1];
          bind[i1]=tmpd;
        }
      }
      __syncthreads();
    }
  }

  for (i0=threadIdx.x; i0<(N-2*BNB*blockIdx.x) && i0<2*BNB; i0+=BNB) {
    // H4D.bin[i0+2*BNB*blockIdx.x]=bin[i0];
    H4D.bin[i0+2*BNB*blockIdx.x]=bin[bind[i0]];
  }
}


__global__
void bitonic_merge1_d(struct_nblist H4D,int stride,int smax)
{
  int N;
  int idx,idx0,idx1;
  int s,smin;
  int lsb,msb;
  int i0,i1;
  int dir,cmp;
  // struct_bin tmp;
  int tmpd;
  __shared__ struct_bin bin[2*BNB];
  __shared__ int bind[2*BNB];
  
  smin=smax/BNB;
  if (smin<2*BNB) {
    smin=2*BNB;
  }

  N=H4D.Nbit;
  
  for (i0=threadIdx.x; (stride*i0+blockIdx.x)<N && i0<2*BNB; i0+=BNB) {
    bin[i0]=H4D.bin[stride*i0+blockIdx.x];
    bind[i0]=i0;
  }

  idx=stride*threadIdx.x+blockIdx.x;

  __syncthreads();

  dir=(idx & smax);
  for (s=smax; s>=smin; s/=2) {
    lsb=(idx&(s-1));
    msb=idx-lsb;
    idx0=2*msb+lsb;
    idx1=idx0+s;
    i0=(idx0-blockIdx.x)/stride;
    i1=(idx1-blockIdx.x)/stride;
    // if (blockIdx.x==0 && threadIdx.x==0) {
    //   H4D.debug[H4D.debug[0]++]=idx0;
    //   H4D.debug[H4D.debug[0]++]=idx1;
    //   H4D.debug[H4D.debug[0]++]=i0;
    //   H4D.debug[H4D.debug[0]++]=i1;
    // }
    if (idx1<N) {
      // cmp=compare_bins_d(bin,i0,i1);
      cmp=compare_binds_d(bin,bind,i0,i1);
      if (cmp == (!dir)) {
        // tmp=bin[i0];
        // bin[i0]=bin[i1];
        // bin[i1]=tmp;
        tmpd=bind[i0];
        bind[i0]=bind[i1];
        bind[i1]=tmpd;
      }
    }
    __syncthreads();
  }

  for (i0=threadIdx.x; (stride*i0+blockIdx.x)<N && i0<2*BNB; i0+=BNB) {
    // H4D.bin[stride*i0+blockIdx.x]=bin[i0];
    H4D.bin[stride*i0+blockIdx.x]=bin[bind[i0]];
  }
}


__global__
void bitonic_merge2_d(struct_nblist H4D,int smax)
{
  int N;
  int idx,idx0,idx1;
  int s;
  int lsb,msb;
  int i0,i1;
  int dir,cmp;
  // struct_bin tmp;
  int tmpd;
  __shared__ struct_bin bin[2*BNB];
  __shared__ int bind[2*BNB];
  
  N=H4D.Nbit;
  
  for (i0=threadIdx.x; i0<(N-2*BNB*blockIdx.x) && i0<2*BNB; i0+=BNB) {
    bin[i0]=H4D.bin[i0+2*BNB*blockIdx.x];
    bind[i0]=i0;
  }

  idx=BNB*blockIdx.x+threadIdx.x;

  __syncthreads();

  dir=(idx & smax);
  for (s=BNB; s>=1; s/=2) {
    lsb=(idx&(s-1));
    msb=idx-lsb;
    idx0=2*msb+lsb;
    idx1=idx0+s;
    i0=idx0-2*BNB*blockIdx.x;
    i1=idx1-2*BNB*blockIdx.x;
    if (idx1<N) {
      // cmp=compare_bins_d(bin,i0,i1);
      cmp=compare_binds_d(bin,bind,i0,i1);
      if (cmp == (!dir)) {
        // tmp=bin[i0];
        // bin[i0]=bin[i1];
        // bin[i1]=tmp;
        tmpd=bind[i0];
        bind[i0]=bind[i1];
        bind[i1]=tmpd;
      }
    }
    __syncthreads();
  }

  for (i0=threadIdx.x; i0<(N-2*BNB*blockIdx.x) && i0<2*BNB; i0+=BNB) {
    // H4D.bin[i0+2*BNB*blockIdx.x]=bin[i0];
    H4D.bin[i0+2*BNB*blockIdx.x]=bin[bind[i0]];
  }
}


void bitonic_sort(cudaStream_t strm,struct_nblist *H4D)
{
  int N, N_2BNB;
  int smax;

  N=H4D->Nbit;

  if (N>(2*BNB*BNB)) {
    char message[MAXLENGTH];
    sprintf(message,"Error, bitonic neighbor sort (%s) not implemented for N>2*BNB*BNB=%d. Doubling BNB in defines.h may help a little\n",__FILE__,2*BNB*BNB);
    fatalerror(message);
  }

  N_2BNB=N/(2*BNB);
  bitonic_init_d <<< N_2BNB, BNB, 0, strm >>> (H4D[0]);
  if (N>2*BNB) {
    for (smax=2*BNB; smax<N; smax*=2) {
      bitonic_merge1_d <<< N_2BNB, BNB, 0, strm >>> (H4D[0],N_2BNB,smax);
      bitonic_merge2_d <<< N_2BNB, BNB, 0, strm >>> (H4D[0],smax);
    }
  }
}
