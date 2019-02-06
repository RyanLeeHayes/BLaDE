#ifndef MD_RNA_IONS_MERSENNE_H
#define MD_RNA_IONS_MERSENNE_H

#include "defines.h"
#include "md.h"
#include "state.h"

typedef struct struct_mtstate
{
  int N;
  unsigned long *mt; /* the array for the state vector  */
  real *gauss;
//   int mti; /* mti==N+1 means mt[N] is not initialized */
} struct_mtstate;

__global__ void init_genrand(unsigned long s, struct_mtstate mts);

__global__ void genrand_int32(struct_mtstate mtsin,struct_mtstate mtsout);

__global__ void genrand_gauss(struct_mtstate mts);

// struct_mtstate* alloc_mtstate(cudaStream_t strm,unsigned long s,int N);
void alloc_mtstate(struct_mtstate* mtstate,cudaStream_t strm,unsigned long s,int N);

void do_mt(struct_md* md, struct_mtstate* mts, int in);

void part_mt(struct_md* md,int step,struct_mtstate* mts);

void free_mtstate(struct_mtstate* mtstate,int N);

/* Period parameters */
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define MT_UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define MT_LOWER_MASK 0x7fffffffUL /* least significant r bits */

__device__ inline void genrand_int32_part_di(int Noff,struct_mtstate mtsin,struct_mtstate mtsout,int mti0,int mti1)
{
    int offBlockIdx=blockIdx.x-Noff;
    int i=offBlockIdx*blockDim.x+threadIdx.x;
    int N=mtsin.N;
    unsigned long *mtin=mtsin.mt;
    unsigned long *mtout=mtsout.mt;
    unsigned long y;
    // static unsigned long mag01[2]={0x0UL, MT_MATRIX_A};
    // unsigned long mag01[2]={0x0UL, MT_MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */
    // int mti=pt_mti[0];

    // if (mti >= MT_N) { /* generate N words at one time */
        int kk;
    // 
    //     if (mti == MT_N+1)   /* if init_genrand() has not been called, */
    //         init_genrand(5489UL,mt,&mti); /* a default initial seed is used */

    if (i < N) {
        for (kk=mti0;kk<MT_N-MT_M&&kk<mti1;kk++) {
            // y = (mtin[kk][i]&MT_UPPER_MASK)|(mtin[kk+1][i]&MT_LOWER_MASK);
            y = (mtin[kk*N+i]&MT_UPPER_MASK)|(mtin[(kk+1)*N+i]&MT_LOWER_MASK);
            // mtout[kk][i] = mtin[kk+MT_M][i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            // mtout[kk*N+i] = mtin[(kk+MT_M)*N+i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            mtout[kk*N+i] = mtin[(kk+MT_M)*N+i] ^ (y >> 1) ^ ((y & 0x1UL)*MT_MATRIX_A);
        }
        for (;kk<MT_N-1&&kk<mti1;kk++) {
            // y = (mtin[kk][i]&MT_UPPER_MASK)|(mtin[kk+1][i]&MT_LOWER_MASK);
            y = (mtin[kk*N+i]&MT_UPPER_MASK)|(mtin[(kk+1)*N+i]&MT_LOWER_MASK);
            // mtout[kk][i] = mtout[kk+(MT_M-MT_N)][i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            // mtout[kk*N+i] = mtout[(kk+(MT_M-MT_N))*N+i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            mtout[kk*N+i] = mtout[(kk+(MT_M-MT_N))*N+i] ^ (y >> 1) ^ ((y & 0x1UL)*MT_MATRIX_A);
        }
        if (kk<mti1) {
            // y = (mtin[MT_N-1][i]&MT_UPPER_MASK)|(mtout[0][i]&MT_LOWER_MASK);
            y = (mtin[(MT_N-1)*N+i]&MT_UPPER_MASK)|(mtout[0*N+i]&MT_LOWER_MASK);
            // mtout[MT_N-1][i] = mtout[MT_M-1][i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            // mtout[(MT_N-1)*N+i] = mtout[(MT_M-1)*N+i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            mtout[(MT_N-1)*N+i] = mtout[(MT_M-1)*N+i] ^ (y >> 1) ^ ((y & 0x1UL)*MT_MATRIX_A);
        }
    }
    // 
    //     mti = 0;
    // }
    // 
    // y = mt[mti++];
    // 
    // /* Tempering */
    // y ^= (y >> 11);
    // y ^= (y << 7) & 0x9d2c5680UL;
    // y ^= (y << 15) & 0xefc60000UL;
    // y ^= (y >> 18);
    // 
    // pt_mti[0]=mti;
    // return y;
}


// Returns two random reals with variance 1 in pointer r
__device__ inline void genrand_gauss_part_di(int Noff,struct_mtstate mts,int mti0,int mti1)
{
  int offBlockIdx=blockIdx.x-Noff;
  int i=offBlockIdx*blockDim.x+threadIdx.x;
  unsigned long y;
  real a,b;
  int mti;
  int N=mts.N;
  unsigned long *mt=mts.mt;
  real *gs=mts.gauss;

  if (i < N) {
    for (mti=mti0; mti<mti1; mti+=2) {
      y = mt[mti*N+i]; // [mti][i];
      /* Tempering */
      y ^= (y >> 11);
      y ^= (y << 7) & 0x9d2c5680UL;
      y ^= (y << 15) & 0xefc60000UL;
      y ^= (y >> 18);    
      a=(((real) y) + 0.5)*(1.0/4294967296.0);

      y = mt[(mti+1)*N+i]; // [mti+1][i];
      /* Tempering */
      y ^= (y >> 11);
      y ^= (y << 7) & 0x9d2c5680UL;
      y ^= (y << 15) & 0xefc60000UL;
      y ^= (y >> 18);    
      b=(((real) y) + 0.5)*(1.0/4294967296.0);

      // a=sqrt(-2*log(a));
      // gs[mti*N+i]=a*cos(2*M_PI*b); // [mti][i]
      // gs[(mti+1)*N+i]=a*sin(2*M_PI*b); // [mti+1][i]
      a=__fsqrt_rn(-2*__logf(a));
      gs[mti*N+i]=a*__cosf(2*M_PI*b); // [mti][i]
      gs[(mti+1)*N+i]=a*__sinf(2*M_PI*b); // [mti+1][i]
    }
  }
}

#endif

