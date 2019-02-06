// Begin Mersenne Twister
// Modified from
// http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html
/* 
   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)  
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER O
R
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

#include "mersenne.h"

#include "defines.h"
#include "md.h"

#include <omp.h>
#include <math.h>
#include <stdlib.h>

// static unsigned long mt[MT_N]; /* the array for the state vector  */
// static int mti=MT_N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
//__global__ void init_genrand(unsigned long s, struct_mtstate mts, int* pt_mti)
__global__ void init_genrand(unsigned long s, struct_mtstate mts)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int mti;
    int N=mts.N;
    unsigned long *mt=mts.mt;

    if (i < N) {
      s+=i;
      mt[i]= s & 0xffffffffUL; // [0][i]
      for (mti=1; mti<MT_N; mti++) {
        mt[mti*N+i] = // [mti][i]
            (1812433253UL * (mt[(mti-1)*N+i] ^ (mt[(mti-1)*N+i] >> 30)) + mti); // [mti-1][i]
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti*N+i] &= 0xffffffffUL; // [mti][i]
        /* for >32 bit machines */
      }
    }

    // if (i==0) {
    //   pt_mti[0]=mti;
    // }
}

/* // initialize by an array with array-length
// init_key is the array for initializing keys
// key_length is its length
// slight change for C++, 2004/2/26
static
void init_by_array(unsigned long init_key[], int key_length, unsigned long mt[MT_N], int* pt_mti)
{
    int i, j, k;
    int mti=pt_mti[0];
    init_genrand(19650218UL,mt,&mti);
    i=1; j=0;
    k = (MT_N>key_length ? MT_N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; // non linear
        mt[i] &= 0xffffffffUL; // for WORDSIZE > 32 machines
        i++; j++;
        if (i>=MT_N) { mt[0] = mt[MT_N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=MT_N-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
          - i; // non linear
        mt[i] &= 0xffffffffUL; // for WORDSIZE > 32 machines
        i++;
        if (i>=MT_N) { mt[0] = mt[MT_N-1]; i=1; }
    }

    mt[0] = 0x80000000UL; // MSB is 1; assuring non-zero initial array
    pt_mti[0]=mti;
}*/

/* generates a random number on [0,0xffffffff]-interval */
// __global__ unsigned long genrand_int32(unsigned long mt[MT_N],int* pt_mti)
__global__ void genrand_int32(struct_mtstate mtsin,struct_mtstate mtsout)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int N=mtsin.N;
    unsigned long *mtin=mtsin.mt;
    unsigned long *mtout=mtsout.mt;
    unsigned long y;
    // static unsigned long mag01[2]={0x0UL, MT_MATRIX_A};
    unsigned long mag01[2]={0x0UL, MT_MATRIX_A};
    // /* mag01[x] = x * MATRIX_A  for x=0,1 */
    // int mti=pt_mti[0];

    // if (mti >= MT_N) { /* generate N words at one time */
        int kk;
    // 
    //     if (mti == MT_N+1)   /* if init_genrand() has not been called, */
    //         init_genrand(5489UL,mt,&mti); /* a default initial seed is used */

    if (i < N) {
        for (kk=0;kk<MT_N-MT_M;kk++) {
            // y = (mtin[kk][i]&MT_UPPER_MASK)|(mtin[kk+1][i]&MT_LOWER_MASK);
            y = (mtin[kk*N+i]&MT_UPPER_MASK)|(mtin[(kk+1)*N+i]&MT_LOWER_MASK);
            // mtout[kk][i] = mtin[kk+MT_M][i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            mtout[kk*N+i] = mtin[(kk+MT_M)*N+i] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<MT_N-1;kk++) {
            // y = (mtin[kk][i]&MT_UPPER_MASK)|(mtin[kk+1][i]&MT_LOWER_MASK);
            y = (mtin[kk*N+i]&MT_UPPER_MASK)|(mtin[(kk+1)*N+i]&MT_LOWER_MASK);
            // mtout[kk][i] = mtout[kk+(MT_M-MT_N)][i] ^ (y >> 1) ^ mag01[y & 0x1UL];
            mtout[kk*N+i] = mtout[(kk+(MT_M-MT_N))*N+i] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        // y = (mtin[MT_N-1][i]&MT_UPPER_MASK)|(mtout[0][i]&MT_LOWER_MASK);
        y = (mtin[(MT_N-1)*N+i]&MT_UPPER_MASK)|(mtout[0*N+i]&MT_LOWER_MASK);
        // mtout[MT_N-1][i] = mtout[MT_M-1][i] ^ (y >> 1) ^ mag01[y & 0x1UL];
        mtout[(MT_N-1)*N+i] = mtout[(MT_M-1)*N+i] ^ (y >> 1) ^ mag01[y & 0x1UL];
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
// End Mersenne Twister


__global__ void genrand_int32_part(struct_mtstate mtsin,struct_mtstate mtsout,int mti0,int mti1)
{
  genrand_int32_part_di(0,mtsin,mtsout,mti0,mti1);
}


// Returns two random reals with variance 1 in pointer r
__global__ void genrand_gauss(struct_mtstate mts)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long y;
  real a,b;
  int mti;
  int N=mts.N;
  unsigned long *mt=mts.mt;
  real *gs=mts.gauss;

  if (i < N) {
    for (mti=0; mti<MT_N; mti+=2) {
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

      a=sqrt(-2*log(a));
      gs[mti*N+i]=a*cos(2*M_PI*b); // [mti][i]
      gs[(mti+1)*N+i]=a*sin(2*M_PI*b); // [mti+1][i]
    }
  }
}


// Returns two random reals with variance 1 in pointer r
__global__ void genrand_gauss_part(struct_mtstate mts,int mti0,int mti1)
{
  genrand_gauss_part_di(0,mts,mti0,mti1);
}


// struct_mtstate* alloc_mtstate(cudaStream_t strm,unsigned long s,int N)
void alloc_mtstate(struct_mtstate* mtstate,cudaStream_t strm,unsigned long s,int N)
{
  // struct_mtstate *mtstate;

  // mtstate=(struct_mtstate*) calloc(2,sizeof(struct_mtstate));

  mtstate[0].N=mtstate[1].N=N;
  cudaMalloc(&(mtstate[0].mt),N*MT_N*sizeof(unsigned long));
  cudaMalloc(&(mtstate[1].mt),N*MT_N*sizeof(unsigned long));
  cudaMalloc(&(mtstate[0].gauss),N*MT_N*sizeof(real));
  cudaMalloc(&(mtstate[1].gauss),N*MT_N*sizeof(real));
  // mtstate[0].mti=mtstate[1].mti=MT_N+1;

  // init_genrand <<< (N+63)/64, 64 >>> (s, mtstate[1], &(mtstate[1].mti));
  init_genrand <<< (N+BU-1)/BU, BU, 0, strm >>> (s, mtstate[1]);
  genrand_int32 <<< (N+BU-1)/BU, BU, 0, strm >>> (mtstate[1], mtstate[0]);
  genrand_int32 <<< (N+BU-1)/BU, BU, 0, strm >>> (mtstate[0], mtstate[1]);
  genrand_gauss <<< (N+BU-1)/BU, BU, 0, strm >>> (mtstate[1]);

  // return mtstate;
}


void do_mt(struct_md* md,struct_mtstate* mts, int in)
{
  int N=mts[0].N;
  int out=(in+1)%2;
  cudaStream_t strm=md->state->stream_default;

  genrand_int32 <<< (N+BU-1)/BU, BU, 0, strm >>> (mts[in], mts[out]);
  genrand_gauss <<< (N+BU-1)/BU, BU, 0, strm >>> (mts[out]);
}


void part_mt(struct_md* md,int step,struct_mtstate* mts)
{
  int N=mts[0].N;
  int mti=4*(step % (MT_N/4)); // Index within the mt array (0 to MT_N, 4 each step)
  int mtact=(step/(MT_N/4))%2; // Two mtstates (0 and 1), this is the active one
  int mtinact=(mtact+1)%2;
  cudaStream_t strm=md->state->stream_default;
  
  // Current step is working on mts[mtact], mti to mti+4. Generate new ints for the current step, well use them to generate gaussians in MT_N/4 steps. Generate new gaussians for the opposite step mts[mtinact]. We'll use those Gaussians in MT_N/4 steps.
  genrand_int32_part <<< (N+BU-1)/BU, BU, 0, strm >>> (mts[mtinact], mts[mtact], mti, mti+4);
  genrand_gauss_part <<< (N+BU-1)/BU, BU, 0, strm >>> (mts[mtinact],mti,mti+4);
}


void free_mtstate(struct_mtstate* mtstate,int N)
{
  cudaFree(mtstate[0].mt);
  cudaFree(mtstate[1].mt);
  cudaFree(mtstate[0].gauss);
  cudaFree(mtstate[1].gauss);

  // free(mtstate);
}

