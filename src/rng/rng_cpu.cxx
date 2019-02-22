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

#include <time.h>
#include <math.h>
#include <stdlib.h>

#include "rng/rng_cpu.h"
 
/* Period parameters */
#define MT_N 624
#define MT_M 397
#define MT_MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define MT_UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define MT_LOWER_MASK 0x7fffffffUL /* least significant r bits */

// static unsigned long mt[MT_N]; /* the array for the state vector  */
// static int mti=MT_N+1; /* mti==N+1 means mt[N] is not initialized */



// Class Constructors
RngCPU::RngCPU()
{
  mtState=alloc_mtstate(time(NULL));
}

RngCPU::~RngCPU()
{
  free_mtstate(mtState);
}



/* initializes mt[N] with a seed */
void RngCPU::init_genrand(unsigned long s, unsigned long *mt, int* pt_mti)
{
    int mti=pt_mti[0];
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<MT_N; mti++) {
        mt[mti] =
            (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
    pt_mti[0]=mti;
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void RngCPU::init_by_array(unsigned long *init_key, int key_length, unsigned long *mt, int* pt_mti)
{
    int i, j, k;
    int mti=pt_mti[0];
    init_genrand(19650218UL,mt,&mti);
    i=1; j=0;
    k = (MT_N>key_length ? MT_N : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1664525UL))
          + init_key[j] + j; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++; j++;
        if (i>=MT_N) { mt[0] = mt[MT_N-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=MT_N-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
          - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i>=MT_N) { mt[0] = mt[MT_N-1]; i=1; }
    }

    mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
    pt_mti[0]=mti;
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long RngCPU::genrand_int32(unsigned long *mt,int* pt_mti)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MT_MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */
    int mti=pt_mti[0];

    if (mti >= MT_N) { /* generate N words at one time */
        int kk;

        if (mti == MT_N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL,mt,&mti); /* a default initial seed is used */

        for (kk=0;kk<MT_N-MT_M;kk++) {
            y = (mt[kk]&MT_UPPER_MASK)|(mt[kk+1]&MT_LOWER_MASK);
            mt[kk] = mt[kk+MT_M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<MT_N-1;kk++) {
            y = (mt[kk]&MT_UPPER_MASK)|(mt[kk+1]&MT_LOWER_MASK);
            mt[kk] = mt[kk+(MT_M-MT_N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[MT_N-1]&MT_UPPER_MASK)|(mt[0]&MT_LOWER_MASK);
        mt[MT_N-1] = mt[MT_M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }

    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    pt_mti[0]=mti;
    return y;
}
// End Mersenne Twister


// Returns a gaussian random number with variance 1
real RngCPU::rand_normal()
{
  real r;
  real a,b;
  if (mtState->bNextReal) {
    r=mtState->nextReal;
    mtState->bNextReal=false;
  } else {
    a=(((real)genrand_int32(mtState->mt,&(mtState->mti))) + 0.5)*(1.0/4294967296.0);
    b=(((real)genrand_int32(mtState->mt,&(mtState->mti))) + 0.5)*(1.0/4294967296.0);
    a=sqrt(-2*log(a));
    r=a*cos(2*M_PI*b);
    mtState->nextReal=a*sin(2*M_PI*b);
    mtState->bNextReal=true;
  }
  return r;
}


// Returns a random number between 0 and 1
real RngCPU::rand_uniform()
{
  return (((real)genrand_int32(mtState->mt,&(mtState->mti))) + 0.5)*(1.0/4294967296.0);
}


struct MTState* RngCPU::alloc_mtstate(unsigned long s)
{
  struct MTState *mts;

  mts=(struct MTState*)malloc(sizeof(struct MTState));
  mts->mt=(unsigned long*)calloc(MT_N,sizeof(unsigned long));
  mts->mti=MT_N+1;
  init_genrand(s,mts->mt,&(mts->mti));
  mts->bNextReal=false;

  return mts;
}


void RngCPU::free_mtstate(struct MTState* mts)
{
  free(mts->mt);
  free(mts);
}

