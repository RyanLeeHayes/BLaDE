#ifndef MAIN_VEC_H
#define MAIN_VEC_H

#include "defines.h"

#include <math.h>



#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// From http://stackoverflow.com/questions/16077464/atomicadd-for-real-on-gpu
// And https://stackoverflow.com/questions/37566987/cuda-atomicadd-for-doubles-definition-error
__device__ static inline
double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ static inline
double atomicAdd(double* address, float val)
{
  return atomicAdd(address,(double)val);
}



// Necessary because CUDA can't take modulus correctly reliably
__device__ static inline
int rectify_modulus(int a,int b)
{
  int c=a%b;
  c-=(c>=b?b:0);
  c+=(c<0?b:0);
  return c;
}

// rectify_modulus might be expensive
__device__ static inline
int nearby_modulus(int a,int b)
{
  a-=(a>=b?b:0);
  a+=(a<0?b:0);
  return a;
}

// rectify_modulus might be expensive
__device__ static inline
int over_modulus(int a,int b)
{
  a-=(a>=b?b:0);
  return a;
}



__host__ __device__ inline
real_x boxxx(real3_x b) {return b.x;}
__host__ __device__ inline
real_x boxxx(real123_x b) {return b.a.x;}
#if defined(DOUBLE) != defined(DOUBLE_X)
__host__ __device__ inline
real boxxx(real3 b) {return b.x;}
#endif
__host__ __device__ inline
real boxxx(real123 b) {return b.a.x;}

__host__ __device__ inline
real_x boxyy(real3_x b) {return b.y;}
__host__ __device__ inline
real_x boxyy(real123_x b) {return b.b.y;}
#if defined(DOUBLE) != defined(DOUBLE_X)
__host__ __device__ inline
real boxyy(real3 b) {return b.y;}
#endif
__host__ __device__ inline
real boxyy(real123 b) {return b.b.y;}

__host__ __device__ inline
real_x boxzz(real3_x b) {return b.z;}
__host__ __device__ inline
real_x boxzz(real123_x b) {return b.c.z;}
#if defined(DOUBLE) != defined(DOUBLE_X)
__host__ __device__ inline
real boxzz(real3 b) {return b.z;}
#endif
__host__ __device__ inline
real boxzz(real123 b) {return b.c.z;}

__host__ __device__ inline
real_x boxyx(real3_x b) {return 0;}
__host__ __device__ inline
real_x boxyx(real123_x b) {return b.b.x;}
#if defined(DOUBLE) != defined(DOUBLE_X)
__host__ __device__ inline
real boxyx(real3 b) {return 0;}
#endif
__host__ __device__ inline
real boxyx(real123 b) {return b.b.x;}

__host__ __device__ inline
real_x boxzx(real3_x b) {return 0;}
__host__ __device__ inline
real_x boxzx(real123_x b) {return b.c.x;}
#if defined(DOUBLE) != defined(DOUBLE_X)
__host__ __device__ inline
real boxzx(real3 b) {return 0;}
#endif
__host__ __device__ inline
real boxzx(real123 b) {return b.c.x;}

__host__ __device__ inline
real_x boxzy(real3_x b) {return 0;}
__host__ __device__ inline
real_x boxzy(real123_x b) {return b.c.y;}
#if defined(DOUBLE) != defined(DOUBLE_X)
__host__ __device__ inline
real boxzy(real3 b) {return 0;}
#endif
__host__ __device__ inline
real boxzy(real123 b) {return b.c.y;}

__host__ __device__ inline
real boxxx(real321 b) {return b.a.x;}
__host__ __device__ inline
real boxyy(real321 b) {return b.b.x;}
__host__ __device__ inline
real boxzz(real321 b) {return b.c.x;}
__host__ __device__ inline
real boxxy(real321 b) {return b.a.y;}
__host__ __device__ inline
real boxxz(real321 b) {return b.a.z;}
__host__ __device__ inline
real boxyz(real321 b) {return b.b.y;}

__host__ __device__ inline
real boxxy(real3 b) {return 0;}
__host__ __device__ inline
real boxxz(real3 b) {return 0;}
__host__ __device__ inline
real boxyz(real3 b) {return 0;}



template<typename real3_out,typename real_in,typename real3_in>
__device__ static inline
void at_real3_scaleinc(real3_out *a,real_in f,real3_in x)
{
  atomicAdd(&(a[0].x),f*x.x);
  atomicAdd(&(a[0].y),f*x.y);
  atomicAdd(&(a[0].z),f*x.z);
}

template<typename real3_out,typename real3_in>
__device__ static inline
void at_real3_inc(real3_out *a,real3_in x)
{
  atomicAdd(&(a[0].x),x.x);
  atomicAdd(&(a[0].y),x.y);
  atomicAdd(&(a[0].z),x.z);
}

template<typename real3_out,typename real3_in>
__device__ static inline
void at_real3_dec(real3_out *a,real3_in x)
{
  atomicAdd(&(a[0].x),-x.x);
  atomicAdd(&(a[0].y),-x.y);
  atomicAdd(&(a[0].z),-x.z);
}

template<typename real_out,typename real3_in>
__host__ __device__ static inline
real_out real3_mag(real3_in a)
{
  return sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}

template<typename real_out,typename real3_in>
__host__ __device__ static inline
real_out real3_mag2(real3_in a)
{
  return a.x*a.x+a.y*a.y+a.z*a.z;
}

template<bool flagBox,typename real3_type,typename box_type>
__host__ __device__ static inline
real3_type real3_subpbc(real3_type a,real3_type b,box_type box)
{
  real3_type c;
  c.x=a.x-b.x;
  c.y=a.y-b.y;
  c.z=a.z-b.z;
  if (flagBox) {
    int i=floor(c.z/boxzz(box)+0.5f);
    c.x-=boxzx(box)*i; // box.c.x*i;
    c.y-=boxzy(box)*i; // box.c.y*i;
    c.z-=boxzz(box)*i; // box.c.z*i;
    i=floor(c.y/boxyy(box)+0.5f);
    c.x-=boxyx(box)*i; // box.b.x*i;
    c.y-=boxyy(box)*i; // box.b.y*i;
    i=floor(c.x/boxxx(box)+0.5f);
    c.x-=boxxx(box)*i; // box.a.x*i;
  } else {
    c.x-=boxxx(box)*floor(c.x/boxxx(box)+0.5f); // Getting type of real3_type is hard on intel, 0.5f will cast up if necessary
    c.y-=boxyy(box)*floor(c.y/boxyy(box)+0.5f);
    c.z-=boxzz(box)*floor(c.z/boxzz(box)+0.5f);
  }
  return c;
}

template<typename real3_type>
__host__ __device__ static inline
real3_type real3_sub(real3_type a,real3_type b)
{
  real3_type c;
  c.x=a.x-b.x;
  c.y=a.y-b.y;
  c.z=a.z-b.z;
  return c;
}

template<typename real3_type>
__host__ __device__ static inline
real3_type real3_add(real3_type a,real3_type b)
{
  real3_type c;
  c.x=a.x+b.x;
  c.y=a.y+b.y;
  c.z=a.z+b.z;
  return c;
}

template<typename real_out,typename real3_a,typename real3_b>
__host__ __device__ static inline
real_out real3_dot(real3_a a,real3_b b)
{
  return a.x*b.x+a.y*b.y+a.z*b.z;
}

template<typename real3_out,typename real3_in>
__host__ __device__ static inline
real3_out real3_cast(real3_in a)
{
  real3_out c;
  c.x=a.x;
  c.y=a.y;
  c.z=a.z;
  return c;
}

template<typename real3_type>
__host__ __device__ static inline
real3_type real3_cross(real3_type a,real3_type b)
{
  real3_type c;
  c.x=a.y*b.z-a.z*b.y;
  c.y=a.z*b.x-a.x*b.z;
  c.z=a.x*b.y-a.y*b.x;
  return c;
}

template<typename real3_out,typename real_in,typename real3_in>
__host__ __device__ static inline
real3_out real3_scale(real_in f,real3_in a)
{
  real3_out c;
  c.x=f*a.x;
  c.y=f*a.y;
  c.z=f*a.z;
  return c;
}

template<typename real3_out,typename real_in,typename real3_in>
__host__ __device__ static inline
void real3_scaleinc(real3_out *a,real_in f,real3_in x)
{
  a[0].x+=f*x.x;
  a[0].y+=f*x.y;
  a[0].z+=f*x.z;
}

template<typename real3_type,typename real_type>
__host__ __device__ static inline
void real3_scaleself(real3_type *a,real_type f)
{
  a[0].x*=f;
  a[0].y*=f;
  a[0].z*=f;
}

template<typename real3_type>
__host__ __device__ static inline
void real3_inc(real3_type *a,real3_type x)
{
  a[0].x+=x.x;
  a[0].y+=x.y;
  a[0].z+=x.z;
}

template<typename real3_type>
__host__ __device__ static inline
void real3_dec(real3_type *a,real3_type x)
{
  a[0].x-=x.x;
  a[0].y-=x.y;
  a[0].z-=x.z;
}

// fmod rounds toward zero, remainder is in the range -0.5 to 0.5, either one must be rectified.
template<bool flagBox,typename real3_type,typename box_type>
__host__ __device__ static inline
real3_type real3_modulus(real3_type a,box_type b)
{
  real3_type c;
  if (flagBox) {
    c=a;
    int i=floor(c.z/boxzz(b));
    c.x-=boxzx(b)*i;
    c.y-=boxzy(b)*i;
    c.z-=boxzz(b)*i;
    i=floor(c.y/boxyy(b));
    c.x-=boxyx(b)*i;
    c.y-=boxyy(b)*i;
    i=floor(c.x/boxxx(b));
    c.x-=boxxx(b)*i;
  } else {
    c.x=fmod(a.x,boxxx(b));
    c.y=fmod(a.y,boxyy(b));
    c.z=fmod(a.z,boxzz(b));
    c.x+=(c.x<0?boxxx(b):0);
    c.y+=(c.y<0?boxyy(b):0);
    c.z+=(c.z<0?boxzz(b):0);
  }
  return c;
}

template<typename real3_type>
__host__ __device__ static inline
real3_type real3_reset()
{
  real3_type out;
  out.x=0;
  out.y=0;
  out.z=0;
  return out;
}

template<typename real3_type,typename real_type>
__host__ __device__ static inline
void real3_rotation_matrix(real3_type *R,real3_type axis,real_type c,real_type s)
{
  real_type t=1-c;
  R[0].x=t*axis.x*axis.x + c;
  R[0].y=t*axis.x*axis.y - s*axis.z;
  R[0].z=t*axis.x*axis.z + s*axis.y;
  R[1].x=t*axis.y*axis.x + s*axis.z;
  R[1].y=t*axis.y*axis.y + c;
  R[1].z=t*axis.y*axis.z - s*axis.x;
  R[2].x=t*axis.z*axis.x - s*axis.y;
  R[2].y=t*axis.z*axis.y + s*axis.x;
  R[2].z=t*axis.z*axis.z + c;
}

template<typename real3_out,typename real_out,typename real3_a,typename real3_b>
__host__ __device__ static inline
real3_out real3_rotate(real3_a *R,real3_b in)
{
  real3_out out;
  // out.x=real3_dot<decltype(out.x)>(R[0],in);
  // out.y=real3_dot<decltype(out.y)>(R[1],in);
  // out.z=real3_dot<decltype(out.z)>(R[2],in);
  // Intel wasn't smart enough to compile that...
  out.x=real3_dot<real_out>(R[0],in);
  out.y=real3_dot<real_out>(R[1],in);
  out.z=real3_dot<real_out>(R[2],in);
  return out;
}



template <typename real_type>
__device__ static inline
void real_sum_reduce(real input,real *shared,real_type *global)
{
  real local=input;
  local+=__shfl_down_sync(0xFFFFFFFF,local,1);
  local+=__shfl_down_sync(0xFFFFFFFF,local,2);
  local+=__shfl_down_sync(0xFFFFFFFF,local,4);
  local+=__shfl_down_sync(0xFFFFFFFF,local,8);
  local+=__shfl_down_sync(0xFFFFFFFF,local,16);
  __syncthreads();
  if ((0x1F & threadIdx.x)==0) {
    shared[threadIdx.x>>5]=local;
  }
  __syncthreads();
  local=0;
  if (threadIdx.x < (blockDim.x>>5)) {
    local=shared[threadIdx.x];
  }
  if (threadIdx.x < 32) {
    if (blockDim.x>=64) local+=__shfl_down_sync(0xFFFFFFFF,local,1);
    if (blockDim.x>=128) local+=__shfl_down_sync(0xFFFFFFFF,local,2);
    if (blockDim.x>=256) local+=__shfl_down_sync(0xFFFFFFFF,local,4);
    if (blockDim.x>=512) local+=__shfl_down_sync(0xFFFFFFFF,local,8);
    if (blockDim.x>=1024) local+=__shfl_down_sync(0xFFFFFFFF,local,16);
  }
  if (threadIdx.x==0) {
    atomicAdd(global,(real_type)local);
  }
}

template <typename real_type>
__device__ static inline
void real_sum_reduce(real input,real_type *global)
{
  real local=input;
  local+=__shfl_down_sync(0xFFFFFFFF,local,1);
  local+=__shfl_down_sync(0xFFFFFFFF,local,2);
  local+=__shfl_down_sync(0xFFFFFFFF,local,4);
  local+=__shfl_down_sync(0xFFFFFFFF,local,8);
  local+=__shfl_down_sync(0xFFFFFFFF,local,16);
  __syncthreads();
  if ((0x1F & threadIdx.x)==0) {
    atomicAdd(global,(real_type)local);
  }
}
#endif
