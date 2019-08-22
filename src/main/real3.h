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



__device__ static inline
void at_real3_scaleinc(real3 *a,real f,real3 x)
{
  atomicAdd(&(a[0].x),f*x.x);
  atomicAdd(&(a[0].y),f*x.y);
  atomicAdd(&(a[0].z),f*x.z);
}

__device__ static inline
void at_real3_inc(real3 *a,real3 x)
{
  atomicAdd(&(a[0].x),x.x);
  atomicAdd(&(a[0].y),x.y);
  atomicAdd(&(a[0].z),x.z);
}

__host__ __device__ static inline
real real3_mag(real3 a)
{
  return sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
}

__host__ __device__ static inline
real real3_mag2(real3 a)
{
  return a.x*a.x+a.y*a.y+a.z*a.z;
}

__host__ __device__ static inline
real3 real3_subpbc(real3 a,real3 b,real3 box)
{
  real3 c;
  c.x=a.x-b.x;
  c.y=a.y-b.y;
  c.z=a.z-b.z;
  c.x-=box.x*floor(c.x/box.x+0.5);
  c.y-=box.x*floor(c.y/box.y+0.5);
  c.z-=box.x*floor(c.z/box.z+0.5);
  return c;
}

__host__ __device__ static inline
real3 real3_sub(real3 a,real3 b)
{
  real3 c;
  c.x=a.x-b.x;
  c.y=a.y-b.y;
  c.z=a.z-b.z;
  return c;
}

__host__ __device__ static inline
real3 real3_add(real3 a,real3 b)
{
  real3 c;
  c.x=a.x+b.x;
  c.y=a.y+b.y;
  c.z=a.z+b.z;
  return c;
}

__host__ __device__ static inline
real real3_dot(real3 a,real3 b)
{
  return a.x*b.x+a.y*b.y+a.z*b.z;
}

__host__ __device__ static inline
real3 real3_cross(real3 a,real3 b)
{
  real3 c;
  c.x=a.y*b.z-a.z*b.y;
  c.y=a.z*b.x-a.x*b.z;
  c.z=a.x*b.y-a.y*b.x;
  return c;
}

__host__ __device__ static inline
real3 real3_scale(real f,real3 a)
{
  real3 c;
  c.x=f*a.x;
  c.y=f*a.y;
  c.z=f*a.z;
  return c;
}

__host__ __device__ static inline
void real3_scaleinc(real3 *a,real f,real3 x)
{
  a[0].x+=f*x.x;
  a[0].y+=f*x.y;
  a[0].z+=f*x.z;
}

__host__ __device__ static inline
void real3_scaleself(real3 *a,real f)
{
  a[0].x*=f;
  a[0].y*=f;
  a[0].z*=f;
}

__host__ __device__ static inline
void real3_inc(real3 *a,real3 x)
{
  a[0].x+=x.x;
  a[0].y+=x.y;
  a[0].z+=x.z;
}

__host__ __device__ static inline
void real3_dec(real3 *a,real3 x)
{
  a[0].x-=x.x;
  a[0].y-=x.y;
  a[0].z-=x.z;
}

__host__ __device__ static inline
real3 real3_modulus(real3 a,real3 b)
{
  real3 c;
  c.x=fmod(a.x,b.x);
  c.y=fmod(a.y,b.y);
  c.z=fmod(a.z,b.z);
  c.x+=(c.x<0?b.x:0);
  c.y+=(c.y<0?b.y:0);
  c.z+=(c.z<0?b.z:0);
  return c;
}

__host__ __device__ static inline
real3 real3_reset()
{
  real3 out;
  out.x=0;
  out.y=0;
  out.z=0;
  return out;
}

__host__ __device__ static inline
void real3_rotation_matrix(real3 *R,real3 axis,real c,real s)
{
  real t=1-c;
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

__host__ __device__ static inline
real3 real3_rotate(real3 *R,real3 in)
{
  real3 out;
  out.x=real3_dot(R[0],in);
  out.y=real3_dot(R[1],in);
  out.z=real3_dot(R[2],in);
  return out;
}



__device__ static inline
void real_sum_reduce(real input,real *shared,real *global)
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
    atomicAdd(global,local);
  }
}

__device__ static inline
void real_sum_reduce(real input,real *global)
{
  real local=input;
  local+=__shfl_down_sync(0xFFFFFFFF,local,1);
  local+=__shfl_down_sync(0xFFFFFFFF,local,2);
  local+=__shfl_down_sync(0xFFFFFFFF,local,4);
  local+=__shfl_down_sync(0xFFFFFFFF,local,8);
  local+=__shfl_down_sync(0xFFFFFFFF,local,16);
  __syncthreads();
  if ((0x1F & threadIdx.x)==0) {
    atomicAdd(global,local);
  }
}
#endif
