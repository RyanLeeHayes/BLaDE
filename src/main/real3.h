#ifndef MAIN_VEC_H
#define MAIN_VEC_H

#include "defines.h"

#include <math.h>

__device__ static inline
void at_real3_scaleinc(real3 *a,real f,real3 x)
{
  realAtomicAdd(&(a[0].x),f*x.x);
  realAtomicAdd(&(a[0].y),f*x.y);
  realAtomicAdd(&(a[0].z),f*x.z);
}

__host__ __device__ static inline
real real3_mag(real3 a)
{
  return sqrt(a.x*a.x+a.y*a.y+a.z*a.z);
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

#ifdef COMMENTED
__host__ __device__ static inline
void vec_subtract(vec a,vec b,vec c)
{
  c[0]=a[0]-b[0];
  c[1]=a[1]-b[1];
  c[2]=a[2]-b[2];
}


__host__ __device__ static inline
void vec_subpbc(vec a,vec b,vec box,vec c)
{
  int i;
  for (i=0; i<DIM3; i++) {
    c[i]=a[i]-b[i];
    c[i]-=box[i]*floor(c[i]/box[i]+0.5);
  }
}


__host__ __device__ static inline
void vec_subsaveshift(vec a,vec b,vec box,vec shift,vec c)
{
  int i;
  for (i=0; i<DIM3; i++) {
    c[i]=a[i]-b[i];
    shift[i]=-box[i]*floor(c[i]/box[i]+0.5);
    c[i]+=shift[i];
  }
}


__host__ __device__ static inline
void vec_subshift_ns(vec a,vec b,vec sab,vec c) {
  c[0]=a[0]-b[0]+sab[0];
  c[1]=a[1]-b[1]+sab[1];
  c[2]=a[2]-b[2]+sab[2];
}


__host__ __device__ static inline
void vec_subshift(vec a,vec b,vec sa,vec sb,vec sab,vec c) {
  c[0]=a[0]+sa[0]-b[0]-sb[0]+sab[0];
  c[1]=a[1]+sa[1]-b[1]-sb[1]+sab[1];
  c[2]=a[2]+sa[2]-b[2]-sb[2]+sab[2];
}


__host__ __device__ static inline
void vec_subquick(vec a,vec b,vec box,vec c)
{
  int i;
  for (i=0; i<DIM3; i++) {
    c[i]=a[i]-b[i];
    c[i]-=box[i]*((c[i]>box[i]/2)-(c[i]<-box[i]/2));
  }
}


__host__ __device__ static inline
void fixvec_sub(unsignedfixreal* a,unsignedfixreal* b,vec fr2r,vec c)
{
  int i;
  for (i=0; i<DIM3; i++) {
    c[i]=fr2r[i]*((fixreal) (a[i]-b[i]));
  }
}


__host__ __device__ static inline
real vec_quickdist2(vec a,vec b,vec box)
{
  int i;
  real c,s=0;
  for (i=0; i<DIM3; i++) {
    c=a[i]-b[i];
    c-=box[i]*((c>box[i]/2)-(c<-box[i]/2));
    s+=c*c;
  }
  return s;
}


__host__ __device__ static inline
real vec_shiftdist2(vec a,vec b,vec shift)
{
  int i;
  real c,s=0;
  for (i=0; i<DIM3; i++) {
    c=a[i]-b[i]+shift[i];
    s+=c*c;
  }
  return s;
}


__host__ __device__ static inline
real vec_mag(vec a)
{
  return sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]);
}


__host__ __device__ static inline
real vec_mag2(vec a)
{
  return a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
}


__host__ __device__ static inline
real vec_dot(vec a,vec b)
{
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}


__host__ __device__ static inline
void vec_cross(vec a,vec b,vec c)
{
  c[0]=a[1]*b[2]-a[2]*b[1];
  c[1]=a[2]*b[0]-a[0]*b[2];
  c[2]=a[0]*b[1]-a[1]*b[0];
}


__host__ __device__ static inline
void vec__add(vec a,vec b,vec c) // extra underscore needed on BGQ
{
  c[0]=a[0]+b[0];
  c[1]=a[1]+b[1];
  c[2]=a[2]+b[2];
}


__host__ __device__ static inline
void vec_scale(vec a,real f,vec x)
{
  a[0]=f*x[0];
  a[1]=f*x[1];
  a[2]=f*x[2];
}


__host__ __device__ static inline
void vec_scaleself(vec a,real f)
{
  a[0]*=f;
  a[1]*=f;
  a[2]*=f;
}


__host__ __device__ static inline
void vec_inc(vec a,vec x)
{
  a[0]+=x[0];
  a[1]+=x[1];
  a[2]+=x[2];
}


__host__ __device__ static inline
void vec_dec(vec a,vec x)
{
  a[0]-=x[0];
  a[1]-=x[1];
  a[2]-=x[2];
}


__host__ __device__ static inline
void vec_scaleinc(vec a,real f,vec x)
{
  a[0]+=f*x[0];
  a[1]+=f*x[1];
  a[2]+=f*x[2];
}


// From http://stackoverflow.com/questions/16077464/atomicadd-for-real-on-gpu
__device__ static inline double doubleAtomicAdd(double* address, double val)
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

#ifdef DOUBLEFLAG
#define realAtomicAdd doubleAtomicAdd
#else
#define realAtomicAdd atomicAdd
#endif

__device__ static inline
void at_vec_inc(vec a,vec x)
{
  realAtomicAdd(&(a[0]),x[0]);
  realAtomicAdd(&(a[1]),x[1]);
  realAtomicAdd(&(a[2]),x[2]);
}


__device__ static inline
void at_vec_dec(vec a,vec x)
{
  realAtomicAdd(&(a[0]),-x[0]);
  realAtomicAdd(&(a[1]),-x[1]);
  realAtomicAdd(&(a[2]),-x[2]);
}


__device__ static inline
void at_vec_scaleinc(vec a,real f,vec x)
{
  realAtomicAdd(&(a[0]),f*x[0]);
  realAtomicAdd(&(a[1]),f*x[1]);
  realAtomicAdd(&(a[2]),f*x[2]);
}


__host__ __device__ static inline
void vec_reset(vec a)
{
  a[0]=0;
  a[1]=0;
  a[2]=0;
}


__host__ __device__ static inline
void vec_copy(vec a,vec b)
{
  a[0]=b[0];
  a[1]=b[1];
  a[2]=b[2];
}


__host__ __device__ static inline
void ivec_copy(int* a,int* b)
{
  a[0]=b[0];
  a[1]=b[1];
  a[2]=b[2];
}


__host__ __device__ static inline
int ivec_eq(int* a,int* b)
{
  return a[0]==b[0] && a[1]==b[1] && a[2]==b[2];
}

#endif
#endif
