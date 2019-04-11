#include "holonomic/holonomic.h"
#include "system/system.h"
#include "run/run.h"
#include "system/potential.h"
#include "system/state.h"

#include "main/real3.h"



__global__ void holonomic_rectify_triangle_kernel(int N,struct TriangleCons *cons,struct LeapState ls,real3 box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 x[3];
  real3 xShift[3];
  real3 dx[2];
  real b02[3];
  real3 intCoord[3]; // xHat along x[1]-x[0], zHat along (x[1]-xPrev[0]) cross (xPrev[2]-xPrev[0])
  int j;

  if (i<N) {
    for (j=0; j<3; j++) {
      x[j]=((real3*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx[j-1]=real3_subpbc(x[j],x[0],box);
        xShift[j]=real3_sub(dx[j-1],real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset();
      }
      x[j]=real3_add(x[j],xShift[j]);
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
    }

    // Define internal coordinates
    intCoord[0]=dx[0];
    real3_scaleself(&intCoord[0],1/real3_mag(intCoord[0]));
    intCoord[2]=real3_cross(dx[0],dx[1]);
    real3_scaleself(&intCoord[2],1/real3_mag(intCoord[2]));
    intCoord[1]=real3_cross(intCoord[2],intCoord[0]);

    // Rectify starting positions
    dx[0]=real3_scale(sqrt(b02[0]),intCoord[0]);
    real dx1proj=(b02[0]+b02[1]-b02[2])/(2*sqrt(b02[0]));
    dx[1]=real3_scale(dx1proj,intCoord[0]);
    real3_scaleinc(&dx[1],sqrt(b02[1]-dx1proj*dx1proj),intCoord[1]);
    x[1]=real3_add(x[0],dx[0]);
    x[2]=real3_add(x[0],dx[1]);

    // Finish up
    for (j=1; j<3; j++) {
      ((real3*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

__global__ void holonomic_rectify_branch1_kernel(int N,struct Branch1Cons *cons,struct LeapState ls,real3 box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 x[2];
  real3 xShift[2];
  real3 dx;
  real b02;
  int j;

  if (i<N) {
    for (j=0; j<2; j++) {
      x[j]=((real3*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx=real3_subpbc(x[j],x[0],box);
        xShift[j]=real3_sub(dx,real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset();
      }
      x[j]=real3_add(x[j],xShift[j]);
    }

    b02=cons[i].b0[0];
    b02*=b02;
    real3_scaleself(&dx,sqrt(b02)/real3_mag(dx));
    x[1]=real3_add(x[0],dx);
  
    // Finish up
    for (j=1; j<2; j++) {
      ((real3*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

__global__ void holonomic_rectify_branch2_kernel(int N,struct Branch2Cons *cons,struct LeapState ls,real3 box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 x[3];
  real3 xShift[3];
  real3 dx[2];
  real b02[2];
  int j;

  if (i<N) {
    for (j=0; j<3; j++) {
      x[j]=((real3*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx[j-1]=real3_subpbc(x[j],x[0],box);
        xShift[j]=real3_sub(dx[j-1],real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset();
      }
      x[j]=real3_add(x[j],xShift[j]);
    }

    for (j=0; j<2; j++) {
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
      real3_scaleself(&dx[j],sqrt(b02[j])/real3_mag(dx[j]));
      x[j+1]=real3_add(x[0],dx[j]);
    }

    // Finish up
    for (j=1; j<3; j++) {
      ((real3*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

__global__ void holonomic_rectify_branch3_kernel(int N,struct Branch3Cons *cons,struct LeapState ls,real3 box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3 x[4];
  real3 xShift[4];
  real3 dx[3];
  real b02[3];
  int j;

  if (i<N) {
    for (j=0; j<4; j++) {
      x[j]=((real3*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx[j-1]=real3_subpbc(x[j],x[0],box);
        xShift[j]=real3_sub(dx[j-1],real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset();
      }
      x[j]=real3_add(x[j],xShift[j]);
    }

    for (j=0; j<3; j++) {
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
      real3_scaleself(&dx[j],sqrt(b02[j])/real3_mag(dx[j]));
      x[j+1]=real3_add(x[0],dx[j]);
    }

    // Finish up
    for (j=1; j<4; j++) {
      ((real3*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

void holonomic_rectify(System *system)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;
  int N;

  N=p->triangleConsCount;
  if (N) holonomic_rectify_triangle_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->triangleCons_d,s->leapState[0],s->orthBox);

  N=p->branch1ConsCount;
  if (N) holonomic_rectify_branch1_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch1Cons_d,s->leapState[0],s->orthBox);

  N=p->branch2ConsCount;
  if (N) holonomic_rectify_branch2_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch2Cons_d,s->leapState[0],s->orthBox);

  N=p->branch3ConsCount;
  if (N) holonomic_rectify_branch3_kernel<<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch3Cons_d,s->leapState[0],s->orthBox);
}
