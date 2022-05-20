#include "holonomic/holonomic.h"
#include "system/system.h"
#include "run/run.h"
#include "system/potential.h"
#include "system/state.h"

#include "main/real3.h"



template <bool flagBox,typename box_type>
__global__ void holonomic_rectify_triangle_kernel(int N,struct TriangleCons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x[3];
  real3_x xShift[3];
  real3_x dx[2];
  real b02[3];
  real3_x intCoord[3]; // xHat along x[1]-x[0], zHat along (x[1]-xPrev[0]) cross (xPrev[2]-xPrev[0])
  int j;

  if (i<N) {
    for (j=0; j<3; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx[j-1]=real3_subpbc<flagBox>(x[j],x[0],box);
        xShift[j]=real3_sub(dx[j-1],real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      x[j]=real3_add(x[j],xShift[j]);
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
    }

    // Define internal coordinates
    intCoord[0]=dx[0];
    real3_scaleself(&intCoord[0],1/real3_mag<real_x>(intCoord[0]));
    intCoord[2]=real3_cross(dx[0],dx[1]);
    real3_scaleself(&intCoord[2],1/real3_mag<real_x>(intCoord[2]));
    intCoord[1]=real3_cross(intCoord[2],intCoord[0]);

    // Rectify starting positions
    dx[0]=real3_scale<real3_x>(sqrt(b02[0]),intCoord[0]);
    real_x dx1proj=(b02[0]+b02[1]-b02[2])/(2*sqrt(b02[0]));
    dx[1]=real3_scale<real3_x>(dx1proj,intCoord[0]);
    real3_scaleinc(&dx[1],sqrt(b02[1]-dx1proj*dx1proj),intCoord[1]);
    x[1]=real3_add(x[0],dx[0]);
    x[2]=real3_add(x[0],dx[1]);

    // Finish up
    for (j=1; j<3; j++) {
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_rectify_branch1_kernel(int N,struct Branch1Cons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x[2];
  real3_x xShift[2];
  real3_x dx;
  real b02;
  int j;

  if (i<N) {
    for (j=0; j<2; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx=real3_subpbc<flagBox>(x[j],x[0],box);
        xShift[j]=real3_sub(dx,real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      x[j]=real3_add(x[j],xShift[j]);
    }

    b02=cons[i].b0[0];
    b02*=b02;
    real3_scaleself(&dx,sqrt(b02)/real3_mag<real_x>(dx));
    x[1]=real3_add(x[0],dx);
  
    // Finish up
    for (j=1; j<2; j++) {
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_rectify_branch2_kernel(int N,struct Branch2Cons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x[3];
  real3_x xShift[3];
  real3_x dx[2];
  real b02[2];
  int j;

  if (i<N) {
    for (j=0; j<3; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx[j-1]=real3_subpbc<flagBox>(x[j],x[0],box);
        xShift[j]=real3_sub(dx[j-1],real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      x[j]=real3_add(x[j],xShift[j]);
    }

    for (j=0; j<2; j++) {
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
      real3_scaleself(&dx[j],sqrt(b02[j])/real3_mag<real_x>(dx[j]));
      x[j+1]=real3_add(x[0],dx[j]);
    }

    // Finish up
    for (j=1; j<3; j++) {
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_rectify_branch3_kernel(int N,struct Branch3Cons *cons,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x[4];
  real3_x xShift[4];
  real3_x dx[3];
  real b02[3];
  int j;

  if (i<N) {
    for (j=0; j<4; j++) {
      x[j]=((real3_x*)ls.x)[cons[i].idx[j]];
      if (j>0) {
        dx[j-1]=real3_subpbc<flagBox>(x[j],x[0],box);
        xShift[j]=real3_sub(dx[j-1],real3_sub(x[j],x[0]));
      } else {
        xShift[0]=real3_reset<real3_x>();
      }
      x[j]=real3_add(x[j],xShift[j]);
    }

    for (j=0; j<3; j++) {
      b02[j]=cons[i].b0[j];
      b02[j]*=b02[j];
      real3_scaleself(&dx[j],sqrt(b02[j])/real3_mag<real_x>(dx[j]));
      x[j+1]=real3_add(x[0],dx[j]);
    }

    // Finish up
    for (j=1; j<4; j++) {
      ((real3_x*)ls.x)[cons[i].idx[j]]=real3_sub(x[j],xShift[j]);
    }
  }
}

template <bool flagBox,typename box_type>
void holonomic_rectifyT(System *system,box_type box)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;
  int N;

  N=p->triangleConsCount;
  if (N) holonomic_rectify_triangle_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->triangleCons_d,s->leapState[0],box);

  N=p->branch1ConsCount;
  if (N) holonomic_rectify_branch1_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch1Cons_d,s->leapState[0],box);

  N=p->branch2ConsCount;
  if (N) holonomic_rectify_branch2_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch2Cons_d,s->leapState[0],box);

  N=p->branch3ConsCount;
  if (N) holonomic_rectify_branch3_kernel<flagBox><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch3Cons_d,s->leapState[0],box);
}

void holonomic_rectify(System *system)
{
  if (system->state->typeBox) {
    holonomic_rectifyT<true>(system,system->state->tricBox);
  } else {
    holonomic_rectifyT<false>(system,system->state->orthBox);
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_rectifyback_triangle_kernel(int N,struct TriangleCons *cons,real3_x *position,box_type box,real3_x *positionBack,box_type boxBack)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x0,xb0,x,xb,dx,xNew,xShift;
  int j;

  if (i<N) {
    x0=position[cons[i].idx[0]];
    xb0=positionBack[cons[i].idx[0]];
    for (j=1; j<3; j++) {
      x=position[cons[i].idx[j]];
      xb=positionBack[cons[i].idx[j]];
      dx=real3_subpbc<flagBox>(xb,xb0,boxBack);
      xNew=real3_add(x0,dx);
      dx=real3_subpbc<flagBox>(xNew,x,box);
      xShift=real3_sub(dx,real3_sub(xNew,x));
      position[cons[i].idx[j]]=real3_add(xNew,xShift);
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_rectifyback_branch1_kernel(int N,struct Branch1Cons *cons,real3_x *position,box_type box,real3_x *positionBack,box_type boxBack)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x0,xb0,x,xb,dx,xNew,xShift;
  int j;
  
  if (i<N) {
    x0=position[cons[i].idx[0]];
    xb0=positionBack[cons[i].idx[0]];
    for (j=1; j<2; j++) {
      x=position[cons[i].idx[j]];
      xb=positionBack[cons[i].idx[j]];
      dx=real3_subpbc<flagBox>(xb,xb0,boxBack);
      xNew=real3_add(x0,dx);
      dx=real3_subpbc<flagBox>(xNew,x,box);
      xShift=real3_sub(dx,real3_sub(xNew,x));
      position[cons[i].idx[j]]=real3_add(xNew,xShift);
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_rectifyback_branch2_kernel(int N,struct Branch2Cons *cons,real3_x *position,box_type box,real3_x *positionBack,box_type boxBack)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x0,xb0,x,xb,dx,xNew,xShift;
  int j;
  
  if (i<N) {
    x0=position[cons[i].idx[0]];
    xb0=positionBack[cons[i].idx[0]];
    for (j=1; j<3; j++) {
      x=position[cons[i].idx[j]];
      xb=positionBack[cons[i].idx[j]];
      dx=real3_subpbc<flagBox>(xb,xb0,boxBack);
      xNew=real3_add(x0,dx);
      dx=real3_subpbc<flagBox>(xNew,x,box);
      xShift=real3_sub(dx,real3_sub(xNew,x));
      position[cons[i].idx[j]]=real3_add(xNew,xShift);
    }
  }
}

template <bool flagBox,typename box_type>
__global__ void holonomic_rectifyback_branch3_kernel(int N,struct Branch3Cons *cons,real3_x *position,box_type box,real3_x *positionBack,box_type boxBack)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x x0,xb0,x,xb,dx,xNew,xShift;
  int j;
  
  if (i<N) {
    x0=position[cons[i].idx[0]];
    xb0=positionBack[cons[i].idx[0]];
    for (j=1; j<4; j++) {
      x=position[cons[i].idx[j]];
      xb=positionBack[cons[i].idx[j]];
      dx=real3_subpbc<flagBox>(xb,xb0,boxBack);
      xNew=real3_add(x0,dx);
      dx=real3_subpbc<flagBox>(xNew,x,box);
      xShift=real3_sub(dx,real3_sub(xNew,x));
      position[cons[i].idx[j]]=real3_add(xNew,xShift);
    }
  }
}

template <bool flagBox,typename box_type>
void holonomic_rectifybackT(System *system,box_type box,box_type boxBackup)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;
  int N;

  N=p->triangleConsCount;
  if (N) holonomic_rectifyback_triangle_kernel<flagBox,box_type><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->triangleCons_d,(real3_x*)s->position_d,box,(real3_x*)s->positionb_d,boxBackup);

  N=p->branch1ConsCount;
  if (N) holonomic_rectifyback_branch1_kernel<flagBox,box_type><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch1Cons_d,(real3_x*)s->position_d,box,(real3_x*)s->positionb_d,boxBackup);

  N=p->branch2ConsCount;
  if (N) holonomic_rectifyback_branch2_kernel<flagBox,box_type><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch2Cons_d,(real3_x*)s->position_d,box,(real3_x*)s->positionb_d,boxBackup);

  N=p->branch3ConsCount;
  if (N) holonomic_rectifyback_branch3_kernel<flagBox,box_type><<<(N+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(N,p->branch3Cons_d,(real3_x*)s->position_d,box,(real3_x*)s->positionb_d,boxBackup);
}

void holonomic_rectifyback(System *system)
{
  if (system->state->typeBox) {
    holonomic_rectifybackT<true>(system,system->state->tricBox,system->state->tricBoxBackup);
  } else {
    holonomic_rectifybackT<false>(system,system->state->orthBox,system->state->orthBoxBackup);
  }
}
