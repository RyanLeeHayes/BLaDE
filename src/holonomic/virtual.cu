#include "holonomic/holonomic.h"
#include "system/system.h"
#include "run/run.h"
#include "system/potential.h"
#include "system/state.h"
#include "main/real3.h"



template <bool rectify>
__global__ void calc_virtualSite2_position_kernel(int N,struct VirtualSite2 *virt2,struct LeapState ls,real3_x box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x virtx;
  real3_x hostx[2];
  real3_x dx;
  real s,dxinv;

  if (i<N) {
    // virtx=((real3_x*)ls.x)[virt2[i].vidx];
    hostx[0]=((real3_x*)ls.x)[virt2[i].hidx[0]];
    hostx[1]=((real3_x*)ls.x)[virt2[i].hidx[1]];
    dx=real3_subpbc(hostx[0],hostx[1],box);

    dxinv=1/real3_mag<real>(dx);
    s=virt2[i].dist*dxinv+virt2[i].scale;
    virtx=hostx[0];
    real3_scaleinc(&virtx,s,dx);

    if (!rectify) { // put it close to where it was before
      real3_x prevx=((real3_x*)ls.x)[virt2[i].vidx]; // Where it was
      dx=real3_sub(real3_subpbc(virtx,prevx,box),real3_sub(virtx,prevx));
      real3_inc(&virtx,dx);
    }
    ((real3_x*)ls.x)[virt2[i].vidx]=virtx;
  }
}

template <bool rectify>
__global__ void calc_virtualSite3_position_kernel(int N,struct VirtualSite3 *virt3,struct LeapState ls,real3_x box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x virtx;
  real3_x hostx[3];
  real3_x a,b,c;
  real3_x dx;

  if (i<N) {
    // virtx=((real3_x*)ls.x)[virt2[i].vidx];
    for (int j=0; j<3; j++) {
      hostx[j]=((real3_x*)ls.x)[virt3[i].hidx[j]];
    }

    // Set up coordinate axes: a is along 0 to 1 bond, b is perpendicular to a and away from 2, c is bxa (why the left handed coordinate system charmm?)
    b=real3_subpbc(hostx[2],hostx[1],box);
    if (virt3[i].dist<0) { // Negative distance is used as a bisector flag (indicating to use the bisector of 1 and 2 as the 1 position)
      real3_scaleself(&b,(real_x)(0.5));
      real3_inc(&hostx[1],b);
    }
    a=real3_subpbc(hostx[1],hostx[0],box);
    real3_scaleself(&a,1/real3_mag<real_x>(a));
    c=real3_cross(b,a);
    real3_scaleself(&c,1/real3_mag<real_x>(c));
    b=real3_cross(a,c); // Already normalized

    virtx=real3_scale<real3_x>(sin(virt3[i].phi),c);
    real3_scaleinc(&virtx,cos(virt3[i].phi),b);
    real3_scaleself(&virtx,sin(virt3[i].theta));
    real3_scaleinc(&virtx,cos(virt3[i].theta),a);
    real3_scaleself(&virtx,abs(virt3[i].dist));
    real3_inc(&virtx,hostx[0]);

    if (!rectify) { // put it close to where it was before
      real3_x prevx=((real3_x*)ls.x)[virt3[i].vidx]; // Where it was
      dx=real3_sub(real3_subpbc(virtx,prevx,box),real3_sub(virtx,prevx));
      real3_inc(&virtx,dx);
    }
    ((real3_x*)ls.x)[virt3[i].vidx]=virtx;
  }
}

void calc_virtual_position(System *system,bool rectify)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;

  if (system->id) return; // head process only

  if (p->virtualSite2Count!=0) {
    if (rectify) {
      calc_virtualSite2_position_kernel<true><<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],s->orthBox);
    } else {
      calc_virtualSite2_position_kernel<false><<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],s->orthBox);
    }
  }

  if (p->virtualSite3Count!=0) {
    if (rectify) {
      calc_virtualSite3_position_kernel<true><<<(p->virtualSite3Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite3Count,p->virtualSite3_d,s->leapState[0],s->orthBox);
    } else {
      calc_virtualSite3_position_kernel<false><<<(p->virtualSite3Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite3Count,p->virtualSite3_d,s->leapState[0],s->orthBox);
    }
  }
}

__global__ void calc_virtualSite2_force_kernel(int N,struct VirtualSite2 *virt2,struct LeapState ls,real3_x box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  // real3_x virtx;
  real3_x hostx[2];
  real3_f virtf;
  // real3_f hostf[2];
  real3_x dx;
  real3_f df;
  real s,dxinv;

  if (i<N) {
    // virtx=((real3_x*)ls.x)[virt2[i].vidx];
    hostx[0]=((real3_x*)ls.x)[virt2[i].hidx[0]];
    hostx[1]=((real3_x*)ls.x)[virt2[i].hidx[1]];
    dx=real3_subpbc(hostx[0],hostx[1],box);

    virtf=((real3_f*)ls.f)[virt2[i].vidx];
    // hostf[0]=((real3_f*)ls.f)[virt2[i].hidx[0]];
    // hostf[1]=((real3_f*)ls.f)[virt2[i].hidx[1]];

    dxinv=1/real3_mag<real>(dx);
    s=virt2[i].dist*dxinv+virt2[i].scale;

    df=real3_scale<real3_f>(s-virt2[i].dist*dxinv*dxinv*dxinv*real3_dot<real>(dx,virtf),virtf);
    real3_dec(&((real3_f*)ls.f)[virt2[i].hidx[1]],df);
    real3_inc(&df,virtf);
    real3_inc(&((real3_f*)ls.f)[virt2[i].hidx[0]],df);
  }
}

__global__ void calc_virtualSite3_force_kernel(int N,struct VirtualSite3 *virt3,struct LeapState ls,real3_x box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x virtx;
  real3_x hostx[3];
  real3_x r,s,t;
  real_x rmag,smag;
  real3_f virtf;
  real3_f rf,tf,pf;
  real3_f hostf[3];

  if (i<N) {
    // r = radial vector
    // p = out-of-plane vector (unnormalized)
    // s = j-k vector
    // t = k-l vector
    // rf = radial force
    // pf = torsional force
    // tf = angular force
    virtx=((real3_x*)ls.x)[virt3[i].vidx];
    virtf=((real3_f*)ls.f)[virt3[i].vidx];
    for (int j=0; j<3; j++) {
      hostx[j]=((real3_x*)ls.x)[virt3[i].hidx[j]];
      hostf[j]=real3_reset<real3_f>();
    }

    r=real3_subpbc(virtx,hostx[0],box);
    rf=real3_scale<real3_f>(real3_dot<real_x>(r,virtf)/real3_mag2<real_x>(r),r);
    real3_inc(&hostf[0],rf);

    t=real3_subpbc(hostx[2],hostx[1],box);
    if (virt3[i].dist<0) { // Negative distance is used as a bisector flag (indicating to use the bisector of 1 and 2 as the 1 position)
      real3_scaleself(&t,(real_x)(0.5));
      real3_inc(&hostx[1],t);
    }
    s=real3_subpbc(hostx[0],hostx[1],box);

    // p=real3_cross(r,s);
    // q=real3_cross(s,t);

    // Assume no torsional force (there are input guards)
    rmag=real3_mag<real_x>(r);
    smag=real3_mag<real_x>(s);
    tf=real3_sub(virtf,rf);
    real3_scaleinc(&hostf[0],(smag-rmag)/smag,tf);
    if (virt3[i].dist<0) {
      real3_scaleinc(&hostf[1],((real_x)(0.5))*rmag/smag,tf);
      real3_scaleinc(&hostf[2],((real_x)(0.5))*rmag/smag,tf);
    } else {
      real3_scaleinc(&hostf[1],rmag/smag,tf);
    }

    for (int j=0; j<3; j++) {
      real3_inc(&((real3_f*)ls.f)[virt3[i].hidx[j]],hostf[j]);
    }
  }
}

void calc_virtual_force(System *system)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;

  if (system->id) return; // head process only

  if (p->virtualSite2Count!=0) {
    calc_virtualSite2_force_kernel<<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],s->orthBox);
  }

  if (p->virtualSite3Count!=0) {
    calc_virtualSite3_force_kernel<<<(p->virtualSite3Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite3Count,p->virtualSite3_d,s->leapState[0],s->orthBox);
  }
}
