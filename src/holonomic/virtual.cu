#include "holonomic/holonomic.h"
#include "system/system.h"
#include "run/run.h"
#include "system/potential.h"
#include "system/state.h"
#include "main/real3.h"



template <bool flagBox,bool rectify,typename box_type>
__global__ void calc_virtualSite2_position_kernel(int N,struct VirtualSite2 *virt2,struct LeapState ls,box_type box)
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
    dx=real3_subpbc<flagBox>(hostx[0],hostx[1],box);

    dxinv=1/real3_mag<real>(dx);
    s=virt2[i].dist*dxinv+virt2[i].scale;
    virtx=hostx[0];
    real3_scaleinc(&virtx,s,dx);

    if (!rectify) { // put it close to where it was before
      real3_x prevx=((real3_x*)ls.x)[virt2[i].vidx]; // Where it was
      dx=real3_sub(real3_subpbc<flagBox>(virtx,prevx,box),real3_sub(virtx,prevx));
      real3_inc(&virtx,dx);
    }
    ((real3_x*)ls.x)[virt2[i].vidx]=virtx;
  }
}

template <bool flagBox,bool rectify,typename box_type>
__global__ void calc_virtualSite3_position_kernel(int N,struct VirtualSite3 *virt3,struct LeapState ls,box_type box)
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
    b=real3_subpbc<flagBox>(hostx[2],hostx[1],box);
    if (virt3[i].dist<0) { // Negative distance is used as a bisector flag (indicating to use the bisector of 1 and 2 as the 1 position)
      real3_scaleself(&b,(real_x)(0.5));
      real3_inc(&hostx[1],b);
    }
    a=real3_subpbc<flagBox>(hostx[1],hostx[0],box);
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
      dx=real3_sub(real3_subpbc<flagBox>(virtx,prevx,box),real3_sub(virtx,prevx));
      real3_inc(&virtx,dx);
    }
    ((real3_x*)ls.x)[virt3[i].vidx]=virtx;
  }
}

template <bool flagBox,typename box_type>
void calc_virtual_positionT(System *system,box_type box,bool rectify)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;

  if (system->id) return; // head process only

  if (p->virtualSite2Count!=0) {
    if (rectify) {
      calc_virtualSite2_position_kernel<flagBox,true><<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],box);
    } else {
      calc_virtualSite2_position_kernel<flagBox,false><<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],box);
    }
  }

  if (p->virtualSite3Count!=0) {
    if (rectify) {
      calc_virtualSite3_position_kernel<flagBox,true><<<(p->virtualSite3Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite3Count,p->virtualSite3_d,s->leapState[0],box);
    } else {
      calc_virtualSite3_position_kernel<flagBox,false><<<(p->virtualSite3Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite3Count,p->virtualSite3_d,s->leapState[0],box);
    }
  }
}

void calc_virtual_position(System *system,bool rectify)
{
  if (system->state->typeBox) {
    calc_virtual_positionT<true>(system,system->state->tricBox,rectify);
  } else {
    calc_virtual_positionT<true>(system,system->state->orthBox,rectify);
  }
}



template <bool flagBox,typename box_type>
__global__ void calc_virtualSite2_force_kernel(int N,struct VirtualSite2 *virt2,struct LeapState ls,box_type box)
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
    dx=real3_subpbc<flagBox>(hostx[0],hostx[1],box);

    virtf=((real3_f*)ls.f)[virt2[i].vidx];
    // hostf[0]=((real3_f*)ls.f)[virt2[i].hidx[0]];
    // hostf[1]=((real3_f*)ls.f)[virt2[i].hidx[1]];

    dxinv=1/real3_mag<real>(dx);
    s=virt2[i].dist*dxinv+virt2[i].scale;

    df=real3_scale<real3_f>(s-virt2[i].dist*dxinv*dxinv*dxinv*real3_dot<real>(dx,virtf),virtf);
    at_real3_dec(&((real3_f*)ls.f)[virt2[i].hidx[1]],df);
    real3_inc(&df,virtf);
    at_real3_inc(&((real3_f*)ls.f)[virt2[i].hidx[0]],df);
  }
}

template <bool flagBox,typename box_type>
__global__ void calc_virtualSite3_force_kernel(int N,struct VirtualSite3 *virt3,struct LeapState ls,box_type box)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real3_x virtx;
  real3_x hostx[3];
  real3_x r,s,t;
  real3_x p,q,rr,tt;
  real_x sinv2,st;
  real3_f virtf;
  real3_f tfresult,pfresult; // angular and torsional results, respectively
  real_f pfmag;
  real3_f hostf[3];

  if (i<N) {
    // r = radial vector
    // s = j-k vector
    // t = k-l vector
    // p = r x s out-of-plane vector (unnormalized)
    // q = s x t
    // rr = r component perpendicular to s
    // tt = t component perpendicular to s
    // pfmag = torsional force magnitude
    virtx=((real3_x*)ls.x)[virt3[i].vidx];
    virtf=((real3_f*)ls.f)[virt3[i].vidx];
    for (int j=0; j<3; j++) {
      hostx[j]=((real3_x*)ls.x)[virt3[i].hidx[j]];
      hostf[j]=real3_reset<real3_f>();
    }

    r=real3_subpbc<flagBox>(virtx,hostx[0],box);
    t=real3_subpbc<flagBox>(hostx[1],hostx[2],box);
    if (virt3[i].dist<0) { // Negative distance is used as a bisector flag (indicating to use the bisector of 1 and 2 as the 1 position)
      real3_scaleself(&t,(real_x)(0.5));
      real3_dec(&hostx[1],t);
    }
    s=real3_subpbc<flagBox>(hostx[0],hostx[1],box);
    sinv2=1/real3_mag2<real_x>(s);

    real3_inc(&hostf[0],virtf);

    tfresult=real3_scale<real3_f>((real_f)sinv2,real3_cross(real3_cast<real3_f>(s),real3_cross(real3_cast<real3_f>(r),virtf))); // (s x (r x f)) / (s^2)
    real3_inc(&hostf[1],tfresult);
    real3_dec(&hostf[0],tfresult);

    if (sin(virt3[i].theta)>((real)1e-6)) {
      p=real3_cross(r,s);
      q=real3_cross(s,t);
      rr=real3_scale<real3_x>(sinv2,real3_cross<real3_x>(s,p));
      tt=real3_scale<real3_x>(sinv2,real3_cross<real3_x>(q,s));

      pfmag=real3_dot<real_f>(p,virtf)/real3_mag<real_f>(p);
      pfresult=real3_scale<real3_f>((real3_mag<real_f>(rr)*(pfmag))/
        (real3_mag<real_f>(tt)*real3_mag<real_f>(q)),q);
      st=real3_dot<real_x>(s,t)*sinv2;
      real3_inc(&hostf[2],pfresult);
      real3_scaleinc(&hostf[0],st,pfresult);
      real3_scaleinc(&hostf[1],-1-st,pfresult);
    }

    if (virt3[i].dist<0) {
      real3_scaleself(&hostf[1],(real_f)0.5);
      real3_inc(&hostf[2],hostf[1]);
    }

    for (int j=0; j<3; j++) {
      at_real3_inc(&((real3_f*)ls.f)[virt3[i].hidx[j]],hostf[j]);
    }
  }
}

template <bool flagBox,typename box_type>
void calc_virtual_forceT(System *system,box_type box)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;

  if (system->id) return; // head process only

  if (p->virtualSite2Count!=0) {
    calc_virtualSite2_force_kernel<flagBox><<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],box);
  }

  if (p->virtualSite3Count!=0) {
    calc_virtualSite3_force_kernel<flagBox><<<(p->virtualSite3Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite3Count,p->virtualSite3_d,s->leapState[0],box);
  }
}

void calc_virtual_force(System *system)
{
  if (system->state->typeBox) {
    calc_virtual_forceT<true>(system,system->state->tricBox);
  } else {
    calc_virtual_forceT<false>(system,system->state->orthBox);
  }
}
