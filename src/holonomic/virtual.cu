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

void calc_virtual_position(System *system,bool rectify)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;

  if (system->id) return; // head process only
  if (p->virtualSite2Count==0) return;

  if (rectify) {
    calc_virtualSite2_position_kernel<true><<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],s->orthBox);
  } else {
    calc_virtualSite2_position_kernel<false><<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],s->orthBox);
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

void calc_virtual_force(System *system)
{
  Run *r=system->run;
  State *s=system->state;
  Potential *p=system->potential;

  if (system->id) return; // head process only
  if (p->virtualSite2Count==0) return;

  calc_virtualSite2_force_kernel<<<(p->virtualSite2Count+BLUP-1)/BLUP,BLUP,0,r->updateStream>>>(p->virtualSite2Count,p->virtualSite2_d,s->leapState[0],s->orthBox);
}
