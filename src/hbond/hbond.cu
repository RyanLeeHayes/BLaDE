#include <cuda_runtime.h>
#include <math.h>

#include "hbond/hbond.h"
#include "domdec/hbondlist.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "main/real3.h"



//Hyres HBOND Force Evaluator
template <bool flagBox,typename box_type>
__global__ void getforce_hbond_kernel(int capacity,int *hbondCount_d,HbondPotential *hbonds,HbondParameters hp,real3 *position,real3_f *force,real *lambda,real_f *lambdaForce,box_type box,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii,jj,kk;
  real3 drij,drkj,dr;
  real ijinv,kjinv,cosijk,dota;
  real r,fr,fc;
  real kij,kkj;
  real g;
  real rinv;
  real U,dUdr; 
  real s2,s4,s10,s12;
  real mij,mkj;
  real c2,dgdc;
  HbondPotential dh;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,xj,xk;
  real3 fO,fN,fH;
  int hbondCount=min(*hbondCount_d,capacity);

  if (i<hbondCount) {
    // Geometry
    dh=hbonds[i];
    ii=dh.idx[0]; //O
    jj=dh.idx[1]; //H
    kk=dh.idx[2]; //N
    xi=position[ii];
    xj=position[jj];
    xk=position[kk];

    drij=real3_subpbc<flagBox>(xi,xj,box);
    drkj=real3_subpbc<flagBox>(xk,xj,box);
    dota=real3_dot<real>(drij,drkj);
    mij=real3_mag<real>(drij);
    mkj=real3_mag<real>(drkj);
    ijinv=1/mij;
    kjinv=1/mkj;
    cosijk=dota*ijinv*kjinv;

    dr = real3_subpbc<flagBox>(xi,xk,box); //O-N distance
    r=real3_mag<real>(dr);
    //Interaction
    if (r<hp.rCut && cosijk<0) {
      rinv = 1/r;
      s2=hp.sigma*rinv;
      s2=s2*s2;
      s4=s2*s2;
      s10=s4*s4*s2;
      s12=s10*s2;
      U=hp.epsilon*(5*s12-6*s10);
      dUdr=60*hp.epsilon*(s10-s12)*rinv;

      c2=cosijk*cosijk;
      g=-c2*cosijk;
      dgdc = -3*c2;

      fr=g*dUdr*rinv;
      fc=U*dgdc;
      
      kij=fc*ijinv;
      kkj=fc*kjinv;

      fO.x=fr*dr.x+kij*(drkj.x*kjinv-cosijk*drij.x*ijinv);
      fO.y=fr*dr.y+kij*(drkj.y*kjinv-cosijk*drij.y*ijinv);
      fO.z=fr*dr.z+kij*(drkj.z*kjinv-cosijk*drij.z*ijinv);
      fN.x=-fr*dr.x+kkj*(drij.x*ijinv-cosijk*drkj.x*kjinv);
      fN.y=-fr*dr.y+kkj*(drij.y*ijinv-cosijk*drkj.y*kjinv);
      fN.z=-fr*dr.z+kkj*(drij.z*ijinv-cosijk*drkj.z*kjinv);
      fH.x=-(fO.x+fN.x);
      fH.y=-(fO.y+fN.y);
      fH.z=-(fO.z+fN.z);

      int b[2];
      real l[2]={1,1};
      b[0]=0xFFFF & dh.siteBlock[0];
      b[1]=0xFFFF & dh.siteBlock[1];
      if (b[0]) {
        l[0]=lambda[b[0]];
        if (b[1]) l[1]=lambda[b[1]];
      }
      real E=U*g;                 // unscaled energy = dE/d(lambda product)
      real scale=l[0]*l[1];

      // Lambda force: d(l0*l1*E)/dl0 = l1*E, d/dl1 = l0*E
      if (b[0]) {
        atomicAdd(&lambdaForce[b[0]],l[1]*E);
        if (b[1]) atomicAdd(&lambdaForce[b[1]],l[0]*E);
      }

      // Spatial force scaled by the lambda product (scale fO and fN before fH=-(fO+fN))
     
      at_real3_scaleinc(&force[ii],scale,fO);
      at_real3_scaleinc(&force[jj],scale,fH);
      at_real3_scaleinc(&force[kk],scale,fN);
      
      lEnergy=scale*E;
    }
  }
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox, typename box_type>
void getforce_hbondT(System *system, box_type box, bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  struct HbondList *hl=&p->hbondList;
  int N=hl->capacity;
  int shMem=0;
  real_e *pEnergy=NULL;
  if (r->useHbond==false) return;
  if (r->calcTermFlag[eehbond]==false) return;

  if (N==0) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eehbond;
  }

  getforce_hbond_kernel <flagBox> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(N,hl->flags_d,hl->list_d,p->hbondParameters,(real3*)s->position_fd,(real3_f*)s->force_d,s->lambda_fd,s->lambdaForce_d,box,pEnergy);
}

void getforce_hbond(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_hbondT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_hbondT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
