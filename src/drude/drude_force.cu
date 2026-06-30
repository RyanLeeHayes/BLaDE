#include <cuda_runtime.h>
#include <math.h>

#include "drude/drude_plugin.h"
#include "main/real3.h"
#include "run/run.h"
#include "system/potential.h"
#include "system/state.h"
#include "system/system.h"

static __device__ __forceinline__
real drude_msld_scale_nbdirect_pair(int siteBlockA,int siteBlockB,const real *lambda,
  int *blockA,int *blockB,real *dscaleA,real *dscaleB)
{
  int bA=0xFFFF & siteBlockA;
  int bB=0xFFFF & siteBlockB;
  real lA=(bA && lambda? lambda[bA] : (real)1);
  real lB=(bB && lambda? lambda[bB] : (real)1);
  real dsA=(real)0;
  real dsB=(real)0;
  real scale=(real)1;

  if ((siteBlockA&0xFFFF0000)==(siteBlockB&0xFFFF0000)) {
    if (siteBlockA==siteBlockB) {
      scale=lA;
      if (bA) dsA=(real)1;
    } else {
      scale=(real)0;
    }
  } else {
    scale=lA*lB;
    if (bA) dsA=lB;
    if (bB) dsB=lA;
  }

  *blockA=bA;
  *blockB=bB;
  *dscaleA=dsA;
  *dscaleB=dsB;
  return scale;
}

template <bool flagBox,typename box_type>
__global__ void getforce_drude_spring_kernel(int pairCount,const struct DrudeSpringPairPotential *pairs,
  const real3 *position,real3_f *force,box_type box,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  if (i<pairCount) {
    struct DrudeSpringPairPotential pp=pairs[i];
    int id=pp.idx[0];
    int ip=pp.idx[1];

    real3 xd=position[id];
    real3 xp=position[ip];
    real3 dr=real3_subpbc<flagBox>(xd,xp,box);
    bool hasAxis1=(pp.anisoIdx[0]>=0);
    bool hasAxis2=(pp.anisoIdx[1]>=0 && pp.anisoIdx[2]>=0);

    if ((hasAxis1 || hasAxis2) && pp.r0==(real)0) {
      real a1=(hasAxis1? pp.aniso12 : (real)1);
      real a2=(hasAxis2? pp.aniso34 : (real)1);
      real a3=(real)3-a1-a2;
      real k3=pp.k/a3;
      real k1=pp.k/a1-k3;
      real k2=pp.k/a2-k3;

      lEnergy=((real)0.5)*k3*real3_mag2<real>(dr);
      at_real3_scaleinc(&force[id], k3,dr);
      at_real3_scaleinc(&force[ip],-k3,dr);

      if (hasAxis1) {
        int ia=pp.anisoIdx[0];
        real3 axis=real3_subpbc<flagBox>(xp,position[ia],box);
        real axisR=real3_mag<real>(axis);
        if (axisR>(real)0) {
          real invAxisR=(real)1/axisR;
          real3 dir;
          dir.x=axis.x*invAxisR;
          dir.y=axis.y*invAxisR;
          dir.z=axis.z*invAxisR;
          real rprime=real3_dot<real>(dr,dir);
          lEnergy+=((real)0.5)*k1*rprime*rprime;

          real3 f1;
          f1.x=k1*rprime*dir.x;
          f1.y=k1*rprime*dir.y;
          f1.z=k1*rprime*dir.z;
          real3 residual;
          residual.x=dr.x-rprime*dir.x;
          residual.y=dr.y-rprime*dir.y;
          residual.z=dr.z-rprime*dir.z;
          real f2scale=k1*rprime*invAxisR;
          real3 f2;
          f2.x=f2scale*residual.x;
          f2.y=f2scale*residual.y;
          f2.z=f2scale*residual.z;

          at_real3_inc(&force[id],f1);
          at_real3_dec(&force[ip],f1);
          at_real3_inc(&force[ip],f2);
          at_real3_dec(&force[ia],f2);
        }
      }

      if (hasAxis2) {
        int ia=pp.anisoIdx[1];
        int ib=pp.anisoIdx[2];
        real3 axis=real3_subpbc<flagBox>(position[ia],position[ib],box);
        real axisR=real3_mag<real>(axis);
        if (axisR>(real)0) {
          real invAxisR=(real)1/axisR;
          real3 dir;
          dir.x=axis.x*invAxisR;
          dir.y=axis.y*invAxisR;
          dir.z=axis.z*invAxisR;
          real rprime=real3_dot<real>(dr,dir);
          lEnergy+=((real)0.5)*k2*rprime*rprime;

          real3 f1;
          f1.x=k2*rprime*dir.x;
          f1.y=k2*rprime*dir.y;
          f1.z=k2*rprime*dir.z;
          real3 residual;
          residual.x=dr.x-rprime*dir.x;
          residual.y=dr.y-rprime*dir.y;
          residual.z=dr.z-rprime*dir.z;
          real f2scale=k2*rprime*invAxisR;
          real3 f2;
          f2.x=f2scale*residual.x;
          f2.y=f2scale*residual.y;
          f2.z=f2scale*residual.z;

          at_real3_inc(&force[id],f1);
          at_real3_dec(&force[ip],f1);
          at_real3_inc(&force[ia],f2);
          at_real3_dec(&force[ib],f2);
        }
      }
    } else {
      real r=real3_mag<real>(dr);
      real delta=r-pp.r0;
      real fbond=pp.k*delta;
      lEnergy=((real)0.5)*pp.k*delta*delta;

      if (r>0) {
        at_real3_scaleinc(&force[id], fbond/r,dr);
        at_real3_scaleinc(&force[ip],-fbond/r,dr);
      }
    }
  }

  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
__global__ void getforce_drude_screened_kernel(int pairCount,const struct DrudeScreenedPairPotential *pairs,
  const real3 *position,real3_f *force,box_type box,real_e *energy,
  const int *atomSiteBlock,const real *lambda,real_f *lambdaForce)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  if (i<pairCount) {
    struct DrudeScreenedPairPotential pp=pairs[i];
    int first[4]={pp.idx[0],pp.idx[0],pp.idx[1],pp.idx[1]};
    int second[4]={pp.idx[2],pp.idx[3],pp.idx[2],pp.idx[3]};
    real sign[4]={(real)1,(real)-1,(real)-1,(real)1};

    for (int j=0; j<4; j++) {
      int ia=first[j];
      int ib=second[j];
      real3 xa=position[ia];
      real3 xb=position[ib];
      real3 delta=real3_subpbc<flagBox>(xa,xb,box);
      real r=real3_mag<real>(delta);
      if (r<=0) continue;

      real rInv=1/r;
      real u=pp.screeningScale*r;
      real expu=exp(-u);
      real screening=1-((real)1+((real)0.5)*u)*expu;
      real pairEnergy0=sign[j]*pp.energyScale*screening*rInv;
      real pairScale=(real)1;
      int blockA=0;
      int blockB=0;
      real dscaleA=(real)0;
      real dscaleB=(real)0;
      if (atomSiteBlock) {
        pairScale=drude_msld_scale_nbdirect_pair(
          atomSiteBlock[ia],atomSiteBlock[ib],lambda,
          &blockA,&blockB,&dscaleA,&dscaleB);
      }
      real pairEnergy=pairScale*pairEnergy0;
      lEnergy+=pairEnergy;
      if (lambdaForce && pairEnergy0!=(real)0) {
        if (blockA) {
          atomicAdd(&lambdaForce[blockA],(real_f)(dscaleA*pairEnergy0));
        }
        if (blockB) {
          atomicAdd(&lambdaForce[blockB],(real_f)(dscaleB*pairEnergy0));
        }
      }

      real fScale=-pairScale*sign[j]*pp.energyScale*rInv*rInv*
        (screening*rInv-((real)0.5)*((real)1+u)*expu*pp.screeningScale);
      at_real3_scaleinc(&force[ia], fScale,delta);
      at_real3_scaleinc(&force[ib],-fScale,delta);
    }
  }

  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_drudeT(System *system,box_type box,bool calcEnergy)
{
  if (!system->drudePlugin) return;
  if (!system->drudePlugin->enabled) return;
  if (!system->run->calcTermFlag[eedrude]) return;

  DrudePlugin *plugin=system->drudePlugin;
  State *s=system->state;
  Run *r=system->run;
  real_e *pEnergy=NULL;
  int shMem=0;
  if (calcEnergy) {
    pEnergy=s->energy_d+eedrude;
    shMem=BLBO*sizeof(real)/32;
  }

  if (plugin->springPairCount>0) {
    getforce_drude_spring_kernel<flagBox><<<(plugin->springPairCount+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
      plugin->springPairCount,plugin->springPairs_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,pEnergy);
  }
  if (plugin->screenedPairCount>0) {
    getforce_drude_screened_kernel<flagBox><<<(plugin->screenedPairCount+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
      plugin->screenedPairCount,plugin->screenedPairs_d,
      (real3*)s->position_fd,(real3_f*)s->force_d,box,pEnergy,
      plugin->atomSiteBlock_d,s->lambda_fd,s->lambdaForce_d);
  }
  plugin->getforce_nbthole14(system,calcEnergy);
}

void DrudePlugin::getforce(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_drudeT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_drudeT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
