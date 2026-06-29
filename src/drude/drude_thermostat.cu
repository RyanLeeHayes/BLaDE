#include <cuda_runtime.h>
#include <math.h>

#include "drude/drude_plugin.h"
#include "rng/rng_gpu.h"
#include "run/run.h"
#include "system/state.h"
#include "system/system.h"

__global__ void drude_save_pre_thermostat_kernel(int pairCount,const int2 *pairs,
  const real_v *velocity,real_v *preVelocity)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<pairCount) {
    int d=pairs[i].x;
    int p=pairs[i].y;
    int d0=3*d;
    int p0=3*p;
    preVelocity[6*i+0]=velocity[d0+0];
    preVelocity[6*i+1]=velocity[d0+1];
    preVelocity[6*i+2]=velocity[d0+2];
    preVelocity[6*i+3]=velocity[p0+0];
    preVelocity[6*i+4]=velocity[p0+1];
    preVelocity[6*i+5]=velocity[p0+2];
  }
}

__global__ void drude_post_thermostat_kernel(int pairCount,const int2 *pairs,
  real_v *velocity,real_x *position,const real_v *preVelocity,const real *random,const real *ism,
  real halfdt,real frictionSys,real noiseSys,real frictionDrude,real noiseDrude)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<pairCount) {
    int d=pairs[i].x;
    int p=pairs[i].y;
    int d0=3*d;
    int p0=3*p;

    real ismD=ism[d0];
    real ismP=ism[p0];
    real md=(ismD>(real)0? (real)(1/(ismD*ismD)) : (real)0);
    real mp=(ismP>(real)0? (real)(1/(ismP*ismP)) : (real)0);

    real_v vdGen[3]={velocity[d0+0],velocity[d0+1],velocity[d0+2]};
    real_v vpGen[3]={velocity[p0+0],velocity[p0+1],velocity[p0+2]};
    real_v vd0[3]={preVelocity[6*i+0],preVelocity[6*i+1],preVelocity[6*i+2]};
    real_v vp0[3]={preVelocity[6*i+3],preVelocity[6*i+4],preVelocity[6*i+5]};

    real_v vdNew[3]={vdGen[0],vdGen[1],vdGen[2]};
    real_v vpNew[3]={vpGen[0],vpGen[1],vpGen[2]};

    if (md>(real)0 && mp>(real)0) {
      real M=md+mp;
      real invM=(real)1/M;
      real mu=md*mp*invM;
      real invSqrtM=(real)(1/sqrt(M));
      real invSqrtMu=(real)(1/sqrt(mu));
      real fracP=mp*invM;
      real fracD=md*invM;
      for (int j=0; j<3; j++) {
        real vCom0=(md*vd0[j]+mp*vp0[j])*invM;
        real vRel0=vd0[j]-vp0[j];
        real randCom=random[6*i+j];
        real randRel=random[6*i+3+j];
        real vCom1=frictionSys*vCom0+noiseSys*invSqrtM*randCom;
        real vRel1=frictionDrude*vRel0+noiseDrude*invSqrtMu*randRel;
        vdNew[j]=vCom1+fracP*vRel1;
        vpNew[j]=vCom1-fracD*vRel1;
      }
    } else if (md>(real)0 && mp<=(real)0) {
      for (int j=0; j<3; j++) {
        real randCom=random[6*i+j];
        vdNew[j]=frictionSys*vd0[j]+noiseSys*ismD*randCom;
      }
    } else if (mp>(real)0 && md<=(real)0) {
      for (int j=0; j<3; j++) {
        real randCom=random[6*i+j];
        vpNew[j]=frictionSys*vp0[j]+noiseSys*ismP*randCom;
      }
    }

    for (int j=0; j<3; j++) {
      velocity[d0+j]=vdNew[j];
      velocity[p0+j]=vpNew[j];
      position[d0+j]+=halfdt*(vdNew[j]-vdGen[j]);
      position[p0+j]+=halfdt*(vpNew[j]-vpGen[j]);
    }
  }
}

void DrudePlugin::pre_thermostat(System *system)
{
  if (!is_active(system)) return;
  if (!pairs_d || !preThermostatVelocity_d || !thermostatRandom_d) return;
  if (!system->rngGPU || !system->state || !system->run) return;

  system->rngGPU->rand_normal(6*pairCount,thermostatRandom_d,system->run->updateStream);
  drude_save_pre_thermostat_kernel<<<(pairCount+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(
    pairCount,pairs_d,system->state->leapState->v,preThermostatVelocity_d);
}

void DrudePlugin::post_thermostat(System *system)
{
  if (!is_active(system)) return;
  if (!pairs_d || !preThermostatVelocity_d || !thermostatRandom_d) return;
  if (!system->state || !system->run) return;

  real dt=system->run->dt;
  real frictionSys=exp(-system->run->gamma*dt);
  real noiseSys=sqrt(((real)1-frictionSys*frictionSys)*kB*system->run->T);
  real frictionDrude=exp(-system->run->gammaDrude*dt);
  real noiseDrude=sqrt(((real)1-frictionDrude*frictionDrude)*kB*system->run->Tdrude);
  real halfdt=((real)0.5)*dt;

  drude_post_thermostat_kernel<<<(pairCount+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(
    pairCount,pairs_d,
    system->state->leapState->v,
    system->state->leapState->x,
    preThermostatVelocity_d,
    thermostatRandom_d,
    system->state->leapState->ism,
    halfdt,frictionSys,noiseSys,frictionDrude,noiseDrude);
}
