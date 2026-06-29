#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#include "drude/drude_plugin.h"
#include "io/io.h"
#include "main/real3.h"
#include "run/run.h"
#include "system/state.h"
#include "system/structure.h"
#include "system/system.h"

template <bool flagBox,typename box_type>
__device__ static inline
real3 drude_hardwall_delta(const real_x *position,int d0,int p0,box_type box)
{
  real3 xd;
  real3 xp;
  xd.x=(real)position[d0+0];
  xd.y=(real)position[d0+1];
  xd.z=(real)position[d0+2];
  xp.x=(real)position[p0+0];
  xp.y=(real)position[p0+1];
  xp.z=(real)position[p0+2];
  return real3_subpbc<flagBox>(xd,xp,box);
}

template <bool flagBox,typename box_type>
__global__ void drude_hardwall_kernel(int pairCount,const int2 *pairs,
  real_x *position,real_v *velocity,const real *ism,real maxDrudeDistance,
  real stepSize,real hardwallScale,box_type box,int *hardwallCount,int *tooFarCount)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i<pairCount) {
    int d=pairs[i].x;
    int p=pairs[i].y;
    int d0=3*d;
    int p0=3*p;

    real3 delta=drude_hardwall_delta<flagBox>(position,d0,p0,box);
    real r2=real3_mag2<real>(delta);
    if (r2<=(real)0) return;

    real r=sqrt(r2);
    if (r<=maxDrudeDistance) return;
    if (r>(real)2*maxDrudeDistance) {
      atomicAdd(tooFarCount,1);
      return;
    }

    atomicAdd(hardwallCount,1);

    real invR=(real)1/r;
    real bx=delta.x*invR;
    real by=delta.y*invR;
    real bz=delta.z*invR;

    real_v vd[3]={velocity[d0+0],velocity[d0+1],velocity[d0+2]};
    real_v vp[3]={velocity[p0+0],velocity[p0+1],velocity[p0+2]};
    real dotvd=vd[0]*bx+vd[1]*by+vd[2]*bz;
    real dotvp=vp[0]*bx+vp[1]*by+vp[2]*bz;
    real_v vdPerp[3]={vd[0]-dotvd*bx,vd[1]-dotvd*by,vd[2]-dotvd*bz};
    real_v vpPerp[3]={vp[0]-dotvp*bx,vp[1]-dotvp*by,vp[2]-dotvp*bz};

    real ismD=ism[d0];
    real ismP=ism[p0];
    real md=(ismD>(real)0? (real)(1/(ismD*ismD)) : (real)0);
    real mp=(ismP>(real)0? (real)(1/(ismP*ismP)) : (real)0);
    real deltaR=r-maxDrudeDistance;
    real deltaT=stepSize;

    if (md>(real)0 && mp<=(real)0) {
      if (fabs(dotvd)>(real)0) {
        deltaT=fmin(stepSize,deltaR/fabs(dotvd));
      }
      real vBond=hardwallScale/sqrt(md);
      real newDotvd=(fabs(dotvd)>(real)0? -dotvd*vBond/fabs(dotvd) : -vBond);
      real drd=-deltaR+deltaT*newDotvd;
      position[d0+0]+=bx*drd;
      position[d0+1]+=by*drd;
      position[d0+2]+=bz*drd;
      real3 cdelta=drude_hardwall_delta<flagBox>(position,d0,p0,box);
      real cr=real3_mag<real>(cdelta);
      if (cr>maxDrudeDistance && cr>(real)0) {
        real s=maxDrudeDistance/cr;
        position[d0+0]=position[p0+0]+cdelta.x*s;
        position[d0+1]=position[p0+1]+cdelta.y*s;
        position[d0+2]=position[p0+2]+cdelta.z*s;
      }
      velocity[d0+0]=vdPerp[0]+bx*newDotvd;
      velocity[d0+1]=vdPerp[1]+by*newDotvd;
      velocity[d0+2]=vdPerp[2]+bz*newDotvd;
      return;
    }

    if (mp>(real)0 && md<=(real)0) {
      if (fabs(dotvp)>(real)0) {
        deltaT=fmin(stepSize,deltaR/fabs(dotvp));
      }
      real vBond=hardwallScale/sqrt(mp);
      real newDotvp=(fabs(dotvp)>(real)0? -dotvp*vBond/fabs(dotvp) : -vBond);
      real drp=deltaR+deltaT*newDotvp;
      position[p0+0]+=bx*drp;
      position[p0+1]+=by*drp;
      position[p0+2]+=bz*drp;
      real3 cdelta=drude_hardwall_delta<flagBox>(position,d0,p0,box);
      real cr=real3_mag<real>(cdelta);
      if (cr>maxDrudeDistance && cr>(real)0) {
        real s=maxDrudeDistance/cr;
        position[p0+0]=position[d0+0]-cdelta.x*s;
        position[p0+1]=position[d0+1]-cdelta.y*s;
        position[p0+2]=position[d0+2]-cdelta.z*s;
      }
      velocity[p0+0]=vpPerp[0]+bx*newDotvp;
      velocity[p0+1]=vpPerp[1]+by*newDotvp;
      velocity[p0+2]=vpPerp[2]+bz*newDotvp;
      return;
    }

    if (md<=(real)0 || mp<=(real)0) {
      return;
    }

    real invTotal=(real)1/(md+mp);
    real vbCM=(md*dotvd+mp*dotvp)*invTotal;
    real relDotd=dotvd-vbCM;
    real relDotp=dotvp-vbCM;
    real relSpeed=relDotd-relDotp;

    if (fabs(relSpeed)>(real)0) {
      deltaT=fmin(stepSize,deltaR/fabs(relSpeed));
    }

    real vBond=hardwallScale/sqrt(md);
    real newRelDotd=(fabs(relDotd)>(real)0? -relDotd*vBond*mp*invTotal/fabs(relDotd) : -vBond*mp*invTotal);
    real newRelDotp=(fabs(relDotp)>(real)0? -relDotp*vBond*md*invTotal/fabs(relDotp) : -vBond*md*invTotal);
    real drd=-deltaR*mp*invTotal+deltaT*newRelDotd;
    real drp= deltaR*md*invTotal+deltaT*newRelDotp;
    real newDotd=newRelDotd+vbCM;
    real newDotp=newRelDotp+vbCM;

    position[d0+0]+=bx*drd;
    position[d0+1]+=by*drd;
    position[d0+2]+=bz*drd;
    position[p0+0]+=bx*drp;
    position[p0+1]+=by*drp;
    position[p0+2]+=bz*drp;
    real3 cdelta=drude_hardwall_delta<flagBox>(position,d0,p0,box);
    real cdx=cdelta.x;
    real cdy=cdelta.y;
    real cdz=cdelta.z;
    real cr=real3_mag<real>(cdelta);
    if (cr>maxDrudeDistance && cr>(real)0) {
      real s=maxDrudeDistance/cr;
      real tdx=cdx*s;
      real tdy=cdy*s;
      real tdz=cdz*s;
      real udx=position[p0+0]+cdx;
      real udy=position[p0+1]+cdy;
      real udz=position[p0+2]+cdz;
      real cmx=(md*udx+mp*position[p0+0])*invTotal;
      real cmy=(md*udy+mp*position[p0+1])*invTotal;
      real cmz=(md*udz+mp*position[p0+2])*invTotal;
      position[d0+0]=cmx+tdx*mp*invTotal;
      position[d0+1]=cmy+tdy*mp*invTotal;
      position[d0+2]=cmz+tdz*mp*invTotal;
      position[p0+0]=cmx-tdx*md*invTotal;
      position[p0+1]=cmy-tdy*md*invTotal;
      position[p0+2]=cmz-tdz*md*invTotal;
    }

    velocity[d0+0]=vdPerp[0]+bx*newDotd;
    velocity[d0+1]=vdPerp[1]+by*newDotd;
    velocity[d0+2]=vdPerp[2]+bz*newDotd;
    velocity[p0+0]=vpPerp[0]+bx*newDotp;
    velocity[p0+1]=vpPerp[1]+by*newDotp;
    velocity[p0+2]=vpPerp[2]+bz*newDotp;
  }
}

template <bool flagBox,typename box_type>
static real drude_pair_distance_host(System *system,int d,int p,box_type box)
{
  real3 xd;
  real3 xp;
  xd.x=(real)system->state->position[d][0];
  xd.y=(real)system->state->position[d][1];
  xd.z=(real)system->state->position[d][2];
  xp.x=(real)system->state->position[p][0];
  xp.y=(real)system->state->position[p][1];
  xp.z=(real)system->state->position[p][2];
  real3 delta=real3_subpbc<flagBox>(xd,xp,box);
  return real3_mag<real>(delta);
}

template <bool flagBox,typename box_type>
static void drude_compute_temperatures_hostT(System *system,DrudePlugin *plugin,box_type box)
{
  int nVel=system->state->lambdaCount+3*system->state->atomCount;
  int nPos=2*system->state->lambdaCount+3*system->state->atomCount;
  cudaMemcpy(system->state->velocityBuffer,system->state->velocityBuffer_d,nVel*sizeof(real_v),cudaMemcpyDeviceToHost);
  cudaMemcpy(system->state->positionBuffer,system->state->positionBuffer_d,nPos*sizeof(real_x),cudaMemcpyDeviceToHost);

  double keCom=0.0;
  double keRel=0.0;
  int comPairs=0;
  int relPairs=0;
  real maxDist=(real)0;
  for (int i=0; i<plugin->pairCount; i++) {
    int d=plugin->pairs_tmp[i].x;
    int p=plugin->pairs_tmp[i].y;
    real dist=drude_pair_distance_host<flagBox>(system,d,p,box);
    if (dist>maxDist) maxDist=dist;
    double md=system->structure->atomList[d].mass;
    double mp=system->structure->atomList[p].mass;
    if (md>0 && mp>0) {
      double M=md+mp;
      double mu=md*mp/M;
      double vdx=system->state->velocity[d][0];
      double vdy=system->state->velocity[d][1];
      double vdz=system->state->velocity[d][2];
      double vpx=system->state->velocity[p][0];
      double vpy=system->state->velocity[p][1];
      double vpz=system->state->velocity[p][2];
      double vcomx=(md*vdx+mp*vpx)/M;
      double vcomy=(md*vdy+mp*vpy)/M;
      double vcomz=(md*vdz+mp*vpz)/M;
      double vrelx=vdx-vpx;
      double vrely=vdy-vpy;
      double vrelz=vdz-vpz;
      keCom+=0.5*M*(vcomx*vcomx+vcomy*vcomy+vcomz*vcomz);
      keRel+=0.5*mu*(vrelx*vrelx+vrely*vrely+vrelz*vrelz);
      comPairs++;
      relPairs++;
    } else if (md>0 || mp>0) {
      int a=(md>0? d : p);
      double m=(md>0? md : mp);
      double vx=system->state->velocity[a][0];
      double vy=system->state->velocity[a][1];
      double vz=system->state->velocity[a][2];
      keCom+=0.5*m*(vx*vx+vy*vy+vz*vz);
      comPairs++;
    }
  }

  plugin->temperatureCOMLast=(comPairs>0? (real)(2*keCom/(3.0*comPairs*kB)) : (real)0);
  plugin->temperatureRelLast=(relPairs>0? (real)(2*keRel/(3.0*relPairs*kB)) : (real)0);
  plugin->maxPairDistanceLast=maxDist;
}

static void drude_compute_temperatures_host(System *system,DrudePlugin *plugin)
{
  if (!system->state || !system->structure) return;
  if (plugin->pairCount<=0) {
    plugin->temperatureCOMLast=0;
    plugin->temperatureRelLast=0;
    plugin->maxPairDistanceLast=0;
    return;
  }

  if (system->state->typeBox) {
    drude_compute_temperatures_hostT<true>(system,plugin,system->state->tricBox_f);
  } else {
    drude_compute_temperatures_hostT<false>(system,plugin,system->state->orthBox_f);
  }
}

void DrudePlugin::apply_hardwall(System *system,long int step)
{
  if (!is_active(system)) return;
  if (!system->run || !system->state) return;
  if (!pairs_d || !hardwallCount_d || !hardwallTooFarCount_d) return;

  hardwallCountLast=0;
  cudaMemsetAsync(hardwallCount_d,0,sizeof(int),system->run->updateStream);

  if (system->run->maxDrudeDistance>(real)0) {
    real hardwallScale=sqrt(kB*system->run->Tdrude);
    if (system->state->typeBox) {
      drude_hardwall_kernel<true><<<(pairCount+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(
        pairCount,pairs_d,
        system->state->leapState->x,
        system->state->leapState->v,
        system->state->leapState->ism,
        system->run->maxDrudeDistance,
        system->run->dt,
        hardwallScale,
        system->state->tricBox_f,
        hardwallCount_d,
        hardwallTooFarCount_d);
    } else {
      drude_hardwall_kernel<false><<<(pairCount+BLUP-1)/BLUP,BLUP,0,system->run->updateStream>>>(
        pairCount,pairs_d,
        system->state->leapState->x,
        system->state->leapState->v,
        system->state->leapState->ism,
        system->run->maxDrudeDistance,
        system->run->dt,
        hardwallScale,
        system->state->orthBox_f,
        hardwallCount_d,
        hardwallTooFarCount_d);
    }
  }

  bool checkTooFar=(system->run->maxDrudeDistance>(real)0 &&
    (system->run->freqNRG<=0 || (step%system->run->freqNRG)==0));
  if (checkTooFar) {
    int tooFarCount=0;
    cudaStreamSynchronize(system->run->updateStream);
    cudaMemcpy(&tooFarCount,hardwallTooFarCount_d,sizeof(int),cudaMemcpyDeviceToHost);
    if (tooFarCount>0) {
      fatal(__FILE__,__LINE__,
        "Drude particle moved too far beyond hard wall constraint (%d pair%s with r > 2*maxDrudeDistance).\n",
        tooFarCount,(tooFarCount==1?"":"s"));
    }
    cudaMemsetAsync(hardwallTooFarCount_d,0,sizeof(int),system->run->updateStream);
  }

  if (system->id==0 && system->run->freqNRG>0 && (step%system->run->freqNRG)==0) {
    // Ensure hard-wall updates on updateStream are visible before diagnostics.
    cudaStreamSynchronize(system->run->updateStream);
    cudaMemcpy(&hardwallCountLast,hardwallCount_d,sizeof(int),cudaMemcpyDeviceToHost);
    hardwallCountTotal+=hardwallCountLast;
    drude_compute_temperatures_host(system,this);
    fprintf(stdout,"DRUDE DIAG> step %ld Tcom %8.3f Trel %8.3f dmaxA %8.4f hardwall %d total %lld\n",
      step,
      (double)temperatureCOMLast,
      (double)temperatureRelLast,
      (double)(maxPairDistanceLast/ANGSTROM),
      hardwallCountLast,
      hardwallCountTotal);
  }
}
