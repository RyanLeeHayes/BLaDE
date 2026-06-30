#include <cuda_runtime.h>
#include <math.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

#include "drude/drude_plugin.h"
#include "domdec/domdec.h"
#include "main/gpu_check.h"
#include "main/real3.h"
#include "run/run.h"
#include "system/potential.h"
#include "system/state.h"
#include "system/system.h"

static const int DRUDE_NBTHOLE_ACTIVE_LIST_MIN_PAIRS=2048;
static const int DRUDE_NBTHOLE_ACTIVE_LIST_FALLBACK_REBUILD_PERIOD=10;
static const real DRUDE_NBTHOLE_ACTIVE_LIST_MIN_SKIN=(real)(0.25*ANGSTROM);

static bool drude_nbthole_term_active(System *system,const DrudePlugin *plugin)
{
  if (!plugin || !plugin->enabled) return false;
  if (!system || !system->run || !system->state) return false;
  if (!system->run->calcTermFlag[eedrude]) return false;
  return true;
}

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

static __device__ __forceinline__
real drude_msld_scale_nb14_pair(int siteBlockA,int siteBlockB,const real *lambda,
  int *blockA,int *blockB,real *dscaleA,real *dscaleB)
{
  return drude_msld_scale_nbdirect_pair(siteBlockA,siteBlockB,lambda,
    blockA,blockB,dscaleA,dscaleB);
}

template <bool useNb14Scaling>
static __device__ __forceinline__
real drude_msld_scale_pair(int siteBlockA,int siteBlockB,const real *lambda,
  int *blockA,int *blockB,real *dscaleA,real *dscaleB)
{
  if (useNb14Scaling) {
    return drude_msld_scale_nb14_pair(siteBlockA,siteBlockB,lambda,blockA,blockB,dscaleA,dscaleB);
  }
  return drude_msld_scale_nbdirect_pair(siteBlockA,siteBlockB,lambda,blockA,blockB,dscaleA,dscaleB);
}

template <bool flagBox,bool useNb14Scaling,typename box_type>
__global__ void getforce_drude_nbthole_kernel(int pairCount,
  const struct DrudeNBTholePairPotential *pairs,
  const real3 *position,
  real3_f *force,
  box_type box,
  real cutoff2,
  real_e *energy,
  const int *atomSiteBlock,
  const real *lambda,
  real_f *lambdaForce)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  if (i<pairCount) {
    struct DrudeNBTholePairPotential pp=pairs[i];
    int ia=pp.idx[0];
    int ib=pp.idx[1];
    real3 xa=position[ia];
    real3 xb=position[ib];
    real3 delta=real3_subpbc<flagBox>(xa,xb,box);
    real r2=real3_mag2<real>(delta);
    if (r2>(real)0 && r2<cutoff2) {
      real r=sqrt(r2);
      real rInv=1/r;
      real u=pp.screeningScale*r;
      real expu=exp(-u);
      real corr=-(((real)1+((real)0.5)*u)*expu); // S(r)-1
      real dSdr=((real)0.5)*((real)1+u)*expu*pp.screeningScale;

      real pairEnergy0=pp.energyScale*corr*rInv;
      real dEdr0=pp.energyScale*(dSdr*rInv-corr*rInv*rInv);

      real pairScale=(real)1;
      int blockA=0;
      int blockB=0;
      real dscaleA=(real)0;
      real dscaleB=(real)0;
      if (atomSiteBlock) {
        pairScale=drude_msld_scale_pair<useNb14Scaling>(
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

      real fScale=(pairScale*dEdr0)*rInv;
      at_real3_scaleinc(&force[ia], fScale,delta);
      at_real3_scaleinc(&force[ib],-fScale,delta);
    }
  }

  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,bool useNb14Scaling,typename box_type>
__global__ void getforce_drude_nbthole_active_kernel(int activePairCount,
  const int *activePairIndices,
  const struct DrudeNBTholePairPotential *pairs,
  const real3 *position,
  real3_f *force,
  box_type box,
  real cutoff2,
  real_e *energy,
  const int *atomSiteBlock,
  const real *lambda,
  real_f *lambdaForce)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real lEnergy=0;
  extern __shared__ real sEnergy[];

  if (i<activePairCount) {
    int pairIdx=activePairIndices[i];
    struct DrudeNBTholePairPotential pp=pairs[pairIdx];
    int ia=pp.idx[0];
    int ib=pp.idx[1];
    real3 xa=position[ia];
    real3 xb=position[ib];
    real3 delta=real3_subpbc<flagBox>(xa,xb,box);
    real r2=real3_mag2<real>(delta);
    if (r2>(real)0 && r2<cutoff2) {
      real r=sqrt(r2);
      real rInv=1/r;
      real u=pp.screeningScale*r;
      real expu=exp(-u);
      real corr=-(((real)1+((real)0.5)*u)*expu); // S(r)-1
      real dSdr=((real)0.5)*((real)1+u)*expu*pp.screeningScale;

      real pairEnergy0=pp.energyScale*corr*rInv;
      real dEdr0=pp.energyScale*(dSdr*rInv-corr*rInv*rInv);

      real pairScale=(real)1;
      int blockA=0;
      int blockB=0;
      real dscaleA=(real)0;
      real dscaleB=(real)0;
      if (atomSiteBlock) {
        pairScale=drude_msld_scale_pair<useNb14Scaling>(
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

      real fScale=(pairScale*dEdr0)*rInv;
      at_real3_scaleinc(&force[ia], fScale,delta);
      at_real3_scaleinc(&force[ib],-fScale,delta);
    }
  }

  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
struct DrudeNbtholeWithinCutoffPred {
  const struct DrudeNBTholePairPotential *pairs;
  const real3 *position;
  box_type box;
  real cutoff2;

  __host__ __device__
  DrudeNbtholeWithinCutoffPred(const struct DrudeNBTholePairPotential *pairs_,
    const real3 *position_,box_type box_,real cutoff2_)
  : pairs(pairs_), position(position_), box(box_), cutoff2(cutoff2_) {}

  __host__ __device__
  bool operator()(const int pairIdx) const
  {
    struct DrudeNBTholePairPotential pp=pairs[pairIdx];
    real3 xa=position[pp.idx[0]];
    real3 xb=position[pp.idx[1]];
    real3 delta=real3_subpbc<flagBox>(xa,xb,box);
    real r2=real3_mag2<real>(delta);
    return (r2>(real)0 && r2<cutoff2);
  }
};

template <bool flagBox,typename box_type>
static int build_nbthole_active_indices(System *system,box_type box,real cutoff2)
{
  DrudePlugin *plugin=system->drudePlugin;
  if (!plugin || plugin->nbtholePairCount<=0) return 0;
  if (!plugin->nbtholePairs_d) return 0;
  if (!system->state || !system->run) return 0;

  if (!plugin->nbtholeActiveIndices_d || plugin->nbtholeActiveCapacity<plugin->nbtholePairCount) {
    if (plugin->nbtholeActiveIndices_d) cudaFree(plugin->nbtholeActiveIndices_d);
    gpuCheck(cudaMalloc(&plugin->nbtholeActiveIndices_d,plugin->nbtholePairCount*sizeof(int)));
    plugin->nbtholeActiveCapacity=plugin->nbtholePairCount;
  }

  thrust::counting_iterator<int> begin(0);
  thrust::counting_iterator<int> end(plugin->nbtholePairCount);
  thrust::device_ptr<int> outBegin=thrust::device_pointer_cast(plugin->nbtholeActiveIndices_d);
  DrudeNbtholeWithinCutoffPred<flagBox,box_type> pred(plugin->nbtholePairs_d,(real3*)system->state->position_fd,box,cutoff2);
  thrust::device_ptr<int> outEnd=thrust::copy_if(
    thrust::cuda::par.on(system->run->nbdirectStream),
    begin,end,outBegin,pred);

  return (int)(outEnd-outBegin);
}

template <bool flagBox,typename box_type>
static void launch_nbthole_pair_kernel(System *system,box_type box,bool calcEnergy)
{
  DrudePlugin *plugin=system->drudePlugin;
  if (!drude_nbthole_term_active(system,plugin)) return;
  if (plugin->nbtholePairCount<=0) return;
  if (plugin->nbtholeCutoff<=(real)0) return;

  State *s=system->state;
  Run *r=system->run;
  real_e *pEnergy=NULL;
  int shMem=0;
  if (calcEnergy) {
    pEnergy=s->energy_d+eedrude;
    shMem=BLBO*sizeof(real)/32;
  }

  real cutoff2=plugin->nbtholeCutoff*plugin->nbtholeCutoff;
  if (plugin->nbtholePairCount>=DRUDE_NBTHOLE_ACTIVE_LIST_MIN_PAIRS) {
    int rebuildPeriod=DRUDE_NBTHOLE_ACTIVE_LIST_FALLBACK_REBUILD_PERIOD;
    real skin=DRUDE_NBTHOLE_ACTIVE_LIST_MIN_SKIN;
    if (system->domdec) {
      if (system->domdec->freqDomdec>0) {
        rebuildPeriod=system->domdec->freqDomdec;
      }
      if (system->domdec->cullPad>skin) {
        skin=system->domdec->cullPad;
      }
    }
    real buildCutoff=plugin->nbtholeCutoff+skin;
    real buildCutoff2=buildCutoff*buildCutoff;
    long int currentStep=r->step;

    bool needRebuild=!plugin->nbtholeActiveListValid;
    if (!needRebuild) {
      if (plugin->nbtholeActiveBuildCutoffLast<buildCutoff) {
        needRebuild=true;
      } else if (currentStep<plugin->nbtholeActiveBuildStepLast) {
        needRebuild=true;
      } else if ((currentStep-plugin->nbtholeActiveBuildStepLast)>=rebuildPeriod) {
        needRebuild=true;
      }
    }

    int activePairCount=plugin->nbtholeActiveCountLast;
    if (needRebuild) {
      activePairCount=build_nbthole_active_indices<flagBox>(system,box,buildCutoff2);
      plugin->nbtholeActiveListValid=true;
      plugin->nbtholeActiveBuildStepLast=currentStep;
      plugin->nbtholeActiveBuildCutoffLast=buildCutoff;
      plugin->nbtholeActiveRebuildPeriodLast=rebuildPeriod;
    }

    plugin->nbtholeUsedActiveListLast=true;
    plugin->nbtholeActiveCountLast=activePairCount;
    if (activePairCount<=0) return;

    getforce_drude_nbthole_active_kernel<flagBox,false><<<(activePairCount+BLBO-1)/BLBO,BLBO,shMem,r->nbdirectStream>>>(
      activePairCount,
      plugin->nbtholeActiveIndices_d,
      plugin->nbtholePairs_d,
      (real3*)s->position_fd,
      (real3_f*)s->force_d,
      box,
      cutoff2,
      pEnergy,
      plugin->atomSiteBlock_d,
      s->lambda_fd,
      s->lambdaForce_d);
  } else {
    plugin->nbtholeUsedActiveListLast=false;
    plugin->nbtholeActiveListValid=false;
    plugin->nbtholeActiveBuildStepLast=-1;
    plugin->nbtholeActiveBuildCutoffLast=(real)0;
    plugin->nbtholeActiveRebuildPeriodLast=1;
    plugin->nbtholeActiveCountLast=plugin->nbtholePairCount;
    getforce_drude_nbthole_kernel<flagBox,false><<<(plugin->nbtholePairCount+BLBO-1)/BLBO,BLBO,shMem,r->nbdirectStream>>>(
      plugin->nbtholePairCount,
      plugin->nbtholePairs_d,
      (real3*)s->position_fd,
      (real3_f*)s->force_d,
      box,
      cutoff2,
      pEnergy,
      plugin->atomSiteBlock_d,
      s->lambda_fd,
      s->lambdaForce_d);
  }
}

template <bool flagBox,typename box_type>
static void launch_nbthole14_pair_kernel(System *system,box_type box,bool calcEnergy)
{
  DrudePlugin *plugin=system->drudePlugin;
  if (!drude_nbthole_term_active(system,plugin)) return;
  if (plugin->nbthole14PairCount<=0) return;

  State *s=system->state;
  Run *r=system->run;
  real_e *pEnergy=NULL;
  int shMem=0;
  if (calcEnergy) {
    pEnergy=s->energy_d+eedrude;
    shMem=BLBO*sizeof(real)/32;
  }

  real cutoff2=r->cutoffs.rCut*r->cutoffs.rCut;
  getforce_drude_nbthole_kernel<flagBox,true><<<(plugin->nbthole14PairCount+BLBO-1)/BLBO,BLBO,shMem,r->bondedStream>>>(
    plugin->nbthole14PairCount,
    plugin->nbthole14Pairs_d,
    (real3*)s->position_fd,
    (real3_f*)s->force_d,
    box,
    cutoff2,
    pEnergy,
    plugin->atomSiteBlock_d,
    s->lambda_fd,
    s->lambdaForce_d);
}

void DrudePlugin::getforce_nbthole(System *system,bool calcEnergy)
{
  if (!drude_nbthole_term_active(system,this)) {
    nbtholeUsedActiveListLast=false;
    nbtholeActiveListValid=false;
    nbtholeActiveCountLast=0;
    return;
  }
  if (!nbtholePairs_d || nbtholePairCount<=0) {
    nbtholeUsedActiveListLast=false;
    nbtholeActiveListValid=false;
    nbtholeActiveCountLast=0;
    return;
  }

  if (system->state->typeBox) {
    launch_nbthole_pair_kernel<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    launch_nbthole_pair_kernel<false>(system,system->state->orthBox_f,calcEnergy);
  }
}

void DrudePlugin::getforce_nbthole14(System *system,bool calcEnergy)
{
  if (!drude_nbthole_term_active(system,this)) return;
  if (!nbthole14Pairs_d || nbthole14PairCount<=0) return;

  if (system->state->typeBox) {
    launch_nbthole14_pair_kernel<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    launch_nbthole14_pair_kernel<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
