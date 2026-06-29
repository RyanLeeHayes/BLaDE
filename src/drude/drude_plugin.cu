#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <map>
#include <set>
#include <string.h>

#include "drude/drude_plugin.h"
#include "io/io.h"
#include "main/gpu_check.h"
#include "msld/msld.h"
#include "run/run.h"
#include "rng/rng_cpu.h"
#include "system/potential.h"
#include "system/structure.h"
#include "system/state.h"
#include "system/system.h"

static inline bool drude_pair_valid_index(int idx,int atomCount)
{
  return (idx>=0 && idx<atomCount);
}

static inline long long drude_make_ordered_pair_key(int a,int b)
{
  int i=(a<b? a : b);
  int j=(a<b? b : a);
  return (((long long)i)<<32)|(unsigned int)j;
}

static real drude_charmm_spring_k_to_internal(real k)
{
  return (real)(2.0*KCAL_MOL/(ANGSTROM*ANGSTROM))*k;
}

static inline void drude_add_constraint_pair(std::set<long long> &pairs,int a,int b)
{
  if (a==b) return;
  pairs.insert(drude_make_ordered_pair_key(a,b));
}

static void drude_validate_no_holonomic_overlap(System *system,DrudePlugin *plugin)
{
  if (!system || !system->potential) return;
  if (plugin->pairCount<=0) return;

  Potential *pot=system->potential;
  std::set<long long> constrainedPairs;

  for (size_t i=0; i<pot->triangleCons_tmp.size(); i++) {
    const TriangleCons &c=pot->triangleCons_tmp[i];
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[1]);
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[2]);
    drude_add_constraint_pair(constrainedPairs,c.idx[1],c.idx[2]);
  }
  for (size_t i=0; i<pot->branch1Cons_tmp.size(); i++) {
    const Branch1Cons &c=pot->branch1Cons_tmp[i];
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[1]);
  }
  for (size_t i=0; i<pot->branch2Cons_tmp.size(); i++) {
    const Branch2Cons &c=pot->branch2Cons_tmp[i];
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[1]);
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[2]);
  }
  for (size_t i=0; i<pot->branch3Cons_tmp.size(); i++) {
    const Branch3Cons &c=pot->branch3Cons_tmp[i];
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[1]);
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[2]);
    drude_add_constraint_pair(constrainedPairs,c.idx[0],c.idx[3]);
  }

  for (int i=0; i<plugin->pairCount; i++) {
    int drudeIdx=plugin->pairs_tmp[i].x;
    int parentIdx=plugin->pairs_tmp[i].y;
    if (constrainedPairs.count(drude_make_ordered_pair_key(drudeIdx,parentIdx))) {
      fatal(__FILE__,__LINE__,
        "Drude pair (%d,%d) conflicts with a holonomic constraint bond. "
        "Remove this SHAKE constraint or Drude spring pair.\n",
        drudeIdx,parentIdx);
    }
  }
}

static void drude_build_pair_list(DrudePlugin *plugin,System *system)
{
  std::map<int,int> drudeToParent;
  std::set<long long> uniqueSpringPairs;

  plugin->pairs_tmp.clear();
  if (!system->structure) {
    return;
  }
  int atomCount=system->structure->atomCount;

  for (size_t i=0; i<plugin->springPairs_tmp.size(); i++) {
    const DrudeSpringPairPotential &spring=plugin->springPairs_tmp[i];
    int drudeIdx=spring.idx[0];
    int parentIdx=spring.idx[1];
    long long key=((long long)drudeIdx<<32)|(unsigned int)parentIdx;

    if (!drude_pair_valid_index(drudeIdx,atomCount) || !drude_pair_valid_index(parentIdx,atomCount)) {
      fatal(__FILE__,__LINE__,"Drude spring pair index out of range: drude=%d parent=%d atomCount=%d\n",
        drudeIdx,parentIdx,atomCount);
    }
    if (drudeIdx==parentIdx) {
      fatal(__FILE__,__LINE__,"Drude spring pair cannot use identical drude and parent atom index %d\n",drudeIdx);
    }
    if (!isfinite((double)spring.k) || spring.k<=(real)0) {
      fatal(__FILE__,__LINE__,
        "Drude spring pair entry %d has invalid spring constant k=%g\n",
        (int)i,(double)spring.k);
    }
    if (drude_spring_has_anisotropy(&spring)) {
      if (fabs((double)spring.r0)>(double)1e-12) {
        fatal(__FILE__,__LINE__,
          "Anisotropic Drude spring entry %d requires r0=0 (found %g)\n",
          (int)i,(double)spring.r0);
      }
      if ((spring.anisoIdx[1]<0)!=(spring.anisoIdx[2]<0)) {
        fatal(__FILE__,__LINE__,
          "Anisotropic Drude spring entry %d has incomplete axis2 definition: axis3=%d axis4=%d\n",
          (int)i,spring.anisoIdx[1],spring.anisoIdx[2]);
      }
      for (int j=0; j<3; j++) {
        if (spring.anisoIdx[j]>=0 && !drude_pair_valid_index(spring.anisoIdx[j],atomCount)) {
          fatal(__FILE__,__LINE__,
            "Anisotropic Drude spring entry %d axis index out of range: axis[%d]=%d atomCount=%d\n",
            (int)i,j,spring.anisoIdx[j],atomCount);
        }
      }
      bool hasAxis1=(spring.anisoIdx[0]>=0);
      bool hasAxis2=(spring.anisoIdx[1]>=0 && spring.anisoIdx[2]>=0);
      real a1=(hasAxis1? spring.aniso12 : (real)1);
      real a2=(hasAxis2? spring.aniso34 : (real)1);
      if (!isfinite((double)spring.aniso12) || spring.aniso12<=(real)0 ||
          !isfinite((double)spring.aniso34) || spring.aniso34<=(real)0) {
        fatal(__FILE__,__LINE__,
          "Anisotropic Drude spring entry %d has invalid factors aniso12=%g aniso34=%g\n",
          (int)i,(double)spring.aniso12,(double)spring.aniso34);
      }
      if (a1+a2>=(real)3) {
        fatal(__FILE__,__LINE__,
          "Anisotropic Drude spring entry %d has invalid effective factors a1=%g a2=%g\n",
          (int)i,(double)a1,(double)a2);
      }
    }
    if (uniqueSpringPairs.count(key)) {
      fatal(__FILE__,__LINE__,"Duplicate drude spring pair found: drude=%d parent=%d\n",drudeIdx,parentIdx);
    }
    uniqueSpringPairs.insert(key);
    if (drudeToParent.count(drudeIdx)==0) {
      drudeToParent[drudeIdx]=parentIdx;
      plugin->pairs_tmp.push_back(make_int2(drudeIdx,parentIdx));
    } else if (drudeToParent[drudeIdx]!=parentIdx) {
      fatal(__FILE__,__LINE__,
        "Drude atom %d assigned to multiple parents (%d and %d)\n",
        drudeIdx,drudeToParent[drudeIdx],parentIdx);
    }
  }
}

static int drude_count_anisotropic_springs(const DrudePlugin *plugin)
{
  int count=0;
  for (size_t i=0; i<plugin->springPairs_tmp.size(); i++) {
    if (drude_spring_has_anisotropy(&plugin->springPairs_tmp[i])) count++;
  }
  return count;
}

static void drude_validate_screened_pairs(System *system,DrudePlugin *plugin)
{
  if (!system || !system->structure) return;
  if (plugin->screenedPairCount<=0) return;

  int atomCount=system->structure->atomCount;
  std::map<int,int> drudeToParent;
  std::set<std::pair<long long,long long> > uniqueScreenedPairs;

  for (int i=0; i<plugin->pairCount; i++) {
    drudeToParent[plugin->pairs_tmp[i].x]=plugin->pairs_tmp[i].y;
  }

  for (int i=0; i<plugin->screenedPairCount; i++) {
    const DrudeScreenedPairPotential &pp=plugin->screenedPairs_tmp[i];
    int d1=pp.idx[0];
    int p1=pp.idx[1];
    int d2=pp.idx[2];
    int p2=pp.idx[3];

    if (!drude_pair_valid_index(d1,atomCount) ||
        !drude_pair_valid_index(p1,atomCount) ||
        !drude_pair_valid_index(d2,atomCount) ||
        !drude_pair_valid_index(p2,atomCount)) {
      fatal(__FILE__,__LINE__,
        "Drude screened pair index out of range at entry %d: (%d,%d,%d,%d), atomCount=%d\n",
        i,d1,p1,d2,p2,atomCount);
    }
    if (d1==p1 || d2==p2) {
      fatal(__FILE__,__LINE__,
        "Drude screened pair entry %d contains identical drude/parent atom indices: (%d,%d,%d,%d)\n",
        i,d1,p1,d2,p2);
    }
    if (d1==d2 && p1==p2) {
      fatal(__FILE__,__LINE__,
        "Drude screened pair entry %d references the same dipole twice: (%d,%d)\n",
        i,d1,p1);
    }

    std::map<int,int>::const_iterator it1=drudeToParent.find(d1);
    if (it1==drudeToParent.end() || it1->second!=p1) {
      fatal(__FILE__,__LINE__,
        "Drude screened pair entry %d has no matching spring pair for first dipole (%d,%d)\n",
        i,d1,p1);
    }
    std::map<int,int>::const_iterator it2=drudeToParent.find(d2);
    if (it2==drudeToParent.end() || it2->second!=p2) {
      fatal(__FILE__,__LINE__,
        "Drude screened pair entry %d has no matching spring pair for second dipole (%d,%d)\n",
        i,d2,p2);
    }

    if (!isfinite((double)pp.screeningScale) || pp.screeningScale<(real)0) {
      fatal(__FILE__,__LINE__,
        "Drude screened pair entry %d has invalid screeningScale=%g (must be finite and >=0)\n",
        i,(double)pp.screeningScale);
    }
    if (!isfinite((double)pp.energyScale)) {
      fatal(__FILE__,__LINE__,
        "Drude screened pair entry %d has invalid non-finite energyScale=%g\n",
        i,(double)pp.energyScale);
    }

    long long k1=(((long long)d1)<<32)|(unsigned int)p1;
    long long k2=(((long long)d2)<<32)|(unsigned int)p2;
    if (k1>k2) {
      long long t=k1;
      k1=k2;
      k2=t;
    }
    std::pair<long long,long long> pairKey=std::make_pair(k1,k2);
    if (uniqueScreenedPairs.count(pairKey)>0) {
      fatal(__FILE__,__LINE__,
        "Duplicate Drude screened pair entry detected at %d: (%d,%d)-(%d,%d)\n",
        i,d1,p1,d2,p2);
    }
    uniqueScreenedPairs.insert(pairKey);
  }
}

static void drude_validate_nbthole_pair_list(const std::vector<DrudeNBTholePairPotential> &pairs,
  int pairCount,int atomCount,const char *label)
{
  std::set<long long> uniquePairs;
  for (int i=0; i<pairCount; i++) {
    const DrudeNBTholePairPotential &pp=pairs[i];
    int a=pp.idx[0];
    int b=pp.idx[1];
    if (!drude_pair_valid_index(a,atomCount) || !drude_pair_valid_index(b,atomCount)) {
      fatal(__FILE__,__LINE__,
        "Drude %s pair index out of range at entry %d: (%d,%d), atomCount=%d\n",
        label,i,a,b,atomCount);
    }
    if (a==b) {
      fatal(__FILE__,__LINE__,
        "Drude %s pair entry %d contains identical atom indices (%d)\n",
        label,i,a);
    }
    if (!isfinite((double)pp.screeningScale) || pp.screeningScale<(real)0) {
      fatal(__FILE__,__LINE__,
        "Drude %s pair entry %d has invalid screeningScale=%g (must be finite and >=0)\n",
        label,i,(double)pp.screeningScale);
    }
    if (!isfinite((double)pp.energyScale)) {
      fatal(__FILE__,__LINE__,
        "Drude %s pair entry %d has invalid non-finite energyScale=%g\n",
        label,i,(double)pp.energyScale);
    }
    long long key=drude_make_ordered_pair_key(a,b);
    if (uniquePairs.count(key)>0) {
      fatal(__FILE__,__LINE__,
        "Duplicate Drude %s pair entry detected at %d: (%d,%d)\n",
        label,i,a,b);
    }
    uniquePairs.insert(key);
  }
}

static void drude_validate_nbthole_pairs(System *system,DrudePlugin *plugin)
{
  if (!system || !system->structure) return;
  int atomCount=system->structure->atomCount;

  drude_validate_nbthole_pair_list(plugin->nbtholePairs_tmp,plugin->nbtholePairCount,
    atomCount,"NBTHOLE");
  drude_validate_nbthole_pair_list(plugin->nbthole14Pairs_tmp,plugin->nbthole14PairCount,
    atomCount,"NBTHOLE14");
}

static int drude_merge_site_block(int site,int block)
{
  if (site>=(1<<16) || block>=(1<<16) || site<0 || block<0) {
    fatal(__FILE__,__LINE__,
      "Drude MSLD site/block out of range. site=%d block=%d\n",
      site,block);
  }
  return ((site<<16)|block);
}

static void drude_build_atom_siteblock_map(System *system,DrudePlugin *plugin)
{
  plugin->atomSiteBlock_tmp.clear();
  if (!system || !system->structure) return;

  int atomCount=system->structure->atomCount;
  plugin->atomSiteBlock_tmp.assign(atomCount,0);

  if (!system->msld || !system->msld->atomBlock || !system->msld->lambdaSite) {
    return;
  }

  for (int i=0; i<atomCount; i++) {
    int block=system->msld->atomBlock[i];
    if (block<0 || block>=system->msld->blockCount) {
      fatal(__FILE__,__LINE__,
        "Drude MSLD atomBlock out of range for atom %d: block=%d blockCount=%d\n",
        i,block,system->msld->blockCount);
    }
    int site=system->msld->lambdaSite[block];
    plugin->atomSiteBlock_tmp[i]=drude_merge_site_block(site,block);
  }
}

static void drude_apply_mass_transfer(System *system,DrudePlugin *plugin)
{
  if (!system->structure || !system->state) {
    return;
  }
  if (plugin->pairCount<=0) {
    return;
  }

  for (int i=0; i<plugin->pairCount; i++) {
    int drudeIdx=plugin->pairs_tmp[i].x;
    int parentIdx=plugin->pairs_tmp[i].y;
    real md=system->structure->atomList[drudeIdx].mass;
    real mp=system->structure->atomList[parentIdx].mass;
    real targetMass=md;
    bool applyOverride=false;

    if (plugin->explicitMasses.count(drudeIdx)==1) {
      targetMass=plugin->explicitMasses[drudeIdx];
      applyOverride=true;
    } else if (md<=(real)0) {
      targetMass=plugin->defaultDrudeMass;
      applyOverride=true;
    }

    if (applyOverride) {
      real totalMass=md+mp;
      if (targetMass<=(real)0 || !isfinite((double)targetMass)) {
        fatal(__FILE__,__LINE__,
          "Invalid Drude target mass %.6f for atom %d\n",
          (double)targetMass,drudeIdx);
      }
      if (totalMass<=targetMass) {
        fatal(__FILE__,__LINE__,
          "Cannot assign Drude mass %.6f at atom %d: parent atom %d total pair mass %.6f is too small\n",
          (double)targetMass,drudeIdx,parentIdx,(double)totalMass);
      }
      system->structure->atomList[drudeIdx].mass=targetMass;
      system->structure->atomList[parentIdx].mass=totalMass-targetMass;
      plugin->massTransferApplied=true;
    }
  }

  for (int i=0; i<plugin->pairCount; i++) {
    int drudeIdx=plugin->pairs_tmp[i].x;
    int parentIdx=plugin->pairs_tmp[i].y;
    real md=system->structure->atomList[drudeIdx].mass;
    real mp=system->structure->atomList[parentIdx].mass;

    if (md<=(real)0) {
      fatal(__FILE__,__LINE__,
        "Invalid non-positive Drude mass %.6f at atom %d after transfer setup\n",
        (double)md,drudeIdx);
    }
    if (mp<=(real)0) {
      fatal(__FILE__,__LINE__,
        "Invalid non-positive parent mass %.6f at atom %d after transfer setup\n",
        (double)mp,parentIdx);
    }
    if (mp<=md) {
      fatal(__FILE__,__LINE__,
        "Invalid Drude/parent mass pair for atoms (%d,%d): parent mass %.6f must exceed Drude mass %.6f\n",
        drudeIdx,parentIdx,(double)mp,(double)md);
    }
    for (int j=0; j<3; j++) {
      system->state->invsqrtMass[drudeIdx][j]=(real)(1/sqrt(md));
      system->state->invsqrtMass[parentIdx][j]=(real)(1/sqrt(mp));
    }
  }

  // pairCount>0 here (early return above), so the upload always runs.
  int n=system->state->lambdaCount+3*system->state->atomCount;
  gpuCheck(cudaMemcpy(system->state->invsqrtMassBuffer_d,system->state->invsqrtMassBuffer,
    n*sizeof(real),cudaMemcpyHostToDevice));
}

static void drude_initialize_pair_velocities(System *system,DrudePlugin *plugin)
{
  if (!system->state || !system->run || !system->rngCPU) {
    return;
  }
  if (plugin->pairCount<=0) {
    return;
  }

  // Keep checkpoint restart velocities untouched.
  if (system->run->fnmCPI!="") {
    return;
  }

  real Tsys=system->run->T;
  real Tdrude=system->run->Tdrude;
  for (int i=0; i<plugin->pairCount; i++) {
    int drudeIdx=plugin->pairs_tmp[i].x;
    int parentIdx=plugin->pairs_tmp[i].y;
    real md=system->structure->atomList[drudeIdx].mass;
    real mp=system->structure->atomList[parentIdx].mass;

    if (md>(real)0 && mp>(real)0) {
      real M=md+mp;
      real mu=md*mp/M;
      real fracP=mp/M;
      real fracD=md/M;
      real sigmaCom=sqrt(kB*Tsys/M);
      real sigmaRel=sqrt(kB*Tdrude/mu);
      for (int j=0; j<3; j++) {
        real vCom=sigmaCom*system->rngCPU->rand_normal();
        real vRel=sigmaRel*system->rngCPU->rand_normal();
        system->state->velocity[drudeIdx][j]=vCom+fracP*vRel;
        system->state->velocity[parentIdx][j]=vCom-fracD*vRel;
      }
    } else if (md>(real)0 && mp<=(real)0) {
      real sigma=sqrt(kB*Tsys/md);
      for (int j=0; j<3; j++) {
        system->state->velocity[drudeIdx][j]=sigma*system->rngCPU->rand_normal();
      }
    } else if (mp>(real)0 && md<=(real)0) {
      real sigma=sqrt(kB*Tsys/mp);
      for (int j=0; j<3; j++) {
        system->state->velocity[parentIdx][j]=sigma*system->rngCPU->rand_normal();
      }
    }
  }

  int n=system->state->lambdaCount+3*system->state->atomCount;
  gpuCheck(cudaMemcpy(system->state->velocityBuffer_d,system->state->velocityBuffer,
    n*sizeof(real_v),cudaMemcpyHostToDevice));
  plugin->velocityInitialized=true;
}

static void drude_release_device_buffers(DrudePlugin *plugin)
{
  if (plugin->springPairs_d) cudaFree(plugin->springPairs_d);
  if (plugin->screenedPairs_d) cudaFree(plugin->screenedPairs_d);
  if (plugin->nbtholePairs_d) cudaFree(plugin->nbtholePairs_d);
  if (plugin->nbtholeActiveIndices_d) cudaFree(plugin->nbtholeActiveIndices_d);
  if (plugin->nbthole14Pairs_d) cudaFree(plugin->nbthole14Pairs_d);
  if (plugin->atomSiteBlock_d) cudaFree(plugin->atomSiteBlock_d);
  if (plugin->pairs_d) cudaFree(plugin->pairs_d);
  if (plugin->preThermostatVelocity_d) cudaFree(plugin->preThermostatVelocity_d);
  if (plugin->thermostatRandom_d) cudaFree(plugin->thermostatRandom_d);
  if (plugin->hardwallCount_d) cudaFree(plugin->hardwallCount_d);
  if (plugin->hardwallTooFarCount_d) cudaFree(plugin->hardwallTooFarCount_d);

  plugin->springPairs_d=NULL;
  plugin->screenedPairs_d=NULL;
  plugin->nbtholePairs_d=NULL;
  plugin->nbtholeActiveIndices_d=NULL;
  plugin->nbtholeActiveCapacity=0;
  plugin->nbtholeActiveCountLast=0;
  plugin->nbtholeUsedActiveListLast=false;
  plugin->nbtholeActiveListValid=false;
  plugin->nbtholeActiveBuildStepLast=-1;
  plugin->nbtholeActiveBuildCutoffLast=(real)0;
  plugin->nbtholeActiveRebuildPeriodLast=1;
  plugin->nbthole14Pairs_d=NULL;
  plugin->atomSiteBlock_d=NULL;
  plugin->pairs_d=NULL;
  plugin->preThermostatVelocity_d=NULL;
  plugin->thermostatRandom_d=NULL;
  plugin->hardwallCount_d=NULL;
  plugin->hardwallTooFarCount_d=NULL;
}

DrudePlugin::DrudePlugin()
{
  enabled=false;

  springPairCount=0;
  springPairs_d=NULL;
  screenedPairCount=0;
  screenedPairs_d=NULL;
  nbtholePairCount=0;
  nbtholePairs_d=NULL;
  nbtholeActiveIndices_d=NULL;
  nbtholeActiveCapacity=0;
  nbtholeActiveCountLast=0;
  nbtholeUsedActiveListLast=false;
  nbtholeActiveListValid=false;
  nbtholeActiveBuildStepLast=-1;
  nbtholeActiveBuildCutoffLast=(real)0;
  nbtholeActiveRebuildPeriodLast=1;
  nbthole14PairCount=0;
  nbthole14Pairs_d=NULL;
  nbtholeCutoff=(real)(5.0*ANGSTROM);
  defaultDrudeMass=(real)0.4;
  defaultThole=(real)1.3;
  explicitMasses.clear();
  explicitPolarizabilities.clear();
  explicitTholes.clear();
  atomSiteBlock_tmp.clear();
  atomSiteBlock_d=NULL;

  pairCount=0;
  pairs_d=NULL;
  preThermostatVelocity_d=NULL;
  thermostatRandom_d=NULL;
  hardwallCount_d=NULL;
  hardwallTooFarCount_d=NULL;

  massTransferApplied=false;
  velocityInitialized=false;
  hardwallCountLast=0;
  hardwallCountTotal=0;
  temperatureCOMLast=0;
  temperatureRelLast=0;
  maxPairDistanceLast=0;
}

DrudePlugin::~DrudePlugin()
{
  clear();
}

bool DrudePlugin::is_active(System *system) const
{
  if (!enabled) return false;
  if (!system || !system->run) return false;
  if (!system->run->calcTermFlag[eedrude]) return false;
  if (pairCount<=0) return false;
  return true;
}

void DrudePlugin::clear()
{
  springPairs_tmp.clear();
  screenedPairs_tmp.clear();
  nbtholePairs_tmp.clear();
  nbthole14Pairs_tmp.clear();
  pairs_tmp.clear();
  atomSiteBlock_tmp.clear();

  springPairCount=0;
  screenedPairCount=0;
  nbtholePairCount=0;
  nbthole14PairCount=0;
  pairCount=0;

  drude_release_device_buffers(this);

  enabled=false;
  massTransferApplied=false;
  velocityInitialized=false;
  hardwallCountLast=0;
  hardwallCountTotal=0;
  temperatureCOMLast=0;
  temperatureRelLast=0;
  maxPairDistanceLast=0;
  nbtholeCutoff=(real)(5.0*ANGSTROM);
  defaultDrudeMass=(real)0.4;
  defaultThole=(real)1.3;
  explicitMasses.clear();
  explicitPolarizabilities.clear();
  explicitTholes.clear();
}

void DrudePlugin::initialize(System *system)
{
  drude_release_device_buffers(this);

  springPairCount=springPairs_tmp.size();
  screenedPairCount=screenedPairs_tmp.size();
  nbtholePairCount=nbtholePairs_tmp.size();
  nbthole14PairCount=nbthole14Pairs_tmp.size();
  massTransferApplied=false;
  velocityInitialized=false;
  hardwallCountLast=0;
  hardwallCountTotal=0;
  temperatureCOMLast=0;
  temperatureRelLast=0;
  maxPairDistanceLast=0;

  if (springPairCount>0) {
    gpuCheck(cudaMalloc(&springPairs_d,springPairCount*sizeof(struct DrudeSpringPairPotential)));
    gpuCheck(cudaMemcpy(springPairs_d,springPairs_tmp.data(),
      springPairCount*sizeof(struct DrudeSpringPairPotential),cudaMemcpyHostToDevice));
  }
  if (screenedPairCount>0) {
    gpuCheck(cudaMalloc(&screenedPairs_d,screenedPairCount*sizeof(struct DrudeScreenedPairPotential)));
    gpuCheck(cudaMemcpy(screenedPairs_d,screenedPairs_tmp.data(),
      screenedPairCount*sizeof(struct DrudeScreenedPairPotential),cudaMemcpyHostToDevice));
  }
  if (nbtholePairCount>0) {
    gpuCheck(cudaMalloc(&nbtholePairs_d,nbtholePairCount*sizeof(struct DrudeNBTholePairPotential)));
    gpuCheck(cudaMemcpy(nbtholePairs_d,nbtholePairs_tmp.data(),
      nbtholePairCount*sizeof(struct DrudeNBTholePairPotential),cudaMemcpyHostToDevice));
    gpuCheck(cudaMalloc(&nbtholeActiveIndices_d,nbtholePairCount*sizeof(int)));
    nbtholeActiveCapacity=nbtholePairCount;
  }
  if (nbthole14PairCount>0) {
    gpuCheck(cudaMalloc(&nbthole14Pairs_d,nbthole14PairCount*sizeof(struct DrudeNBTholePairPotential)));
    gpuCheck(cudaMemcpy(nbthole14Pairs_d,nbthole14Pairs_tmp.data(),
      nbthole14PairCount*sizeof(struct DrudeNBTholePairPotential),cudaMemcpyHostToDevice));
  }

  drude_build_pair_list(this,system);
  pairCount=pairs_tmp.size();
  drude_validate_screened_pairs(system,this);
  drude_validate_nbthole_pairs(system,this);
  drude_validate_no_holonomic_overlap(system,this);
  drude_build_atom_siteblock_map(system,this);
  if (!atomSiteBlock_tmp.empty()) {
    gpuCheck(cudaMalloc(&atomSiteBlock_d,atomSiteBlock_tmp.size()*sizeof(int)));
    gpuCheck(cudaMemcpy(atomSiteBlock_d,atomSiteBlock_tmp.data(),
      atomSiteBlock_tmp.size()*sizeof(int),cudaMemcpyHostToDevice));
  }
  if (pairCount>0) {
    gpuCheck(cudaMalloc(&pairs_d,pairCount*sizeof(int2)));
    gpuCheck(cudaMemcpy(pairs_d,pairs_tmp.data(),pairCount*sizeof(int2),cudaMemcpyHostToDevice));
    gpuCheck(cudaMalloc(&preThermostatVelocity_d,6*pairCount*sizeof(real_v)));
    gpuCheck(cudaMalloc(&thermostatRandom_d,6*pairCount*sizeof(real)));
    gpuCheck(cudaMalloc(&hardwallCount_d,sizeof(int)));
    gpuCheck(cudaMalloc(&hardwallTooFarCount_d,sizeof(int)));
    gpuCheck(cudaMemset(hardwallCount_d,0,sizeof(int)));
    gpuCheck(cudaMemset(hardwallTooFarCount_d,0,sizeof(int)));
  }

  drude_apply_mass_transfer(system,this);
  drude_initialize_pair_velocities(system,this);
}

void parse_drude(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  io_nexta(line,token);
  if (!system->drudePlugin) {
    system->drudePlugin=new DrudePlugin();
  }

  if (strcmp(token,"help")==0) {
    fprintf(stdout,"?drude help> Print drude plugin commands\n");
    fprintf(stdout,"?drude enable [0|1]> Enable or disable drude plugin features\n");
    fprintf(stdout,"?drude clear> Remove all configured drude terms\n");
    fprintf(stdout,"?drude spring [drudeAtomIdx] [parentAtomIdx] [kCharmm] [r0A]> Add one drude-parent harmonic pair using CHARMM Kb units\n");
    fprintf(stdout,"?drude anisotropy [drudeAtomIdx] [axis1Idx] [axis3Idx] [axis4Idx] [aniso12] [aniso34]> Add OpenMM-style anisotropy to an existing spring\n");
    fprintf(stdout,"?drude screenedpair [d1] [p1] [d2] [p2] [screeningScale] [energyScale]> Add one Thole-screened dipole pair\n");
    fprintf(stdout,"?drude nbtholepair [a1] [a2] [screeningScale] [energyScale]> Add one NBTHOLE correction pair (non-excluded)\n");
    fprintf(stdout,"?drude nbthole14 [a1] [a2] [screeningScale] [energyScale]> Add one NBTHOLE correction pair for 1-4 excluded interactions\n");
    fprintf(stdout,"?drude nbtholecutoff [distanceA]> Set NBTHOLE nonbonded correction cutoff in Angstrom\n");
    fprintf(stdout,"?drude defaultmass [massInAMU]> Set default Drude particle mass used for massless Drude sites\n");
    fprintf(stdout,"?drude mass [drudeAtomIdx] [massInAMU]> Set per-atom Drude mass override\n");
    fprintf(stdout,"?drude defaultthole [thole]> Set default per-atom Thole value for autobuild\n");
    fprintf(stdout,"?drude polarizability [atomIdx] [alphaA3]> Set per-atom Drude polarizability for autobuild\n");
    fprintf(stdout,"?drude thole [atomIdx] [thole]> Set per-atom Thole value for autobuild\n");
    fprintf(stdout,"?drude autobuild [all|springs|screened|nbthole]> Auto-build Drude pairs from loaded PSF+PRM data\n");
    fprintf(stdout,"?drude print> Print configured drude counts and diagnostics\n");
  } else if (strcmp(token,"enable")==0) {
    system->drudePlugin->enabled=io_nextb(line);
  } else if (strcmp(token,"clear")==0) {
    system->drudePlugin->clear();
  } else if (strcmp(token,"spring")==0) {
    struct DrudeSpringPairPotential pp;
    pp.idx[0]=io_nexti(line);
    pp.idx[1]=io_nexti(line);
    drude_spring_clear_anisotropy(&pp);
    pp.k=drude_charmm_spring_k_to_internal(io_nextf(line));
    pp.r0=ANGSTROM*io_nextf(line);
    system->drudePlugin->springPairs_tmp.push_back(pp);
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"anisotropy")==0) {
    int drudeIdx=io_nexti(line);
    int axis1Idx=io_nexti(line);
    int axis3Idx=io_nexti(line);
    int axis4Idx=io_nexti(line);
    real aniso12=io_nextf(line);
    real aniso34=io_nextf(line);
    int springEntry=-1;
    for (size_t i=0; i<system->drudePlugin->springPairs_tmp.size(); i++) {
      if (system->drudePlugin->springPairs_tmp[i].idx[0]==drudeIdx) {
        if (springEntry>=0) {
          fatal(__FILE__,__LINE__,
            "drude anisotropy found multiple spring entries for drude atom %d\n",drudeIdx);
        }
        springEntry=(int)i;
      }
    }
    if (springEntry<0) {
      fatal(__FILE__,__LINE__,
        "drude anisotropy requires an existing spring for drude atom %d\n",drudeIdx);
    }
    DrudeSpringPairPotential &pp=system->drudePlugin->springPairs_tmp[springEntry];
    pp.anisoIdx[0]=axis1Idx;
    pp.anisoIdx[1]=axis3Idx;
    pp.anisoIdx[2]=axis4Idx;
    pp.aniso12=aniso12;
    pp.aniso34=aniso34;
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"screenedpair")==0) {
    struct DrudeScreenedPairPotential pp;
    pp.idx[0]=io_nexti(line);
    pp.idx[1]=io_nexti(line);
    pp.idx[2]=io_nexti(line);
    pp.idx[3]=io_nexti(line);
    pp.screeningScale=io_nextf(line);
    pp.energyScale=io_nextf(line);
    system->drudePlugin->screenedPairs_tmp.push_back(pp);
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"nbtholepair")==0 || strcmp(token,"nbthole")==0) {
    struct DrudeNBTholePairPotential pp;
    pp.idx[0]=io_nexti(line);
    pp.idx[1]=io_nexti(line);
    pp.screeningScale=io_nextf(line);
    pp.energyScale=io_nextf(line);
    system->drudePlugin->nbtholePairs_tmp.push_back(pp);
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"nbthole14")==0 || strcmp(token,"nbthole14pair")==0) {
    struct DrudeNBTholePairPotential pp;
    pp.idx[0]=io_nexti(line);
    pp.idx[1]=io_nexti(line);
    pp.screeningScale=io_nextf(line);
    pp.energyScale=io_nextf(line);
    system->drudePlugin->nbthole14Pairs_tmp.push_back(pp);
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"nbtholecutoff")==0) {
    system->drudePlugin->nbtholeCutoff=io_nextf(line)*ANGSTROM;
    if (system->drudePlugin->nbtholeCutoff<(real)0) {
      fatal(__FILE__,__LINE__,"drude nbtholecutoff must be >= 0 Angstrom\n");
    }
  } else if (strcmp(token,"defaultmass")==0) {
    real mass=io_nextf(line);
    if (!isfinite((double)mass) || mass<=(real)0) {
      fatal(__FILE__,__LINE__,"drude defaultmass must be finite and > 0 (found %g)\n",(double)mass);
    }
    system->drudePlugin->defaultDrudeMass=mass;
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"mass")==0) {
    int drudeIdx=io_nexti(line);
    real mass=io_nextf(line);
    if (drudeIdx<0) {
      fatal(__FILE__,__LINE__,"drude mass requires non-negative atom index (found %d)\n",drudeIdx);
    }
    if (!isfinite((double)mass) || mass<=(real)0) {
      fatal(__FILE__,__LINE__,"drude mass must be finite and > 0 (found %g)\n",(double)mass);
    }
    system->drudePlugin->explicitMasses[drudeIdx]=mass;
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"defaultthole")==0) {
    real thole=io_nextf(line);
    if (!isfinite((double)thole) || thole<(real)0) {
      fatal(__FILE__,__LINE__,"drude defaultthole must be finite and >= 0 (found %g)\n",(double)thole);
    }
    system->drudePlugin->defaultThole=thole;
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"polarizability")==0 || strcmp(token,"alpha")==0) {
    int atomIdx=io_nexti(line);
    real alpha=io_nextf(line)*ANGSTROM*ANGSTROM*ANGSTROM;
    if (atomIdx<0) {
      fatal(__FILE__,__LINE__,"drude polarizability requires non-negative atom index (found %d)\n",atomIdx);
    }
    if (!isfinite((double)alpha) || alpha<=(real)0) {
      fatal(__FILE__,__LINE__,"drude polarizability must be finite and > 0 Angstrom^3\n");
    }
    system->drudePlugin->explicitPolarizabilities[atomIdx]=alpha;
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"thole")==0) {
    int atomIdx=io_nexti(line);
    real thole=io_nextf(line);
    if (atomIdx<0) {
      fatal(__FILE__,__LINE__,"drude thole requires non-negative atom index (found %d)\n",atomIdx);
    }
    if (!isfinite((double)thole) || thole<(real)0) {
      fatal(__FILE__,__LINE__,"drude thole must be finite and >= 0 (found %g)\n",(double)thole);
    }
    system->drudePlugin->explicitTholes[atomIdx]=thole;
    system->drudePlugin->enabled=true;
  } else if (strcmp(token,"autobuild")==0) {
    drude_autobuild(line,system);
  } else if (strcmp(token,"print")==0) {
    fprintf(stdout,"DRUDE PRINT> enabled=%d\n",(int)system->drudePlugin->enabled);
    fprintf(stdout,"DRUDE PRINT> springPairCount=%d screenedPairCount=%d nbtholePairCount=%d nbthole14PairCount=%d pairCount=%d anisotropicSpringCount=%d\n",
      (int)system->drudePlugin->springPairs_tmp.size(),
      (int)system->drudePlugin->screenedPairs_tmp.size(),
      (int)system->drudePlugin->nbtholePairs_tmp.size(),
      (int)system->drudePlugin->nbthole14Pairs_tmp.size(),
      (int)system->drudePlugin->pairs_tmp.size(),
      drude_count_anisotropic_springs(system->drudePlugin));
    fprintf(stdout,"DRUDE PRINT> nbtholeCutoffA=%g\n",
      (double)(system->drudePlugin->nbtholeCutoff/ANGSTROM));
    fprintf(stdout,"DRUDE PRINT> defaultDrudeMass=%g explicitMassOverrides=%d\n",
      (double)system->drudePlugin->defaultDrudeMass,
      (int)system->drudePlugin->explicitMasses.size());
    fprintf(stdout,"DRUDE PRINT> defaultThole=%g polarizabilityOverrides=%d tholeOverrides=%d\n",
      (double)system->drudePlugin->defaultThole,
      (int)system->drudePlugin->explicitPolarizabilities.size(),
      (int)system->drudePlugin->explicitTholes.size());
    fprintf(stdout,"DRUDE PRINT> massTransferApplied=%d velocityInitialized=%d\n",
      (int)system->drudePlugin->massTransferApplied,
      (int)system->drudePlugin->velocityInitialized);
    fprintf(stdout,"DRUDE PRINT> last Tcom=%g Trel=%g hardwallLast=%d hardwallTotal=%lld\n",
      (double)system->drudePlugin->temperatureCOMLast,
      (double)system->drudePlugin->temperatureRelLast,
      system->drudePlugin->hardwallCountLast,
      system->drudePlugin->hardwallCountTotal);
    fprintf(stdout,"DRUDE PRINT> last maxPairDistance=%g\n",
      (double)(system->drudePlugin->maxPairDistanceLast/ANGSTROM));
    fprintf(stdout,"DRUDE PRINT> last nbtholeUsedActiveList=%d nbtholeActiveCount=%d\n",
      (int)system->drudePlugin->nbtholeUsedActiveListLast,
      system->drudePlugin->nbtholeActiveCountLast);
    fprintf(stdout,"DRUDE PRINT> last nbtholeActiveListValid=%d nbtholeActiveBuildStep=%ld nbtholeActiveBuildCutoffA=%g nbtholeActiveRebuildPeriod=%d\n",
      (int)system->drudePlugin->nbtholeActiveListValid,
      system->drudePlugin->nbtholeActiveBuildStepLast,
      (double)(system->drudePlugin->nbtholeActiveBuildCutoffLast/ANGSTROM),
      system->drudePlugin->nbtholeActiveRebuildPeriodLast);
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized drude token: %s\n",token);
  }
}

void blade_init_drude(System *system)
{
  system+=omp_get_thread_num();
  if (system->drudePlugin) delete(system->drudePlugin);
  system->drudePlugin=new DrudePlugin();
}

void blade_dest_drude(System *system)
{
  system+=omp_get_thread_num();
  if (system->drudePlugin) delete(system->drudePlugin);
  system->drudePlugin=NULL;
}

void blade_clear_drude(System *system)
{
  system+=omp_get_thread_num();
  if (!system->drudePlugin) return;
  system->drudePlugin->clear();
}

void blade_set_drude_enabled(System *system,int enabled)
{
  system+=omp_get_thread_num();
  if (!system->drudePlugin) system->drudePlugin=new DrudePlugin();
  system->drudePlugin->enabled=(enabled!=0);
}

void blade_add_drude_spring(System *system,int drudeIdx,int parentIdx,double k,double r0)
{
  system+=omp_get_thread_num();
  if (!system->drudePlugin) system->drudePlugin=new DrudePlugin();

  struct DrudeSpringPairPotential pp;
  pp.idx[0]=drudeIdx;
  pp.idx[1]=parentIdx;
  drude_spring_clear_anisotropy(&pp);
  pp.k=drude_charmm_spring_k_to_internal((real)k);
  pp.r0=ANGSTROM*(real)r0;
  system->drudePlugin->springPairs_tmp.push_back(pp);
  system->drudePlugin->enabled=true;
}

void blade_add_drude_screened_pair(System *system,
  int drude1,int parent1,int drude2,int parent2,double screeningScale,double energyScale)
{
  system+=omp_get_thread_num();
  if (!system->drudePlugin) system->drudePlugin=new DrudePlugin();

  struct DrudeScreenedPairPotential pp;
  pp.idx[0]=drude1;
  pp.idx[1]=parent1;
  pp.idx[2]=drude2;
  pp.idx[3]=parent2;
  pp.screeningScale=(real)screeningScale;
  pp.energyScale=(real)energyScale;
  system->drudePlugin->screenedPairs_tmp.push_back(pp);
  system->drudePlugin->enabled=true;
}
