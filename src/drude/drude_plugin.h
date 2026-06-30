#ifndef BLADE_DRUDE_PLUGIN_H
#define BLADE_DRUDE_PLUGIN_H

#include <map>
#include <vector>

#include "main/defines.h"
#include "drude/drude_types.h"

// Forward declaration
class System;

class DrudePlugin {
  public:
  bool enabled;

  int springPairCount;
  std::vector<struct DrudeSpringPairPotential> springPairs_tmp;
  struct DrudeSpringPairPotential *springPairs_d;

  int screenedPairCount;
  std::vector<struct DrudeScreenedPairPotential> screenedPairs_tmp;
  struct DrudeScreenedPairPotential *screenedPairs_d;

  int nbtholePairCount;
  std::vector<struct DrudeNBTholePairPotential> nbtholePairs_tmp;
  struct DrudeNBTholePairPotential *nbtholePairs_d;
  int *nbtholeActiveIndices_d;
  int nbtholeActiveCapacity;
  int nbtholeActiveCountLast;
  bool nbtholeUsedActiveListLast;
  bool nbtholeActiveListValid;
  long int nbtholeActiveBuildStepLast;
  real nbtholeActiveBuildCutoffLast;
  int nbtholeActiveRebuildPeriodLast;

  int nbthole14PairCount;
  std::vector<struct DrudeNBTholePairPotential> nbthole14Pairs_tmp;
  struct DrudeNBTholePairPotential *nbthole14Pairs_d;

  real nbtholeCutoff;
  real defaultDrudeMass;
  real defaultThole;
  std::map<int,real> explicitMasses;
  std::map<int,real> explicitPolarizabilities;
  std::map<int,real> explicitTholes;

  // Per-atom encoded (site<<16|block) values used for MSLD scaling in Drude pair kernels.
  std::vector<int> atomSiteBlock_tmp;
  int *atomSiteBlock_d;

  int pairCount;
  std::vector<int2> pairs_tmp; // [drude, parent]
  int2 *pairs_d;

  real_v *preThermostatVelocity_d; // [pair][drude xyz, parent xyz]
  real *thermostatRandom_d; // [pair][com xyz, rel xyz]
  int *hardwallCount_d;
  int *hardwallTooFarCount_d;

  bool massTransferApplied;
  bool velocityInitialized;
  int hardwallCountLast;
  long long hardwallCountTotal;
  real temperatureCOMLast;
  real temperatureRelLast;
  real maxPairDistanceLast;

  DrudePlugin();
  ~DrudePlugin();

  bool is_active(System *system) const;

  void clear();
  void initialize(System *system);
  void getforce(System *system,bool calcEnergy);
  void getforce_nbthole(System *system,bool calcEnergy);
  void getforce_nbthole14(System *system,bool calcEnergy);
  void pre_thermostat(System *system);
  void post_thermostat(System *system);
  void apply_hardwall(System *system,long int step);
};

void parse_drude(char *line,System *system);
void drude_autobuild(char *line,System *system);

// Library functions
extern "C" {
  void blade_init_drude(System *system);
  void blade_dest_drude(System *system);
  void blade_clear_drude(System *system);
  void blade_set_drude_enabled(System *system,int enabled);
  void blade_add_drude_spring(System *system,int drudeIdx,int parentIdx,double k,double r0);
  void blade_add_drude_screened_pair(System *system,
    int drude1,int parent1,int drude2,int parent2,double screeningScale,double energyScale);
}

#endif
