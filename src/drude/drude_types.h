#ifndef BLADE_DRUDE_TYPES_H
#define BLADE_DRUDE_TYPES_H

#include "main/defines.h"

struct DrudeSpringPairPotential {
  int idx[2]; // [drude, parent]
  int anisoIdx[3]; // [axis1, axis3, axis4], negative disables anisotropy
  real k;
  real r0;
  real aniso12;
  real aniso34;
};

static inline void drude_spring_clear_anisotropy(struct DrudeSpringPairPotential *pp)
{
  pp->anisoIdx[0]=-1;
  pp->anisoIdx[1]=-1;
  pp->anisoIdx[2]=-1;
  pp->aniso12=(real)1;
  pp->aniso34=(real)1;
}

static inline bool drude_spring_has_anisotropy(const struct DrudeSpringPairPotential *pp)
{
  return (pp->anisoIdx[0]>=0 || (pp->anisoIdx[1]>=0 && pp->anisoIdx[2]>=0));
}

struct DrudeScreenedPairPotential {
  int idx[4]; // [drude1, parent1, drude2, parent2]
  real screeningScale;
  real energyScale;
};

struct DrudeNBTholePairPotential {
  int idx[2]; // [atom1, atom2]
  real screeningScale;
  real energyScale;
};

#endif
