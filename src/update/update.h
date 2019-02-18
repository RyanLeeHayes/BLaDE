#ifndef UPDATE_UPDATE_H
#define UPDATE_UPDATE_H

// See https://aip.scitation.org/doi/abs/10.1063/1.1420460 for barostat
// Nevermind, this one is better:
// dx.doi.org/10.1016/j.cplett.2003.12.039

#include "main/defines.h"

// Forward declarations
class System;

struct LeapParms
{
  real dt;
  real gamma;
  real kT;
  // real mDiff;
  real SigmaVsVs;
  real XsVsCoeff;
  real CondSigmaXsXs;
  real SigmaVrVr;
  real XrVrCoeff;
  real CondSigmaXrXr;
  // real vhalf_vscale;
  // real vhalf_ascale;
  // real vhalf_Vsscale;
  real v_vscale;
  real v_ascale;
  real v_Vsscale;
  real x_vscale;
  real x_Xrscale;
};

class Update {
  public:
  struct LeapParms *leapParms;

  Update()
  {
    leapParms=NULL;
  }

  ~Update()
  {
    if (leapParms) free(leapParms);
  }

  void initialize(System *system);
  void update(int step,System *system);

  void calcABCD(real slambda,real* A,real* B,real* C,real* D);
  struct LeapParms* alloc_leapparms(real dt,real gamma,real T);
};

#endif
