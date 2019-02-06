#ifndef MD_RNA_IONS_LEAPPARMS_H
#define MD_RNA_IONS_LEAPPARMS_H

#include "defines.h"

typedef struct struct_leapparms
{
  real dt;
  real gamma;
  real kT;
  real kTeff;
  real mDiff;
  real SigmaVsVs;
  real XsVsCoeff;
  real CondSigmaXsXs;
  real SigmaVrVr;
  real XrVrCoeff;
  real CondSigmaXrXr;
  real vhalf_vscale;
  real vhalf_ascale;
  real vhalf_Vsscale;
  real v_vscale;
  real v_ascale;
  real v_Vsscale;
  real x_vscale;
  real x_Xrscale;
} struct_leapparms;

struct_leapparms* alloc_leapparms(real dt,real gamma,real kT);

void free_leapparms(struct_leapparms* leapparms) ;

#endif

