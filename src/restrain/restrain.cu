#include <cuda_runtime.h>
#include <math.h>

#include "restrain.h"
#include "main/defines.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

#include "main/real3.h"

__global__ void getforce_harm_kernel(int harmCount,struct HarmonicPotential *harms,real3 *position,real3_f *force,real3 box,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  int ii;
  real r2;
  real3 dr;
  HarmonicPotential hp;
  real krnm2;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  real3 xi,x0;
  
  if (i<harmCount) {
    // Geometry
    hp=harms[i];
    ii=hp.idx;
    xi=position[ii];
    x0=hp.r0;
// NOTE #warning "Unprotected division"
    dr=real3_subpbc(xi,x0,box);
    r2=real3_mag2<real>(dr);
    krnm2=hp.k*pow(r2,((real)0.5)*hp.n-1);
    
    if (energy) {
      lEnergy=krnm2*r2;
    }
    at_real3_scaleinc(&force[ii], hp.n*krnm2,dr);
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void getforce_harm(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eebias]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eebias;
  }

  N=p->harmCount;
  if (N>0) getforce_harm_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->harms_d,(real3*)s->position_fd,(real3_f*)s->force_d,s->orthBox_f,pEnergy);
}
