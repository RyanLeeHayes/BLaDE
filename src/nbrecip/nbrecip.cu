#include <cuda_runtime.h>
#include <math.h>

#include "nbrecip/nbrecip.h"
#include "main/defines.h"
#include "system/system.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

#include "main/real3.h"



// getforce_ewaldself_kernel<<<(N+BLNB-1)/BLNB,BLNB,shMem,p->bondedStream>>>(N,p->charge_d,prefactor,m->atomBlock_d,m->lambda_d,m->lambdaForce_d,pEnergy);
__global__ void getforce_ewaldself_kernel(int atomCount,real *charge,real prefactor,int *atomBlock,real *lambda,real *lambdaForce,real *energy)
{
// NYI - maybe energy should be a double
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  real q;
  real lEnergy=0;
  extern __shared__ real sEnergy[];
  int b;
  real l=1;
  
  if (i<atomCount) {
    q=charge[i];
    
    // Scaling
    b=atomBlock[i];
    if (b) {
      l=lambda[b];
    }

    // interaction
    if (b || energy) {
      lEnergy=prefactor*q*q;
    }

    // Lambda force
    if (b) {
      realAtomicAdd(&lambdaForce[b],2*l*lEnergy);
    }
  }

  // Energy, if requested
  if (energy) {
    lEnergy*=l*l;
    real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

void getforce_ewaldself(System *system,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Msld *m=system->msld;
  int N=p->atomCount;
  int shMem=0;
  real *pEnergy=NULL;
  real prefactor=system->run->betaEwald*(kELECTRIC/sqrt(M_PI));

  if (calcEnergy) {
    shMem=BLNB*sizeof(real)/32;
    pEnergy=s->energy_d+eenbrecip;
  }

  getforce_ewaldself_kernel<<<(N+BLNB-1)/BLNB,BLNB,shMem,p->nbrecipStream>>>(N,p->charge_d,prefactor,m->atomBlock_d,m->lambda_d,m->lambdaForce_d,pEnergy);
}



// getforce_angle_kernel<<<(N+BLBO-1)/BLBO,BLBO,shMem,p->bondedStream>>>(N,p->angles_d,(real3*)s->position_d,(real3*)s->force_d,s->orthBox,m->lambda_d,m->lambdaForce_d,pEnergy);
// __global__ void getforce_angle_kernel(int angleCount,struct AnglePotential *angles,real3 *position,real3 *force,real3 box,real *lambda,real *lambdaForce,real *energy)
