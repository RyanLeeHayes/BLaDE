#include <cuda_runtime.h>
#include <math.h>

#include "restrain/restrain.h"
#include "main/defines.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"

#include "main/real3.h"
#include "main/gpu_check.h"

__device__ void cross_covariance(int RmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms)
{
  // set up Bjk = sum_{i in f} wi * xi_jth column * x0i_kth column
  // returns 3x3
}

__device__ void quaternion_matrix(int RmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms,real *B) // this is 3x3 matrix of 3x3 elements i.e. the cross-covariance
{
  // input is 
  // set up 4x4 F from B
}

__device__ void quaternion_to_rotation(int RmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms,real *q) // this is 1x4 array of quaternion
{
  // q -> R
}

__device__ void fit(int RmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms)
{
  // calculate optimal R
}

__device__ void potential(int RmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms)
{
  // flat bottom harmonic
}


template <bool flagBox,typename box_type>
__global__ void getforce_rmsdRestraint_kernel(int rmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real_e *energy)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;

  real lEnergy=0;
  //
  // initialize fcoords, rcoords, k, d0, wfs, wrs, Wf, Wr
  //
  RmsdRestPotentialDev rr;
  

  if (i<rmsdRestCount) {
    rr=rmsdRests[i];
    // calcs 
  }

  // Energy, if requested
  if (energy) {
    //lEnergy*=l;
    //real_sum_reduce(lEnergy,sEnergy,energy);
  }
}

template <bool flagBox,typename box_type>
void getforce_rmsdRestT(System *system,box_type box,bool calcEnergy)
{
  Potential *p=system->potential;
  State *s=system->state;
  Run *r=system->run;
  int N;
  int shMem=0;
  real_e *pEnergy=NULL;

  if (r->calcTermFlag[eemmfp]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eemmfp;
  }

  N=p->rmsdRestCount;
  if (N>0) getforce_rmsdRestraint_kernel <flagBox> <<<(N+BLBO-1)/BLBO,BLBO,shMem,r->biaspotStream>>>(N,p->rmsdRests_d,p->rmsdRestAtoms_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy); gpuCheck(cudaGetLastError());
}

void getforce_rmsdRest(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_rmsdRestT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_rmsdRestT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
