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


__device__ void block_sum(real *input,int dim,real* shared) {
  int lane=threadIdx.x&31;    // cheaper than % 32
  int warp=threadIdx.x>>5;    // 5 bit shift to right, equals int divisional by 2^5
  int nwarps=blockDim.x>>5;   // 256 / 32 = 8 total warps

  for (int i=0; i<dim; i++) {
    real local=input[i];

    local+=__shfl_down_sync(0xFFFFFFFF,local,1);
    local+=__shfl_down_sync(0xFFFFFFFF,local,2);
    local+=__shfl_down_sync(0xFFFFFFFF,local,4);
    local+=__shfl_down_sync(0xFFFFFFFF,local,8);
    local+=__shfl_down_sync(0xFFFFFFFF,local,16);

    // each lane 0 within each warp now holds partial warp sum
    if (lane==0) { shared[warp]=local; }
    __syncthreads();

    local=0;
    for (int w=0; w<nwarps; w++) {
      local+=shared[w];
    }
    input[i]=local;
    __syncthreads();
  }
}

__device__ void calculate_quaternion_matrix_from_crosscovariance(real* B[3][3]) {
  // F is symmetric 4x4 matrix of linear summation of B elements
  //
  // Inputs:
  // B: symmetric 3x3 matrix of cross-covariance
  //
  // Procedure:
  // Just sum and assign each entry. Check writeup for full 4x4 matrix definition
  //
  // Outputs:
  // F: symmetric 4x4 matrix of quaternion coefficients

  // initialize eigenvectors as identity matrix
  //real eigenvecs[4][4];
  //for (int i=0; i<eigenvectors.size(); i++) {
  // for (int j=0; j<eigenvectors.size(); j++) {
  //    if (i==j) { eigenvectors[i][j] = 1; }
  //    else { eigenvectors[i][j] = 0; }
  //  }
  //}
  //return F;
}

__device__ void calculate_quaternion_from_quaternion_matrix(real F[4][4]) {
  // F*eigenvecs = mu*eigenvecs
  //
  // Inputs:
  // F: 4x4 quaternion coefficient matrix
  //
  // Procedure:
  // Jacobi iteration to find eigenvalues mu, then find eigenvectors eigenvecs
  //
  // Output:
  // eigenvecs: 4x4 of eigenvectors (q = eigenvecs[0], what to calcualate optimal R from)
  // mu: eigenvalues
  //
  //return std::make_pair(mu, eigenvecs);
}


__device__ void quaternion_to_rotation(real q[4]) // this is 1x4 array of quaternion
{
  // R has a quadratic summation relationship to q with pre-defined coefficients
  //
  // Inputs:
  // q: 4x1 array, optimal quaternion quaternion
  // 
  // Procedure:
  // Use write-up coefficients
  //
  // Outputs:
  // R: 3x3 optimal rotation matrix
}

__device__ void calculate_optimal_rotation(RmsdRestAtom* frameStart, int fCount)
{
  // calculate optimal R
}

__device__ void potential(int RmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms)
{
  // flat bottom harmonic
}


template <bool flagBox,typename box_type>
__global__ void getforce_rmsdRestraint_kernel(int rmsdRestCount,RmsdRestPotentialDev *rmsdRests,RmsdRestAtom *rmsdRestAtoms,real3 *position,real3_f *force,box_type box,real *lambda,real_f *lambdaForce,real_e *energy) {
  real lEnergy=0;
  __shared__ real sred[32];
  extern __shared__ real sEnergy[];

  RmsdRestPotentialDev rmsdRest=rmsdRests[blockIdx.x];

  // STAGE 1: CENTROID
  // wrapping has been a pain to implement. the following approachc computes the centroid
  // in two passes: anchor frame[0] -> approx. centroid (some atoms could be out of box)
  // -> real centroid

  real3 anchor=position[rmsdRestAtoms[rmsdRest.fStartIdx].idx]; 
  for (int pass=0; pass<2; pass++) {
    real cpart[3]={0};
    // each thread gets frame atom index (threadIdx.x + n*256)
    for (int k=threadIdx.x; k<rmsdRest.fCount; k+=blockDim.x) {
      RmsdRestAtom fatom=rmsdRestAtoms[rmsdRest.fStartIdx+k];
      real3 xp=real3_add(anchor, real3_subpbc<flagBox>(position[fatom.idx], anchor, box));
      cpart[0]+=fatom.w*xp.x;
      cpart[1]+=fatom.w*xp.y;
      cpart[2]+=fatom.w*xp.z;
    }
    block_sum(cpart,3,sred);
    anchor.x=cpart[0]/rmsdRest.Wf;
    anchor.y=cpart[1]/rmsdRest.Wf;
    anchor.z=cpart[2]/rmsdRest.Wf;
  }
  real3 c=anchor;
  real carr[3]={c.x,c.y,c.z};
  
  // STAGE 2: CROSS-COVARIANCE 
  real Bpart[9]={0};
  real B[9]={0};

  for (int k=threadIdx.x; k<rmsdRest.fCount; k+=blockDim.x) {
    RmsdRestAtom fatom=rmsdRestAtoms[rmsdRest.fStartIdx+k];
    real3 xp=real3_subpbc<flagBox>(position[fatom.idx],c,box); // centered coords
    real3 xp0=fatom.refPos;

    real x[3]={xp.x,xp.y,xp.z};
    real x0[3]={xp0.x,xp0.y,xp0.z};
    for (int a=0; a<3; a++) {
      for (int b=0; b<3; b++) {
        Bpart[3*a+b]+=fatom.w*x[a]*x0[b];
      }
    }
  }
  block_sum(Bpart,9,sred);

  for (int j=0; j<9; j++) {
    B[j]=Bpart[j];
  }


  if (threadIdx.x==0) {
  printf("RMSD %d c  %.6f %.6f %.6f  Wf %.4f\n",blockIdx.x,carr[0],carr[1],carr[2],rmsdRest.Wf);
  printf("RMSD %d B  %.5f %.5f %.5f  %.5f %.5f %.5f  %.5f %.5f %.5f\n",blockIdx.x,
         B[0],B[1],B[2],B[3],B[4],B[5],B[6],B[7],B[8]);
}


  // STOPPAGE


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

  if (r->calcTermFlag[eermsd]==false) return;

  if (calcEnergy) {
    shMem=BLBO*sizeof(real)/32;
    pEnergy=s->energy_d+eermsd;
  }

  N=p->rmsdRestCount;
  if (N>0) getforce_rmsdRestraint_kernel <flagBox> <<<N,BLBO,shMem,r->biaspotStream>>>(N,p->rmsdRests_d,p->rmsdRestAtoms_d,(real3*)s->position_fd,(real3_f*)s->force_d,box,s->lambda_fd,s->lambdaForce_d,pEnergy); gpuCheck(cudaGetLastError());
}

void getforce_rmsdRest(System *system,bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_rmsdRestT<true>(system,system->state->tricBox_f,calcEnergy);
  } else {
    getforce_rmsdRestT<false>(system,system->state->orthBox_f,calcEnergy);
  }
}
