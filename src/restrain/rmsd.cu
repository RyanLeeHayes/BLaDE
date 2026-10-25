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


__device__ void block_sum(real* input, int dim, real* shared) {
  int lane = threadIdx.x&31;
  int warp = threadIdx.x>>5;
  int nwarps = blockDim.x>>5; 

  for (int i = 0; i < dim; i++) {
    real local = input[i];
    local += __shfl_down_sync(0xFFFFFFFF,local,1);
    local += __shfl_down_sync(0xFFFFFFFF,local,2);
    local += __shfl_down_sync(0xFFFFFFFF,local,4);
    local += __shfl_down_sync(0xFFFFFFFF,local,8);
    local += __shfl_down_sync(0xFFFFFFFF,local,16);

    if (lane == 0) { shared[warp] = local; }
    __syncthreads();

    local = 0;
    for (int w = 0; w < nwarps; w++) {
      local += shared[w];
    }

    input[i] = local;
    __syncthreads();
  }
}


__device__ real* quaternion_matrix(const real* B, real* F) {
  F[0] = B[0] + B[4] + B[8];  // B11+B22+B33
  F[1] = B[7] - B[5];         // B32-B23
  F[2] = B[2] - B[6];         // B13-B31
  F[3] = B[3] - B[1];         // B21-B12

  F[4] = F[1];
  F[5] = B[0] - B[4] - B[8];  // B11-B22-B33
  F[6] = B[1] + B[3];         // B12+B21
  F[7] = B[2] + B[6];         // B13+B31

  F[8] = F[2];
  F[9] = F[6];
  F[10] = -B[0] + B[4] - B[8]; // -B11+B22-B33
  F[11] = B[5] + B[7];         // B23+B32

  F[12] = F[3];
  F[13] = F[7];
  F[14] = F[11];
  F[15] = -B[0] - B[4] + B[8];  // -B11-B22+B33

  return F;
}

__device__ void jacobi_eigensolver(real* F, real* evals, real* evecs, int dim, int max_iter=100, double tol=1e-12) {
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      evecs[dim*i + j] = static_cast<double>(i == j);
    }
  }

  for (int iter = 0; iter < max_iter; iter++) {
    double max_off_diag = 0.0;
    int p = 0;
    int q = 0;

    for (int i = 0; i < dim; i++) {
      for (int j = i+1; j < dim; j++) {
        if (fabs(F[dim*i +j]) > max_off_diag) {
          max_off_diag = fabs(F[dim*i + j]); 
          p = i;
          q = j;
        }
      }
    }

    if (max_off_diag < tol) { break; }

    double fpp = F[dim*p + p];
    double fqq = F[dim*q + q];
    double fpq = F[dim*p + q];

    double theta = 0.5*atan2(2.0*fpq, fqq - fpp);
    double c = cos(theta);
    double s = sin(theta);

    F[dim*p + p] = c*c*fpp - 2*s*c*fpq + s*s*fqq;
    F[dim*q + q] = s*s*fpp + 2*s*c*fpq + c*c*fqq;
    F[dim*p + q] = 0.0;
    F[dim*q + p] = 0.0;

    for (int i = 0; i < dim; i++) {
      if (i != p && i != q) {
        double fip = F[dim*i + p];
        double fiq = F[dim*i + q];
        F[dim*i + p] = c*fip - s*fiq;
        F[dim*p + i] = F[dim*i + p];
        F[dim*i + q] = s*fip + c*fiq;
        F[dim*q + i] = F[dim*i + q];
      }
    }

    // extract eigenvectors
    for (int i = 0; i < dim; i++) {
      double vip = evecs[dim*i + p];
      double viq = evecs[dim*i + q];
      evecs[dim*i + p] = c*vip - s*viq;
      evecs[dim*i + q] = s*vip + c*viq;
    }
  }

  // accumulate eigenvalues from diagonal of F
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      if (i == j) { evals[i] = F[dim*i + j]; }
    }
  }
}


__device__ void quaternion_to_rotation(real* q, real* R) {
  real q0 = q[0];
  real q1 = q[1];
  real q2 = q[2];
  real q3 = q[3];

  R[0] = q0*q0 + q1*q1 - q2*q2 -q3*q3;
  R[1] = 2*(q1*q2 - q0*q3);
  R[2] = 2*(q1*q3 + q0*q2);

  R[3] = 2*(q1*q2 + q0*q3);
  R[4] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
  R[5] = 2*(q2*q3 - q0*q1);

  R[6] = 2*(q1*q3 - q0*q2);
  R[7] = 2*(q2*q3 + q0*q1);
  R[8] = q0*q0 - q1*q1 - q2*q2 + q3*q3;
}

__device__ void potential(int RmsdRestCount, RmsdRestPotentialDev* rmsdRests, RmsdRestAtom* rmsdRestAtoms) {
  // flat bottom harmonic
}


template <bool flagBox, typename box_type>
__global__ void getforce_rmsdRestraint_kernel(int rmsdRestCount, RmsdRestPotentialDev* rmsdRests, RmsdRestAtom* rmsdRestAtoms, real3* position, real3_f* force, box_type box, real* lambda, real_f* lambdaForce, real_e* energy) {
  real lEnergy = 0;
  __shared__ real sred[32];
  extern __shared__ real sEnergy[];

  RmsdRestPotentialDev rmsdRest = rmsdRests[blockIdx.x];

  // STAGE 1: CENTROID
  // wrapping has been a pain to implement. the following approachc computes the centroid
  // in two passes: anchor frame[0] -> approx. centroid (some atoms could be out of box)
  // -> real centroid

  real3 anchor = position[rmsdRestAtoms[rmsdRest.fStartIdx].idx]; 
  for (int pass = 0; pass < 2; pass++) {
    real cpart[3] = {0};
    // each thread gets frame atom index (threadIdx.x + n*256)
    for (int k = threadIdx.x; k < rmsdRest.fCount; k += blockDim.x) {
      RmsdRestAtom fatom = rmsdRestAtoms[rmsdRest.fStartIdx+k];
      real3 xp_3 = real3_add(anchor, real3_subpbc<flagBox>(position[fatom.idx], anchor, box));
      cpart[0] += fatom.w*xp_3.x;
      cpart[1] += fatom.w*xp_3.y;
      cpart[2] += fatom.w*xp_3.z;
    }
    block_sum(cpart,3,sred);
    anchor.x = cpart[0]/rmsdRest.Wf;
    anchor.y = cpart[1]/rmsdRest.Wf;
    anchor.z = cpart[2]/rmsdRest.Wf;
  }
  real3 c_3 = anchor;
  real c[3] = {c_3.x, c_3.y, c_3.z};
  
  // STAGE 2: CROSS-COVARIANCE 
  real Bpart[9] = {0};
  real B[9] = {0};

  for (int k = threadIdx.x; k < rmsdRest.fCount; k += blockDim.x) {
    RmsdRestAtom fatom = rmsdRestAtoms[rmsdRest.fStartIdx + k];
    real3 xp_3 = real3_subpbc<flagBox>(position[fatom.idx], c_3, box); // centered coords
    real3 xp0_3 = fatom.refPos;

    real xp[3] = {xp_3.x, xp_3.y, xp_3.z};
    real xp0[3] = {xp0_3.x, xp0_3.y, xp0_3.z};
    for (int a = 0; a < 3; a++) {
      for (int b = 0; b < 3; b++) {
        Bpart[3*a + b] += fatom.w*xp[a]*xp0[b];
      }
    }
  }
  block_sum(Bpart,9,sred);

  for (int j = 0; j < 9; j++) {
    B[j] = Bpart[j];
  }

  // STAGE 3: QUATERNION MATRIX
  real F[16];
  quaternion_matrix(B, F);

  // STAGE 4: QUATERNION
  real evals[4];
  real evecs[16];
  jacobi_eigensolver(F, evals, evecs, 4);

  int max_eval_idx = 0;

  for (int i = 1; i < 4; i++) {
    if (evals[i] > evals[max_eval_idx]) {
      max_eval_idx = i;
    }
  }

  real q[4];
  for (int j = 0; j < 4; j++) { q[j] = evecs[4*j + max_eval_idx]; }

  // STAGE 5: OPTIMAL ROTATION
  real R[9] = {0};
  quaternion_to_rotation(q, R);
            


  // Energy, if requested
  if (energy) {
    //lEnergy*=l;
    //real_sum_reduce(lEnergy,sEnergy,energy);
  }
}


template <bool flagBox, typename box_type>
void getforce_rmsdRestT(System* system, box_type box, bool calcEnergy)
{
  Potential *p = system->potential;
  State* s = system->state;
  Run* r = system->run;
  int N;
  int shMem = 0;
  real_e* pEnergy = NULL;

  if (r->calcTermFlag[eermsd] == false) { return; }

  if (calcEnergy) {
    shMem = BLBO*sizeof(real)/32;
    pEnergy = s->energy_d + eermsd;
  }

  N = p->rmsdRestCount;
  if (N > 0) {
    getforce_rmsdRestraint_kernel <flagBox> <<<N, BLBO, shMem, r->biaspotStream>>>(N, p->rmsdRests_d, p->rmsdRestAtoms_d, (real3*)s->position_fd, (real3_f*)s->force_d, box, s->lambda_fd, s->lambdaForce_d, pEnergy);
  gpuCheck(cudaGetLastError());
  }
}


void getforce_rmsdRest(System* system, bool calcEnergy)
{
  if (system->state->typeBox) {
    getforce_rmsdRestT<true>(system, system->state->tricBox_f, calcEnergy);
  } else {
    getforce_rmsdRestT<false>(system, system->state->orthBox_f, calcEnergy);
  }
}
