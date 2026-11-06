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
    local += __shfl_down_sync(0xFFFFFFFF, local, 1);
    local += __shfl_down_sync(0xFFFFFFFF, local, 2);
    local += __shfl_down_sync(0xFFFFFFFF, local, 4);
    local += __shfl_down_sync(0xFFFFFFFF, local, 8);
    local += __shfl_down_sync(0xFFFFFFFF, local, 16);

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


__device__ void quaternion_matrix(const real* B, real* F) {
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
  F[15] = -B[0] - B[4] + B[8]; // -B11-B22+B33
}


__device__ void jacobi_eigensolver(real* F, real* evals, real* evecs, const int dim, const int max_iter=100, const double tol=1e-12) {
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
        if (fabs(F[dim*i + j]) > max_off_diag) {
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


__device__ void quaternion_to_rotation(const real* q, real* R) {
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


__device__ real3 rotate_vector(const real* R, const real3* v) {
  real3 rotated;
  rotated.x = R[0]*v->x + R[1]*v->y + R[2]*v->z;
  rotated.y = R[3]*v->x + R[4]*v->y + R[5]*v->z;
  rotated.z = R[6]*v->x + R[7]*v->y + R[8]*v->z;
  return rotated;
}


__device__ void mat3_times_transposed_mat3(const real* B, real* R, real* G) {
  for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        G[3*i + j] = B[3*i + 0]*R[3*j + 0] + B[3*i + 1]*R[3*j + 1] + B[3*i + 2]*R[3*j + 2];
      }
  }
}


__device__ void mat3_inverse(const real* M, real* K) {
  real a = M[0];
  real b = M[1];
  real c = M[2];
  real d = M[3];
  real e = M[4];
  real f = M[5];
  real g = M[6];
  real h = M[7];
  real i = M[8];
  real det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
  real inv = 1/det;

  K[0] = (e*i - f*h)*inv;
  K[1] = (c*h - b*i)*inv;
  K[2] = (b*f - c*e)*inv;
  K[3] = (f*g - d*i)*inv;
  K[4] = (a*i - c*g)*inv;
  K[5] = (c*d - a*f)*inv;
  K[6] = (d*h - e*g)*inv;  
  K[7] = (b*g - a*h)*inv;
  K[8] = (a*e - b*d)*inv;
}


template <bool flagBox, typename box_type>
__global__ void getforce_rmsdRestraint_kernel(const int rmsdRestCount, const RmsdRestPotentialDev* rmsdRests, const RmsdRestAtom* rmsdRestAtoms, const real3* position, real3_f* force, const box_type box, real* lambda, real_f* lambdaForce, real_e* energy) {
  __shared__ real sred[32];
  real lEnergy = 0;
  real l = 1;
  extern __shared__ real sEnergy[];

  RmsdRestPotentialDev rmsdRest = rmsdRests[blockIdx.x];
  real Wf = rmsdRest.Wf;
  real Wr = rmsdRest.Wr;
  real D0 = rmsdRest.D0;
  int block = rmsdRest.block;

  // Stage 1: Centroid
  // Computes the centroid in two passes to minimize effects of out of box atoms.
  real3 anchor = position[rmsdRestAtoms[rmsdRest.fStartIdx].idx]; 
  for (int pass = 0; pass < 2; pass++) {
    real cpart[3] = {0};
    for (int k = threadIdx.x; k < rmsdRest.fCount; k += blockDim.x) {
      RmsdRestAtom fatom = rmsdRestAtoms[rmsdRest.fStartIdx+k];
      real3 xp_3 = real3_add(anchor, real3_subpbc<flagBox>(position[fatom.idx], anchor, box));
      cpart[0] += fatom.w*xp_3.x;
      cpart[1] += fatom.w*xp_3.y;
      cpart[2] += fatom.w*xp_3.z;
    }
    block_sum(cpart, 3, sred);
    anchor.x = cpart[0]/Wf;
    anchor.y = cpart[1]/Wf;
    anchor.z = cpart[2]/Wf;
  }
  real3 c_3 = anchor;
  real c[3] = {c_3.x, c_3.y, c_3.z};
  
  // Stage 2: Cross-covariance
  real B[9] = {0};

  for (int k = threadIdx.x; k < rmsdRest.fCount; k += blockDim.x) {
    RmsdRestAtom fatom = rmsdRestAtoms[rmsdRest.fStartIdx + k];
    real3 xp_3 = real3_subpbc<flagBox>(position[fatom.idx], c_3, box); // centered coords
    real3 xp0_3 = fatom.refPos;

    real xp[3] = {xp_3.x, xp_3.y, xp_3.z};
    real xp0[3] = {xp0_3.x, xp0_3.y, xp0_3.z};
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        B[3*i + j] += fatom.w*xp[i]*xp0[j];
      }
    }
  }
  block_sum(B, 9, sred);

  // Stage 3: Quaternion matrix
  real F[16];
  quaternion_matrix(B, F);

  // Stage 4: Quaternion
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

  // Stage 5: Optimal rotation matrix
  real R[9] = {0};
  quaternion_to_rotation(q, R);
  
  // Stage 6: Forces
  // 6.1: Forces on restrained
  real part[7] = {0}; // u, S.x, S.y, S.z, tau.x, tau.y, tau.z
  for (int k = threadIdx.x; k < rmsdRest.rCount; k += blockDim.x) {
    RmsdRestAtom ratom = rmsdRestAtoms[rmsdRest.rStartIdx + k];
    real3 xp_3 = real3_subpbc<flagBox>(position[ratom.idx], c_3, box); // centered coords
    real3 rxp0_3 = rotate_vector(R, &ratom.refPos); // rotated ref centered position, r_j
    real3 delta = real3_sub(xp_3, rxp0_3);
    real3 cr = real3_cross(rxp0_3, delta);
    real w = ratom.w;

    part[0] += w*real3_mag2<real>(delta);
    part[1] += 2*w*delta.x;
    part[2] += 2*w*delta.y;
    part[3] += 2*w*delta.z;
    part[4] += 2*w*cr.x;
    part[5] += 2*w*cr.y;
    part[6] += 2*w*cr.z;
  }
  block_sum(part, 7, sred);
  real u = part[0];
  real3 S = {part[1], part[2], part[3]};
  real3 tau = {part[4], part[5], part[6]};

  real D = sqrt(u/Wr); // RMSD of ligand from its fitted ref pose
  real V = 0;          // flat-bottom harmonic
  real dV = 0;         // flat-bottom harmonic derivative w.r.t u

  if (D > D0) {
    V = 0.5*rmsdRest.k*(D - D0)*(D - D0);
    dV = rmsdRest.k*(D - D0)/(2*Wr*D);
  }

  if (block) { l = lambda[block]; }

  if (threadIdx.x == 0) {
    if (block) {
      atomicAdd(&lambdaForce[block], V); // dU/dlambda, unscaled
    }
    lEnergy = l*V;
  }

  for (int k = threadIdx.x; k < rmsdRest.rCount; k += blockDim.x) {
    RmsdRestAtom ratom = rmsdRestAtoms[rmsdRest.rStartIdx + k];
    real3 xp_3 = real3_subpbc<flagBox>(position[ratom.idx], c_3, box); // centered coords
    real3 delta = real3_sub<real3>(xp_3, rotate_vector(R, &ratom.refPos));
    at_real3_scaleinc(&force[ratom.idx], l*dV*2*ratom.w, delta); // force[] accumulates +dU/dx, integrator applies sign
  }

  // 6.2: Forces on frame
  // dU/dx_i = -l*dV * [(w_i/W_f) S + w_i kappa x r_i], kappa = K tau, K = (Tr(G) I - G)^-1, G = B R^T, r_i = R x'_{i0}
  real G[9];
  mat3_times_transposed_mat3(B, R, G);
  real trG = G[0] + G[4] + G[8];

  real M[9];
  for (int j = 0; j < 9; j++) { M[j] = -G[j]; }
  M[0] += trG;
  M[4] += trG;
  M[8] += trG;

  real K[9];
  mat3_inverse(M, K);
  real3 kappa = rotate_vector(K, &tau);

  if (dV != 0) {
    for (int k = threadIdx.x; k < rmsdRest.fCount; k += blockDim.x) {
      RmsdRestAtom fatom = rmsdRestAtoms[rmsdRest.fStartIdx + k];
      real3 ri = rotate_vector(R, &fatom.refPos);
      real3 f = real3_scale<real3>(fatom.w/Wf, S);
      real3_scaleinc(&f, fatom.w, real3_cross(kappa, ri));
      at_real3_scaleinc(&force[fatom.idx], -l*dV, f);
    }
  }

  // Energy, if requested
  if (energy) {
    real_sum_reduce(lEnergy, sEnergy, energy);
  }
}


template <bool flagBox, typename box_type>
void getforce_rmsdRestT(System* system, box_type box, bool calcEnergy) {
  Potential* p = system->potential;
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
    getforce_rmsdRestraint_kernel <flagBox> <<<N, BLBO, shMem, r->biaspotStream>>> (N, p->rmsdRests_d, p->rmsdRestAtoms_d, (real3*)s->position_fd, (real3_f*)s->force_d, box, s->lambda_fd, s->lambdaForce_d, pEnergy);
  gpuCheck(cudaGetLastError());
  }
}


void getforce_rmsdRest(System* system, bool calcEnergy) {
  if (system->state->typeBox) {
    getforce_rmsdRestT<true>(system, system->state->tricBox_f, calcEnergy);
  } else {
    getforce_rmsdRestT<false>(system, system->state->orthBox_f, calcEnergy);
  }
}
