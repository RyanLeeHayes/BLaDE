#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <cstdio>

#include <cuda_runtime.h>
#include "update/lbfgs.h"
#include "main/real3.h" // for real_sum_reduce

// C = u*A + w*(d^n)*B, C and A and B can be related
__global__ void vector_add(
    int N, real u, real *A, real w, real* d, real n, real *B, 
    real* C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if(d){
            C[i] = u * A[i] + w * pow(*d, n) * B[i];
        } else {
            C[i] = u * A[i] + w * B[i];
        }
    }
}

    // *dot = w*u^p*(A).(B), w is cpu real, u is gpu real, p is cpu real
__global__ void dot_product(
    int N, real w, real* u, real p, real *A, real *B, 
    real* dot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    real lDot = 0;
    extern __shared__ real sDot[];
    if(i == 0){
        *dot = 0;
    }
    if (i < N) {
        if(u){
            lDot = w*powf(*u, p)*A[i]*B[i];
        } else {
            lDot = w*A[i]*B[i];
        }
    }

    real_sum_reduce(lDot, sDot, dot);
}

//  <position precision, real precision>
LBFGS::LBFGS(int m, int DOF, std::function<real()> user_grad, real *position, real *gradient)
    : m(m), DOF(DOF), system_grad(user_grad), X_d(position), G_d(gradient) {
    step = 0;
    cudaMalloc(&tmp_d, sizeof(real));
    cudaMalloc(&gamma_d, sizeof(real));
    cudaMalloc(&rho_inv_d, m*sizeof(real));
    cudaMalloc(&alpha_d, m*sizeof(real));
    cudaMalloc(&beta_d, m*sizeof(real));
    cudaMalloc(&q_d, DOF*sizeof(real));
    cudaMalloc(&prev_positions_d, DOF*sizeof(real));
    cudaMalloc(&prev_gradient_d, DOF*sizeof(real));
    cudaMalloc(&s_d, m*DOF*sizeof(real));
    cudaMalloc(&y_d, m*DOF*sizeof(real));

    U0 = user_grad(); // first energy call by lbfgs
}

LBFGS::~LBFGS() {
    if (tmp_d) cudaFree(tmp_d);
    if (rho_inv_d) cudaFree(rho_inv_d);
    if (alpha_d) cudaFree(alpha_d);
    if (gamma_d) cudaFree(gamma_d);
    if (beta_d) cudaFree(beta_d);
    if (prev_positions_d) cudaFree(prev_positions_d);
    if (prev_gradient_d) cudaFree(prev_gradient_d);
    if (q_d) cudaFree(q_d);
    if (s_d) cudaFree(s_d);
    if (y_d) cudaFree(y_d);
}

void LBFGS::minimize_step(real f0) { // f0 from outer loop
    cudaMemcpy(q_d, G_d, DOF*sizeof(real), cudaMemcpyDefault); // G from outer loop
    //vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, -1, G_d, 0, NULL, 1, q_d, q_d);
    if(step != 0 && m != 0) update_sk();
    // Two-loop recursion - Ch7.4, p178 of Nocedal & Wright
    for (int i = step-1; i >= step-m; i--) {
        if(i < 0) break;
        int index = i % m;
        // alpha.i = rho.i*s.i*q
        dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32, 0>>>(DOF, 1, rho_inv_d+index, -1, s_d+index*DOF, q_d, alpha_d+index);
        // q = q - alpha.i*y.i
        vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, q_d, -1, alpha_d+index, 1, y_d+index*DOF, q_d);
    }
    if(step != 0 && m != 0){
        int index = (step-1) % m;
        // norm = y.i*y.i
        dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32, 0>>>(DOF, 1, NULL, 1, y_d+index*DOF, y_d+index*DOF, tmp_d);
        // dot = s.i*y.i/norm
        dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32, 0>>>(DOF, 1, tmp_d, -1, s_d+index*DOF, y_d+index*DOF, gamma_d);
        // q = gamma*q
        vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 0, q_d, 1, gamma_d, 1, q_d, q_d);
    }
    for (int i = step-m; i < step; i++) {
        if(step == 0) break;
        if(i < 0) continue;
        int index = i % m;
        // B = rho.i*y.i*q
        dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32, 0>>>(DOF, 1, rho_inv_d+index, -1, y_d+index*DOF, q_d, tmp_d);
        // q = q + s.i*alpha.i
        vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, q_d, +1, alpha_d+index, 1, s_d+index*DOF, q_d); 
        // q = q - B*s.i
        vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, q_d, -1, tmp_d, 1, s_d+index*DOF, q_d);
    }
    // q = -q
    vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, -1, q_d, 0, NULL, 1, q_d, q_d);
    // min_a f(X + a*q)
    linesearch(f0); // backtracking, leaves X & G at final X+a*q
    step++;
}

void LBFGS::update_sk() {
    int index = (step - 1) % m;
    vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, X_d, -1, NULL, 1, prev_positions_d, s_d+index*DOF);
    vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, G_d, -1, NULL, 1, prev_gradient_d, y_d+index*DOF);
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32, 0>>>(DOF, 1, NULL, 1, y_d+index*DOF, s_d+index*DOF, rho_inv_d+index);

    cudaMemcpy(prev_positions_d, X_d, DOF*sizeof(real), cudaMemcpyDefault);
    cudaMemcpy(prev_gradient_d, G_d, DOF*sizeof(real), cudaMemcpyDefault);
}

// Ch3.2, p37 of Nocedal & Wright 
real LBFGS::linesearch(real f0) {
    int max_it = 15;
    real a = 5e-3; // max step size
    real a_prev = 0;
    real c1 = 1e-4; // Armijo cond
    real tau = 0.25; // labeled rho in reference
    real fi = 0;
    real m0;
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real)/32, 0>>>(DOF, 1, NULL, 1, G_d, q_d, tmp_d); // df(0)/da
    cudaMemcpy(&m0, tmp_d, sizeof(real), cudaMemcpyDefault);
    for (int i = 0; i < max_it; i++) {
        // Bump positions back and forth
        vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, X_d, (a-a_prev), NULL, 1, q_d, X_d);
        fi = system_grad(); // update G
        printf("a: %f, U0-Uf: %f, m0: %f\n", a, f0-fi, m0);
        if (fi <= f0 + c1*a*m0) {
            Uf = fi;
            return a; // position & gradient already updated at new point)
        } else {
            a_prev = a;
            a *= tau;
        }
    }
    // Linesearch failed
    printf("Linesearch Failed!\n", a); // a0*tau^(max_iter)
    Uf = fi;
    return a;
}