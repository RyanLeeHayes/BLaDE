#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <cstdio>
#include <algorithm>

#include <cuda_runtime.h>
#include "update/lbfgs.h"
#include "main/real3.h" // for real_sum_reduce

/*
    Cuda Kernels
*/

// d = 1/sqrt(d) - element-wise
__global__ void inv_sqrt(int N, real_x *d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d[i] = 1.0/sqrt(d[i]);
    }
}

// d = 1/d - element-wise
__global__ void recip(int N, real_x *d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d[i] = 1.0/d[i];
    }
}

// d = abs(d) - element-wise
__global__ void vector_abs(int N, real_x *d){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d[i] = abs(d[i]);
    }
}

// C = u*A + w*d*B, C and A and B can be related
__global__ void vector_add(int N, real_x u, real_x *A, real_x w, real_x *d, real_x *B, real_x *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if(d){
            C[i] = u * A[i] + w * d[0] * B[i];
        } else {
            C[i] = u * A[i] + w * B[i];
        }
    }
}

// specific for 2nd lbfgs loop
// C = A + (u-w)*B, C and A and B can be related
__global__ void vector_add(int N, real_x *A, real_x *u, real_x *w, real_x *B, real_x *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + (u[0]-w[0]) * B[i];
    }
}

// C = A[0]*B - element-wise
__global__ void vector_scale(int N, real_x *A, real_x *B, real_x *C){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[0]*B[i];
    }
}

// B = c*A - element-wise
__global__ void vector_scale(int N, real_x c, real_x *A, real_x *B){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        B[i] = c*A[i];
    }
}

__global__ void dot_product(int N, real_x *A, real_x *B, real_x* dot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    real_x lDot = 0;
    extern __shared__ real sDot[];
    if (i < N) {
        lDot = A[i]*B[i];
    }

    real_sum_reduce((real)lDot, sDot, dot);
}

/*
    Class setup
*/

//  <position precision, real_x precision>
LBFGS::LBFGS(int m, real_x eps, int DOF, bool verbose, std::function<real_x()> user_grad, real_x *position, real_x *gradient)
    : m(m), eps_tol(eps), DOF(DOF), verbose(verbose), system_grad(user_grad), X_d(position), G_d(gradient) {
    k = 0;
    cudaMalloc(&tmp_d, sizeof(real_x));
    cudaMalloc(&gamma_d, sizeof(real_x));
    cudaMalloc(&rho_d, m*sizeof(real_x));
    cudaMalloc(&alpha_d, m*sizeof(real_x));
    cudaMalloc(&q_d, DOF*sizeof(real_x));
    cudaMalloc(&prev_positions_d, DOF*sizeof(real_x));
    cudaMalloc(&prev_gradient_d, DOF*sizeof(real_x));
    cudaMalloc(&s_d, m*DOF*sizeof(real_x));
    cudaMalloc(&y_d, m*DOF*sizeof(real_x));
    cudaMalloc(&s_tmp_d, DOF*sizeof(real_x));
    cudaMalloc(&y_tmp_d, DOF*sizeof(real_x));

    U0 = user_grad(); // first energy call by lbfgs
    Uf = U0;

    // Normalize first s.d. step
    gamma_norm();

    // Set up s.d. step
    cudaMemcpy(q_d, G_d, DOF*sizeof(real_x), cudaMemcpyDefault);
    vector_scale<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, gamma_d, q_d, q_d);
    vector_scale<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, -1, q_d, q_d);
}

LBFGS::~LBFGS() {
    if (tmp_d) cudaFree(tmp_d);
    if (gamma_d) cudaFree(gamma_d);
    if (rho_d) cudaFree(rho_d);
    if (alpha_d) cudaFree(alpha_d);
    if (q_d) cudaFree(q_d);
    if (prev_positions_d) cudaFree(prev_positions_d);
    if (prev_gradient_d) cudaFree(prev_gradient_d);
    if (s_d) cudaFree(s_d);
    if (y_d) cudaFree(y_d);
    if (s_tmp_d) cudaFree(s_tmp_d);
    if (y_tmp_d) cudaFree(y_tmp_d);
}

/*
    L-BFGS iteration
*/

bool LBFGS::converged(){
    // return true if rmsg minimized
    rmsg = 0;
    grad_mag = 0;
    grad_pos_mag = 0;

    cudaMemset(tmp_d, 0, sizeof(real_x));
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, G_d, G_d, tmp_d);
    cudaMemcpy(&rmsg, tmp_d, sizeof(real_x), cudaMemcpyDefault);
    grad_mag = sqrt(rmsg);
    rmsg = sqrt(rmsg/DOF);
    cudaMemset(tmp_d, 0, sizeof(real_x));
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, X_d, X_d, tmp_d);
    cudaMemcpy(&grad_pos_mag, tmp_d, sizeof(real_x), cudaMemcpyDefault);
    grad_pos_mag = grad_mag / std::max(1.0, sqrt(grad_pos_mag));

    if (verbose){
      printf("   rmsg = %f\n", rmsg);
      printf("    |g| = %f\n", grad_mag);
      printf("|g|/|x| = %f\n", grad_pos_mag);
    }
    if(rmsg < eps_tol){
        if (verbose) printf("Structure minimized!!\n");
        minimized=true;
    }
    return rmsg < eps_tol;
}

// set gamma = 1/|g_d|
void LBFGS::gamma_norm(){
    cudaMemset(tmp_d, 0, sizeof(real_x));
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, G_d, G_d, tmp_d);
    inv_sqrt<<<1,1>>>(1, tmp_d);
    cudaMemcpy(gamma_d, tmp_d, sizeof(real_x), cudaMemcpyDefault);
}

// Ch7.4, p178 of Nocedal & Wright (Algorithm 7.4)
void LBFGS::minimize_step(real_x f0) { // f0 & G filled from class initializer
    // Copy kth positions & gradients
    cudaMemcpy(prev_positions_d, X_d, DOF*sizeof(real_x), cudaMemcpyDefault);
    cudaMemcpy(prev_gradient_d, G_d, DOF*sizeof(real_x), cudaMemcpyDefault);
    if(converged()) return;

    // min_a f(X + a*q)
    step_size = linesearch(f0); // X & G left at & evaluated at f(X+a*q)

    // kth position and gradient deltas
    update_sk_yk(); // potentially skip decrement k or set k=0 & clear s & y memory
    k++;

    // Two-loop recursion
    cudaMemcpy(q_d, G_d, DOF*sizeof(real_x), cudaMemcpyDefault); 
    for (int i = k-1; i >= std::max(0, k-m); i--) {
        int index = i % m;
        // alpha.i = rho.i*(s.i dot q)
        cudaMemset(alpha_d+index, 0, sizeof(real_x));
        dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, s_d+index*DOF, q_d, alpha_d+index);
        vector_scale<<<1,1>>>(1, rho_d+index, alpha_d+index, alpha_d+index);
        // q = q - alpha.i*y.i
        vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, q_d, -1, alpha_d+index, y_d+index*DOF, q_d);
    }
    // q = gamma.k*q = r
    vector_scale<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, gamma_d, q_d, q_d);
    for (int i = std::max(0, k-m); i <= k-1; i++) {
        int index = i % m;
        // B = rho.i*(y.i dot q)
        cudaMemset(tmp_d, 0, sizeof(real_x));
        dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, y_d+index*DOF, q_d, tmp_d);
        vector_scale<<<1,1>>>(1, rho_d+index, tmp_d, tmp_d);
        // q = q + (alpha.i - B)*s.i = q - B*s.i + alpha.i*s.i
        vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, q_d, alpha_d+index, tmp_d, s_d+index*DOF, q_d);
   }

    // q = -q
    vector_scale<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, -1, q_d, q_d);
    step_count++;
}

// this method is overly cautious & slow, likely doesn't need the cpu grad checks or resets
void LBFGS::update_sk_yk() {
    if(m == 0) { // Steepest decent, normalize step size via gamma
        gamma_norm();
        return;
    } 

    // s & y tmp
    // alpha*q = displacement, X_d and prev_position might differ by pbc wrapping
    vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 0, X_d, step_size, NULL, q_d, s_tmp_d); 
    vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, G_d, -1, NULL, prev_gradient_d, y_tmp_d);

    real_x yy = 0;
    cudaMemset(tmp_d, 0, sizeof(real_x));
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, y_tmp_d, y_tmp_d, tmp_d);
    cudaMemcpy(&yy, tmp_d, sizeof(real_x), cudaMemcpyDefault);
    real_x sy = 0;
    cudaMemset(tmp_d, 0, sizeof(real_x));
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, s_tmp_d, y_tmp_d, tmp_d);
    cudaMemcpy(&sy, tmp_d, sizeof(real_x), cudaMemcpyDefault);

    // Check curvature s.T H s = s.T y > 0 for positive def matrix satisfying secant eq Hs=y (required for L-BFGS)
    if(sy < 1e-10){ 
        printf("Curvature condition (sy = %e) not satistied! Clearing L-BFGS memory!\n", sy);
        cudaMemset(s_d, 0, m*DOF*sizeof(real_x));
        cudaMemset(y_d, 0, m*DOF*sizeof(real_x));
        gamma_norm();
        k = -1;
        reset_count++;
        return;
    } 

    // rho = 1/y.s
    real_x rho = 1.0/sy;
    // gamma_k = s.y/(y.y)
    real_x gamma = sy/yy;
    cudaMemcpy(gamma_d, &gamma, sizeof(real_x), cudaMemcpyDefault);
    int index = k % m;
    cudaMemcpy(rho_d+index, &rho, sizeof(real_x), cudaMemcpyDefault);
    cudaMemcpy(s_d + index*DOF, s_tmp_d, DOF*sizeof(real_x), cudaMemcpyDefault);
    cudaMemcpy(y_d + index*DOF, y_tmp_d, DOF*sizeof(real_x), cudaMemcpyDefault);
    //printf("iter: %d, k: %d, index: %d, rho: %f, gamma: %f\n", step_count, k, index, rho, gamma);
}

/*
    Line Search
*/

// return [f(X + alpha*p), df(X+alpha*p)/da]
void LBFGS::phi(real_x alpha, real_x* result){
    cudaMemcpy(X_d, prev_positions_d, DOF*sizeof(real_x), cudaMemcpyDefault);
    vector_add<<<(DOF+BLUP-1)/BLUP,BLUP>>>(DOF, 1, X_d, alpha, NULL, q_d, X_d);
    result[0] = system_grad();
    cudaMemset(tmp_d, 0, sizeof(real_x));
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, G_d, q_d, tmp_d); // df(0)/da
    cudaMemcpy(&result[1], tmp_d, sizeof(real_x), cudaMemcpyDefault);
}

real_x sign(real_x value){
    if(value < 0){
        return -1;
    }
    return 1;
}

// Ch3.5, p59 of Nocedal & Wright (Equation 3.59) 
real_x cubic_interp(real_x a, real_x fa, real_x ga, real_x b, real_x fb, real_x gb){
    real_x d1 = ga + gb - 3*(fa - fb)/(a-b);
    real_x d2 = sign(b - a)*sqrt(d1*d1 - ga*gb);
    real_x arg = b - (b-a)*(gb + d2 - d1)/(gb - ga + 2*d2);
    if(isinf(arg) || isnan(arg) || abs(arg - a) < 1e-7 || abs(arg - b) < 1e-7 || arg < 0){
        //printf("Quadratic interpolation failed! arg: %f, a: %f, b: %f\n", arg, a, b);
        return (a + b) / 2.0;
    }
    return arg;
}

// Ch3.5, p60 of Nocedal & Wright (Algorithm 3.5)
real_x LBFGS::linesearch(real_x f0){
    real_x max_iter = 5;
    real_x aim1 = 0; 
    real_x ai = 1;
    if (step_count == 0) { ai = 1e-2; } // steepest decent step
    real_x amax = 10;
    // Data is already loaded, saves 1 energy & grad call
    real_x phi0[2] = {f0, 0};
    cudaMemset(tmp_d, 0, sizeof(real_x));
    dot_product<<<(DOF+BLUP-1)/BLUP,BLUP,BLUP*sizeof(real_x)/32, 0>>>(DOF, G_d, q_d, tmp_d); // df(0)/da
    cudaMemcpy(&phi0[1], tmp_d, sizeof(real_x), cudaMemcpyDefault);
    real_x phiim1[2];
    memcpy(phiim1, phi0, 2*sizeof(real_x));
    // Evaluate at 1 & iterate using cubic interpolation
    real_x phii[2];
    phi(ai, phii);
    if (verbose){
      printf("alpha: %f, phi: %f, phi': %f\n", aim1, phi0[0], phi0[1]);
      printf("alpha: %f, phi: %f, phi': %f\n", ai, phii[0], phii[1]);
    }
    for(int i = 0; i < max_iter; i++){
        // Check strong wolfe condition
        if (phii[0] <= phi0[0] + c1*ai*phi0[1] && abs(phii[1]) <= c2*abs(phi0[1])){
            Uf = phii[0];
            return ai;
        }
        real_x tmp = ai;
        ai = cubic_interp(aim1, phiim1[0], phiim1[1], ai, phii[0], phii[1]);
        phi(ai, phii);
        if (verbose) printf("alpha: %f, phi: %f, phi': %f\n", ai, phii[0], phii[1]);
        aim1 = tmp;
        memcpy(phiim1, phii, 2*sizeof(real_x));
        if(ai < 0 || ai > amax){
            break;
        }
    }

    // Failed line search
    ai = 1e-4; // default step size, a small enough step should decrease function if s.y > 0 
    phi(ai, phii);
    Uf = phii[0];
    printf("Linesearch failed! Defaulting to ai = %e, phi: %f, phi': %f\n", ai, phii[0], phii[1]);
    return ai;
}