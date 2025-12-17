#pragma once

#include <functional>
#include "main/defines.h"

class LBFGS {
public:
    int m; // Number of previous gradients to use for hessian approximation (5-7)
    int step;
    bool minimized;
    real U0;
    real Uf;

    LBFGS(int m, int DOF, std::function<real()> user_grad, real* position, real* gradient);
    ~LBFGS();
    void minimize_step(real f0);

private:
    int DOF; // Degrees of freedom to minimize
    // All data stored on GPU
    real *beta_d;
    real *tmp_d; // 1 real temp storage
    real *gamma_d; // sk-1 projected onto yk-1
    real *rho_inv_d; // [m] rho_inv[i] = s[i]^T * y[i]
    real *alpha_d; // [m] alpha[i] = rho[i] * s[i]^T * y[i]
    real *search_d; // [DOF] L-BFGS search direction
    real *prev_positions_d; // [DOF] x[i-1]
    real *prev_gradient_d; // [DOF] g[i-1]
    real *X_d; // points to user GPU memory, positions to be optimized
    real *G_d; // points to user GPU memory
    real *q_d;
    real *s_d; // [m, DOF] s[i] = x[i+1] - x[i] (1D array)
    real *y_d; // [m, DOF] y[i] = grad[i+1] - grad[i] (1D array)
    std::function<real()> system_grad;

    void update_sk();
    real linesearch(real f0);
};