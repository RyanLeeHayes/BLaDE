#pragma once

#include <functional>
#include "main/defines.h"

// Implementation based on Nocedal & Wright book with references made to OpenMM extern lib implementation
class LBFGS {
public:
    int m=5; // Number of previous gradients to use for hessian approximation (5-7)
    int step_count; // total number of minimize calls
    int reset_count; // number of times lbfgs memory was cleared
    bool minimized;
    real_x U0;
    real_x Uf;
    // Line search parameters
    real_x c1 = 1e-4; // sufficient decent
    real_x c2 = .9; // curvature cond.
    // TODO: Implement convergence/stability checks
    real_x eps = 1e-4; // Minimization criteria 
    real_x delta = 1e-4; // stopping criteria

    LBFGS(int m, int DOF, std::function<real_x()> user_grad, real_x *position, real_x *gradient);
    ~LBFGS();
    void minimize_step(real_x f0);

private:
    int DOF; // Degrees of freedom to minimize
    real_x step_size = 1; // previous step size
    // All data stored on GPU
    int k = 0; // lbfgs iteration step
    real_x *tmp_d; // 1 real temperary storage
    real_x *gamma_d; // s[k-1] projected onto y[k-1]
    real_x *rho_d; // [m] rho[k] = 1/(s[k]^T * y[k])
    real_x *alpha_d; // [m] alpha[i] = rho[i] * s[i]^T * y[i]
    real_x *search_d; // [DOF] L-BFGS search direction
    real_x *prev_positions_d; // [DOF] x[k-1]
    real_x *prev_gradient_d; // [DOF] g[k-1]
    real_x *X_d; // points to user GPU memory, positions to be optimized
    real_x *G_d; // points to user GPU memory
    real_x *q_d;
    real_x *s_d; // [m, DOF] s[k-1] = x[k] - x[k-1] (1D array)
    real_x *y_d; // [m, DOF] y[k-1] = grad[k] - grad[k-1] (1D array)

    real_x *s_tmp_d; // [DOF] x[k] - x[k-1], used to check curvature 
    real_x *y_tmp_d; // [DOF] g[k] - g[k-1], used to check curvature
    bool skipped_step = false; // skipped hessian update, if 2 updates are skipped, we clear s & y memory

    std::function<real_x()> system_grad;

    void gamma_norm();
    void update_sk_yk();
    void phi(real_x alpha, real_x *result, bool leave_x);

    bool check_convergence();
    
    real_x zoom(real_x al, real_x const * const phi_lower, real_x au, real_x const * const phi_upper, real_x const * const phi0);
    real_x linesearch(real_x f0);
};