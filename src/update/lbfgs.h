#pragma once

#include <functional>
#include "main/defines.h"

// Implementation based on Nocedal & Wright book with references made to OpenMM extern lib implementation
class LBFGS {
public:
    int step_count=0; // total number of minimize calls
    int reset_count=0; // number of times lbfgs memory was cleared
    int m=5; // Number of previous gradients to use for hessian approximation (5-7)
    bool minimized=false;
    real_x U0, Uf;

    LBFGS(int m, real_x eps, int DOF, std::function<real_x()> user_grad, real_x *position, real_x *gradient);
    ~LBFGS();
    void minimize_step(real_x f0);

private:
    // Line search parameters
    real_x c1 = 1e-4; // sufficient decent
    real_x c2 = .9; // curvature cond.
    // Convergence
    real_x eps_tol = 1; // rms criteria (1e-1 is very tight and unlikely to work)
    real_x rmsg = 0; // |g| / sqrt(DOF)
    real_x grad_mag = 0; // |g| 
    real_x grad_pos_mag = 0; // |g| / max(1, |x|)

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

    bool converged();
    void gamma_norm();
    void update_sk_yk();

    void phi(real_x alpha, real_x *result);
    real_x linesearch(real_x f0);
};