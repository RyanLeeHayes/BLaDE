
#ifndef LBFGS_H
#define LBFGS_H

#include <type_traits>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <functional>

using namespace std;

template <typename T> class LBFGS {
    static_assert(std::is_floating_point<T>::value, "L-BFGS can only be used with floating point types");

public:
    LBFGS(int m, int DOF, std::function<T(T*, T*)> user_grad) : m(m), DOF(DOF), grad(user_grad) {
        minimized = false;
        min_step_size = sizeof(T) == 4 ? 1e-4 : 1e-7;
        max_step_size = 1.0;
        rho = (T*) calloc(m, sizeof(T));
        alpha = (T*) calloc(m, sizeof(T));
        gamma = (T*) calloc(m, sizeof(T));
        beta = (T*) calloc(m, sizeof(T));
        step_size = 0.0;
         
        search = (T*) calloc(DOF, sizeof(T));
        prev_positions = (T*) calloc(DOF, sizeof(T));
        prev_gradient = (T*) calloc(DOF, sizeof(T));

        q = (T*) calloc(DOF, sizeof(T));
        x_plus_step = (T*) calloc(DOF, sizeof(T));
        s = (T*) calloc(m*DOF, sizeof(T));
        y = (T*) calloc(m*DOF, sizeof(T));
    }

    ~LBFGS() {
        free(rho);
        free(alpha);
        free(gamma);
        free(beta);
        free(G);
        free(X);
        free(search);
        free(prev_positions);
        free(prev_gradient);
        free(q);
        free(x_plus_step);
        free(s);
        free(y);
    }

/**
 * Steepest decent step
 * 
 * @param X Initial Position
 * @param G Initial Position Grad
 */
void init(T* X, T* G, T steepest_descent_step_size) {
    // f(X0), g(X0)
    grad(X, G);
    // Update
    for (int i = 0; i < DOF; ++i) {
        prev_positions[i] = X[i];
        prev_gradient[i] = G[i];
    }
    for (int i = 0; i < DOF; ++i) {
        s[i + (m - 1) * DOF] = X[i] - prev_positions[i];
        y[i + (m - 1) * DOF] = G[i] - prev_gradient[i];
        X[i] -= steepest_descent_step_size * G[i];
    }
    // f(X0-size*G), g(X0-size*G)
    grad(X, G);
}

/**
 * 1. Compute new L-BFGS step direction
 *   Pseudocode from wikipedia:
 *   q = g.i // search direction to be updated
 *   for j = i-1 to i-m:
 *     alpha.j = rho.j * s.j.T * q // dot product
 *     q = q - alpha.j * y.j // vector scale & subtraction
 *   gamma.i = s.i-1.T * y.i-1 / y.i-1.T * y.i-1 // dot products in numerator and denominator
 *   q = gamma.i * q
 *   for j = i-m to i-1:
 *     beta = rho.j * y.j.T * q // dot product
 *     q = q + (alpha.j - beta) * s.j // vector scale & addition
 *   q = -q  // negate applied above instead of here most likely
 *   gamma = s.i.T * y.i / y.i.T * y.i
 *   rho.j = 1 / (y.j.T * s.j)
 */
void minimize_step(T* X, T* G) {
    if (minimized){
        return;
    }
    // Eval function at X
    T p0 = grad(X, G);
    for (int i = 0; i < DOF; ++i) {
        q[i] = G[i];
    }
    update(X, G);
    for (int i = m - 1; i >= 0; --i) {
        alpha[i] = rho[i] * dot_product(s + i * DOF, q, DOF);
        for (int j = 0; j < DOF; ++j) {
            q[j] -= alpha[i] * y[i * DOF + j];
        }
    }
    T gamma = dot_product(s + (m - 1) * DOF, y + (m - 1) * DOF, DOF) / dot_product(y + (m - 1) * DOF, y + (m - 1) * DOF, DOF);
    for (int j = 0; j < DOF; ++j) {
        q[j] *= gamma;
    }
    for (int i = 0; i < m; ++i) {
        T beta = rho[i] * dot_product(y + i * DOF, q, DOF);
        for (int j = 0; j < DOF; ++j) {
            q[j] += (alpha[i] - beta) * s[i * DOF + j];
        }
    }
    // min_a f(X + a*q)
    step_size = this->linesearch(p0, X, G); 
    std::cout << "step size: " << step_size << "\n";
    // Update system positions
    for (int j = 0; j < DOF; ++j) {
        X[j] += (step_size * -q[j]);
    }
}

void update(T* X, T* G) {
    for (int i = 0; i < ((m - 1) * DOF); ++i) {
        s[i] = s[i + DOF];
        y[i] = y[i + DOF];
    }
    for (int i = 0; i < DOF; ++i) {
        s[(m - 1) * DOF + i] = X[i] - prev_positions[i];
        y[(m - 1) * DOF + i] = G[i] - prev_gradient[i];
    }
    for (int i = 0; i < m - 1; ++i) {
        rho[i] = rho[i + 1];
    }
    double s_dot_y = dot_product((s + (m - 1) * DOF), (y + (m - 1) * DOF), DOF);
    if (s_dot_y == 0) {
        std::cout << "Error: Dividing by zero. Function is likely already at a minimum.\n";
        minimized = true;
    }
    else {
        rho[m - 1] = 1 / s_dot_y;
    }
    for (int i = 0; i < DOF; ++i) {
        prev_positions[i] = X[i];
        prev_gradient[i] = G[i];
    }
}

T linesearch(T p0, T* X, T* G) {
    int max_it = 1000;
    T c = 0.5;
    T tau = 0.75;
    T m = dot_product(G, q, DOF);
    T step_size = 1;
    for (int i = 0; i < max_it; i++) {
        T sum = 0;
        for (int j = 0; j < DOF; ++j) {
            x_plus_step[j] = X[j] - (step_size * q[j]);
        }
        T new_value = grad(x_plus_step, G);
        if (p0 - new_value >= step_size * -c * m) {
            return step_size;
        } else {
            step_size *= tau;
            printf("Step_size: %f, p0: %f, new: %f, x+s: %f\n", step_size, p0, new_value, q[50]);
        }
    }
    return 0;
}

private:
    int m; // Number of previous gradients to use for hessian approximation (5-7)
    int DOF; // Degrees of freedom
    T min_step_size; // terminate minimization with steps smaller than this number
    T max_step_size; // ensures that lbfgs doesn't overshoot minimum
    bool minimized;

    T *beta; 
    T *gamma; // s projected onto y
    T *rho; // [m] rho_{i} = (s_{i}^T * y_{i}
    T *alpha; // [m] alpha_{i} = rho_{i} * s_{i}^T * y_{i}

    T *X;
    T *G;
    T *search; // [DOF] L-BFGS search direction
    T *prev_positions; // [DOF] x_{i-1}
    T *prev_gradient; // [DOF] g_{i-1}

    T* q;
    T* x_plus_step;
    T *s; // [m*DOF] x_{i+1} - x_{i} = s_{i}
    T *y; // [m*DOF] grad_{i+1} - grad_{i} = y_{i}
    T step_size;

    std::function<T(T*, T*)> grad;

    T dot_product(T* a, T* b, int n) {
        T result = 0;
        for (int i = 0; i < n; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
};

#endif // LBFGS