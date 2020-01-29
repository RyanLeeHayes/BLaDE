#ifndef SYSTEM_STATE_H
#define SYSTEM_STATE_H

#include <stdio.h>
#include <cuda_runtime.h>

#include <string>
#include <map>

#include "main/defines.h"

// Forward delcarations
class System;
class RngCPU;
class RngGPU;

struct LeapParms1
{
  real dt;
  real gamma;
  real kT;
};

struct LeapParms2
{
  real sqrta; // a=exp(-gamma*dt)
  real noise; // sqrt((1-a)*kT) - still need to divide by sqrt(m)
  real fscale; // 0.5*b*dt; b=sqrt(tanh(0.5*gamma*dt)/(0.5*gamma*dt));
};

struct LeapState
{
  int N1; // spatial dof
  int N; // spatial dof + alchemical dof
  real *x;
  real *v;
  real *f;
  real *ism; // 1/sqrt(m)
  real *random;
};

class State {
  public:
  int atomCount;
  int lambdaCount;

  // Lambda-Spatial-Theta buffers
  real *positionBuffer;
  real *positionBuffer_d;
#ifdef REPLICAEXCHANGE
  real *positionRExBuffer; // For REx communication
#endif
  real *positionBackup_d; // For NPT
  real *positionBuffer_omp;
  real *forceBuffer;
  real *forceBuffer_d;
  real *forceBackup_d; // For NPT
  real *forceBuffer_omp;

  // Other buffers
  real_e *energy;
  real_e *energy_d;
  real_e *energyBackup_d;
  real_e *energy_omp;

  // Spatial-Theta buffers
  real *velocityBuffer;
  real *velocityBuffer_d;
  real *invsqrtMassBuffer;
  real *invsqrtMassBuffer_d;

  // Constraint stuff
  real *positionCons_d;

  // The box
  real3 orthBox;
  real3 *orthBox_omp;
  real3 orthBoxBackup;

  // Buffer for floating point output
  float (*positionXTC)[3]; // intentional float

  // Labels (do not free)
  real *lambda;
  real *lambda_d;
  real (*position)[3];
  real (*position_d)[3];
  real *theta;
  real *theta_d;
  real (*velocity)[3];
  real (*velocity_d)[3];
  real *thetaVelocity;
  real *thetaVelocity_d;
  real *lambdaForce;
  real *lambdaForce_d;
  real (*force)[3];
  real (*force_d)[3];
  real *thetaForce;
  real *thetaForce_d;
  real (*invsqrtMass)[3];
  real (*invsqrtMass_d)[3];
  real *thetaInvsqrtMass;
  real *thetaInvsqrtMass_d;

  // Leapfrog structures
  struct LeapParms1 *leapParms1;
  struct LeapParms2 *leapParms2;
  struct LeapParms1 *lambdaLeapParms1;
  struct LeapParms2 *lambdaLeapParms2;
  struct LeapState *leapState;

  State(System *system);
  ~State();

  // From system/state.cxx
  void initialize(System *system);
  void save_state(System *system);

  struct LeapParms1* alloc_leapparms1(real dt,real gamma,real T);
  struct LeapParms2* alloc_leapparms2(real dt,real gamma,real T);
  struct LeapState* alloc_leapstate(int N1,int N2,real *x,real *v,real *f,real *ism);
  void free_leapstate(struct LeapState *ls);

  void recv_state();
  void send_state();
  void recv_position();
  void recv_lambda();
  void recv_energy();

  void backup_position();
  void restore_position();

  void broadcast_position(System *system);
  void broadcast_velocity(System *system);
  void broadcast_box(System *system);
  void gather_force(System *system,bool calcEnergy);

  // From update/update.cu
  void update(int step,System *system);
};

#endif
