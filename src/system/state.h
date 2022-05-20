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
  real friction; // a2=exp(-gamma*dt)
  real noise; // sqrt((1-a2*a2)*kT) - still need to divide by sqrt(m)
  real halfdt; // 0.5*dt
};

struct LeapState
{
  int N1; // spatial dof
  int N; // spatial dof + alchemical dof
  real_x *x;
  real_v *v;
  real_f *f;
  real *ism; // 1/sqrt(m)
  real *random;
};

typedef enum ebox {
  ebcubi, // cubic
  ebtetr, // tetragonal
  eborth, // orthorhombic
  ebmono, // monoclinic
  ebtric, // triclinic
  ebhexa, // hexagonal
  ebrhom, // rhombohedral
  ebocta, // truncated octahedron
  ebrhdo, // rhombic dodecahedron
  ebend} EBox;

class State {
  public:
  int atomCount;
  int lambdaCount;

  // Lambda-Spatial-Theta buffers
  real_x *positionBuffer;
  real_x *positionBuffer_d;
  real *positionBuffer_fd; // floating precision device version
#ifdef REPLICAEXCHANGE
  real_x *positionRExBuffer; // For REx communication
#endif
  real_x *positionBackup_d; // For NPT
  real *positionBuffer_omp;
  real_f *forceBuffer;
  real_f *forceBuffer_d;
  real_f *forceBackup_d; // For NPT
  real_f *forceBuffer_omp;

  // Other buffers
  real_e *energy;
  real_e *energy_d;
  real_e *energyBackup_d;
  real_e *energy_omp;

  // Spatial-Theta buffers
  real_v *velocityBuffer;
  real_v *velocityBuffer_d;
  real *invsqrtMassBuffer;
  real *invsqrtMassBuffer_d;

  // Constraint stuff
  real_x *positionCons_d;

  // The box
  int nameBox;
  bool typeBox; // false is orthographic, true is triclinic
  real33_x box;
  real33_x boxBackup;
  real3_x orthBox;
  real3_x orthBoxBackup;
  real3 orthBox_f; // floating precision version
  real3 *orthBox_omp;
  real3 kOrthBox_f;
  real123_x tricBox;
  real123_x tricBoxBackup;
  real123 tricBox_f;
  real123 *tricBox_omp;
  real321 kTricBox_f;

  // Buffer for floating point output
  float (*positionXTC)[3]; // intentional float

  // Labels (do not free)
  real_x *lambda;
  real_x *lambda_d;
  real *lambda_fd;
  real_x (*position)[3];
  real_x (*position_d)[3];
  real (*position_fd)[3];
  real_x (*positionb_d)[3];
  real_x *theta;
  real_x *theta_d;
  real *theta_fd;
  real_v (*velocity)[3];
  real_v (*velocity_d)[3];
  real_v *thetaVelocity;
  real_v *thetaVelocity_d;
  real_f *lambdaForce;
  real_f *lambdaForce_d;
  real_f (*force)[3];
  real_f (*force_d)[3];
  real_f *thetaForce;
  real_f *thetaForce_d;
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
  struct LeapState* alloc_leapstate(int N1,int N2,real_x *x,real_v *v,real_f *f,real *ism);
  void free_leapstate(struct LeapState *ls);

  void recv_state();
  void send_state();
  void recv_position();
  void recv_lambda();
  void recv_energy();

  void backup_position();
  void restore_position();

  void broadcast_position(System *system);
  void broadcast_box(System *system);
  void gather_force(System *system,bool calcEnergy);

  void prettify_position(System *system);
  void check_box(System *system);

  // From update/update.cu
  void set_fd(System *system);
  void update(int step,System *system);
  void kinetic_energy(System *system);
};

#endif
