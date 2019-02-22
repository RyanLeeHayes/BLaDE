#ifndef MSLD_MSLD_H
#define MSLD_MSLD_H

#include <vector>

#include "main/defines.h"

// Forward declarations
class System;

// CHARMM format: LDBV INDEX  I   J  CLASS  REF  CFORCE NPOWER
struct VariableBias {
  int i,j;
  int type;
  real l0;
  real k;
  int n;
};

class Msld {
  public:
  int blockCount;
  int *atomBlock;
  int *lambdaSite;
  real *lambda;
  real *lambdaForce;
  real *lambdaBias;
  real *theta;
  real *thetaVelocity;
  real *thetaMass;
  real *lambdaCharge;

  real *lambda_d;
  real *lambdaForce_d;
  real *theta_d;
  real *thetaVelocity_d;
  real *thetaMass_d;

  bool scaleTerms[6]; // bond,ureyb,angle,dihe,impr,cmap
  std::vector<struct VariableBias> variableBias;
  std::vector<Int2> softBonds;
  std::vector<std::vector<int>> atomRestraints;

  Msld();
  ~Msld();

  void bonded_scaling(int *idx,int *siteBlock,int type,int Nat,int Nsc);
  void bond_scaling(int idx[2],int siteBlock[2]);
  void ureyb_scaling(int idx[3],int siteBlock[2]);
  void angle_scaling(int idx[3],int siteBlock[2]);
  void dihe_scaling(int idx[4],int siteBlock[2]);
  void impr_scaling(int idx[4],int siteBlock[2]);
  void cmap_scaling(int idx[8],int siteBlock[3]);

  void send_real(real *p_d,real *p);
  void recv_real(real *p,real *p_d);
};

void parse_msld(char *line,System *system);

#endif
