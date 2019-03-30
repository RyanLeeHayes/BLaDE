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
  real *lambdaBias;
  real *theta;
  real *thetaVelocity;
  real *thetaMass;
  real *lambdaCharge;

  int *atomBlock_d;
  int *lambdaSite_d;
  real *lambdaBias_d;

  int siteCount;
  int *blocksPerSite;
  int *blocksPerSite_d;
  int *siteBound;
  int *siteBound_d;

  real gamma;
  real fnex;

  bool scaleTerms[6]; // bond,ureyb,angle,dihe,impr,cmap

  int variableBiasCount;
  std::vector<struct VariableBias> variableBias_tmp;
  struct VariableBias *variableBias;
  struct VariableBias *variableBias_d;

  std::vector<Int2> softBonds;
  std::vector<std::vector<int>> atomRestraints;

  bool useSoftCore;
  bool useSoftCore14;

  Msld();
  ~Msld();

  void bonded_scaling(int *idx,int *siteBlock,int type,int Nat,int Nsc);
  void bond_scaling(int idx[2],int siteBlock[2]);
  void ureyb_scaling(int idx[3],int siteBlock[2]);
  void angle_scaling(int idx[3],int siteBlock[2]);
  void dihe_scaling(int idx[4],int siteBlock[2]);
  void impr_scaling(int idx[4],int siteBlock[2]);
  void cmap_scaling(int idx[8],int siteBlock[3]);
  void nb14_scaling(int idx[2],int siteBlock[2]);
  void nbex_scaling(int idx[2],int siteBlock[2]);
  void nbond_scaling(int idx[1],int siteBlock[1]);

  bool interacting(int i,int j);

  void initialize(System *system);

  void calc_lambda_from_theta(cudaStream_t stream,System *system);
  void calc_thetaForce_from_lambdaForce(cudaStream_t stream,System *system);
  void calc_fixedBias(System *system,bool calcEnergy);
  void calc_variableBias(System *system,bool calcEnergy);
};

void parse_msld(char *line,System *system);

#endif
