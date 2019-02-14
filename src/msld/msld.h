#ifndef MSLD_MSLD_H
#define MSLD_MSLD_H

#include <stdio.h>

// Forward declarations
class System;

// CHARMM format: LDBV INDEX  I   J  CLASS  REF  CFORCE NPOWER
struct VariableBias {
  int i,j;
  int type;
  double l0;
  double k;
  int n;
};

class Msld {
  public:
  int blockCount;
  int *atomBlock;
  int *lambdaSite;
  double *lambda;
  double *lambdaForce;
  double *lambdaBias;
  double *theta;
  double *thetaVelocity;
  double *thetaMass;
  double *lambdaCharge;
  bool scaleTerms[6]; // bond,ureyb,angle,dihe,impr,cmap
  std::vector<struct VariableBias> variableBias;
  std::vector<Int2> softBonds;
  std::vector<std::vector<int>> atomRestraints;

  Msld() {
    fprintf(stdout,"IMPLEMENT MSLD create %s %d\n",__FILE__,__LINE__);
    blockCount=1;
    atomBlock=NULL;
    lambdaSite=NULL;
    lambda=NULL;
    lambdaForce=NULL;
    lambdaBias=NULL;
    theta=NULL;
    thetaVelocity=NULL;
    thetaMass=NULL;
    lambdaCharge=NULL;
    variableBias.clear();
    softBonds.clear();
    atomRestraints.clear();
  }

  ~Msld() {
    if (atomBlock) free(atomBlock);
    if (lambdaSite) free(lambdaSite);
    if (lambda) free(lambda);
    if (lambdaForce) free(lambdaForce);
    if (lambdaBias) free(lambdaBias);
    if (theta) free(theta);
    if (thetaVelocity) free(thetaVelocity);
    if (thetaMass) free(thetaMass);
    if (lambdaCharge) free(lambdaCharge);
  }

};

void parse_msld(char *line,System *system);

#endif
