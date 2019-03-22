#ifndef SYSTEM_POTENTIAL_H
#define SYSTEM_POTENTIAL_H

#include <cuda_runtime.h>
#include <cufft.h>

#include <vector>
#include <map>
#include <string>
#include <set>

#include "main/defines.h"

// Forward declarations
class System;
class TypeName8O;
bool operator==(const TypeName8O& a,const TypeName8O& b);
bool operator<(const TypeName8O& a,const TypeName8O& b);

typedef enum eeterm {
  eebond,
  eeangle,
  eedihe,
  eeimpr,
  eecmap,
  eenb14,
  eenbdirect,
  eenbrecip,
  eenbrecipself,
  eenbrecipexcl,
  eelambda,
  eepotential,
  eekinetic,
  eetotal,
  eeend} EETerm;

struct CountType {
  int count;
  std::string type;
};
bool operator<(const CountType& a,const CountType& b);

struct BondPotential {
  int idx[2];
  int siteBlock[2];
  real kb;
  real b0;
};

struct AnglePotential {
  int idx[3];
  int siteBlock[2];
  real kangle;
  real angle0;
  real kureyb;
  real ureyb0;
};

struct DihePotential {
  int idx[4];
  int siteBlock[2];
  real kdih;
  int ndih;
  real dih0;
};

struct ImprPotential {
  int idx[4];
  int siteBlock[2];
  real kimp;
  real imp0;
};

struct CmapPotential {
  int idx[8];
  int siteBlock[3];
  int ngrid;
  real (*kcmapPtr)[4][4];
};

struct Nb14Potential {
  int idx[2];
  int siteBlock[2];
  real qxq;
  real c12;
  real c6;
};

struct NbExPotential {
  int idx[2];
  int siteBlock[2];
  real qxq;
};

struct NbondPotential {
  int siteBlock;
  real q;
  int typeIdx;
};

struct VdwPotential {
  real c12,c6;
};

struct ExclPotential {
  int idx[2];
};

class Potential {
  public:
  int atomCount;

  int bondCount;
  std::vector<struct BondPotential> bonds_tmp;
  struct BondPotential *bonds;
  struct BondPotential *bonds_d;
  int angleCount;
  std::vector<struct AnglePotential> angles_tmp;
  struct AnglePotential *angles;
  struct AnglePotential *angles_d;
  int diheCount;
  std::vector<struct DihePotential> dihes_tmp;
  struct DihePotential *dihes;
  struct DihePotential *dihes_d;
  int imprCount;
  std::vector<struct ImprPotential> imprs_tmp;
  struct ImprPotential *imprs;
  struct ImprPotential *imprs_d;
  int cmapCount;
  std::vector<struct CmapPotential> cmaps_tmp;
  struct CmapPotential *cmaps;
  struct CmapPotential *cmaps_d;

  std::map<TypeName8O,real(*)[4][4]> cmapTypeToPtr;

  int nb14Count;
  std::vector<struct Nb14Potential> nb14s_tmp;
  struct Nb14Potential *nb14s;
  struct Nb14Potential *nb14s_d;
  int nbexCount;
  std::vector<struct NbExPotential> nbexs_tmp;
  struct NbExPotential *nbexs;
  struct NbExPotential *nbexs_d;

  std::set<int> *bondExcl;
  std::set<int> *angleExcl;
  std::set<int> *diheExcl;
  std::set<int> *msldExcl;
  std::set<int> *allExcl;
  // std::vector<Int2> excls_tmp;
  // struct Int2 *excls;
  // struct Int2 *excls_d;

  int exclCount;
  std::vector<struct ExclPotential> excls_tmp;
  struct ExclPotential *excls;
  struct ExclPotential *excls_d;

  real *charge;
  real *charge_d;

  int gridDimPME[3];
  cufftReal *chargeGridPME_d;
  cufftComplex *fourierGridPME_d;
  cufftReal *potentialGridPME_d;
  real *bGridPME, *bGridPME_d;
  cufftHandle planFFTPME, planIFFTPME;
  size_t bufferSizeFFTPME,bufferSizeIFFTPME;

  std::map<std::string,int> typeCount;
  std::set<struct CountType> typeSort;
  std::vector<std::string> typeList;
  std::map<std::string,int> typeLookup;

  struct NbondPotential *nbonds;
  struct NbondPotential *nbonds_d;
  int vdwParameterCount;
  struct VdwPotential *vdwParameters;
  struct VdwPotential *vdwParameters_d;

  Potential();
  ~Potential();

  void initialize(System *system);

  void calc_force(int step,System *system);
};

#endif
