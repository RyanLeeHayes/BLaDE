#ifndef SYSTEM_STRUCTURE_H
#define SYSTEM_STRUCTURE_H

#include <stdio.h>
#include <string>
#include <vector>

#include "main/defines.h"

// Forward declarations:
class System;

struct AtomStructure {
  int atomIdx;
  std::string segName;
  int resIdx;
  std::string resName;
  std::string atomName;
  std::string atomTypeName;
  real charge;
  real mass;
};

struct BondStructure {
  int idx[2];
  real kb;
  real b0;
};

struct AngleStructure {
  int idx[3];
  real kangle;
  real angle0;
  real kureyb;
  real ureyb0;
};

struct DiheStructure {
  int idx[4];
  real kdih;
  int ndih;
  real dih0;
};

struct ImprStructure {
  int idx[4];
  real kimp;
  real imp0;
};

struct CmapStructure {
  int idx[8];
  int ngrid;
  int kcmapIndex;
};

class Structure {
  public:
  int atomCount;
  std::vector<struct AtomStructure> atomList;

  int *atomTypeIdx;
  real *charge;
  real *mass;

  std::vector<struct Int2> bondList; // std::vector<int[2]> bondList;
  int bondCount;
  struct BondStructure *bonds;

  std::vector<struct Int3> angleList;
  int angleCount;
  struct AngleStructure *angles;

  std::vector<struct Int4> diheList;
  int diheCount;
  struct DiheStructure *dihes;

  std::vector<struct Int4> imprList;
  int imprCount;
  struct ImprStructure *imprs;

  std::vector<struct Int8> cmapList;
  int cmapCount;
  struct CmapStructure *cmaps;
  
  Structure() {
    atomCount=0;

    atomTypeIdx=NULL;
    charge=NULL;
    mass=NULL;

    bondCount=0;
    bonds=NULL;
    angleCount=0;
    angles=NULL;
    diheCount=0;
    dihes=NULL;
    imprCount=0;
    imprs=NULL;
    cmapCount=0;
    cmaps=NULL;
  }

  ~Structure() {
    if (atomTypeIdx) free(atomTypeIdx);
    if (charge) free(charge);
    if (mass) free(mass);
    if (bonds) free(bonds);
    if (angles) free(angles);
    if (dihes) free(dihes);
    if (imprs) free(imprs);
    if (cmaps) free(cmaps);
  }

  void add_structure_psf_file(FILE *fp);
  void setup(System *system);
  void dump();
};

void parse_structure(char *line,System *system);

#endif
