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

class Structure {
  public:
  int atomCount;
  std::vector<struct AtomStructure> atomList;

  int *atomTypeIdx;
  real *charge;
  real *mass;

  int bondCount;
  std::vector<struct Int2> bondList;

  int angleCount;
  std::vector<struct Int3> angleList;

  int diheCount;
  std::vector<struct Int4> diheList;

  int imprCount;
  std::vector<struct Int4> imprList;

  int cmapCount;
  std::vector<struct Int8> cmapList;

  bool shakeHbond;
  
  Structure();
  ~Structure();

  void add_structure_psf_file(FILE *fp);
  void dump();
};

void parse_structure(char *line,System *system);

#endif
