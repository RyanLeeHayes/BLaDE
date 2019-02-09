#ifndef SYSTEM_STRUCTURE_H
#define SYSTEM_STRUCTURE_H

#include <stdio.h>
#include <string>
#include <new>

// Forward declarations:
class System;

class Structure {
  public:
  int atomCount;
  int *atomIdx;
  std::string *segName;
  int *resIdx;
  std::string *resName;
  std::string *atomName;
  std::string *atomTypeName;
  int *atomTypeIdx;
  double *charge;
  double *mass;

  int bondCount;
  int (*bonds)[2];
  int angleCount;
  int (*angles)[3];
  int diheCount;
  int (*dihes)[4];
  int imprCount;
  int (*imprs)[4];
  int cmapCount;
  int (*cmaps)[8];
  
  Structure() {
    atomCount=0;
    atomIdx=NULL;
    segName=NULL;
    resIdx=NULL;
    resName=NULL;
    atomName=NULL;
    atomTypeName=NULL;
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
    if (atomIdx!=NULL) free(atomIdx);
    if (segName!=NULL) delete[] segName;
    if (resIdx!=NULL) free(resIdx);
    if (resName!=NULL) delete[]resName;
    if (atomName!=NULL) delete[]atomName;
    if (atomTypeName!=NULL) delete[]atomTypeName;
    if (atomTypeIdx!=NULL) free(atomTypeIdx);
    if (charge!=NULL) free(charge);
    if (mass!=NULL) free(mass);
    if (bonds!=NULL) free(bonds);
    if (angles!=NULL) free(angles);
    if (dihes!=NULL) free(dihes);
    if (imprs!=NULL) free(imprs);
    if (cmaps!=NULL) free(cmaps);
  }

  void add_structure_psf_file(FILE *fp);
  void dump();
};

void parse_structure(char *line,System *system);

#endif
