#ifndef SYSTEM_PARAMETERS_H
#define SYSTEM_PARAMETERS_H

#include <stdio.h>

#include <map>
#include <string>
#include <vector>
#include <set>

#include "main/defines.h"

// Forward declarations:
class System;

class TypeName2 {
  public:
  std::string t[2];
};
bool operator==(const TypeName2& a,const TypeName2& b);
bool operator<(const TypeName2& a,const TypeName2& b);

class TypeName3 {
  public:
  std::string t[3];
};
bool operator==(const TypeName3& a,const TypeName3& b);
bool operator<(const TypeName3& a,const TypeName3& b);

class TypeName4 {
  public:
  std::string t[4];
};
bool operator==(const TypeName4& a,const TypeName4& b);
bool operator<(const TypeName4& a,const TypeName4& b);

class TypeName8O { // O is for ordered, this one is not symmetric upon reversal
  public:
  std::string t[8];
};
bool operator==(const TypeName8O& a,const TypeName8O& b);
bool operator<(const TypeName8O& a,const TypeName8O& b);

struct BondParameter {
  real kb;
  real b0;
};

struct AngleParameter {
  real kangle;
  real angle0;
  real kureyb;
  real ureyb0;
};

struct DiheParameter {
  real kdih;
  int ndih;
  real dih0;
};

struct ImprParameter {
  real kimp;
  real imp0;
};

/*
class CmapParameter {
  int ngrid;
  public:
  real *kcmap;

  void CmapParameter()
  {
    ngrid=0;
    kcmap=NULL;
  }
  void ~CmapParameter()
  {
    if (kcmal!=NULL) {
      delete [] kcmap;
    }
  }
  void setNgrid(int in)
  {
    ngrid=in;
    if (kcmap!=NULL) {
      delete [] kcmap;
    }
    kcmap=new real[ngrid*ngrid];
  }
  real* operator[](std::size_t idx) {return kcmap+ngrid*idx;}
  int getNgrid() {return ngrid;}
};*/
struct CmapParameter {
  int ngrid;
  real kcmap[24][24];
};

struct NbondParameter {
  real eps;
  real sig;
  real eps14;
  real sig14;
};

class Parameters {
  public:
    int atomTypeCount;
    std::map<std::string,int> atomTypeMap;
    std::vector<std::string> atomType;
    std::map<std::string,real> atomMass;

    std::map<TypeName2,struct BondParameter> bondParameter;
    std::map<TypeName3,struct AngleParameter> angleParameter;
    std::map<TypeName4,std::vector<struct DiheParameter>> diheParameter;
    std::map<TypeName4,struct ImprParameter> imprParameter;
    std::map<TypeName8O,struct CmapParameter> cmapParameter;
    std::map<std::string,struct NbondParameter> nbondParameter;
    std::map<TypeName2,struct NbondParameter> nbfixParameter;

    int maxDiheTerms; // Number of Fourier componenets in the biggest dihedral
    std::set<std::string> knownTokens;

  Parameters() {
    atomTypeCount=0;
    atomTypeMap.clear();
    atomTypeMap["X"]=-1;
    atomType.clear();
    atomMass.clear();

    bondParameter.clear();
    angleParameter.clear();
    diheParameter.clear();
    imprParameter.clear();
    cmapParameter.clear();
    nbondParameter.clear();
    nbfixParameter.clear();

    maxDiheTerms=0;
    knownTokens.clear();
    knownTokens.insert("ATOM");
    knownTokens.insert("BOND");
    knownTokens.insert("ANGL");
    knownTokens.insert("DIHE");
    knownTokens.insert("IMPR");
    knownTokens.insert("CMAP");
    knownTokens.insert("NONB");
    knownTokens.insert("NBFI");
    knownTokens.insert("HBON");
    knownTokens.insert("END");
  }

  ~Parameters() {
    atomTypeMap.clear();
    atomType.clear();
    atomMass.clear();

    bondParameter.clear();
    angleParameter.clear();
    diheParameter.clear();
    imprParameter.clear();
    cmapParameter.clear();
    nbondParameter.clear();
    nbfixParameter.clear();
    knownTokens.clear();
  }

  // Defined in parameters.cxx
  void add_parameter_file(FILE *fp);
  void add_parameter_atoms(FILE *fp);
  void add_parameter_bonds(FILE *fp);
  void add_parameter_angles(FILE *fp);
  void add_parameter_dihes(FILE *fp);
  void add_parameter_imprs(FILE *fp);
  void add_parameter_cmaps(FILE *fp);
  void add_parameter_nbonds(char *line,FILE *fp);
  void add_parameter_nbfixs(FILE *fp);
  void dump();
  std::string check_type_name(std::string type,const char *tag);
  std::string require_type_name(std::string type,const char *tag);

};

void parse_parameters(char *line,System *system);

#endif
