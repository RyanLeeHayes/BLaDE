#ifndef SYSTEM_PARAMETERS_H
#define SYSTEM_PARAMETERS_H

#include <stdio.h>

#include <map>
#include <string>
#include <vector>

// Forward declarations:
class System;

struct BondParameter {
  int i;
  int j;
  double kb;
  double b0;
};

struct AngleParameter {
  int i;
  int j;
  int k;
  double kangle;
  double angle0;
  double kureyb;
  double ureyb0;
};

struct DihParameter {
  int i;
  int j;
  int k;
  int l;
  double kdih;
  int ndih;
  double dih0;
};

struct ImpParameter {
  int i;
  int j;
  int k;
  int l;
  double kimp;
  double imp0;
};

struct CmapParameter {
  int i1,j1,k1,l1;
  int i2,j2,k2,l2;
  int ngrid;
  double kcmap[24][24];
};

struct NonbondedParameter {
  int i;
  double eps;
  double sig;
  double eps14;
  double sig14;
};

struct NbfixParameter {
  int i;
  int j;
  double eps;
  double sig;
  double eps14;
  double sig14;
};

class Parameters {
  public:
    int atomTypeCount;
    std::map<std::string,int> atomTypeMap;
    std::vector<std::string> atomType;
    std::vector<double> atomMass;

    int bondParameterCount;
    std::vector<struct BondParameter> bondParameter;

    int angleParameterCount;
    std::vector<struct AngleParameter> angleParameter;

    int dihParameterCount;
    std::vector<struct DihParameter> dihParameter;

    int impParameterCount;
    std::vector<struct ImpParameter> impParameter;

    int cmapParameterCount;
    std::vector<struct CmapParameter> cmapParameter;

    int nonbondedParameterCount;
    std::vector<struct NonbondedParameter> nonbondedParameter;

    int nbfixParameterCount;
    std::vector<struct NbfixParameter> nbfixParameter;

  Parameters() {
    atomTypeCount=0;
    atomTypeMap.clear();
    atomTypeMap["X"]=-1;
    atomType.clear();
    atomMass.clear();

    bondParameterCount=0;
    bondParameter.clear();

    angleParameterCount=0;
    angleParameter.clear();

    dihParameterCount=0;
    dihParameter.clear();

    impParameterCount=0;
    impParameter.clear();

    cmapParameterCount=0;
    cmapParameter.clear();

    nonbondedParameterCount=0;
    nonbondedParameter.clear();

    nbfixParameterCount=0;
    nbfixParameter.clear();
  }

  ~Parameters() {
    atomTypeMap.clear();
    atomType.clear();
    atomMass.clear();

    bondParameter.clear();
    angleParameter.clear();
    dihParameter.clear();
    impParameter.clear();
    cmapParameter.clear();
    nonbondedParameter.clear();
    nbfixParameter.clear();
  }

  // Defined in parameters.cxx
  void add_parameter_file(FILE *fp);
  void add_parameter_atoms(FILE *fp);
  void add_parameter_bonds(FILE *fp);
  void add_parameter_angles(FILE *fp);
  void add_parameter_dihs(FILE *fp);
  void add_parameter_imps(FILE *fp);
  void add_parameter_cmaps(FILE *fp);
  void add_parameter_nonbondeds(char *line,FILE *fp);
  void add_parameter_nbfixs(FILE *fp);
  void dump();

};

void parse_parameters(char *line,System *system);

#endif
