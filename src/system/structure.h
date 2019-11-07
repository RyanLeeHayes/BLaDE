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
  std::map<std::string,void(Structure::*)(char*,char*,System*)> parseStructure;
  std::map<std::string,std::string> helpStructure;

  int atomCount;
  std::vector<struct AtomStructure> atomList;

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

  void setup_parse_structure();
  void help(char *line,char *token,System *system);
  void error(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);
  void file(char *line,char *token,System *system);
  void parse_shake(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);
  void add_structure_psf_file(FILE *fp);
};

void parse_structure(char *line,System *system);

// Library functions
extern "C" {
  void blade_add_atom(System *system,
    int atomIdx,const char *segName,int resIdx,const char *resName,
    const char *atomName,const char *atomTypeName,double charge,double mass);
  void blade_add_bond(System *system,int i,int j);
  void blade_add_angle(System *system,int i,int j,int k);
  void blade_add_dihe(System *system,int i,int j,int k,int l);
  void blade_add_impr(System *system,int i,int j,int k,int l);
  void blade_add_cmap(System *system,int i1,int j1,int k1,int l1,int i2,int j2,int k2,int l2);
}

#endif
