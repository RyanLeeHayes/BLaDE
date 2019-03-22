#ifndef SYSTEM_COORDINATES_H
#define SYSTEM_COORDINATES_H

#include <stdio.h>
#include <cuda_runtime.h>

#include <string>
#include <map>

#include "main/defines.h"

// Forward delcarations
class System;

struct AtomCoordinates {
  std::string segName;
  int resIdx;
  std::string atomName;
};
bool operator<(const struct AtomCoordinates& a,const struct AtomCoordinates& b);
bool operator==(const struct AtomCoordinates& a,const struct AtomCoordinates& b);

class Coordinates {
  public:
  std::map<std::string,void(Coordinates::*)(char*,char*,System*)> parseCoordinates;
  std::map<std::string,std::string> helpCoordinates;

  std::map<struct AtomCoordinates,Real3> fileData;

  int atomCount;
  real (*particleBox)[3]; // [3][3]
  real3 particleOrthBox;
  real (*particlePosition)[3];
  real (*particleVelocity)[3];

  Coordinates(int n,System *system);
  ~Coordinates();

  void setup_parse_coordinates();

  void help(char *line,char *token,System *system);
  void error(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);
  void file(char *line,char *token,System *system);
  void parse_box(char *line,char *token,System *system);
  void parse_velocity(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);

  void file_pdb(FILE *fp,System *system);
};

void parse_coordinates(char *line,System *system);

#endif
