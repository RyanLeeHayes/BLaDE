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
  std::string resIdx;
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
  int particleBoxName;
  real3_x particleBoxABC;
  real3_x particleBoxAlBeGa;
  real_x (*particlePosition)[3];
  real_v (*particleVelocity)[3];

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
  void file_crd(FILE *fp,System *system);
};

void parse_coordinates(char *line,System *system);

// Library functions
extern "C" {
  void blade_init_coordinates(System *system,int n);
  void blade_dest_coordinates(System *system);
  void blade_add_coordinates_position(System *system,int i,double x,double y,double z);
  void blade_add_coordinates_velocity(System *system,int i,double vx,double vy,double vz);
  void blade_add_coordinates_box(System *system,int name,double a,double b,double c,double alpha,double beta,double gamma);
}

#endif
