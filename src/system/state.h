#ifndef SYSTEM_STATE_H
#define SYSTEM_STATE_H

#include <stdio.h>
#include <cuda_runtime.h>

#include <string>
#include <map>

#include "main/defines.h"

// Forward delcarations
class System;
class RngCPU;
class RngGPU;

typedef enum eeterm {
  eebond,
  eeangle,
  eedihe,
  eeimpr,
  eecmap,
  eenbdirect,
  eenbrecip,
  eepotential,
  eekinetic,
  eetotal,
  eeend} EETerm;

struct AtomState {
  std::string segName;
  int resIdx;
  std::string atomName;
};
bool operator<(const struct AtomState& a,const struct AtomState& b);
bool operator==(const struct AtomState& a,const struct AtomState& b);

class State {
  public:
  std::map<std::string,void(State::*)(char*,char*,System*)> parseState;
  std::map<std::string,std::string> helpState;

  std::map<struct AtomState,Real3> fileData;

  RngCPU *rngCPU;
  RngGPU *rngGPU;

  int atomCount;
  real (*box)[3]; // [3][3]
  real (*position)[3];
  float (*fposition)[3]; // Intentional float
  real (*velocity)[3];
  real (*force)[3];
  real (*mass)[3];
  real (*invsqrtMass)[3];
  real *energy;
// Device versions
  real (*box_d)[3];
  real (*position_d)[3];
  real (*velocity_d)[3];
  real (*force_d)[3];
  real (*mass_d)[3];
  real (*invsqrtMass_d)[3];
  real (*random_d)[3];
  real *energy_d;

  real3 orthBox;

  State(int n,System *system);
  ~State();

  void setup_parse_state();

  void help(char *line,char *token,System *system);
  void error(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);
  void file(char *line,char *token,System *system);
  void parse_box(char *line,char *token,System *system);
  void parse_velocity(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);

  void file_pdb(FILE *fp,System *system);

  void send_box();
  void recv_box();
  void send_position();
  void recv_position();
  void send_velocity();
  void recv_velocity();
  void send_invsqrtMass();
  void recv_invsqrtMass();
  void send_energy();
  void recv_energy();
};

void parse_state(char *line,System *system);

#endif
