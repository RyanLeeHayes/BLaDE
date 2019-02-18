#ifndef SYSTEM_STATE_H
#define SYSTEM_STATE_H

#include <stdio.h>
#include <math.h>

#include <string>
#include <map>

// Forward delcarations
class System;

struct AtomState {
  std::string segName;
  int resIdx;
  std::string atomName;
};
bool operator<(const struct AtomState& a,const struct AtomState& b);

class State {
  public:
  std::map<std::string,void(State::*)(char*,char*,System*)> parseState;
  std::map<std::string,std::string> helpState;

  std::map<struct AtomState,Real3> fileData;

  int atomCount;
  real box[3][3];
  real (*position)[3];
  float (*fposition)[3]; // Intentional float
  real (*velocity)[3];
  real (*force)[3];

  State(int n) {
    atomCount=n;
    box[0][0]=NAN;
    position=(real(*)[3])calloc(n,sizeof(real[3]));
#ifdef DOUBLE
    fposition=(float(*)[3])calloc(n,sizoef(float[3]));
#else
    fposition=position;
#endif
    velocity=(real(*)[3])calloc(n,sizeof(real[3]));
    force=(real(*)[3])calloc(n,sizeof(real[3]));
    setup_parse_state();
    fprintf(stdout,"IMPLEMENT State create %s %d\n",__FILE__,__LINE__);
  }

  ~State() {
    if (position) free(position);
#ifdef DOUBLE
    if (fposition) free(fposition);
#endif
    if (velocity) free(velocity);
    if (force) free(force);
  }

  void setup_parse_state();

  void help(char *line,char *token,System *system);
  void error(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);
  void file(char *line,char *token,System *system);
  void parse_box(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);

  void file_pdb(FILE *fp,System *system);
};

void parse_state(char *line,System *system);

#endif
