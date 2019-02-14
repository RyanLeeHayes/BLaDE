#ifndef SYSTEM_STATE_H
#define SYSTEM_STATE_H

#include <stdio.h>
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

  std::map<struct AtomState,Double3> fileData;

  int atomCount;
  double (*position)[3];
  double (*velocity)[3];
  double (*force)[3];

  State(int n) {
    atomCount=n;
    position=(double(*)[3])calloc(n,sizeof(double[3]));
    velocity=(double(*)[3])calloc(n,sizeof(double[3]));
    force=(double(*)[3])calloc(n,sizeof(double[3]));
    setup_parse_state();
    fprintf(stdout,"IMPLEMENT State create %s %d\n",__FILE__,__LINE__);
  }

  ~State() {
    if (position) free(position);
    if (velocity) free(velocity);
    if (force) free(force);
  }

  void setup_parse_state();

  void help(char *line,char *token,System *system);
  void error(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);
  void file(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);

  void file_pdb(FILE *fp,System *system);
};

void parse_state(char *line,System *system);

#endif
