#ifndef RUN_RUN_H
#define RUN_RUN_H

#include <map>
#include <string>

#include "main/defines.h"

#include "xdr/xdrfile.h"
#include "xdr/xdrfile_xtc.h"

// Forward delcaration
class System;

struct Cutoffs {
  real betaEwald;
  real rCut;
  real rSwitch;
};

class Run {
  public:
  std::map<std::string,void(Run::*)(char*,char*,System*)> parseRun;
  std::map<std::string,std::string> helpRun;

  std::string fnmXTC;
  std::string fnmLMD;
  std::string fnmNRG;
  XDRFILE *fpXTC;
  FILE *fpLMD;
  FILE *fpNRG;
  int freqXTC;
  int freqLMD;
  int freqNRG;
// NYI - read and write checkpoint files

  int step; // current step
  int step0; // starting step
  int nsteps; // steps in next dynamics call
  real dt;
  real T;
  real gamma;

  real betaEwald;
  real rCut;
  real rSwitch;
  real gridSpace; // grid spacing for PME calculation
  struct Cutoffs cutoffs;

  cudaStream_t masterStream;
  cudaEvent_t forceComplete;
  cudaEvent_t updateComplete;

  Run();
  ~Run();

  void setup_parse_run();

  void error(char *line,char *token,System *system);
  void help(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);

  void set_variable(char *line,char *token,System *system);
  void dynamics(char *line,char *token,System *system);

  void dynamics_initialize(System *system);
  void dynamics_finalize(System *system);
};

void parse_run(char *line,System *system);

#endif
