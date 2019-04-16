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
  std::string fnmCPI;
  std::string fnmCPO;
  XDRFILE *fpXTC;
  FILE *fpLMD;
  FILE *fpNRG;
  int freqXTC;
  int freqLMD;
  int freqNRG;

  long int step; // current step
  long int step0; // starting step
  int nsteps; // steps in next dynamics call
  real dt;
  real T;
  real gamma;

  real betaEwald;
  real rCut;
  real rSwitch;
  real gridSpace; // grid spacing for PME calculation
  int orderEwald; // interpolation order (4, 6, or 8 typically)
  struct Cutoffs cutoffs;
  real shakeTolerance;

  int freqNPT;
  real volumeFluctuation;
  real pressure;

  cudaStream_t updateStream;
  cudaStream_t bondedStream;
  cudaStream_t biaspotStream;
  cudaStream_t nbdirectStream;
  cudaStream_t nbrecipStream;

  cudaEvent_t forceBegin;
  cudaEvent_t bondedComplete;
  cudaEvent_t biaspotComplete;
  cudaEvent_t nbdirectComplete;
  cudaEvent_t nbrecipComplete;
  cudaEvent_t forceComplete;


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
