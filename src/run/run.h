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
  XDRFILE *fpXLMD;
  FILE *fpLMD;
  FILE *fpNRG;
  int freqXTC;
  int freqLMD;
  int freqNRG;
  bool hrLMD;
  bool prettyXTC;

  long int step; // current step
  long int step0; // starting step
  int nsteps; // steps in next dynamics call
  real dt;
  real T;
  real gamma;

  real betaEwald;
  real rCut;
  real rSwitch;
  bool vfSwitch;   //added by clb3
  bool usePME;
  real gridSpace; // grid spacing for PME calculation
  int grid[3];
  int orderEwald; // interpolation order (4, 6, or 8 typically)
  struct Cutoffs cutoffs;
  real shakeTolerance;

  int freqNPT;
  real volumeFluctuation;
  real pressure;

  bool domdecHeuristic;

  std::map<std::string,int> termStringToInt;
  std::map<int,bool> calcTermFlag;

#ifdef REPLICAEXCHANGE
  std::string fnmREx;
  FILE *fpREx;
  int freqREx;
  int replica;
#endif

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
  // cudaEvent_t forceComplete;
  cudaEvent_t communicate;
  cudaEvent_t *communicate_omp;


  Run(System *system);
  ~Run();

  void setup_parse_run();

  void error(char *line,char *token,System *system);
  void help(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);

  void set_variable(char *line,char *token,System *system);
  void set_term(char *line,char *token,System *system);
  void energy(char *line,char *token,System *system);
  void test(char *line,char *token,System *system);
  void dynamics(char *line,char *token,System *system);

  void dynamics_initialize(System *system);
  void dynamics_finalize(System *system);
};

void parse_run(char *line,System *system);

// Library functions
extern "C" {
  void blade_init_run(System *system);
  void blade_dest_run(System *system);
  void blade_add_run_flags(System *system,
    double gamma, double betaEwald, double rCut, double rSwitch,
    int vdWfSwitch, int elecPME,
    double gridSpace, int gridx, int gridy, int gridz,
    int orderEwald, double shakeTolerance);
  void blade_add_run_dynopts(System *system,
    int step, int step0, int nsteps, double dt, double T,
    int freqNPT, double volumeFluctuation, double pressure);
  void blade_run_energy(System *system);
}

#endif
