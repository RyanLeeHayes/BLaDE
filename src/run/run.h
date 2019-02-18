#ifndef RUN_RUN_H
#define RUN_RUN_H

#include <map>
#include <string>

#include "main/defines.h"

#include "xdr/xdrfile.h"
#include "xdr/xdrfile_xtc.h"

// Forward delcaration
class System;

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

  Run()
  {
    step0=0;
    nsteps=5000;
    dt=0.001; // ps
#warning "Temperature and friction automatically set"
    T=300; // K
    gamma=1.0; // ps^-1
    fnmXTC="default.xtc";
    fnmLMD="default.lmd";
    fnmNRG="default.nrg";
    fpXTC=NULL;
    fpLMD=NULL;
    fpNRG=NULL;
    freqXTC=1000;
    freqLMD=10;
    freqNRG=10;
    setup_parse_run();
  }

  ~Run()
  {
    if (fpXTC) xdrfile_close(fpXTC);
    if (fpLMD) fclose(fpLMD);
    if (fpNRG) fclose(fpNRG);
  }

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
