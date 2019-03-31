#include <string.h>

#include "run/run.h"
#include "system/system.h"
#include "io/io.h"
#include "msld/msld.h"
#include "system/state.h"
#include "system/potential.h"
#include "domdec/domdec.h"



#define PROFILESERIAL

// Class constructors
Run::Run()
{
  step0=0;
  nsteps=5000;
  dt=0.001*PICOSECOND; // ps
  T=300; // K
  gamma=1.0/PICOSECOND; // ps^-1
  fnmXTC="default.xtc";
  fnmLMD="default.lmd";
  fnmNRG="default.nrg";
  fpXTC=NULL;
  fpLMD=NULL;
  fpNRG=NULL;
  freqXTC=1000;
  freqLMD=10;
  freqNRG=10;
// Nonbonded options
  betaEwald=1/(3.2*ANGSTROM); // rCut=10*ANGSTROM, erfc(betaEwald*rCut)=1e-5
  rCut=10*ANGSTROM;
  rSwitch=8.5*ANGSTROM;
  gridSpace=1.0*ANGSTROM;
  orderEwald=6;

  cutoffs.betaEwald=betaEwald;
  cutoffs.rCut=rCut;
  cutoffs.rSwitch=rSwitch;

  shakeTolerance=2e-7; // floating point precision is only 1.2e-7

  freqNPT=50;
  volumeFluctuation=100*ANGSTROM*ANGSTROM*ANGSTROM;
  pressure=1*ATMOSPHERE;

#ifdef PROFILESERIAL
  updateStream=0;
#else
  cudaStreamCreate(&updateStream);
#endif
  cudaEventCreate(&forceBegin);
  cudaEventCreate(&forceComplete);
  setup_parse_run();

#ifdef PROFILESERIAL
  bondedStream=0;
  biaspotStream=0;
  nbdirectStream=0;
  nbrecipStream=0;
#else
  cudaStreamCreate(&bondedStream);
  cudaStreamCreate(&biaspotStream);
  cudaStreamCreate(&nbdirectStream);
  cudaStreamCreate(&nbrecipStream);
#endif
  cudaEventCreate(&bondedComplete);
  cudaEventCreate(&biaspotComplete);
  cudaEventCreate(&nbdirectComplete);
  cudaEventCreate(&nbrecipComplete);
}

Run::~Run()
{
  if (fpXTC) xdrfile_close(fpXTC);
  if (fpLMD) fclose(fpLMD);
  if (fpNRG) fclose(fpNRG);
#ifndef PROFILESERIAL
  cudaStreamDestroy(updateStream);
#endif
  cudaEventDestroy(forceBegin);
  cudaEventDestroy(forceComplete);

#ifndef PROFILESERIAL
  cudaStreamDestroy(bondedStream);
  cudaStreamDestroy(biaspotStream);
  cudaStreamDestroy(nbdirectStream);
  cudaStreamDestroy(nbrecipStream);
#endif
  cudaEventDestroy(bondedComplete);
  cudaEventDestroy(biaspotComplete);
  cudaEventDestroy(nbdirectComplete);
  cudaEventDestroy(nbrecipComplete);
}



// Parsing functions
void parse_run(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->run) {
    system->run=new Run();
  }

  io_nexta(line,token);
  name=token;
  if (system->run->parseRun.count(name)==0) name="";
  // So much for function pointers being elegant.
  // call the function pointed to by: system->run->parseRun[name]
  // within the object: system->run
  // with arguments: (line,token,system)
  (system->run->*(system->run->parseRun[name]))(line,token,system);
}

void Run::setup_parse_run()
{
  parseRun[""]=&Run::error;
  helpRun[""]="If you see this string, something went wrong.\n";
  parseRun["help"]=&Run::help;
  helpRun["help"]="?run help [directive]> Prints help on run directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
  parseRun["print"]=&Run::dump;
  helpRun["print"]="?run print> Prints out some of the data in the run data structure\n";
  parseRun["reset"]=&Run::reset;
  helpRun["reset"]="?run reset> Resets the run data structure to it's default values\n";
  parseRun["setvariable"]=&Run::set_variable;
  helpRun["setvariable"]="?run setvariable \"name\" \"value\"> Set the variable \"name\" to \"value\". Available \"name\"s are: dt (time step in ps), nsteps (number of steps of dynamics to run), fnmxtc (filename for the coordinate output), fnmlmd (filename for the lambda output), fnmnrg (filename for the energy output)\n";
  parseRun["dynamics"]=&Run::dynamics;
  helpRun["dynamics"]="?run dynamics> Run dynamics with the options set by \"run setvariable\"\n";
}

void Run::help(char *line,char *token,System *system)
{
  std::string name=io_nexts(line);
  if (name=="") {
    fprintf(stdout,"?run> Available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpRun.begin(); ii!=helpRun.end(); ii++) {
      fprintf(stdout," %s",ii->first.c_str());
    }
    fprintf(stdout,"\n");
  } else if (helpRun.count(token)==1) {
    fprintf(stdout,helpRun[name].c_str());
  } else {
    error(line,token,system);
  }
}

void Run::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token after run: %s\n",token);
}

void Run::dump(char *line,char *token,System *system)
{
  fprintf(stdout,"RUN PRINT> dt=%f (time step input in ps)\n",dt);
  fprintf(stdout,"RUN PRINT> T=%f (temperature in K)\n",T);
  fprintf(stdout,"RUN PRINT> gamma=%f (friction input in ps^-1)\n",gamma);
  fprintf(stdout,"RUN PRINT> nsteps=%d (number of time steps for dynamics)\n",nsteps);
  fprintf(stdout,"RUN PRINT> fnmxtc=%s (file name for coordinate trajectory)\n",fnmXTC.c_str());
  fprintf(stdout,"RUN PRINT> fnmlmd=%s (file name for lambda trajectory)\n",fnmLMD.c_str());
  fprintf(stdout,"RUN PRINT> fnmnrg=%s (file name for energy output)\n",fnmNRG.c_str());
  fprintf(stdout,"RUN PRINT> betaEwald=%f (input invbetaewald in A)\n",betaEwald);
  fprintf(stdout,"RUN PRINT> rcut=%f (input in A)\n",rCut);
  fprintf(stdout,"RUN PRINT> rswitch=%f (input in A)\n",rSwitch);
  fprintf(stdout,"RUN PRINT> gridspace=%f (For PME - input in A)\n",gridSpace);
  fprintf(stdout,"RUN PRINT> orderewald=%d (PME interpolation order, dimensionless. 4, 6, 8, or 10 supported, 6 recommended)\n",orderEwald);
  fprintf(stdout,"RUN PRINT> shaketolerance=%f (For use with shake - dimensionless - do not go below 1e-7 with single precision)\n",shakeTolerance);
  fprintf(stdout,"RUN PRINT> freqnpt=%d (frequency of pressure coupling moves. 10 or less reproduces bulk dynamics, OpenMM often uses 100)\n",freqNPT);
  fprintf(stdout,"RUN PRINT> volumefluctuation=%f (rms volume move for pressure coupling, input in A^3, recommend sqrt(V*(1 A^3)), rms fluctuations are typically sqrt(V*(2 A^3))\n",volumeFluctuation);
  fprintf(stdout,"RUN PRINT> pressure=%f (pressure for pressure coupling, input in atmospheres)\n",pressure);
}

void Run::reset(char *line,char *token,System *system)
{
  delete system->run;
  system->run=NULL;
}

void Run::set_variable(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"dt")==0) {
    dt=PICOSECOND*io_nextf(line);
  } else if (strcmp(token,"nsteps")==0) {
    nsteps=io_nexti(line);
  } else if (strcmp(token,"fnmxtc")==0) {
    fnmXTC=io_nexts(line);
  } else if (strcmp(token,"fnmlmd")==0) {
    fnmLMD=io_nexts(line);
  } else if (strcmp(token,"fnmnrg")==0) {
    fnmNRG=io_nexts(line);
  } else if (strcmp(token,"freqxtc")==0) {
    freqXTC=io_nexti(line);
  } else if (strcmp(token,"freqlmd")==0) {
    freqLMD=io_nexti(line);
  } else if (strcmp(token,"freqnrg")==0) {
    freqNRG=io_nexti(line);
  } else if (strcmp(token,"T")==0) {
    T=io_nextf(line);
  } else if (strcmp(token,"gamma")==0) {
    gamma=io_nextf(line)/PICOSECOND;
  } else if (strcmp(token,"invbetaewald")==0) {
    betaEwald=1/(io_nextf(line)*ANGSTROM);
    cutoffs.betaEwald=betaEwald;
  } else if (strcmp(token,"rcut")==0) {
    rCut=io_nextf(line)*ANGSTROM;
    cutoffs.rCut=rCut;
  } else if (strcmp(token,"rswitch")==0) {
    rSwitch=io_nextf(line)*ANGSTROM;
    cutoffs.rSwitch=rSwitch;
  } else if (strcmp(token,"gridspace")==0) {
    gridSpace=io_nextf(line)*ANGSTROM;
  } else if (strcmp(token,"orderewald")==0) {
    orderEwald=io_nexti(line);
    if ((orderEwald/2)*2!=orderEwald) fatal(__FILE__,__LINE__,"orderEwald (%d) must be even\n",orderEwald);
    if (orderEwald<4 || orderEwald>10) fatal(__FILE__,__LINE__,"orderEwald (%d) must be 4, 6, 8, or 10\n",orderEwald);
  } else if (strcmp(token,"shaketolerance")==0) {
    shakeTolerance=io_nextf(line);
  } else if (strcmp(token,"freqnpt")==0) {
    freqNPT=io_nexti(line);
  } else if (strcmp(token,"volumefluctuation")==0) {
    volumeFluctuation=io_nextf(line)*ANGSTROM*ANGSTROM*ANGSTROM;
  } else if (strcmp(token,"pressure")==0) {
    pressure=io_nextf(line)*ATMOSPHERE;
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token %s in run setvariable command\n",token);
  }
}

void Run::dynamics(char *line,char *token,System *system)
{
  // Initialize data structures
  dynamics_initialize(system);

  // Run dynamics
  for (step=step0; step<step0+nsteps; step++) {
    fprintf(stdout,"Step %d\n",step);
    system->domdec->update_domdec(system,(step%system->domdec->freqDomdec)==0);
    system->potential->calc_force(step,system);
    system->state->update(step,system);
#warning "Need to copy coordinates before update"
    print_dynamics_output(step,system);

    // NYI check gpu
    if (cudaPeekAtLastError() != cudaSuccess) {
      cudaError_t err=cudaPeekAtLastError();
      fatal(__FILE__,__LINE__,"GPU error code %d during run propogation of MPI rank %d\n%s\n",err,system->id,cudaGetErrorString(err));
    }
  }

  dynamics_finalize(system);
}

void Run::dynamics_initialize(System *system)
{
  // Open files
  if (!fpXTC) {
    fpXTC=xdrfile_open(fnmXTC.c_str(),"w");
    if (!fpXTC) {
      fatal(__FILE__,__LINE__,"Failed to open XTC file %s\n",fnmXTC.c_str());
    }
  }
  if (!fpLMD) fpLMD=fpopen(fnmLMD.c_str(),"w");
  if (!fpNRG) fpNRG=fpopen(fnmNRG.c_str(),"w");

  // Finish setting up MSLD
  system->msld->initialize(system); 

  // Set up update structures
  if (system->state) delete system->state;
  system->state=new State(system);
  system->state->initialize(system);

  // Set up potential structures
  if (system->potential) delete system->potential;
  system->potential=new Potential();
  system->potential->initialize(system);

  //NYI read checkpoint

  // Set up domain decomposition
  if (system->domdec) delete system->domdec;
  system->domdec=new Domdec();
  system->domdec->initialize(system);

  // NYI check gpu
  if (cudaPeekAtLastError() != cudaSuccess) {
    cudaError_t err=cudaPeekAtLastError();
    fatal(__FILE__,__LINE__,"GPU error code %d during run initialization of MPI rank %d\n%s\n",err,system->id,cudaGetErrorString(err));
  }
}

void Run::dynamics_finalize(System *system)
{
  //NYI write checkpoint
  step0=step;
}

