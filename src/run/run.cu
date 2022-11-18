#include <omp.h>
#include <cuda_runtime.h>
#include <string.h>

#include "run/run.h"
#include "system/system.h"
#include "io/io.h"
#include "msld/msld.h"
#include "system/state.h"
#include "system/potential.h"
#include "system/selections.h"
#include "holonomic/rectify.h"
#include "domdec/domdec.h"

#ifdef REPLICAEXCHANGE
#include <mpi.h>
#endif



// #warning "Hardcoded serial kernels"
// #define PROFILESERIAL

// Class constructors
Run::Run(System *system)
{
  step0=0;
  nsteps=5000;
  dt=0.001*PICOSECOND; // ps
  T=300; // K
  gamma=1.0/PICOSECOND; // ps^-1
  fnmXTC="default.xtc";
  fnmLMD="default.lmd";
  fnmNRG="default.nrg";
  fnmCPI="";
  fnmCPO="default.cpt";
  fpXTC=NULL;
  fpXLMD=NULL;
  fpLMD=NULL;
  fpNRG=NULL;
  freqXTC=1000;
  freqLMD=10;
  freqNRG=10;
  hrLMD=true;
  prettyXTC=false;
// Nonbonded options
  betaEwald=1/(3.2*ANGSTROM); // rCut=10*ANGSTROM, erfc(betaEwald*rCut)=1e-5
  rCut=10*ANGSTROM;
  rSwitch=8.5*ANGSTROM;
  vfSwitch=true;
  usePME=true;
  gridSpace=1.0*ANGSTROM;
  grid[0]=-1;
  grid[1]=-1;
  grid[2]=-1;
  orderEwald=6;

  cutoffs.betaEwald=betaEwald;
  cutoffs.rCut=rCut;
  cutoffs.rSwitch=rSwitch;

  shakeTolerance=2e-7; // floating point precision is only 1.2e-7

  freqNPT=50;
  volumeFluctuation=100*ANGSTROM*ANGSTROM*ANGSTROM;
  pressure=1*ATMOSPHERE;

  domdecHeuristic=true;

  termStringToInt.clear();
  termStringToInt["bond"]=eebond;
  termStringToInt["angle"]=eeangle;
  termStringToInt["urey"]=eeurey;
  termStringToInt["dihe"]=eedihe;
  termStringToInt["impr"]=eeimpr;
  termStringToInt["cmap"]=eecmap;
  termStringToInt["nb14"]=eenb14;
  termStringToInt["nbdirect"]=eenbdirect;
  termStringToInt["nbrecip"]=eenbrecip;
  termStringToInt["nbrecipself"]=eenbrecipself;
  termStringToInt["nbrecipexcl"]=eenbrecipexcl;
  termStringToInt["lambda"]=eelambda;
  termStringToInt["bias"]=eebias;
  termStringToInt["potential"]=eepotential;
  termStringToInt["kinetic"]=eekinetic;
  termStringToInt["total"]=eetotal;
  calcTermFlag.clear();
  for (int i=0; i<eeend; i++) {
    calcTermFlag[i]=true;
  }

#ifdef REPLICAEXCHANGE
  fnmREx="default.rex";
  fpREx=NULL;
  freqREx=-1;
  MPI_Comm_rank(MPI_COMM_WORLD, &replica);
#endif

#ifdef PROFILESERIAL
  updateStream=0;
  bondedStream=0;
  biaspotStream=0;
  nbdirectStream=0;
  nbrecipStream=0;
#else
  cudaStreamCreate(&updateStream);
  cudaStreamCreate(&bondedStream);
  cudaStreamCreate(&biaspotStream);
  cudaStreamCreate(&nbdirectStream);
  cudaStreamCreate(&nbrecipStream);
  // Set priorities if desired:
  // int low,high;
  // cudaDeviceGetStreamPriorityRange(&low,&high);
  // cudaStreamCreateWithPriority(&nbdirectStream,cudaStreamDefault,low);
  // cudaStreamCreateWithPriority(&nbrecipStream,cudaStreamDefault,high);
#endif
  cudaEventCreate(&forceBegin);
  cudaEventCreate(&bondedComplete);
  cudaEventCreate(&biaspotComplete);
  cudaEventCreate(&nbdirectComplete);
  cudaEventCreate(&nbrecipComplete);
  // cudaEventCreate(&forceComplete);
  cudaEventCreate(&communicate);

  if (system->idCount>0) {
    communicate_omp=(cudaEvent_t*)calloc(system->idCount,sizeof(cudaEvent_t));
#pragma omp barrier
    system->message[system->id]=(void*)&communicate;
#pragma omp barrier
    for (int i=0; i<system->idCount; i++) {
      communicate_omp[i]=((cudaEvent_t*)(system->message[i]))[0];
    }
#pragma omp barrier
  } else {
    communicate_omp=NULL;
  }

  setup_parse_run();
}

Run::~Run()
{
  if (fpXTC) xdrfile_close(fpXTC);
  if (fpXLMD) xdrfile_close(fpXLMD);
  if (fpLMD) fclose(fpLMD);
  if (fpNRG) fclose(fpNRG);
#ifndef PROFILESERIAL
  cudaStreamDestroy(updateStream);
#endif
  cudaEventDestroy(forceBegin);
  // cudaEventDestroy(forceComplete);

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
  cudaEventDestroy(communicate);
  if (communicate_omp) free(communicate_omp);
}



// Parsing functions
void parse_run(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->run) {
    system->run=new Run(system);
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
  parseRun["setterm"]=&Run::set_term;
  helpRun["setterm"]="?run setterm [term] [on|off]> Turn terms (including bond, angle, dihe, impr, nb14 nbdirect, nbrecip, nbrecipself, nbrecipexcl, lambda, and bias) on or off\n";
  parseRun["energy"]=&Run::energy;
  helpRun["energy"]="?run energy> Calculate energy of current conformation or from fnmcpi checkpoint in file\"\n";
  parseRun["test"]=&Run::test;
  helpRun["test"]="?run test [arguments]> Test first derivatives using finite differences. Valid arguments are \"alchemical [difference]\" and \"spatial [selection] [difference]\"\n";
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
  fprintf(stdout,"RUN PRINT> dt=%f (time step input in ps)\n",dt/PICOSECOND);
  fprintf(stdout,"RUN PRINT> T=%f (temperature in K)\n",T);
  fprintf(stdout,"RUN PRINT> gamma=%f (friction input in ps^-1)\n",gamma*PICOSECOND);
  fprintf(stdout,"RUN PRINT> nsteps=%d (number of time steps for dynamics)\n",nsteps);
  fprintf(stdout,"RUN PRINT> fnmxtc=%s (file name for coordinate trajectory)\n",fnmXTC.c_str());
  fprintf(stdout,"RUN PRINT> fnmlmd=%s (file name for lambda trajectory)\n",fnmLMD.c_str());
  fprintf(stdout,"RUN PRINT> fnmnrg=%s (file name for energy output)\n",fnmNRG.c_str());
  fprintf(stdout,"RUN PRINT> fnmcpi=%s (file name for reading checkpoint in, null means start without checkpoint)\n",fnmCPI.c_str());
  fprintf(stdout,"RUN PRINT> fnmcpo=%s (file name for writing out checkpoint file for later continuation)\n",fnmCPO.c_str());
  fprintf(stdout,"RUN PRINT> betaEwald=%f (input 1/invbetaewald in A^-1)\n",betaEwald*ANGSTROM);
  fprintf(stdout,"RUN PRINT> rcut=%f (input in A)\n",rCut/ANGSTROM);
  fprintf(stdout,"RUN PRINT> rswitch=%f (input in A)\n",rSwitch/ANGSTROM);
  fprintf(stdout,"RUN PRINT> vfswitch=%d\n",vfSwitch);
  fprintf(stdout,"RUN PRINT> usepme=%d\n",usePME);
  fprintf(stdout,"RUN PRINT> gridspace=%f (For PME - input in A)\n",gridSpace/ANGSTROM);
  fprintf(stdout,"RUN PRINT> grid=[%d %d %d] (For PME if gridspace<0)\n",grid[0],grid[1],grid[2]);
  fprintf(stdout,"RUN PRINT> orderewald=%d (PME interpolation order, dimensionless. 4, 6, 8, or 10 supported, 6 recommended)\n",orderEwald);
  fprintf(stdout,"RUN PRINT> shaketolerance=%f (For use with shake - dimensionless - do not go below 1e-7 with single precision)\n",shakeTolerance);
  fprintf(stdout,"RUN PRINT> freqnpt=%d (frequency of pressure coupling moves. 10 or less reproduces bulk dynamics, OpenMM often uses 100)\n",freqNPT);
  fprintf(stdout,"RUN PRINT> volumefluctuation=%f (rms volume move for pressure coupling, input in A^3, recommend sqrt(V*(1 A^3)), rms fluctuations are typically sqrt(V*(2 A^3))\n",volumeFluctuation/(ANGSTROM*ANGSTROM*ANGSTROM));
  fprintf(stdout,"RUN PRINT> pressure=%f (pressure for pressure coupling, input in atmospheres)\n",pressure/ATMOSPHERE);
  fprintf(stdout,"RUN PRINT> domdecheuristic=%d (use heuristics for domdec limits without checking their validity)\n",(int)domdecHeuristic);
#ifdef REPLICAEXCHANGE
  fprintf(stdout,"RUN PRINT> fnmrex=%s (file name for replica exchange)\n",fnmREx.c_str());
  fprintf(stdout,"RUN PRINT> freqrex=%d (frequency of replica exchange attempts. Use {rexrank} (NYI) to access 0 ordinalized replica index in script)\n",freqREx);
#endif
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
    if (fpXTC) xdrfile_close(fpXTC);
    fpXTC=NULL;
    fnmXTC=io_nexts(line);
  } else if (strcmp(token,"fnmlmd")==0) {
    if (fpXLMD) xdrfile_close(fpXLMD);
    fpXLMD=NULL;
    if (fpLMD) fclose(fpLMD);
    fpLMD=NULL;
    fnmLMD=io_nexts(line);
  } else if (strcmp(token,"fnmnrg")==0) {
    if (fpNRG) fclose(fpNRG);
    fpNRG=NULL;
    fnmNRG=io_nexts(line);
  } else if (strcmp(token,"fnmcpi")==0) {
    fnmCPI=io_nexts(line);
  } else if (strcmp(token,"fnmcpo")==0) {
    fnmCPO=io_nexts(line);
  } else if (strcmp(token,"freqxtc")==0) {
    freqXTC=io_nexti(line);
  } else if (strcmp(token,"freqlmd")==0) {
    freqLMD=io_nexti(line);
  } else if (strcmp(token,"freqnrg")==0) {
    freqNRG=io_nexti(line);
  } else if (strcmp(token,"hrlmd")==0) {
    hrLMD=io_nextb(line);
  } else if (strcmp(token,"prettyxtc")==0) {
    prettyXTC=io_nextb(line);
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
  } else if (strcmp(token,"vfswitch")==0) {
    vfSwitch=io_nextb(line);
  } else if (strcmp(token,"usepme")==0) {
    usePME=io_nextb(line);
  } else if (strcmp(token,"gridspace")==0) {
    gridSpace=io_nextf(line)*ANGSTROM;
  } else if (strcmp(token,"grid")==0) {
    grid[0]=io_nexti(line);
    grid[1]=io_nexti(line);
    grid[2]=io_nexti(line);
    gridSpace=-1;
  } else if (strcmp(token,"orderewald")==0) {
    orderEwald=io_nexti(line);
    if ((orderEwald/2)*2!=orderEwald) fatal(__FILE__,__LINE__,"orderEwald (%d) must be even\n",orderEwald);
    if (orderEwald<4 || orderEwald>8) fatal(__FILE__,__LINE__,"orderEwald (%d) must be 4, 6, or 8\n",orderEwald);
  } else if (strcmp(token,"shaketolerance")==0) {
    shakeTolerance=io_nextf(line);
  } else if (strcmp(token,"freqnpt")==0) {
    freqNPT=io_nexti(line);
  } else if (strcmp(token,"volumefluctuation")==0) {
    volumeFluctuation=io_nextf(line)*ANGSTROM*ANGSTROM*ANGSTROM;
  } else if (strcmp(token,"pressure")==0) {
    pressure=io_nextf(line)*ATMOSPHERE;
  } else if (strcmp(token,"domdecheuristic")==0) {
    domdecHeuristic=io_nextb(line);
#ifdef REPLICAEXCHANGE
  } else if (strcmp(token,"fnmrex")==0) {
    if (fpREx) fclose(fpREx);
    fpREx=NULL;
    fnmREx=io_nexts(line);
  } else if (strcmp(token,"freqrex")==0) {
    freqREx=io_nexti(line);
#endif
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token %s in run setvariable command\n",token);
  }
}

void Run::set_term(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (termStringToInt.count(token)) {
    calcTermFlag[termStringToInt[token]]=io_nextb(line);
  } else {
    fatal(__FILE__,__LINE__,"No such energy term %s to be turned on or off\n",token);
  }
}

__global__
void shift_kernel(real_x *x,real_x dx)
{
  int i=blockDim.x*blockIdx.x+threadIdx.x;
  if (i==0) {
    x[0]+=dx;
  }
}

void Run::energy(char *line,char *token,System *system)
{
  dynamics_initialize(system);
  system->potential->calc_force(0,system);
  system->state->recv_energy();
  print_nrg(0,system);
  dynamics_finalize(system);
}

void Run::test(char *line,char *token,System *system)
{
  std::string testType=io_nexts(line);
  std::string name; // (selection name for spatial test)
  real dx;
  int i,j,ij,s;
  int ij0,imax,jmax;
  real_e F,E[2];

  // Initialize data structures
  dynamics_initialize(system);

  // Calculate forces
  system->potential->calc_force(0,system);
  // Save position and forces
  system->state->backup_position();

  if (testType=="alchemical") {
    dx=io_nextf(line); // dimensionless
    ij0=0;
    imax=system->state->lambdaCount;
    jmax=1;
  } else if (testType=="spatial") {
    name=io_nexts(line);
    if (system->selections->selectionMap.count(name)==0) {
      fatal(__FILE__,__LINE__,"Error: selection %s not found for spatial derivative testing\n",name.c_str());
    }
    dx=io_nextf(line)*ANGSTROM; // ANGSTROM units
    ij0=system->state->lambdaCount;
    imax=system->state->atomCount;
    jmax=3;
  } else {
    fatal(__FILE__,__LINE__,"Error: test type %s does not match alchemical or spatial\n",testType.c_str());
  }

  for (i=0; i<imax; i++) {
    if (jmax==1 || system->selections->selectionMap[name].boolSelection[i]) {
      for (j=0; j<jmax; j++) {
        ij=ij0+i*jmax+j;
        for (s=0; s<2; s++) {
          // Shift ij by (s-0.5)*dx
          shift_kernel<<<1,1>>>(&system->state->positionBuffer_d[ij],(s-0.5)*dx);
          
          // Calculate energy
          system->domdec->update_domdec(system,0);
          system->potential->calc_force(0,system);

          // Save relevant data
          if (system->id==0) {
            system->state->recv_energy();
            E[s]=system->state->energy[eepotential];
          }

          // Restore positions
          system->state->restore_position();
        }
        if (system->id==0) {
          cudaMemcpy(&F,&system->state->forceBuffer_d[ij],sizeof(real),cudaMemcpyDeviceToHost);
          fprintf(stdout,"ij=%7d, Emin=%20.16g, Emax=%20.16g, (Emax-Emin)/dx=%20.16g, force=%20.16g\n",ij,E[0],E[1],(E[1]-E[0])/dx,F);
        }
      }
    }
  }

  dynamics_finalize(system);
}

void Run::dynamics(char *line,char *token,System *system)
{
  clock_t t1,t2;

  // Initialize data structures
  dynamics_initialize(system);

  // Run dynamics
  t1=clock();
  for (step=step0; step<step0+nsteps; step++) {
    if (system->verbose>0) {
      fprintf(stdout,"Step %d\n",step);
    }
    system->domdec->update_domdec(system,(step%system->domdec->freqDomdec)==0);
    system->potential->calc_force(step,system);
    system->state->update(step,system);
#warning "Need to copy coordinates before update"
    print_dynamics_output(step,system);

    // NYI check gpu
    if (cudaPeekAtLastError() != cudaSuccess) {
      cudaError_t err=cudaPeekAtLastError();
      fatal(__FILE__,__LINE__,"GPU error code %d during run propogation of OMP rank %d\n%s\n",err,system->id,cudaGetErrorString(err));
    }
  }
  t2=clock();
  fprintf(stdout,"Elapsed dynamics time: %f\n",(t2-t1)*1.0/CLOCKS_PER_SEC);

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
  if (hrLMD) {
    if (!fpLMD) fpLMD=fpopen(fnmLMD.c_str(),"w");
  } else {
    if (!fpXLMD) fpXLMD=xdrfile_open(fnmLMD.c_str(),"w");
  }
  if (!fpNRG) fpNRG=fpopen(fnmNRG.c_str(),"w");
#ifdef REPLICAEXCHANGE
  if (!fpREx && freqREx>0) fpREx=fpopen(fnmREx.c_str(),"w");
#endif

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

  // Rectify bond constraints
  holonomic_rectify(system);

  // Read checkpoint
  if (fnmCPI!="") {
    read_checkpoint_file(fnmCPI.c_str(),system);
  }

  // Set up domain decomposition
  if (system->domdec) delete system->domdec;
  system->domdec=new Domdec();
  system->domdec->initialize(system);

  // NYI check gpu
  cudaDeviceSynchronize();
#pragma omp barrier
  if (cudaPeekAtLastError() != cudaSuccess) {
    cudaError_t err=cudaPeekAtLastError();
    fatal(__FILE__,__LINE__,"GPU error code %d during run initialization of OMP rank %d\n%s\n",err,system->id,cudaGetErrorString(err));
  }
#pragma omp barrier
}

void Run::dynamics_finalize(System *system)
{
  step0=step;
  write_checkpoint_file(fnmCPO.c_str(),system);
  system->state->save_state(system);  
}



void blade_init_run(System *system)
{
  system+=omp_get_thread_num();
  if (system->run) {
    delete(system->run);
  }
  system->run=new Run(system);
}

void blade_dest_run(System *system)
{
  system+=omp_get_thread_num();
  if (system->run) {
    delete(system->run);
  }
  system->run=NULL;
}

void blade_add_run_flags(System *system,
  double gamma,
  double betaEwald,
  double rCut,
  double rSwitch,
  int vdWfSwitch,
  int elecPME,
  double gridSpace,
  int gridx,
  int gridy,
  int gridz,
  int orderEwald,
  double shakeTolerance)
{
  system+=omp_get_thread_num();
  system->run->gamma=gamma;

  system->run->betaEwald=betaEwald;
  system->run->rCut=rCut;
  system->run->rSwitch=rSwitch;
  system->run->vfSwitch=vdWfSwitch==1;
  system->run->usePME=elecPME==1;
  system->run->gridSpace=gridSpace; // grid spacing for PME calculation
  system->run->grid[0]=gridx; // if gridSpace is negative, use these values
  system->run->grid[1]=gridy; // if gridSpace is negative, use these values
  system->run->grid[2]=gridz; // if gridSpace is negative, use these values
  system->run->orderEwald=orderEwald; // interpolation order (4, 6, or 8 typically)
  system->run->shakeTolerance=shakeTolerance;

  system->run->cutoffs.betaEwald=betaEwald;
  system->run->cutoffs.rCut=rCut;
  system->run->cutoffs.rSwitch=rSwitch;
}

void blade_add_run_dynopts(System *system,
  int step,
  int step0,
  int nsteps,
  double dt,
  double T,
  int freqNPT,
  double volumeFluctuation,
  double pressure)
{
  system+=omp_get_thread_num();
  system->run->step=step; // current step
  system->run->step0=step0; // starting step
  system->run->nsteps=nsteps; // steps in next dynamics call
  system->run->dt=dt;
  system->run->T=T;

  system->run->freqNPT=freqNPT;
  system->run->volumeFluctuation=volumeFluctuation;
  system->run->pressure=pressure;
}

void blade_run_energy(System *system)
{
  system+=omp_get_thread_num();
  
  if (!system->run) {
    system->run=new Run(system);
  }
  system->run->dynamics_initialize(system);
  system->potential->calc_force(0,system);
  system->state->recv_energy();
  system->run->dynamics_finalize(system);
}
