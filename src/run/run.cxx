#include <string.h>

#include "run/run.h"
#include "system/system.h"
#include "io/io.h"



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
  fprintf(stdout,"RUN PRINT> dt=%f (time step in ps)\n",dt);
  fprintf(stdout,"RUN PRINT> nsteps=%d (number of time steps for dynamics)\n",nsteps);
  fprintf(stdout,"RUN PRINT> fnmxtc=%s (file name for coordinate trajectory)\n",fnmXTC.c_str());
  fprintf(stdout,"RUN PRINT> fnmlmd=%s (file name for lambda trajectory)\n",fnmLMD.c_str());
  fprintf(stdout,"RUN PRINT> fnmnrg=%s (file name for energy output)\n",fnmNRG.c_str());
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
    dt=io_nextf(line);
  } else if (strcmp(token,"nsteps")==0) {
    nsteps=io_nexti(line);
  } else if (strcmp(token,"fnmxtc")==0) {
    fnmXTC=io_nexts(line);
  } else if (strcmp(token,"fnmlmd")==0) {
    fnmLMD=io_nexts(line);
  } else if (strcmp(token,"fnmnrg")==0) {
    fnmNRG=io_nexts(line);
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token %s in run setvariable command\n",token);
  }
}

void Run::dynamics(char *line,char *token,System *system)
{
  int step;

  if (!fpXTC) fpXTC=fpopen(fnmXTC.c_str(),"w");
  if (!fpLMD) fpLMD=fpopen(fnmLMD.c_str(),"w");
  if (!fpNRG) fpNRG=fpopen(fnmNRG.c_str(),"w");

  // Initialize data structures
  dynamics_initialize(system);

  // Run dynamics
  for (step=step0; step<step0+nsteps; step++) {
    system->potential->calc_force(step,system);
    system->update->update(step,system);
    print_dynamics_output(step,system);
  }

  dynamics_finalize(system);
}

void Run::dynamics_initialize(System *system)
{
  if (system->update==NULL) {
    system->update=new Update();
  }
  //NYI read checkpoint
}

void Run::dynamics_finalize(System *system)
{
  //NYI write checkpoint
  step0=step;
}
