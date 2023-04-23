#include <omp.h>
#include <map>
#include <string>

#include "system/system.h"
#include "io/io.h"
#include "io/variables.h"
#include "io/control.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "system/selections.h"
#include "msld/msld.h"
#include "system/coordinates.h"
#include "run/run.h"
#include "rng/rng_cpu.h"
#include "rng/rng_gpu.h"
#include "system/potential.h"
#include "system/state.h"
#include "domdec/domdec.h"



// Class constructors
System::System() {
  fprintf(stdout,"Creating a copy of system\n");
  verbose=1;
  variables=new Variables;
  parameters=NULL;
  structure=NULL;
  selections=NULL;
  msld=NULL;
  coordinates=NULL;
  run=NULL;
  rngCPU=new RngCPU;
  // rngGPU=new RngGPU; // Have to be careful to allocate and free for correct GPU
  rngGPU=NULL;
  potential=NULL;
  state=NULL;
  domdec=NULL;
  setup_parse_system();
}

System::~System() {
  fprintf(stdout,"Destroying a copy of system\n");
  if (variables) delete(variables);
  if (parameters) delete(parameters);
  if (structure) delete(structure);
  if (selections) delete(selections);
  if (msld) delete(msld);
  if (coordinates) delete(coordinates);
  if (run) delete(run);
  if (rngCPU) delete(rngCPU);
  if (rngGPU) delete(rngGPU);
  if (potential) delete(potential);
  if (state) delete(state);
  if (domdec) delete(domdec);
}



// Parsing functions
void System::parse_system(char *line,char *token,System *system)
{
  std::string name;

  // System already exists.

  name=token;
  if (system->parseSystem.count(name)==0) system->error(line,token,system);
  // So much for function pointers being elegant.
  // call the function pointed to by: system->parseSystem[name]
  // within the object: system
  // with arguments: (line,token,system)
  (system->*(system->parseSystem[name]))(line,token,system);
}

void System::setup_parse_system()
{
  parseSystem[""]=&System::pass;
  helpSystem[""]="If you see this string, something went wrong.\n";
  parseSystem["help"]=&System::help;
  helpSystem["help"]="?help [directive]> Prints general help, including a list of directives. If a directive is listed, this prints help on that specific directive.\n";
  parseSystem["parameters"]=&System::parse_system_parameters;
  helpSystem["parameters"]="?parameters> Loads parameters for an MSLD simulation. Should generally be done before anything else\n";
  parseSystem["structure"]=&System::parse_system_structure;
  helpSystem["structure"]="?structure> Sets up a structure/topology/bond-connectivity, either using a sequence and .rtf (CHARMM residue topology file) or using a .psf (CHARMM protein structure file). Should be done second, after parameters have been loaded.\n";
  parseSystem["selection"]=&System::parse_system_selection;
  helpSystem["selection"]="?selection> Sets selections of atoms for use in various other commands, including structure (you may want to delete a selection of atoms) and msld (you need to indicate which atoms are in which groups).\n";
  parseSystem["msld"]=&System::parse_system_msld;
  helpSystem["msld"]="?msld> Set up MSLD data structures. Determines which atoms are in which groups, how to scale their interactions, and so on. Should be done after calls of structure, because if the indices of atoms change after calls to msld, an error will occur.\n";
  parseSystem["coordinates"]=&System::parse_system_coordinates;
  helpSystem["coordinates"]="?coordinates> Sets the initial conditions of the system, including starting spatial coordinates. Must be called after structure is complete, must be called before run.\n";
  parseSystem["run"]=&System::parse_system_run;
  helpSystem["run"]="?run> Run a calculation, such as a dynamics simulation, on the system. Must have parameters, structure, msld, and coordinates set up first.\n";
  parseSystem["stream"]=&System::parse_system_stream;
  helpSystem["stream"]="?stream [filename]> Read commands from filename until they are finished, and then return to this point in this script.\n";
  parseSystem["arrest"]=&System::parse_system_arrest;
  helpSystem["arrest"]="?arrest [int]> Hang the process for int seconds so a debugger like gdb can be attached. Defaults to 30 seconds if no argument is given. For code development only\n";
  parseSystem["variables"]=&System::parse_system_variables;
  helpSystem["variables"]="?variables> Used for storing tokens and their values, for later use in parsing commands. Access the value by surrounding the token name in {}. For example:\nvariables set token1 help variables\n{token1}\nwill print this help message.\n";
  parseSystem["if"]=&System::parse_system_if;
  helpSystem["if"]="?if [conditional]\nelseif [conditional]\nelse\nendif> Used for conditional execution in script\n";
  parseSystem["elseif"]=&System::parse_system_elseif;
  helpSystem["elseif"]="?if [conditional]\nelseif [conditional]\nelse\nendif> Used for conditional execution in script\n";
  parseSystem["else"]=&System::parse_system_else;
  helpSystem["else"]="?if [conditional]\nelse\nendif> Used for conditional execution in script\n";
  parseSystem["endif"]=&System::parse_system_endif;
  helpSystem["endif"]="?if [conditional]\nelseif [conditional]\nelse\nendif> Used for conditional execution in script\n";
  parseSystem["while"]=&System::parse_system_while;
  helpSystem["while"]="?while [conditional]\nendwhile> Used for loops within script. To use a for loop, place initialization before while and increment operation before endwhile.\n";
  parseSystem["endwhile"]=&System::parse_system_endwhile;
  helpSystem["endwhile"]="?while [conditional]\nendwhile> Used for loops within script. To use a for loop, place initialization before while and increment operation before endwhile.\n";
  parseSystem["verbose"]=&System::parse_system_verbose;
  helpSystem["verbose"]="?verbose [int]> Set a verbose level for output. verbose=1 is default, verbose=0 is less output.\n";
}

// MOVE
void System::pass(char *line,char *token,System *system)
{
  ;
}

void System::parse_system_parameters(char *line,char *token,System *system)
{
  parse_parameters(line,system);
}

void System::parse_system_structure(char *line,char *token,System *system)
{
  parse_structure(line,system);
}

void System::parse_system_selection(char *line,char *token,System *system)
{
  parse_selection(line,system);
}

void System::parse_system_msld(char *line,char *token,System *system)
{
  parse_msld(line,system);
}

void System::parse_system_coordinates(char *line,char *token,System *system)
{
  parse_coordinates(line,system);
}

void System::parse_system_run(char *line,char *token,System *system)
{
  parse_run(line,system);
}

void System::parse_system_stream(char *line,char *token,System *system)
{
  io_nexta(line,token);
  interpretter(token,system);
}

void System::parse_system_arrest(char *line,char *token,System *system)
{
  arrested_development(system,io_nexti(line,30));
}

void System::parse_system_variables(char *line,char *token,System *system)
{
  parse_variables(line,system);
}

void System::parse_system_if(char *line,char *token,System *system)
{
  parse_if(line,system);
}

void System::parse_system_elseif(char *line,char *token,System *system)
{
  parse_elseif(line,system);
}

void System::parse_system_else(char *line,char *token,System *system)
{
  parse_else(line,system);
}

void System::parse_system_endif(char *line,char *token,System *system)
{
  parse_endif(line,system);
}

void System::parse_system_while(char *line,char *token,System *system)
{
  parse_while(line,system);
}

void System::parse_system_endwhile(char *line,char *token,System *system)
{
  parse_endwhile(line,system);
}

void System::parse_system_verbose(char *line,char *token,System *system)
{
  verbose=io_nexti(line);
}

// MOVE

void System::help(char *line,char *token,System *system)
{
  std::string name=io_nexts(line);
  if (name=="") {
    fprintf(stdout,"?> This program uses a script to allow you to set up and run alchemical molecular simulations. Each line of the script starts with a directive, followed by other subdirectives. Comments after ! characters are ignored. At any point in the script, you may put help after a directive or subdirective to get documentation on how to use it. Top level directives are listed at the end of this help section. Directives can be divided into directives that set up and manipulate the system, and directives that alter the control flow and allow you to script the set up.\n\nSystem manipulation directives:\nThe general work flow is to set up the potential energy function parameters with calls to parameters, then set up the atoms, bond connectivity, and other potential terms with calls to structure, then set up the msld alchemical treatment with calls to msld, then set up the initial conditions or starting structure with calls to state. After all of that you are ready to call run to calculate energy, minimize the structure, or run dynamics.\n\nScripting control directives:\nStream allows you to start execution of another script from this point in the current script. Set allows you to set internal variables which can be accessed in subsequent commands by enclosing the variable name in {}. (Variable names can also contain variables via nested {{}}). Functions (function/endfunction) can be defined for later use and called (call), if (if/elseif/else/endif) and while (while/endwhile) loops are also available. Some of these features may not yet be implemented, see the listing below for what's available.\n\nSeveral available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpSystem.begin(); ii!=helpSystem.end(); ii++) {
      fprintf(stdout," %s",ii->first.c_str());
    }
    fprintf(stdout,"\n");
  } else if (helpSystem.count(token)==1) {
    fprintf(stdout,helpSystem[name].c_str());
  } else {
    error(line,token,system);
  }
}

void System::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token: %s\n",token);
}

System* init_system()
{
  System *system;
  int id;
  int idCount=omp_get_max_threads(); // omp_get_num_threads();
  void **message; // OMP

  int available;
  int notAvailable=cudaGetDeviceCount(&available);
  if (notAvailable==1) fatal(__FILE__,__LINE__,"No GPUs available\n");
  if (available<omp_get_max_threads()) fatal(__FILE__,__LINE__,"Running with %d omp threads but only %d GPUs\n",omp_get_max_threads(),available);

  system=new System[idCount];

  message=(void**)calloc(idCount,sizeof(void*));
  for (id=0; id<idCount; id++) {
    system[id].id=id;
    system[id].idCount=idCount;
    system[id].message=message;
    cudaSetDevice(id);
    if (id!=0) {
      int accessible;
      cudaDeviceCanAccessPeer(&accessible, id, 0);
      fprintf(stdout,"Device %d %s access device %d directly\n",id,(accessible?"can":"cannot"),0);
      if (accessible) {
        cudaDeviceEnablePeerAccess(0,0); // host 0, required 0
      }
    }
    system[id].rngGPU=new RngGPU;
  }

  return system;
}

void dest_system(System *system)
{
  int id;
  int idCount=omp_get_max_threads(); // omp_get_num_threads();

  free(system->message);
  for (id=0; id<idCount; id++) {
    cudaSetDevice(id);
    if (id!=0) {
      int accessible;
      cudaDeviceCanAccessPeer(&accessible, id, 0);
      if (accessible) {
        cudaDeviceDisablePeerAccess(0); // host 0, have to disable on free, otherwise an error is thrown on reallocation
      }
    }
    delete(system[id].rngGPU);
    system[id].rngGPU=NULL;
  }

  delete[] system;
}

System* blade_init_system()
{
  return init_system();
}

void blade_dest_system(System *system)
{
  dest_system(system);
}

void blade_set_device()
{
  cudaSetDevice(omp_get_thread_num());
}

void blade_set_verbose(System *system,int v)
{
  system+=omp_get_thread_num();
  system->verbose=v;
}
