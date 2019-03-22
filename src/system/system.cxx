#include <map>
#include <string>

#include "system/system.h"
#include "io/io.h"
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
  parameters=NULL;
  structure=NULL;
  selections=NULL;
  msld=NULL;
  coordinates=NULL;
  run=NULL;
  rngCPU=new RngCPU;
  rngGPU=new RngGPU;
  potential=NULL;
  state=NULL;
  domdec=NULL;
  setup_parse_system();
}

System::~System() {
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
void System::parse_system(char *line,char *token,System *system,Control *control)
{
  std::string name;

  // System already exists.

  name=token;
  if (system->parseSystem.count(name)==0) system->error(line,token,system,control);
  // So much for function pointers being elegant.
  // call the function pointed to by: system->parseSystem[name]
  // within the object: system
  // with arguments: (line,token,system,control)
  (system->*(system->parseSystem[name]))(line,token,system,control);
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
  parseSystem["set"]=&System::parse_system_set;
  helpSystem["set"]="?set [token] [value]> Set token to value, for later use in parsing commands. Access the value by surrounding the token name in {}. For example:\nset token1 help set\n{token1}\nwill print this help message.\n";
}

// MOVE
void System::pass(char *line,char *token,System *system,Control *control)
{
  ;
}

void System::parse_system_parameters(char *line,char *token,System *system,Control *control)
{
  parse_parameters(line,system);
}

void System::parse_system_structure(char *line,char *token,System *system,Control *control)
{
  parse_structure(line,system);
}

void System::parse_system_selection(char *line,char *token,System *system,Control *control)
{
  parse_selection(line,system);
}

void System::parse_system_msld(char *line,char *token,System *system,Control *control)
{
  parse_msld(line,system);
}

void System::parse_system_coordinates(char *line,char *token,System *system,Control *control)
{
  parse_coordinates(line,system);
}

void System::parse_system_run(char *line,char *token,System *system,Control *control)
{
  parse_run(line,system);
}

void System::parse_system_stream(char *line,char *token,System *system,Control *control)
{
  io_nexta(line,token);
  interpretter(token,system,control->level+1);
}

void System::parse_system_arrest(char *line,char *token,System *system,Control *control)
{
  arrested_development(system,io_nexti(line,30));
}

void System::parse_system_set(char *line,char *token,System *system,Control *control)
{
  fatal(__FILE__,__LINE__,"set is not yet implemented\n"); // NYI
}

// NYI also define, if, for, {} parsing, etc...
// MOVE

void System::help(char *line,char *token,System *system,Control *control)
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
    error(line,token,system,control);
  }
}

void System::error(char *line,char *token,System *system,Control *control)
{
  fatal(__FILE__,__LINE__,"Unrecognized token: %s\n",token);
}
