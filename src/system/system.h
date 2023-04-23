#ifndef SYSTEM_SYSTEM_H
#define SYSTEM_SYSTEM_H

#include <cuda_runtime.h>

#include <string>
#include <map>
#include <vector>



// Forward declarations - not allowed when calling delete on objects
class Control; // io/io.h
class Variables;
class Parameters;
class Structure;
class Selections;
class Msld;
class Coordinates;
class Run;
class RngCPU;
class RngGPU;
class Potential;
class State;
class Domdec;

class System {
  public:
  int id,idCount;
  int verbose;
  void **message; // (shared variable for message passing of pointers)
// Have command parsing
  std::vector<Control> control;
  Variables *variables;
  Parameters *parameters;
  Structure *structure;
  Selections *selections;
  Msld *msld;
  Coordinates *coordinates;
  Run *run;
// No command parsing
  RngCPU *rngCPU;
  RngGPU *rngGPU;
  State *state;
  Potential *potential;
  Domdec *domdec;

  std::map<std::string,void (System::*)(char*,char*,System*)> parseSystem;
  std::map<std::string,std::string> helpSystem;

  System();
  ~System();

  void setup_parse_system();
  void parse_system(char *line,char *token,System *system);

  void help(char *line,char *token,System *system);
  void error(char *line,char *token,System *system);

  void pass(char *line,char *token,System *system);
  void parse_system_parameters(char *line,char *token,System *system);
  void parse_system_structure(char *line,char *token,System *system);
  void parse_system_selection(char *line,char *token,System *system);
  void parse_system_msld(char *line,char *token,System *system);
  void parse_system_coordinates(char *line,char *token,System *system);
  void parse_system_run(char *line,char *token,System *system);
  void parse_system_stream(char *line,char *token,System *system);
  void parse_system_arrest(char *line,char *token,System *system);
  void parse_system_variables(char *line,char *token,System *system);

  void parse_system_if(char *line,char *token,System *system);
  void parse_system_elseif(char *line,char *token,System *system);
  void parse_system_else(char *line,char *token,System *system);
  void parse_system_endif(char *line,char *token,System *system);
  void parse_system_while(char *line,char *token,System *system);
  void parse_system_endwhile(char *line,char *token,System *system);

  void parse_system_verbose(char *line,char *token,System *system);
};

// Library functions
extern "C" {
  System* blade_init_system();
  void blade_dest_system(System *system);
  void blade_set_device();
  void blade_set_verbose(System *system,int v);
}

#endif
