#ifndef SYSTEM_SYSTEM_H
#define SYSTEM_SYSTEM_H

#include <string>
#include <map>


// Forward declarations - not allowed when calling delete on objects
class Control; // io/io.h
class Parameters;
class Structure;
class Selections;
class Msld;
class State;
class Run;
class Potential;
class Update;

class System {
  public:
// Have command parsing
  Parameters *parameters;
  Structure *structure;
  Selections *selections;
  Msld *msld;
  State *state;
  Run *run;
// No command parsing
  Potential *potential;
  Update *update;

  std::map<std::string,void (System::*)(char*,char*,System*,Control*)> parseSystem;
  std::map<std::string,std::string> helpSystem;

  System();
  ~System();

  void setup_parse_system();
  void parse_system(char *line,char *token,System *system,Control *control);

  void help(char *line,char *token,System *system,Control *control);
  void error(char *line,char *token,System *system,Control *control);

  void pass(char *line,char *token,System *system,Control *control);
  void parse_system_parameters(char *line,char *token,System *system,Control *control);
  void parse_system_structure(char *line,char *token,System *system,Control *control);
  void parse_system_selection(char *line,char *token,System *system,Control *control);
  void parse_system_msld(char *line,char *token,System *system,Control *control);
  void parse_system_state(char *line,char *token,System *system,Control *control);
  void parse_system_run(char *line,char *token,System *system,Control *control);
  void parse_system_stream(char *line,char *token,System *system,Control *control);
  void parse_system_set(char *line,char *token,System *system,Control *control);
};

#endif
