#ifndef SYSTEM_SYSTEM_H
#define SYSTEM_SYSTEM_H

#include "io/io.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "system/selections.h"
#include "msld/msld.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "update/update.h"

// Forward declarations - not allowed when calling delete on objects

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

  System() {
    parameters=NULL;
    structure=NULL;
    selections=NULL;
    msld=NULL;
    state=NULL;
    run=NULL;
    potential=NULL;
    update=NULL;
    setup_parse_system();
  }

  ~System() {
    if (parameters) delete(parameters);
    if (structure) delete(structure);
    if (selections) delete(selections);
    if (msld) delete(msld);
    if (state) delete(state);
    if (run) delete(run);
    if (potential) delete(potential);
    if (update) delete(update);
  }

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
