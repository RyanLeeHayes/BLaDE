#ifndef IO_VARIABLES_H
#define IO_VARIABLES_H

#include <map>
#include <string>

// Forward declarations
class System;

class Variables {
  public:
  std::map<std::string,void(Variables::*)(char*,char*,System*)> parseVariables;
  std::map<std::string,std::string> helpVariables;

  std::map<std::string,std::string> data;

  Variables();
  ~Variables();

  void setup_parse_variables();

  void error(char *line,char *token,System *system);
  void reset(char *line,char *token,System *system);
  void set(char *line,char *token,System *system);
  void calculate(char *line,char *token,System *system);
  void dump(char *line,char *token,System *system);
  void help(char *line,char *token,System *system);

  void substitute(char *line);
};

void parse_variables(char *line,System *system);

#endif
