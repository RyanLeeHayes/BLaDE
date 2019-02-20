#ifndef IO_IO_H
#define IO_IO_H

#include <stdio.h>
#include <string>
#include <map>
#include <vector>

#include "main/defines.h"

// Forward declarations 
class System;

struct Frame {
  std::string type;
  fpos_t fp_pos;
};

class Control {
  public:
  int level;
  int controlLevel;
  std::map<std::string,fpos_t> functions;
  std::vector<struct Frame> backtrace;

  Control()
  {
    level=0;
    controlLevel=0;
    functions.clear();
    backtrace.clear();
  }

  ~Control()
  {
    ;
  }
};

void fatal(const char* fnm,int i,const char* format, ...);
FILE* fpopen(const char* fnm,const char* type);

void io_nexta(char *line,char *token);
std::string io_peeks(char *line);
std::string io_nexts(char *line);
int io_nexti(char *line);
int io_nexti(char *line,int output);
int io_nexti(char *line,FILE *fp,const char *tag);
real io_nextf(char *line);
real io_nextf(char *line,real input);
real io_nextf(char *line,FILE *fp,const char *tag);

void io_strncpy(char *targ,char *dest,int n);

void interpretter(const char *fnm,System *system,int level);

void print_dynamics_output(int step,System *system);
void print_xtc(int step,System *system);

#endif
