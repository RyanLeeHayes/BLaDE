#ifndef IO_IO_H
#define IO_IO_H

#include <stdio.h> // For FILE
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
  FILE *fp;
  fpos_t fp_pos;
  // std::map<std::string,fpos_t> functions;
  std::vector<struct Frame> backtrace;

  Control()
  {
    // functions.clear();
    backtrace.clear();
  }

  ~Control()
  {
    ;
  }
};

void fatal(const char* fnm,int i,const char* format, ...);
void arrested_development(System *system,int howLong);
FILE* fpopen(const char* fnm,const char* type);

void io_shift(char *line,int i);
void io_nexta(char *line,char *token);
std::string io_peeks(char *line);
std::string io_nexts(char *line);
bool io_nextb(char *line);
int io_nexti(char *line);
int io_nexti(char *line,int output);
int io_nexti(char *line,FILE *fp,const char *tag);
real io_nextf(char *line);
real io_nextf(char *line,real input);
real io_nextf(char *line,FILE *fp,const char *tag);

void io_strncpy(char *targ,char *dest,int n);

void interpretter(const char *fnm,System *system);

void print_nrg(int step,System *system);
void print_dynamics_output(int step,System *system);

void write_checkpoint_file(const char *fnm,System *system);
void read_checkpoint_file(const char *fnm,System *system);

// Library functions
extern "C" {
  void blade_interpretter(const char *fnm,System *system);
}

#endif
