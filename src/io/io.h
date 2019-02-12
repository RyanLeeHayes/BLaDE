#ifndef IO_IO_H
#define IO_IO_H

#include <string>

// Forward declarations 
class System;

void fatal(const char* fnm,int i,const char* format, ...);
FILE* fpopen(const char* fnm,const char* type);

void io_nexta(char *line,char *token);
std::string io_peeks(char *line);
std::string io_nexts(char *line);
int io_nexti(char *line);
int io_nexti(char *line,int output);
int io_nexti(char *line,FILE *fp,const char *tag);
double io_nextf(char *line);
double io_nextf(char *line,double output);
double io_nextf(char *line,FILE *fp,const char *tag);

void interpretter(const char *fnm,System *system,int level);

#endif
