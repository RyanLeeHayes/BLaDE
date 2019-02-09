#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "main/defines.h"
#include "system/system.h"
#include "system/parameters.h"

void fatal(const char* fnm,int i,const char* format, ...)
{
  va_list args;

  va_start(args,format);
  fprintf(stdout,"FATAL ERROR:\n");
  fprintf(stdout,"%s:%d\n",fnm,i);
  vfprintf(stdout,format,args);
  va_end(args);

  exit(1);
}

FILE* fpopen(const char* fnm,const char* type)
{
  FILE *fp;

  fprintf(stdout,"Opening file %s for %s\n",fnm,type);
  fp=fopen(fnm,type);
  if (fp==NULL) {
    fatal(__FILE__,__LINE__,"Error: Unable to open file %s\n",fnm);
  }

  return fp;
}

void io_shift(char *line,int i)
{
  memmove(line,line+i,strlen(line+i)+1);
}

// Read the next string in the line to token, and shift line to after string
void io_nexta(char *line,char *token)
{
  int ntoken, nchar;

  ntoken=sscanf(line,"%s%n",token,&nchar);
  if (ntoken==1 && token[0]!='!') {
    io_shift(line,nchar);
  } else {
    token[0]='\0';
  }
}

std::string io_nexts(char *line)
{
  char token[MAXLENGTHSTRING];
  int ntoken, nchar;
  std::string output;

  ntoken=sscanf(line,"%s%n",token,&nchar);
  if (ntoken==1 && token[0]!='!') {
    io_shift(line,nchar);
  } else {
    token[0]='\0';
  }

  output=token;
  return output;
}

// Read the next int in the line and shift line to after string
int io_nexti(char *line)
{
  int ntoken, nchar;
  int output;

  ntoken=sscanf(line,"%d%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  } else {
    fatal(__FILE__,__LINE__,"Error: failed to read int from: %s\n",line);
  }
  return output;
}

int io_nexti(char *line,int output)
{
  int ntoken, nchar;

  ntoken=sscanf(line,"%d%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  }
  return output;
}

int io_nexti(char *line,FILE *fp,const char *tag)
{
  int ntoken, nchar;
  int output;

  ntoken=0;
  while (ntoken!=1) {
    ntoken=sscanf(line,"%d%n",&output,&nchar);
    if (ntoken==1) {
      io_shift(line,nchar);
      return output;
    }
    if(!fgets(line, MAXLENGTHSTRING, fp)) {
      fatal(__FILE__,__LINE__,"End of file while searching for int value for %s\n",tag);
    }
  }
  return output;
}

double io_nextf(char *line)
{
  int ntoken, nchar;
  double output;

  ntoken=sscanf(line,"%lg%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  } else {
    fatal(__FILE__,__LINE__,"Error: failed to read double from: %s\n",line);
  }
  return output;
}

double io_nextf(char *line,double output)
{
  int ntoken, nchar;

  ntoken=sscanf(line,"%lg%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  }
  return output;
}

double io_nextf(char *line,FILE *fp,const char *tag)
{
  int ntoken, nchar;
  double output;

  ntoken=0;
  while (ntoken!=1) {
    ntoken=sscanf(line,"%lg%n",&output,&nchar);
    if (ntoken==1) {
      io_shift(line,nchar);
      return output;
    }
    if(!fgets(line, MAXLENGTHSTRING, fp)) {
      fatal(__FILE__,__LINE__,"End of file while searching for real value for %s\n",tag);
    }
  }
  return output;
}

void interpretter(FILE *fp,System *system,int level)
{
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];

  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    fprintf(stdout,"IN%d> %s",level,line);
    io_nexta(line,token);
    if (strcmp(token,"")==0) {
      ;
    } else if (strcmp(token,"parameters")==0) {
      parse_parameters(line,system);
    } else if (strcmp(token,"structure")==0) {
      parse_structure(line,system);
    } else if (strcmp(token,"stream")==0) {
      FILE *fp2;
      char token2[MAXLENGTHSTRING];
      io_nexta(line,token);
      fp2=fpopen(token,"r");
      interpretter(fp2,system,level+1);
      fclose(fp2);
    } else {
      fatal(__FILE__,__LINE__,"Unrecognized token: %s\n",token); // FIXIT add token name
    }
  }
}
