#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#include "main/defines.h"
#include "system/system.h"

// parse_whatever
#include "io/io.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "system/selections.h"
#include "msld/msld.h"
#include "system/potential.h"
#include "system/state.h"
#include "run/run.h"
#include "xdr/xdrfile.h"
#include "xdr/xdrfile_xtc.h"

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

// for positive i, deletes i characters from beginning of line
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

// Get next word without advancing
std::string io_peeks(char *line)
{
  char token[MAXLENGTHSTRING];
  int ntoken, nchar;
  std::string output;

  ntoken=sscanf(line,"%s%n",token,&nchar);
  if (ntoken==1 && token[0]!='!') {
    ;
  } else {
    token[0]='\0';
  }

  output=token;
  return output;
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

real io_nextf(char *line)
{
  int ntoken, nchar;
  double output; // Intentional double

  ntoken=sscanf(line,"%lg%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  } else {
    fatal(__FILE__,__LINE__,"Error: failed to read double from: %s\n",line);
  }
  return output; // Cast to real
}

real io_nextf(char *line,real input)
{
  int ntoken, nchar;
  double output=input; // Intentional double

  ntoken=sscanf(line,"%lg%n",&output,&nchar);
  if (ntoken==1) {
    io_shift(line,nchar);
  }
  return output; // Cast to real
}

real io_nextf(char *line,FILE *fp,const char *tag)
{
  int ntoken, nchar;
  double output; // Intentional double

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
  return output; // Cast to real
}

void io_strncpy(char *targ,char *dest,int n)
{
  strncpy(targ,dest,n);
  targ[n]='\0';
}

void interpretter(const char *fnm,System *system,int level)
{
  FILE *fp;
  fpos_t fp_pos;
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];
  Control control;
  control.level=level;

  fp=fpopen(fnm,"r");

  fgetpos(fp,&fp_pos);
  // fsetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    fprintf(stdout,"IN%d> %s",level,line);
    io_nexta(line,token);
    system->parse_system(line,token,system,&control);
    fgetpos(fp,&fp_pos);
    // fsetpos(fp,&fp_pos);
  }

  fclose(fp);
}

void print_xtc(int step,System *system)
{
  XDRFILE *fp=system->run->fpXTC;
  real (*x)[3]=system->state->position;
  float (*fx)[3]=system->state->fposition;
  float fbox[3][3];
  int i,j,N;
  char fnm[100];

  N=system->state->atomCount;
  if (fx != x) {
    for (i=0; i<N; i++) {
      for (j=0; j<3; j++) {
        fx[i][j]=x[i][j];
      }
    }
  }
  for (i=0; i<3; i++) {
    for (j=0; j<3; j++) {
      fbox[i][j]=system->state->box[i][j];
    }
  }

//  extern int write_xtc(XDRFILE *xd,
//                       int natoms,int step,real time,
//                       matrix box,rvec *x,real prec);
  write_xtc(fp,N,step,(float) (step*system->run->dt),fbox,fx,1000.0);
}

void print_lmd(int step,System *system)
{
  FILE *fp=system->run->fpLMD;
  real *l=system->msld->lambda;
  int i;

  fprintf(fp,"%10d",step);
  for (i=1; i<system->msld->blockCount; i++) {
    fprintf(fp," %8.6f",l[i]);
  }
  fprintf(fp,"\n");
}

void print_nrg(int step,System *system)
{
  FILE *fp=system->run->fpNRG;
  real *e=system->state->energy;
  int i;

  for (i=0; i<eepotential; i++) {
    e[eepotential]+=e[i];
  }
  e[eetotal]=e[eepotential]+e[eekinetic];

  fprintf(fp,"%10d",step);
  for (i=0; i<eeend; i++) {
    fprintf(fp," %12.4f",e[i]);
  }
  fprintf(fp,"\n");
}

void print_dynamics_output(int step,System *system)
{
  if (step % system->run->freqXTC == 0) {
    system->state->recv_position();
    print_xtc(step,system);
  }
  if (step % system->run->freqLMD == 0) {
    system->msld->recv_real(system->msld->lambda,system->msld->lambda_d);
    print_lmd(step,system);
  }
  if (step % system->run->freqNRG == 0) {
    system->state->recv_energy();
    print_nrg(step,system);
  }
}
