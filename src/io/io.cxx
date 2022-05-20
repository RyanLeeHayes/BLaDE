#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
// For arrested_development
#include <signal.h>
#include <unistd.h>

#include "main/defines.h"
#include "system/system.h"

// parse_whatever
#include "io/io.h"
#include "io/variables.h"
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

void arrested_development(System *system,int howLong) {
  int i;
  char hostname[MAXLENGTHSTRING];
  for (i=0; i<system->idCount; i++) {
#pragma omp barrier
    if (i==system->id) {
      gethostname(hostname,MAXLENGTHSTRING);
      fprintf(stderr,"PID %d rank %d host %s\n",getpid(),i,hostname);
    }
#pragma omp barrier
  }
  sleep(howLong);
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
  if (i>0) {
    memmove(line,line+i,strlen(line+i)+1);
  } else {
    memmove(line-i,line,strlen(line)+1);
  }
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

bool io_nextb(char *line)
{
  std::string booleanString=io_nexts(line);

  if (booleanString=="on" || booleanString=="true" || booleanString=="yes" || booleanString=="1" || booleanString=="T") {
    return true;
  } else if (booleanString=="off" || booleanString=="false" || booleanString=="no" || booleanString=="0" || booleanString=="F") {
    return false;
  } else {
    fatal(__FILE__,__LINE__,"Error: could not convert string %s to boolean value\n",booleanString.c_str());
  }
  return false;
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

void interpretter(const char *fnm,System *system)
{
  FILE *fp;
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];
  system->control.push_back(Control());
  int level=system->control.size();

  fp=fpopen(fnm,"r");
  system->control[level-1].fp=fp;

  fgetpos(fp,&system->control[level-1].fp_pos);
  // fsetpos(fp,&fp_pos);
  while (fgets(line, MAXLENGTHSTRING, fp) != NULL) {
    fprintf(stdout,"IN%d> %s",level,line);
    system->variables->substitute(line);
    io_nexta(line,token);
    system->parse_system(line,token,system);
    fgetpos(fp,&system->control[level-1].fp_pos);
    // fsetpos(fp,&fp_pos);
  }

  fclose(fp);
  system->control.pop_back();
}

void print_xtc(int step,System *system)
{
  XDRFILE *fp=system->run->fpXTC;
  float box[3][3]={{0,0,0},{0,0,0},{0,0,0}};
  int i,j,N;

  N=system->state->atomCount;
  if (system->state->typeBox) {
    box[0][0]=system->state->tricBox.a.x/(10*ANGSTROM);
    box[1][0]=system->state->tricBox.b.x/(10*ANGSTROM);
    box[1][1]=system->state->tricBox.b.y/(10*ANGSTROM);
    box[2][0]=system->state->tricBox.c.x/(10*ANGSTROM);
    box[2][1]=system->state->tricBox.c.y/(10*ANGSTROM);
    box[2][2]=system->state->tricBox.c.z/(10*ANGSTROM);
  } else {
    box[0][0]=system->state->orthBox.x/(10*ANGSTROM);
    box[1][1]=system->state->orthBox.y/(10*ANGSTROM);
    box[2][2]=system->state->orthBox.z/(10*ANGSTROM);
  }

  real_x (*x)[3]=system->state->position;
  float (*xXTC)[3]=system->state->positionXTC;
  for (i=0; i<N; i++) {
    for (j=0; j<3; j++) {
      xXTC[i][j]=x[i][j]/(10*ANGSTROM);
    }
  }
//  extern int write_xtc(XDRFILE *xd,
//                       int natoms,int step,real time,
//                       matrix box,rvec *x,real prec);
  write_xtc(fp,N,step,(float) (step*system->run->dt/PICOSECOND),box,xXTC,1000.0);
}

void print_lmd(int step,System *system)
{
  real_x *l=system->state->lambda;
  int i;

  if (system->run->hrLMD) {
    FILE *fp=system->run->fpLMD;
    fprintf(fp,"%10d",step);
    for (i=1; i<system->state->lambdaCount; i++) {
      fprintf(fp," %8.6f",(real)l[i]);
    }
    fprintf(fp,"\n");
  } else {
    XDRFILE *fp=system->run->fpXLMD;
    xdrfile_write_int(&system->state->lambdaCount-1,1,fp);
#if defined DOUBLE || defined DOUBLE_X
    for (i=1; i<system->state->lambdaCount; i++) {
      float lf=l[i];
      xdrfile_write_float(&lf,1,fp);
    }
#else
    xdrfile_write_float(&l[1],system->state->lambdaCount-1,fp);
#endif
  }
}

void print_nrg(int step,System *system)
{
  FILE *fp=system->run->fpNRG;
  real_e *e=system->state->energy;
  int i;

  fprintf(fp,"%10d",step);
  for (i=0; i<eeend; i++) {
    fprintf(fp," %12.4f",e[i]);
  }
  fprintf(fp,"\n");
}

void print_dynamics_output(int step,System *system)
{
  if (system->id==0) {
    if (step % system->run->freqXTC == 0) {
      system->state->recv_position();
      system->state->prettify_position(system);
      print_xtc(step,system);
    }
    if (step % system->run->freqLMD == 0) {
      system->state->recv_lambda();
      print_lmd(step,system);
    }
    if (step % system->run->freqNRG == 0) {
      system->state->recv_energy();
      print_nrg(step,system);
    }
  }
}

void write_checkpoint_file(const char *fnm,System *system)
{
  FILE *fp;
  int i;

  if (system->id==0) {
    fp=fpopen(fnm,"w");

    system->state->recv_state();

    fprintf(fp,"Step %d\n",system->run->step0);

    fprintf(fp,"Position %d\n",system->state->atomCount);
    for (i=0; i<system->state->atomCount; i++) {
      fprintf(fp,"%f ",system->state->position[i][0]);
      fprintf(fp,"%f ",system->state->position[i][1]);
      fprintf(fp,"%f\n",system->state->position[i][2]);
    }

    fprintf(fp,"Velocity %d\n",system->state->atomCount);
    for (i=0; i<system->state->atomCount; i++) {
      fprintf(fp,"%f ",system->state->velocity[i][0]);
      fprintf(fp,"%f ",system->state->velocity[i][1]);
      fprintf(fp,"%f\n",system->state->velocity[i][2]);
    }

    fprintf(fp,"ThetaPos %d\n",system->state->lambdaCount);
    for (i=0; i<system->state->lambdaCount; i++) {
      fprintf(fp,"%f\n",system->state->theta[i]);
    }

    fprintf(fp,"ThetaVel %d\n",system->state->lambdaCount);
    for (i=0; i<system->state->lambdaCount; i++) {
      fprintf(fp,"%f\n",system->state->thetaVelocity[i]);
    }

    fprintf(fp,"Box\n");
    fprintf(fp,"%f %f %f\n",system->state->box.a.x,system->state->box.a.y,system->state->box.a.z);
    fprintf(fp,"%f %f %f\n",system->state->box.b.x,system->state->box.b.y,system->state->box.b.z);

    fclose(fp);
  }
}

void read_checkpoint_file(const char *fnm,System *system)
{
  FILE *fp;
  int i;
  double v;

  if (system->id==0) {
    fp=fpopen(fnm,"r");
    fscanf(fp,"Step %ld\n",&system->run->step0);

    fscanf(fp,"Position %d\n",&i);
    if (i!=system->state->atomCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of atoms\n");
    }
    for (i=0; i<system->state->atomCount; i++) {
      fscanf(fp,"%lf ",&v);
      system->state->position[i][0]=v;
      fscanf(fp,"%lf ",&v);
      system->state->position[i][1]=v;
      fscanf(fp,"%lf\n",&v);
      system->state->position[i][2]=v;
    }

    fscanf(fp,"Velocity %d\n",&i);
    if (i!=system->state->atomCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of atoms\n");
    }
    for (i=0; i<system->state->atomCount; i++) {
      fscanf(fp,"%lf ",&v);
      system->state->velocity[i][0]=v;
      fscanf(fp,"%lf ",&v);
      system->state->velocity[i][1]=v;
      fscanf(fp,"%lf\n",&v);
      system->state->velocity[i][2]=v;
    }

    fscanf(fp,"ThetaPos %d\n",&i);
    if (i!=system->state->lambdaCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of alchemical coordinates\n");
    }
    for (i=0; i<system->state->lambdaCount; i++) {
      fscanf(fp,"%lf\n",&v);
      system->state->theta[i]=v;
    }

    fscanf(fp,"ThetaVel %d\n",&i);
    if (i!=system->state->lambdaCount) {
      fatal(__FILE__,__LINE__,"Checkpoint file is not compatible with system setup. Wrong number of alchemical coordinates\n");
    }
    for (i=0; i<system->state->lambdaCount; i++) {
      fscanf(fp,"%lf\n",&v);
      system->state->thetaVelocity[i]=v;
    }

    fscanf(fp,"Box\n");
    fscanf(fp,"%lf\n",&v);
    system->state->box.a.x=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.a.y=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.a.z=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.b.x=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.b.y=v;
    fscanf(fp,"%lf\n",&v);
    system->state->box.b.z=v;

    fclose(fp);

    system->state->send_state();
    if (system->msld->fix) { // ffix
      cudaMemcpy(system->state->lambda_d,system->state->theta,system->state->lambdaCount*sizeof(real_x),cudaMemcpyHostToDevice);
    }
    system->msld->calc_lambda_from_theta(0,system);
  }
}

void blade_interpretter(const char *fnm,System *system)
{
  system+=omp_get_thread_num();
  interpretter(fnm,system);
}
