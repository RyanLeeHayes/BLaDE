
#include "parms.h"

#include "defines.h"

#include "leapparms.h"
#include "unspecific.h"
#include "error.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>

#ifndef NOMPI
#include <mpi.h>
#endif


void read_parameter_b(int *A,FILE *fp,const char *token,int def,int id,FILE *log) {
  char s[MAXLENGTH],ss[MAXLENGTH],token_list[MAXLENGTH];
  char *sbuf;
  char a[MAXLENGTH];
  int i,spos;
  int found=0;

  fprintf(log,"Searching for %s [boolean]\n",token);

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    // sscanf(s,"%s = %s",&ss,&a);
    sscanf(s,"%s = %s",ss,a);
    if (strcmp(ss,token) == 0) {
      found=1;
      break;
    }
  }
  rewind(fp);

  if (found==0) {
    strcpy(token_list,token);
    strcat(token_list,"-list");
    while (fgets(s,MAXLENGTH,fp) != NULL) {
      // sscanf(s,"%s = %n",&ss,&spos);
      sscanf(s,"%s = %n",ss,&spos);
      if (strcmp(ss,token_list) == 0) {
        found=1;
        sbuf=s+spos;
        for (i=0;i<=id;i++) {
          // sscanf(sbuf,"%s%n",&a,&spos);
          sscanf(sbuf,"%s%n",a,&spos);
          sbuf+=spos;
        }
        break;
      }
    }
  }
  rewind(fp);

  if (found) {
    if (strcmp(a,"yes")==0) {
      *A=1;
      fprintf(log,"Found 1\n");
    } else if(strcmp(a,"no")==0) {
      *A=0;
      fprintf(log,"Found 0\n");
    } else {
      *A=def;
      fprintf(log,"Default to %d\n",def);
    }
  } else {
    *A=def;
    fprintf(log,"Default to %d\n",def);
  }
}


void read_parameter_i(int *A,FILE *fp,const char *token,int def,int id,FILE *log) {
  char s[MAXLENGTH],ss[MAXLENGTH],token_list[MAXLENGTH];
  char *sbuf;
  int a;
  int i,spos;
  int found=0;

  fprintf(log,"Searching for %s [integer]\n",token);

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    // sscanf(s,"%s = %d",&ss,&a);
    sscanf(s,"%s = %d",ss,&a);
    if (strcmp(ss,token) == 0) {
      found=1;
      break;
    }
  }
  rewind(fp);

  if (found==0) {
    strcpy(token_list,token);
    strcat(token_list,"-list");
    while (fgets(s,MAXLENGTH,fp) != NULL) {
      // sscanf(s,"%s = %n",&ss,&spos);
      sscanf(s,"%s = %n",ss,&spos);
      if (strcmp(ss,token_list) == 0) {
        found=1;
        sbuf=s+spos;
        for (i=0;i<=id;i++) {
          sscanf(sbuf,"%d%n",&a,&spos);
          sbuf+=spos;
        }
        break;
      }
    }
  }
  rewind(fp);

  if (found) {
    *A=a;
    fprintf(log,"Found %d\n",a);
  } else {
    *A=def;
    fprintf(log,"Default to %d\n",def);
  }
}


void read_parameter_d(real *A,FILE *fp,const char *token,real def,int id,FILE *log) {
  char s[MAXLENGTH],ss[MAXLENGTH],token_list[MAXLENGTH];
  char *sbuf;
  double a_d;
  real a;
  int i,spos;
  int found=0;

  fprintf(log,"Searching for %s [real]\n",token);

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    // sscanf(s,"%s = %lg",&ss,&a);
    sscanf(s,"%s = %lg",ss,&a_d);
    a=a_d;
    if (strcmp(ss,token) == 0) {
      found=1;
      break;
    }
  }
  rewind(fp);

  if (found==0) {
    strcpy(token_list,token);
    strcat(token_list,"-list");
    while (fgets(s,MAXLENGTH,fp) != NULL) {
      // sscanf(s,"%s = %n",&ss,&spos);
      sscanf(s,"%s = %n",ss,&spos);
      if (strcmp(ss,token_list) == 0) {
        found=1;
        sbuf=s+spos;
        for (i=0;i<=id;i++) {
          sscanf(sbuf,"%lg%n",&a_d,&spos);
          a=a_d;
          sbuf+=spos;
        }
        break;
      }
    }
  }
  rewind(fp);

  if (found) {
    *A=a;
    fprintf(log,"Found %g\n",a);
  } else {
    *A=def;
    fprintf(log,"Default to %g\n",def);
  }
}


void read_parameter_s(char **A,FILE *fp,const char *token,const char *def,int id,FILE *log) {
  char s[MAXLENGTH],ss[MAXLENGTH],token_list[MAXLENGTH];
  char *sbuf;
  char a[MAXLENGTH];
  int i,spos;
  int found=0;

  fprintf(log,"Searching for %s [string]\n",token);

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    // sscanf(s,"%s = %s",&ss,&a);
    sscanf(s,"%s = %s",ss,a);
    if (strcmp(ss,token) == 0) {
      found=1;
      break;
    }
  }
  rewind(fp);

  if (found==0) {
    strcpy(token_list,token);
    strcat(token_list,"-list");
    while (fgets(s,MAXLENGTH,fp) != NULL) {
      // sscanf(s,"%s = %n",&ss,&spos);
      sscanf(s,"%s = %n",ss,&spos);
      if (strcmp(ss,token_list) == 0) {
        found=1;
        sbuf=s+spos;
        for (i=0;i<=id;i++) {
          // sscanf(sbuf,"%s%n",&a,&spos);
          sscanf(sbuf,"%s%n",a,&spos);
          sbuf+=spos;
        }
        break;
      }
    }
  }
  rewind(fp);

  if (found) {
    strcpy(*A,a);
    fprintf(log,"Found to %s\n",a);
  } else {
    strcpy(*A,def);
    fprintf(log,"Default to %s\n",def);
  }
}

// Allocation/free functions
struct_parms* alloc_parms(int argc, char *argv[], FILE *log)
{
  struct_parms *parms;
  FILE *fp;
  char buffer[MAXLENGTH];
  char *bufferp=buffer;
  real gamma;

  int id=0;
  real T;
  real box;

  #ifndef NOMPI
  MPI_Comm_rank(MPI_COMM_WORLD,&id);
  #endif

  fp=fopen(argv[1],"r");
  if (fp==NULL) {
    char message[MAXLENGTH];
    sprintf(message,"input file %s could not be opened\n(Probably doesn't exist)\n",argv[1]);
    fatalerror(message);
  }

  parms=(struct_parms*) malloc(sizeof(struct_parms));

  read_parameter_i(&(parms->offset),fp,"offset",0,id,log);
  id+=parms->offset;
  parms->id=id;
  parms->phase=0; // Which phase, 0 is no previous checkpoint. Updated elsewhere

  read_parameter_d(&(parms->maxh),fp,"maxh",168000.0,id,log);

  read_parameter_d(&box,fp,"box",100,id,log);
  parms->arg_box=box;

  parms->kbgo=0;
  parms->arg_topfile=(char*) calloc(MAXLENGTH,sizeof(char));
  read_parameter_s(&(parms->arg_topfile),fp,"topfile","",id,log);
  parms->arg_grofile=(char*) calloc(MAXLENGTH,sizeof(char));
  read_parameter_s(&(parms->arg_grofile),fp,"grofile","",id,log);

  if (strcmp("",parms->arg_topfile)==0 && strcmp("",parms->arg_grofile)==0) {
    parms->kbgo=1;
    parms->arg_paramfile=(char*) calloc(MAXLENGTH,sizeof(char));
    read_parameter_s(&(parms->arg_topfile),fp,"kbgo_topfile","",id,log);
    read_parameter_s(&(parms->arg_paramfile),fp,"kbgo_paramfile","",id,log);
    read_parameter_s(&(parms->arg_grofile),fp,"kbgo_pdbfile","",id,log);
  }

  parms->arg_outdir=(char*) calloc(MAXLENGTH,sizeof(char));
  read_parameter_s(&(parms->arg_outdir),fp,"outdir","outfiles",id,log);
  mkdir(parms->arg_outdir,0777);
  read_parameter_d(&(parms->outputguard),fp,"outputguard",5e10,id,log);
  read_parameter_s(&(bufferp),fp,"trajfmt","xtc",id,log);
  if (strcmp(buffer,"gro")==0) {
    parms->trajfmt=0;
  } else if (strcmp(buffer,"xtc")==0) {
    parms->trajfmt=1;
  } else {
    char message[MAXLENGTH];
    sprintf(message,"Fatal Error: unrecognized trajfmt option %s.\nRecognized trajectory formats are gro or xtc\n",parms->trajfmt);
    fatalerror(message);
  }

  // Temperatures
  parms->kB=0.0083144; // Boltzmann Constant
  read_parameter_d(&T,fp,"ref_t",90,id,log);
  parms->kT=parms->kB*T; // System temperature (Kelvin)

  // Excluded Volume
  fprintf(stderr,"WARNING: excluded volume size is hardcoded to 0.596046e-09 (%s line %d)\n",__FILE__,__LINE__);
  parms->k12=0.596046e-09;

  // parms->s2[0][0]=pow(5.96046e-10,1.0/6.0);
  // parms->s2[0][1]=0.34*0.34;
  // parms->s2[1][0]=0.34*0.34;
  // parms->s2[1][1]=0.56*0.56;

  fprintf(stderr,"WARNING: LJ cutoff is hardcoded\n");
#warning "LJ cutoff hardcoded from 0.7 to 0.8 in alloc_parms"
  if (parms->kbgo==0) {
    parms->rcother=0.8; // 0.7; // nm - excluded volume force cutoff
  } else {
    parms->rcother=2.2; // 7e-6 errors for tryptophan
  }
  // parms->rcother=0.8; // nm - excluded volume force cutoff
  if (((int) (parms->arg_box/parms->rcother)) < 3) {
    char message[MAXLENGTH];
    sprintf(message,"box size %g is less than three times larger than\nexcluded volume cutoff of %g. Increase box size. It may be\nsafe to run with the box only twice as large as the cutoff if 4*N entries\nare allocated to checklist and shiftlist in alloc_nlist in neigh.c\n",parms->arg_box,parms->rcother);
    fatalerror(message);
  }

  // Run Control
  read_parameter_i(&(parms->t_output),fp,"nstxout",5000,id,log); // output frequency
#warning "Set neighbor search frequency to 10 in alloc_parms"
  parms->t_ns=10; // 20; // neighbor search frequency
  read_parameter_i(&(parms->steplim),fp,"nsteps",200000000,id,log); // Number of steps
  read_parameter_i(&(parms->stepmeta),fp,"nstepsmeta",100000000,id,log); // Number of steps of metadynamics

  // Leapfrog parameters
  // gamma=1e-5 works well for constantish energy simulations
  read_parameter_d(&(parms->dt),fp,"dt",0.002,id,log); // ps - timestep
  gamma=1; // ps^-1 - drag coefficient for atoms (60 ps^-1 in water)
  read_parameter_d(&gamma,fp,"gamma",gamma,id,log);
  parms->leapparms=alloc_leapparms(parms->dt,gamma,parms->kT);

  #warning "Hardcoded Qdrive drag coefficient"
  parms->Qleapparms=alloc_leapparms(parms->dt,0.01,parms->kT); // CONST

  // Biasing parameters
  read_parameter_b(&(parms->hessian),fp,"hessian",0,id,log); // Whether to compute orthogonal space force with Hessians
  read_parameter_i(&(parms->umbrella_function),fp,"umbrella_function",0,id,log);
  if (parms->umbrella_function<1 || parms->umbrella_function>3) {
    char message[MAXLENGTH];
    sprintf(message,"umbrella_function type set incorrectly. Use 1 (gaussian), 2(lorentzian) or 3 (sqrt lorentzian)\n");
    fatalerror(message);
  }
  read_parameter_d(&(parms->bias_A),fp,"bias_A",0.03,id,log);
  read_parameter_d(&(parms->bias_sigQ),fp,"bias_sigQ",2,id,log);
  read_parameter_d(&(parms->bias_sigFQ),fp,"bias_sigFQ",5,id,log);
  parms->arg_biasfile=(char*) calloc(MAXLENGTH,sizeof(char));
  read_parameter_s(&(parms->arg_biasfile),fp,"biasfile","",id,log);

  fclose(fp);

  return parms;
}


void free_parms(struct_parms* parms)
{
  free_leapparms(parms->leapparms);
  free(parms);
}

