
#include "files.h"

#include "defines.h"
#include "atoms.h"
#include "md.h"
#include "state.h"
#include "parms.h"
#include "topol.h"
#include "vec.h"
#include "times.h"
#include "atoms.h"
#include "nblist.h"
#include "checkpt.h"
#include "version.h"
#include "xdr/xdrfile.h"
#include "xdr/xdrfile_xtc.h"
#include "error.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

struct_files* alloc_files(void)
{
  struct_files *files;
  int i;

  files=(struct_files*) malloc(sizeof(struct_files));
  files->N=eF_MAX;
  files->fps=(FILE**) calloc(files->N,sizeof(FILE*));
  for (i=0; i<files->N; i++) {
    files->fps[i]=NULL;
  }
  files->xtc=NULL;

  return files;
}


void free_files(struct_files* files)
{
  int i;
  for (i=0; i<files->N; i++) {
    if (files->fps[i] != NULL) {
      // if (i==eF_xtc) {
      //   xdrfile_close((XDRFILE*) files->fps[i]);
      // } else {
      fclose(files->fps[i]);
      // }
    }
  }
  if (files->xtc != NULL) {
    xdrfile_close(files->xtc);
  }
  free(files);
}


int count_gro(char *fnm)
{
  FILE *fp;
  char s[MAXLENGTH];
  int N;

  fp=fopen(fnm,"r");

  if (fp==NULL) {
    char message[MAXLENGTH];
    sprintf(message,"Fatal error: grofile %s does not exist\n",fnm);
    fatalerror(message);
  }

  fgets(s,MAXLENGTH,fp);
  fgets(s,MAXLENGTH,fp);
  sscanf(s,"%d",&N);

  fclose(fp);

  return N;
}


// Load gro
void read_gro(char *fnm,struct_state* state)
{
  FILE *fp;
  char s[MAXLENGTH];
  int i,N;
  double x_d,y_d,z_d;

  fp=fopen(fnm,"r");

  if (fp==NULL) {
    char message[MAXLENGTH];
    sprintf(message,"Fatal error: grofile %s does not exist\n",fnm);
    fatalerror(message);
  }

  fgets(s,MAXLENGTH,fp);
  fgets(s,MAXLENGTH,fp);
  sscanf(s,"%d",&N);

  for (i=0; i<N; i++) {
    fgets(s,MAXLENGTH,fp);
    // sscanf(s,"%5d%5c%5c%5d%8.3lg%8.3lg%8.3lg",
    sscanf(s,
           "%5d%5c%5c%5d%8lg%8lg%8lg",
           &(state->res_id[i]),&(state->res_name[i][0]),
           &(state->atom_name[i][0]),&(state->atom_id[i]),
           &x_d,&y_d,&z_d);

    state->xhost[DIM3*i]=x_d;
    state->xhost[DIM3*i+1]=y_d;
    state->xhost[DIM3*i+2]=z_d;
  }

  fclose(fp);
}


int count_pdb(char *fnm)
{
  FILE *fp;
  char s[MAXLENGTH];
  int N=0;

  fp=fopen(fnm,"r");

  if (fp==NULL) {
    char message[MAXLENGTH];
    sprintf(message,"Fatal error: kbgo_pdbfile %s does not exist\n",fnm);
    fatalerror(message);
  }

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (strncmp("ATOM",s,4)==0) {
      N++;
    }
  }

  fclose(fp);

  return N;
}


void read_pdb(char *fnm,struct_state* state)
{
  FILE *fp;
  char s[MAXLENGTH];
  int i=0;
  double x_d,y_d,z_d;
  real x,y,z;

  fp=fopen(fnm,"r");

  if (fp==NULL) {
    char message[MAXLENGTH];
    sprintf(message,"Fatal error: kbgo_pdbfile %s does not exist\n",fnm);
    fatalerror(message);
  }

  while (fgets(s,MAXLENGTH,fp) != NULL) {
    if (strncmp("ATOM",s,4)==0) {
      sscanf(s,
             "ATOM%7d%5c%5c%5d %8lg%8lg%8lg",
             &(state->atom_id[i]),&(state->atom_name[i][0]),
             &(state->res_name[i][0]),&(state->res_id[i]),
             &x_d,&y_d,&z_d);
      x=x_d;
      y=y_d;
      z=z_d;
      state->xhost[DIM3*i]=x/10.0;
      state->xhost[DIM3*i+1]=y/10.0;
      state->xhost[DIM3*i+2]=z/10.0;
      i++;
    }
  }

  fclose(fp);
}


static
void printgro(struct_md* md)
{
  FILE *fp=md->files->fps[eF_gro];
  real *x=md->state->xhost;
  real *box=md->state->boxhost;
  // char *typeR, *typeA;
  int *res_id=md->state->res_id;
  char (*res_name)[5]=md->state->res_name;
  int *atom_id=md->state->atom_id;
  char (*atom_name)[5]=md->state->atom_name;
  int i;
  int id;
  int phase=md->parms->phase;
  char fnm[100];

  if (fp == NULL) {
    id=md->parms->id;
    sprintf(fnm,"%s/traj.%d.%d.gro",md->parms->arg_outdir,id,phase);
    fp=fopen(fnm,"w");
    md->files->fps[eF_gro]=fp;
  }

  fprintf(fp,"Manning RNA trajectory, t= %g\n",md->state->step*md->parms->dt);
  // fprintf(fp,"Manning RNA trajectory (Gt= %g , Kt= %g , theta[0]= %g ), t= %g\n",md->state->Gt->master[0],md->state->Kt->master[0],md->state->qqlist->theta[0],md->state->step*md->parms->dt);
  fprintf(fp,"%d\n",md->state->N_all);
  for (i=0; i<md->state->N_all; i++) {
    fprintf(fp,"%5d%.5s%.5s%5d%8.3f%8.3f%8.3f\n",
      // "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n",
      res_id[i],res_name[i],atom_name[i],atom_id[i],
      x[DIM3*i],x[DIM3*i+1],x[DIM3*i+2]);
  }
  fprintf(fp,"   %g   %g   %g\n",box[0],box[1],box[2]);
}


void printgroframe(struct_md* md,const char* tag)
{
  FILE *fp;
  real *x=md->state->xhost;
  real *box=md->state->boxhost;
  // char *typeR, *typeA;
  int *res_id=md->state->res_id;
  char (*res_name)[5]=md->state->res_name;
  int *atom_id=md->state->atom_id;
  char (*atom_name)[5]=md->state->atom_name;
  int i;
  int id;
  int phase=md->parms->phase;
  char fnm[100];

  id=md->parms->id;
  sprintf(fnm,"%s/%s.%d.%d.gro",md->parms->arg_outdir,tag,id,phase);
  fp=fopen(fnm,"w");

  fprintf(fp,"Manning RNA trajectory, t= %g\n",md->state->step*md->parms->dt);
  // fprintf(fp,"Manning RNA trajectory (Gt= %g , Kt= %g , theta[0]= %g ), t= %g\n",md->state->Gt->master[0],md->state->Kt->master[0],md->state->qqlist->theta[0],md->state->step*md->parms->dt);
  fprintf(fp,"%d\n",md->state->N_all);
  for (i=0; i<md->state->N_all; i++) {
    fprintf(fp,"%5d%.5s%.5s%5d%8.3f%8.3f%8.3f\n",
      // "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n",
      res_id[i],res_name[i],atom_name[i],atom_id[i],
      x[DIM3*i],x[DIM3*i+1],x[DIM3*i+2]);
  }
  fprintf(fp,"   %g   %g   %g\n",box[0],box[1],box[2]);

  fclose(fp);
}


static
void printxtc(struct_md* md)
{
  // XDRFILE *fp=(XDRFILE*) md->files->fps[eF_xtc];
  XDRFILE *fp=md->files->xtc;
  real *x=md->state->xhost;
  rvec *fx=(rvec*) md->state->floatx;
  matrix box={{md->state->boxhost[0],0,0},{0,md->state->boxhost[1],0},{0,0,md->state->boxhost[2]}};
  int i,j,N;
  int id;
  int phase=md->parms->phase;
  char fnm[100];

  if (fp == NULL) {
    printgroframe(md,"confin");
    id=md->parms->id;
    sprintf(fnm,"%s/traj.%d.%d.xtc",md->parms->arg_outdir,id,phase);
    // fp=fopen(fnm,"w");
    fp=xdrfile_open(fnm,"w");
    // md->files->fps[eF_xtc]=(FILE*) fp;
    md->files->xtc=fp;
  }

  N=md->state->N_all;
  for (i=0; i<N; i++) {
    for (j=0; j<DIM3; j++) {
      fx[i][j]=(float) x[DIM3*i+j];
    }
  }
//  extern int write_xtc(XDRFILE *xd,
//                       int natoms,int step,real time,
//                       matrix box,rvec *x,real prec);
  write_xtc(fp,N,md->state->step,(float) (md->state->step*md->parms->dt),
    box,fx,1000.0);
}


/*void printxyz(struct_md* md)
{
  FILE *fp=md->files->fp_xyz;
  real *x=md->state->x;
  int i;

  if (fp == NULL) {
    fp=fopen("pos.xyz","w");
    md->files->fp_xyz=fp;
  }

  fprintf(fp,"%d\nComment - Frame %d\n",md->state->N_all,md->state->step);
  for (i=0; i<DIM3*md->state->N_all; i+=3) {
    fprintf(fp,"P %g %g %g\n",x[i],x[i+1],x[i+2]);
  }

  // fclose(fp); is called in free_files
}*/

/*
// printdata(md->state->qqlist->theta,md->state->qqlist->nqlist_rna,"theta",&(md->files->fp_theta))
static
void printdata(struct_md *md,real* data,int N,const char* s,FILE** fp)
{
  int i, id;
  int phase=md->parms->phase;
  char fnm[100];

  if (*fp == NULL) {
    id=md->parms->id;
    sprintf(fnm,"%s/%s.%d.%d.dat",md->parms->arg_outdir,s,id,phase);
    *fp=fopen(fnm,"w");
  }

  for (i=0; i<N; i++) {
    fprintf(*fp," %12g",data[i]);
  }
  fprintf(*fp,"\n");
}
*/

static
void printdatadevice(struct_md *md,real* data,real* datahost,int N,const char* s,FILE** fp)
{
  int i, id;
  int phase=md->parms->phase;
  char fnm[100];

  if (*fp == NULL) {
    id=md->parms->id;
    sprintf(fnm,"%s/%s.%d.%d.dat",md->parms->arg_outdir,s,id,phase);
    *fp=fopen(fnm,"w");
  }

  cudaMemcpy(datahost,data,N*sizeof(real),cudaMemcpyDeviceToHost);

  for (i=0; i<N; i++) {
    fprintf(*fp," %12g",datahost[i]);
  }
  fprintf(*fp,"\n");
}


void printheader(struct_md* md)
{
  int id;
  int phase=md->parms->phase;
  char fnm[100];
  
  id=md->parms->id;
  sprintf(fnm,"%s/md.%d.%d.log",md->parms->arg_outdir,id,phase);
  md->files->fps[eF_log]=fopen(fnm,"w");

  printversion(md->files->fps[eF_log]);
  md->times->secpcyc=gmx_cycles_calibrate(1.0);
  fprintf(md->files->fps[eF_log],"1 second took %g cycles\n",1.0/md->times->secpcyc);
  md->parms->maxc=(gmx_cycles_t) (0.99*3600.0*md->parms->maxh/md->times->secpcyc);

  fprintf(md->files->fps[eF_log],"grofile = %s\n",md->parms->arg_grofile);
  fprintf(md->files->fps[eF_log],"topfile = %s\n",md->parms->arg_topfile);
  fprintf(md->files->fps[eF_log],"outdir = %s\n",md->parms->arg_outdir);
  fprintf(md->files->fps[eF_log],"box = %g nm\n",md->parms->arg_box);
  fprintf(md->files->fps[eF_log],"Temperature = %g\n",md->parms->kT/md->parms->kB);

  if (md->parms->phase > 0) {
    fprintf(md->files->fps[eF_log],"Read checkpoint file %s/state.%d.%d.cpt\n",md->parms->arg_outdir,md->parms->id,md->parms->phase-1);
  } else {
    fprintf(md->files->fps[eF_log],"Found no checkpoint file, starting a new run.\n");
  }
}


void printoutput(struct_md* md)
{
  FILE **fps=md->files->fps;
  real Q,EQ;

  md->times->start=gmx_cycles_read();

  cudaMemcpy(md->state->xhost, md->state->atoms->x, md->state->atoms->N*sizeof(real), cudaMemcpyDeviceToHost);
  cudaMemcpy(md->state->boxhost, md->state->box, DIM3*sizeof(real), cudaMemcpyDeviceToHost);

  if (md->parms->trajfmt==0) {
    printgro(md);
  } else {
    printxtc(md);
  }
  fprintf(fps[eF_log],"Step %d\n",md->state->step);

  printdatadevice(md,md->state->Gt,md->state->Gthost,1,"Gt",&(fps[eF_Gt]));
  printdatadevice(md,md->state->Kt,md->state->Kthost,1,"Kt",&(fps[eF_Kt]));

  if (md->parms->umbrella) {
    printdatadevice(md,md->parms->umbrella->Q,&Q,1,"Q",&(fps[eF_Q]));
    printdatadevice(md,md->parms->umbrella->EQ,&EQ,1,"EQ",&(fps[eF_EQ]));
  }

  md->times->write+=(gmx_cycles_read()-md->times->start);
}


void printsummary(struct_md* md,gmx_cycles_t start)
{
  gmx_cycles_t stop;
  FILE *fp=md->files->fps[eF_log];

  stop=gmx_cycles_read();
  fprintf(fp,"Took %lld cycles\n",stop-start);
  fprintf(fp,"Took %g second\n",(stop-start)*md->times->secpcyc);

  fprintf(fp,"Neighbor search %lld cycles\n",md->times->nsearch);
  fprintf(fp,"Force calculation %lld cycles\n",md->times->force);
    fprintf(fp,"  Bond  Force calculation %lld cycles\n",md->times->force_bond);
    fprintf(fp,"  Angle Force calculation %lld cycles\n",md->times->force_angle);
    fprintf(fp,"  Dih   Force calculation %lld cycles\n",md->times->force_dih);
    fprintf(fp,"  Pair  Force calculation %lld cycles\n",md->times->force_pair);
    fprintf(fp,"  Elec  Force calculation %lld cycles\n",md->times->force_elec);
    fprintf(fp,"  Other Force calculation %lld cycles\n",md->times->force_other);
    fprintf(fp,"  Sum   Force calculation %lld cycles\n",md->times->force_sum);
  fprintf(fp,"Update %lld cycles\n",md->times->update);
  fprintf(fp,"Writing files %lld cycles\n",md->times->write);

  fprintf(fp,"No longer printing neighbor searching timing details. See %s, line %d to add them.\n",__FILE__,__LINE__);
  fprintf(fp,"nblist_receive %16lld\n",md->times->nblist_receive);
  fprintf(fp,"nblist_bin     %16lld\n",md->times->nblist_bin);
  fprintf(fp,"nblist_block   %16lld\n",md->times->nblist_block);
  fprintf(fp,"nblist_tile    %16lld\n",md->times->nblist_tile);
  fprintf(fp,"nblist_check   %16lld\n",md->times->nblist_check);
  fprintf(fp,"nblist_send    %16lld\n",md->times->nblist_send);

  printcheckpoint(md);
}

