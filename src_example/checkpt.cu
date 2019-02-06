
#include "checkpt.h"

#include "defines.h"

#include "md.h"
#include "atoms.h"
#include "files.h"
#include "parms.h"
#include "state.h"

#include "leapparms.h"
#include "sort.h"
#include "specific.h"
#include "unspecific.h"
#include "nblist.h"
#include "update.h"
#include "mersenne.h"


#include <stdio.h>

void printcheckpoint(struct_md *md)
{
  FILE *fp;
  char fnm[MAXLENGTH];
  struct_atoms *atoms;
  int N;
  real *buf=md->state->bufhost;

  sprintf(fnm,"%s/state.%d.%d.cpt",md->parms->arg_outdir,md->parms->id,md->parms->phase);
  fp=fopen(fnm,"w");
  fprintf(md->files->fps[eF_log],"Writing checkpoint file %s\n",fnm);

  fwrite(&(md->state->step),sizeof(int),1,fp);
  
  atoms=md->state->atoms;
  N=atoms->N;
  cudaMemcpy(buf,atoms->x,N*sizeof(real),cudaMemcpyDeviceToHost);
  fwrite(buf,sizeof(real),N,fp);
  cudaMemcpy(buf,atoms->v,N*sizeof(real),cudaMemcpyDeviceToHost);
  fwrite(buf,sizeof(real),N,fp);
  cudaMemcpy(buf,atoms->Vs_delay,N*sizeof(real),cudaMemcpyDeviceToHost);
  fwrite(buf,sizeof(real),N,fp);

  fclose(fp);
}


int readcheckpoint(struct_md *md)
{
  FILE *fp;
  char fnm[MAXLENGTH];
  struct_atoms *atoms;
  int N;
  real *buf=md->state->bufhost;
  
  md->parms->phase=0;
  sprintf(fnm,"%s/state.%d.%d.cpt",md->parms->arg_outdir,md->parms->id,md->parms->phase);
  fp=fopen(fnm,"r");
  while(fp != NULL) {
    fclose(fp);
    md->parms->phase++;
    sprintf(fnm,"%s/state.%d.%d.cpt",md->parms->arg_outdir,md->parms->id,md->parms->phase);
    fp=fopen(fnm,"r");
  }

  if (md->parms->phase>0) {
    sprintf(fnm,"%s/state.%d.%d.cpt",md->parms->arg_outdir,md->parms->id,md->parms->phase-1);
    fp=fopen(fnm,"r");
    // fprintf(md->files->fps[eF_log],"Reading checkpoint file %s\n",fnm);
    fprintf(stderr,"Reading checkpoint file %s\n",fnm);

    fread(&(md->state->step),sizeof(int),1,fp);
  
    atoms=md->state->atoms;
    N=atoms->N;
    fread(buf,sizeof(real),N,fp);
    cudaMemcpy(atoms->x,buf,N*sizeof(real),cudaMemcpyHostToDevice);
    float2fix <<< (N+BU-1)/BU, BU >>> (atoms[0]);
    fix2float <<< (N+BU-1)/BU, BU >>> (atoms[0]);
    fread(buf,sizeof(real),N,fp);
    cudaMemcpy(atoms->v,buf,N*sizeof(real),cudaMemcpyHostToDevice);
    fread(buf,sizeof(real),N,fp);
    cudaMemcpy(atoms->Vs_delay,buf,N*sizeof(real),cudaMemcpyHostToDevice);
 
    fclose(fp);
  }

  fprintf(stderr,"Running production\n");
  
  // Returns current phase
  return md->parms->phase;
}


void uploadkernelargs(struct_md* md)
{
  upload_bonded_d(
      md->parms->N_bond,
      md->parms->bondparms,
      md->parms->bondblock,
      md->parms->N_angle,
      md->parms->angleparms,
      md->parms->angleblock,
      md->parms->N_dih,
      md->parms->dihparms,
      md->parms->dihblock,
      md->parms->N_pair1,
      md->parms->pair1parms,
      md->parms->N_pair6,
      md->parms->pair6parms,
      md->state->atoms[0],
      md->state->box,
      md->state->Gt);
  upload_other_d(
      md->parms->k12,
      md->parms->rcother*md->parms->rcother,
      md->state->atoms[0],
      md->state->nblist_H4D[0],
      md->state->Gt);
  upload_update_d(md->state->atoms[0]);
  if (md->parms->kbgo!=0) {
    upload_kbgo_d(md->parms->nbparms);
  }
}

