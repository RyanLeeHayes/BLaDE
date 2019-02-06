#ifndef MD_RNA_IONS_FILES_H
#define MD_RNA_IONS_FILES_H

#include "defines.h"

#include "xdr/xdrfile.h"

#include <stdio.h>

typedef enum {
  eF_log,
  eF_gro,
//  eF_xtc,
  eF_Gt,
  eF_Gts,
  eF_Kt,
  eF_Q,
  eF_EQ,
  eF_debug1,
  eF_debug2,
//  eF_loclog,
//  eF_locend=eF_loclog+7,
// Example usage:
// fprintf(md->files->fps[eF_loclog+omp_get_thread_num()],"Node %d, step %d, cycle %lld, before barrier, line %d\n",omp_get_thread_num(),md->state->step,gmx_cycles_read(),__LINE__);
  eF_MAX
} eFILE;

typedef struct struct_files
{
  FILE **fps;
  XDRFILE *xtc;
  int N;
} struct_files;

struct_files* alloc_files(void);

void free_files(struct_files* files);

#include "state.h"


int count_gro(char *fnm);
void read_gro(char *fnm,struct_state* state);

int count_pdb(char *fnm);
void read_pdb(char *fnm,struct_state* state);

#include "version.h"
#include "md.h"
#include "times.h"

void printgroframe(struct_md* md,char* tag);

void printheader(struct_md* md);

// void printmoreheader(struct_md *md);

void printoutput(struct_md* md);

void printsummary(struct_md* md,gmx_cycles_t start);

#endif
