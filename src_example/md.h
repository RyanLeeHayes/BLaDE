#ifndef MD_RNA_IONS_MD_H
#define MD_RNA_IONS_MD_H

#include <stdio.h>
#include "defines.h"

struct struct_parms;
struct struct_state;
struct struct_files;
struct struct_times;

typedef struct
{
  FILE *log;
  struct struct_parms *parms;
  struct struct_state *state;
  struct struct_files *files;
  struct struct_times *times;
} struct_md;

struct_md* alloc_md(int argc, char *argv[]);

void free_md(struct_md* md);

#endif

