#ifndef MD_RNA_IONS_CHECKPT_H
#define MD_RNA_IONS_CHECKPT_H

#include "defines.h"

#include "md.h"

void printcheckpoint(struct_md *md);

int readcheckpoint(struct_md *md);

void uploadkernelargs(struct_md* md);

#endif

